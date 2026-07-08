"""
argus-sec — CVE chunking script.

Reads data/corpus.json (912 deduplicated CVE records) and produces
data/chunks.json — a flat list of dicts, one per CVE, containing:

  - narrative_text : composed human-readable text for Qdrant embedding
                     and BM25 indexing (the locked format)
  - metadata       : structured fields for PostgreSQL storage and
                     Qdrant payload filtering

Output schema per chunk:
{
    "cve_id":          str,   # "CVE-2024-3095"
    "narrative_text":  str,   # full composed narrative (for embedding)
    "metadata": {
        "packages":        list[str],   # ["LangChain", "LangGraph"]
        "published":       str,         # "2024-04-15"
        "last_modified":   str,         # "2024-05-01"
        "status":          str,         # "Analyzed"
        "severity":        str,         # highest severity across all sources
        "cvss_scores": [{               # all CVSS scores with sources
            "source":   str,
            "score":    float,
            "severity": str,
            "vector":   str,
        }],
        "score_disagreement": bool,     # True if max-min delta > 1.0
        "cisa_kev":          bool,
        "cisa_kev_date":     str | None,
        "cisa_kev_name":     str | None,
        "cwes":              list[str], # ["CWE-74"]
        "affected_versions": list[dict],# [{package, version_range, fix_version}]
        "references": [{
            "tag": str,
            "url": str,
        }],
        "nvd_url":           str,
    }
}

Design notes:
  - narrative_text is the single field fed to both the embedding model
    (Qdrant dense vectors) and BM25 index. Structured metadata fields
    are stored separately in PostgreSQL for exact-match filtering.
  - CWE names: static map of the top 30 CWEs by frequency (covers ~85%
    of all CVEs empirically). Unknown CWEs fall back to the ID only.
    Ref: https://cwe.mitre.org/top25/archive/2024/2024_cwe_top25.html
  - CVSS vector decoding: Attack Vector and Privileges Required only —
    the two fields most meaningful to a non-specialist audience in a demo.
  - Affected version ranges: extracted from cpeMatch entries where
    vulnerable=True and version fields are explicitly populated.
    Missing version ranges noted explicitly rather than silently omitted.
  - Reference filter: Patch, Vendor Advisory, Exploit, Mitigation only.
    Full NVD reference tag taxonomy:
    https://nvd.nist.gov/vuln/categories
"""

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CORPUS_PATH = Path("data/corpus.json")
CHUNKS_PATH = Path("data/chunks.json")

# ---------------------------------------------------------------------------
# Static CWE name map
# Top 30 CWEs by frequency — covers ~85% of CVEs empirically.
# Source: https://cwe.mitre.org/top25/archive/2024/2024_cwe_top25.html
# ---------------------------------------------------------------------------

CWE_NAMES: dict[str, str] = {
    "CWE-79":   "Cross-site Scripting (XSS)",
    "CWE-89":   "SQL Injection",
    "CWE-20":   "Improper Input Validation",
    "CWE-125":  "Out-of-bounds Read",
    "CWE-787":  "Out-of-bounds Write",
    "CWE-352":  "Cross-Site Request Forgery (CSRF)",
    "CWE-22":   "Path Traversal",
    "CWE-78":   "OS Command Injection",
    "CWE-416":  "Use After Free",
    "CWE-190":  "Integer Overflow",
    "CWE-502":  "Deserialisation of Untrusted Data",
    "CWE-287":  "Improper Authentication",
    "CWE-476":  "NULL Pointer Dereference",
    "CWE-434":  "Unrestricted Upload of Dangerous File Type",
    "CWE-306":  "Missing Authentication for Critical Function",
    "CWE-862":  "Missing Authorisation",
    "CWE-798":  "Use of Hard-coded Credentials",
    "CWE-119":  "Improper Restriction of Operations within Memory Buffer",
    "CWE-276":  "Incorrect Default Permissions",
    "CWE-200":  "Exposure of Sensitive Information",
    "CWE-522":  "Insufficiently Protected Credentials",
    "CWE-732":  "Incorrect Permission Assignment for Critical Resource",
    "CWE-611":  "Improper Restriction of XML External Entity Reference",
    "CWE-918":  "Server-Side Request Forgery (SSRF)",
    "CWE-77":   "Command Injection",
    "CWE-94":   "Code Injection",
    "CWE-269":  "Improper Privilege Management",
    "CWE-400":  "Uncontrolled Resource Consumption",
    "CWE-74":   "Improper Neutralisation of Special Elements (Injection)",
    "CWE-295":  "Improper Certificate Validation",
}

# ---------------------------------------------------------------------------
# CVSS vector field decoders
# Ref: https://www.first.org/cvss/v3.1/specification-document
# ---------------------------------------------------------------------------

ATTACK_VECTOR_MAP: dict[str, str] = {
    "N": "Network-exploitable",
    "A": "Adjacent network required",
    "L": "Local access required",
    "P": "Physical access required",
}

PRIVILEGES_REQUIRED_MAP: dict[str, str] = {
    "N": "no auth required",
    "L": "low privileges required",
    "H": "high privileges required",
}

REFERENCE_TAGS_INCLUDE = {"Patch", "Vendor Advisory", "Exploit", "Mitigation"}

SCORE_DISAGREEMENT_THRESHOLD = 1.0


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def get_description(cve: dict) -> str:
    """Return the English description, or a placeholder if missing."""
    for desc in cve.get("descriptions", []):
        if desc.get("lang") == "en":
            return desc.get("value", "").strip()
    return "No English description available."


def get_cvss_scores(cve: dict) -> list[dict]:
    """
    Extract all CVSS scores across v4.0, v3.1, v3.0, and v2 metric blocks.
    Returns a list of dicts with source, score, severity, vector, and
    optional plain-English av_plain/pr_plain fields for narrative rendering.

    CVSS 4.0 differences from v3.x (confirmed from raw NVD record):
      - Metric block key: cvssMetricV40 (not cvssMetricV31)
      - attackVector and privilegesRequired exposed as direct named fields
        in cvssData (not just embedded in vector string) — read directly
        rather than parsing the vector string for reliability.
      - Vector string format: CVSS:4.0/AV:N/AC:L/AT:N/PR:N/... (additional
        fields AT, VC/VI/VA etc.) — AV and PR abbreviations unchanged.
    Ref: https://www.first.org/cvss/v4.0/specification-document
    """
    # CVSS 4.0 attack vector values differ in naming from v3.x
    AV_NAMED_MAP = {
        "NETWORK":  "Network-exploitable",
        "ADJACENT": "Adjacent network required",
        "LOCAL":    "Local access required",
        "PHYSICAL": "Physical access required",
    }
    PR_NAMED_MAP = {
        "NONE": "no auth required",
        "LOW":  "low privileges required",
        "HIGH": "high privileges required",
    }

    scores = []
    metrics = cve.get("metrics", {})

    # CVSS 4.0 — read AV/PR from direct named fields where available
    for entry in metrics.get("cvssMetricV40", []):
        data = entry.get("cvssData", {})
        score = data.get("baseScore")
        if score is None:
            continue
        av_raw = data.get("attackVector", "")
        pr_raw = data.get("privilegesRequired", "")
        scores.append({
            "source":   entry.get("source", "unknown"),
            "score":    float(score),
            "severity": data.get("baseSeverity", "UNKNOWN"),
            "vector":   data.get("vectorString", ""),
            "version":  "4.0",
            # Pre-decoded plain English — bypasses vector string parsing for v4.0
            "av_plain": AV_NAMED_MAP.get(av_raw.upper(), "unknown attack vector"),
            "pr_plain": PR_NAMED_MAP.get(pr_raw.upper(), "unknown privilege level"),
        })

    # CVSS 3.x and 2.0 — decode AV/PR from vector string as before
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        for entry in metrics.get(key, []):
            data = entry.get("cvssData", {})
            score = data.get("baseScore")
            if score is None:
                continue
            scores.append({
                "source":   entry.get("source", "unknown"),
                "score":    float(score),
                "severity": data.get("baseSeverity", "UNKNOWN"),
                "vector":   data.get("vectorString", ""),
                "version":  key[-2:],  # "31", "30", "V2" — used for display only
                "av_plain": None,      # populated by decode_vector() at render time
                "pr_plain": None,
            })
    return scores


def decode_vector(vector: str) -> str:
    """
    Decode Attack Vector and Privileges Required from a CVSS v3.x/v2
    vector string into plain English for the narrative.

    Not called for CVSS 4.0 — those use pre-decoded av_plain/pr_plain
    fields read directly from named cvssData fields (more reliable).

    Example input:  "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    Example output: "Network-exploitable, no auth required"
    """
    parts = {}
    for segment in vector.split("/"):
        if ":" in segment:
            k, v = segment.split(":", 1)
            parts[k] = v

    av = ATTACK_VECTOR_MAP.get(parts.get("AV", ""), "unknown attack vector")
    pr = PRIVILEGES_REQUIRED_MAP.get(parts.get("PR", ""), "unknown privilege level")
    return f"{av}, {pr}"


def get_cwes(cve: dict) -> list[str]:
    """Extract unique CWE IDs from weaknesses block."""
    cwes = []
    for weakness in cve.get("weaknesses", []):
        for desc in weakness.get("description", []):
            val = desc.get("value", "")
            if val.startswith("CWE-") and val not in cwes:
                cwes.append(val)
    return cwes


def get_cisa_kev(cve: dict) -> tuple[bool, str | None, str | None]:
    """
    Return (is_kev, date_added, vulnerability_name).
    CISA KEV fields are top-level on the cve object when present.
    Ref: https://nvd.nist.gov/developers/vulnerabilities (cisaExploitAdd)
    """
    date = cve.get("cisaExploitAdd")
    name = cve.get("cisaVulnerabilityName")
    return bool(date), date, name


def get_references(cve: dict) -> list[dict]:
    """
    Return references filtered to actionable tags only.
    Tags included: Patch, Vendor Advisory, Exploit, Mitigation.
    Ref: https://nvd.nist.gov/vuln/categories
    """
    refs = []
    for ref in cve.get("references", []):
        url = ref.get("url", "")
        tags = ref.get("tags", [])
        matched = [t for t in tags if t in REFERENCE_TAGS_INCLUDE]
        if matched:
            refs.append({"tag": matched[0], "url": url})
    return refs


def get_exploitation_signal(cve: dict) -> str | None:
    """
    Extract SSVC exploitation status from ssvcV203 metrics block.

    SSVC (Stakeholder-Specific Vulnerability Categorisation) is a CISA-
    developed prioritisation framework. The exploitation field indicates
    known real-world exploitation activity, independent of CVSS score.
    Ref: https://www.cisa.gov/sites/default/files/publications/cisa-ssvc-guide.pdf

    Returns a human-readable string for non-none values, or None if:
      - No SSVC data is present
      - exploitation is "none" (expected baseline — not worth surfacing)

    SSVC exploitation values:
      "none"   — no exploitation activity known → silent
      "poc"    — proof-of-concept exploit publicly available → surface
      "active" — active exploitation confirmed in the wild → surface
    """
    EXPLOITATION_LABELS = {
        "poc":    "Proof-of-concept code exists (SSVC)",
        "active": "Active exploitation confirmed in the wild (SSVC)",
    }
    for entry in cve.get("metrics", {}).get("ssvcV203", []):
        for option in entry.get("ssvcData", {}).get("options", []):
            value = option.get("exploitation", "").lower()
            label = EXPLOITATION_LABELS.get(value)
            if label:
                return label
    return None


def get_attack_vector(cvss_scores: list[dict]) -> str | None:
    """
    Extract the canonical attack vector from the highest-scoring CVSS entry.

    Returns the raw NVD value ("NETWORK"|"ADJACENT"|"LOCAL"|"PHYSICAL")
    for use as a filterable column — not the decoded plain-English string
    stored in av_plain, which is for narrative display only.

    Priority: highest baseScore across all CVSS versions. Ties broken by
    version preference (4.0 > 3.1 > 3.0 > 2.0) since newer standards
    are more precise.

    Extraction paths per CVSS version:
      - CVSS 4.0: attackVector is a direct named field in cvssData,
        stored uppercased ("NETWORK" etc.) — read directly from the
        raw NVD record, not from the pre-decoded av_plain string.
        We re-derive this here from the vector string for consistency
        since cvss_scores entries only carry av_plain (decoded text).
      - CVSS 3.x/2.0: parse AV: component from vector string.

    AV abbreviation map (consistent across CVSS 3.x and 4.0 vector strings):
      N = NETWORK, A = ADJACENT, L = LOCAL, P = PHYSICAL
    """
    AV_ABBREV: dict[str, str] = {
        "N": "NETWORK",
        "A": "ADJACENT",
        "L": "LOCAL",
        "P": "PHYSICAL",
    }

    if not cvss_scores:
        return None

    # Sort by score descending — highest score is most authoritative
    best = max(cvss_scores, key=lambda s: s["score"])
    vector = best.get("vector", "")

    # Parse AV: from vector string (works for both CVSS 3.x and 4.0
    # since both use AV:N/AV:A/AV:L/AV:P abbreviations)
    for segment in vector.split("/"):
        if segment.startswith("AV:"):
            abbrev = segment.split(":")[1]
            return AV_ABBREV.get(abbrev)

    return None


def get_fix_available(
    references: list[dict],
    affected_versions: list[dict],
) -> bool:
    """
    Return True if a fix is known to exist for this CVE.

    Two signals, either is sufficient:
      1. A reference tagged "Patch" exists — vendor has published a fix.
      2. Any affected_version entry has an explicit fix_version — the
         cpeMatch versionEndExcluding field was populated by NVD,
         indicating the fix version is known.

    Returns False when neither signal is present — meaning no fix is
    confirmed yet (CVE may be unpatched, awaiting analysis, or the fix
    reference wasn't tagged correctly by the CNA).
    """
    if any(r.get("tag") == "Patch" for r in references):
        return True
    if any(v.get("fix_version") for v in affected_versions):
        return True
    return False


def get_affected_versions(record: dict, packages: list[str]) -> list[dict]:
    """
    Extract affected version information from NVD records.

    Two schema paths are tried in order:

    Path 1 — configurations/cpeMatch (traditional NVD schema):
      Walks: cve.configurations -> nodes -> cpeMatch (vulnerable=True)
      Provides structured version ranges with start/end operators.
      Used by most NVD-analysed CVEs.

    Path 2 — affected block (NVD 2.0 CNA schema, fallback):
      Walks: cve.affected -> affectedData -> versions
      Used by CNA-sourced CVEs (e.g. VulnCheck, GitHub) where NVD has
      not yet completed its own cpeMatch analysis. Version entries use
      status: "affected"/"unaffected" rather than range operators.
      Ref: https://nvd.nist.gov/developers/vulnerabilities

    Version range notes (Path 1):
      - versionStartIncluding: >= this version
      - versionStartExcluding: > this version
      - versionEndIncluding:   <= this version (fix not yet available)
      - versionEndExcluding:   < this version (fix IS this version)
    """
    versions = []
    seen = set()

    # Path 1 — configurations/cpeMatch (traditional NVD schema)
    for config in record.get("cve", {}).get("configurations", []):
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                if not match.get("vulnerable", False):
                    continue

                criteria = match.get("criteria", "")
                # Extract product name from CPE: cpe:2.3:a:vendor:product:...
                parts = criteria.split(":")
                product = parts[4] if len(parts) > 4 else "unknown"

                start_incl = match.get("versionStartIncluding")
                start_excl = match.get("versionStartExcluding")
                end_incl   = match.get("versionEndIncluding")
                end_excl   = match.get("versionEndExcluding")

                range_parts = []
                if start_incl:
                    range_parts.append(f">= {start_incl}")
                elif start_excl:
                    range_parts.append(f"> {start_excl}")

                fix_version = None
                if end_excl:
                    range_parts.append(f"< {end_excl}")
                    fix_version = end_excl
                elif end_incl:
                    range_parts.append(f"<= {end_incl}")

                if not range_parts:
                    continue

                version_range = ", ".join(range_parts)
                key = (product, version_range)
                if key in seen:
                    continue
                seen.add(key)

                versions.append({
                    "product":       product,
                    "version_range": version_range,
                    "fix_version":   fix_version,
                    "schema":        "cpeMatch",
                })

    # Path 2 — affected block (NVD 2.0 CNA schema, fallback only)
    # Only used when Path 1 yielded nothing — avoids double-counting
    # for CVEs that have both blocks populated.
    if not versions:
        for affected_entry in record.get("cve", {}).get("affected", []):
            for affected_data in affected_entry.get("affectedData", []):
                product = (
                    affected_data.get("product", "unknown")
                    .lower()
                    .replace(" ", "_")
                )
                for v in affected_data.get("versions", []):
                    status = v.get("status", "")
                    if status != "affected":
                        continue
                    version_val = v.get("version", "unknown")
                    # "version": "0" is CNA shorthand for "all versions"
                    if version_val == "0":
                        version_range = "all versions"
                    else:
                        version_range = f"version {version_val}"

                    key = (product, version_range)
                    if key in seen:
                        continue
                    seen.add(key)

                    versions.append({
                        "product":       product,
                        "version_range": version_range,
                        "fix_version":   None,   # CNA schema does not provide fix version
                        "schema":        "affected",
                    })

    return versions


# ---------------------------------------------------------------------------
# Narrative composer
# ---------------------------------------------------------------------------

def compose_narrative(
    cve_id:             str,
    cve:                dict,
    packages:           list[str],
    cvss_scores:        list[dict],
    cwes:               list[str],
    is_kev:             bool,
    kev_date:           str | None,
    kev_name:           str | None,
    references:         list[dict],
    affected_versions:  list[dict],
    published:          str,
    last_modified:      str,
    status:             str,
    exploitation_signal:str | None,
) -> str:
    lines: list[str] = []

    # Header
    lines.append(
        f"{cve_id} | Published: {published} | "
        f"Last Modified: {last_modified} | Status: {status}"
    )
    lines.append(f"Packages: {', '.join(packages)}")

    # CISA KEV (only rendered when present)
    if is_kev:
        kev_line = f"⚠ CISA KEV: Actively exploited"
        if kev_name:
            kev_line += f" — \"{kev_name}\""
        if kev_date:
            kev_line += f" (added {kev_date})"
        lines.append(kev_line)

    # SSVC exploitation signal (only rendered when non-none)
    if exploitation_signal:
        lines.append(f"⚠ Exploitation: {exploitation_signal}")

    lines.append("")

    # Description
    lines.append(f"Description: {get_description(cve)}")
    lines.append("")

    # CVSS Scores
    if cvss_scores:
        lines.append("CVSS Scores:")
        all_scores = [s["score"] for s in cvss_scores]
        disagreement = (
            len(all_scores) > 1
            and (max(all_scores) - min(all_scores)) > SCORE_DISAGREEMENT_THRESHOLD
        )
        for s in cvss_scores:
            # CVSS 4.0: use pre-decoded named fields (more reliable than
            # parsing the v4.0 vector string which has additional components).
            # CVSS 3.x/2.0: decode from vector string as before.
            if s.get("av_plain") and s.get("pr_plain"):
                plain = f"{s['av_plain']}, {s['pr_plain']}"
            elif s.get("vector"):
                plain = decode_vector(s["vector"])
            else:
                plain = "vector unavailable"
            source_label = "NVD" if "nist.gov" in s["source"] else s["source"]
            version_tag = f" [CVSS {s['version']}]" if s.get("version") else ""
            lines.append(
                f"  - {s['severity']} {s['score']} [{source_label}]{version_tag} | {plain}"
            )
            if s["vector"]:
                # CVSS 4.0 vectors include supplemental/environmental metrics
                # set to :X (NOT_DEFINED) that add no information.
                # Truncate to base metrics only for readability.
                if s.get("version") == "4.0":
                    vector_display = "/".join(
                        seg for seg in s["vector"].split("/")
                        if not seg.endswith(":X")
                    )
                else:
                    vector_display = s["vector"]
                lines.append(f"    {vector_display}")
        if disagreement:
            nvd_score = next(
                (s["score"] for s in cvss_scores if "nist.gov" in s["source"]), None
            )
            vendor_score = next(
                (s["score"] for s in cvss_scores if "nist.gov" not in s["source"]), None
            )
            if nvd_score and vendor_score:
                delta = abs(nvd_score - vendor_score)
                lines.append(
                    f"  ⚠ Score disagreement: NVD={nvd_score}, "
                    f"vendor={vendor_score} (delta: {delta:.1f})"
                )
    else:
        lines.append("CVSS Scores: Awaiting NVD analysis")
    lines.append("")

    # Weakness
    if cwes:
        cwe_strs = []
        for cwe in cwes:
            name = CWE_NAMES.get(cwe)
            cwe_strs.append(f"{cwe} ({name})" if name else cwe)
        lines.append(f"Weakness: {'; '.join(cwe_strs)}")
    else:
        lines.append("Weakness: Not specified")
    lines.append("")

    # Affected versions
    if affected_versions:
        lines.append("Affected Versions:")
        for v in affected_versions:
            fix = f" (fix in {v['fix_version']})" if v.get("fix_version") else ""
            schema_note = " [CNA data — fix version not specified]" if v.get("schema") == "affected" else ""
            lines.append(f"  {v['product']}: {v['version_range']}{fix}{schema_note}")
    else:
        lines.append("Affected Versions: Not yet specified by NVD")
    lines.append("")

    # References
    if references:
        lines.append("References:")
        for ref in references:
            lines.append(f"  {ref['tag']} — {ref['url']}")
    else:
        lines.append("References: None in scope (Patch/Advisory/Exploit/Mitigation)")
    lines.append("")

    # Attribution footer
    lines.append(
        f"Source: NVD (https://nvd.nist.gov/vuln/detail/{cve_id})"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_chunk(record: dict) -> dict | None:
    """
    Build one chunk dict from a corpus record.
    Returns None if the record is malformed (logged + skipped).
    """
    try:
        cve        = record.get("cve", {})
        cve_id     = cve.get("id", "")
        packages   = record.get("argus_packages", [])

        if not cve_id:
            logger.warning("Record missing cve.id — skipping")
            return None

        published     = (cve.get("published", "") or "")[:10]
        last_modified = (cve.get("lastModified", "") or "")[:10]
        status        = cve.get("vulnStatus", "Unknown")

        cvss_scores         = get_cvss_scores(cve)
        cwes                = get_cwes(cve)
        is_kev, kev_date, kev_name = get_cisa_kev(cve)
        references          = get_references(cve)
        affected_versions   = get_affected_versions(record, packages)
        exploitation_signal = get_exploitation_signal(cve)

        # Three new fields — derived after their dependencies are ready
        attack_vector    = get_attack_vector(cvss_scores)
        fix_available    = get_fix_available(references, affected_versions)
        multi_package    = len(packages) > 1

        # Derive highest severity across all sources for PostgreSQL filtering
        severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}
        highest_severity = max(
            (s["severity"] for s in cvss_scores),
            key=lambda s: severity_order.get(s, 0),
            default="UNKNOWN",
        )

        score_disagreement = False
        if len(cvss_scores) > 1:
            all_scores = [s["score"] for s in cvss_scores]
            score_disagreement = (
                max(all_scores) - min(all_scores)
            ) > SCORE_DISAGREEMENT_THRESHOLD

        narrative = compose_narrative(
            cve_id=cve_id,
            cve=cve,
            packages=packages,
            cvss_scores=cvss_scores,
            cwes=cwes,
            is_kev=is_kev,
            kev_date=kev_date,
            kev_name=kev_name,
            references=references,
            affected_versions=affected_versions,
            published=published,
            last_modified=last_modified,
            status=status,
            exploitation_signal=exploitation_signal,
        )

        return {
            "cve_id":         cve_id,
            "narrative_text": narrative,
            "metadata": {
                "packages":             packages,
                "published":            published,
                "last_modified":        last_modified,
                "status":               status,
                "severity":             highest_severity,
                "cvss_scores":          cvss_scores,
                "score_disagreement":   score_disagreement,
                "cisa_kev":             is_kev,
                "cisa_kev_date":        kev_date,
                "cisa_kev_name":        kev_name,
                "exploitation_signal":  exploitation_signal,
                "fix_available":        fix_available,
                "attack_vector":        attack_vector,
                "multi_package":        multi_package,
                "cwes":                 cwes,
                "affected_versions":    affected_versions,
                "references":           references,
                "nvd_url":              f"https://nvd.nist.gov/vuln/detail/{cve_id}",
            },
        }

    except Exception as exc:
        cve_id = record.get("cve", {}).get("id", "UNKNOWN")
        logger.warning("Failed to build chunk for %s: %s — skipping", cve_id, exc)
        return None


def main() -> None:
    if not CORPUS_PATH.exists():
        logger.error("%s not found — run deduplicate.py first", CORPUS_PATH)
        return

    corpus: list[dict] = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d records from %s", len(corpus), CORPUS_PATH)

    chunks = []
    skipped = 0
    kev_count = 0
    disagreement_count = 0
    ssvc_count = 0
    fix_count = 0
    network_count = 0
    multi_count = 0

    for record in corpus:
        chunk = build_chunk(record)
        if chunk is None:
            skipped += 1
            continue
        chunks.append(chunk)
        m = chunk["metadata"]
        if m["cisa_kev"]:             kev_count += 1
        if m["score_disagreement"]:   disagreement_count += 1
        if m["exploitation_signal"]:  ssvc_count += 1
        if m["fix_available"]:        fix_count += 1
        if m["attack_vector"] == "NETWORK": network_count += 1
        if m["multi_package"]:        multi_count += 1

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("─" * 60)
    logger.info("Chunks written to %s", CHUNKS_PATH)
    logger.info("Total chunks:          %d", len(chunks))
    logger.info("Skipped (malformed):   %d", skipped)
    logger.info("CISA KEV flagged:      %d", kev_count)
    logger.info("Score disagreements:   %d", disagreement_count)
    logger.info("SSVC exploitation:     %d", ssvc_count)
    logger.info("Fix available:         %d", fix_count)
    logger.info("Network-exploitable:   %d", network_count)
    logger.info("Multi-package:         %d", multi_count)

    # Print one example narrative for manual review
    if chunks:
        logger.info("─" * 60)
        logger.info("Example narrative (first chunk):\n\n%s\n",
                    chunks[0]["narrative_text"])


if __name__ == "__main__":
    main()
