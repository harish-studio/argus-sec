# ADR-003: Strict Grounding over Augmented Synthesis

**Status:** Accepted  
**Date:** 2026-07-07  
**Deciders:** Harry (solo portfolio project)

---

## Context

The synthesis LLM (gpt-5.4-nano) generates answers from retrieved CVE context. Two grounding strategies were considered:

1. **Strict grounding** — the LLM answers only from retrieved CVE records. If the answer is not in the context, it says so.
2. **Augmented grounding** — the LLM uses retrieved CVE records as its primary source but may supplement with general cybersecurity knowledge, clearly distinguishing between corpus content and general knowledge.

---

## Decision

**Strict grounding** — the synthesis prompt enforces: *"Answer ONLY from the provided CVE records. Do not add any information not present in the context."*

Citations are extracted from retrieved chunk CVE IDs, not from LLM output — preventing the model from fabricating CVE references even when it knows about a CVE from training data.

---

## Rationale

**Domain-specific failure modes justify strict grounding.**

In a general-purpose assistant, augmented grounding improves answer completeness by filling gaps with parametric knowledge. In a security tool, the same behaviour causes specific, dangerous failure modes:

1. **Wrong CVSS scores** — a model trained on CVE data may recall an outdated score (e.g. 7.5 before NVD revised it to 9.8). Presenting this as fact misleads a security analyst's prioritisation.
2. **Wrong affected versions** — version ranges are frequently updated after initial publication. Parametric knowledge reflects training-time data, not current NVD state.
3. **Hallucinated CVE IDs** — LLMs are known to generate plausible-looking but non-existent CVE identifiers when they cannot find an exact match in context.
4. **False confidence** — an augmented answer blending corpus content with parametric knowledge is harder to audit than a strictly grounded one.

**RAGAS Faithfulness score validates the decision.**

Faithfulness of 0.9265 (threshold 0.85) confirms that 93% of claims in generated answers are directly supported by retrieved context. This score would be meaningless to report if the model was freely drawing on parametric knowledge.

**The confidence gate reinforces strict grounding.**

When top rerank score < 0.4 (sigmoid-normalised), synthesis is skipped entirely and a plain-language explanation is returned. This makes the system's uncertainty explicit rather than generating a low-confidence answer that appears authoritative.

---

## Consequences

**Positive:**
- Faithfulness: 0.9265 — verifiable grounding, suitable for security-critical contexts
- Citations extracted from retrieval (not LLM output) — prevents CVE ID hallucination
- System explicitly surfaces its own uncertainty (confidence gate) rather than hiding it
- Answers are auditable — every claim can be traced to a specific CVE chunk

**Negative:**
- Answers may be incomplete if the corpus has gaps (e.g. recently published CVEs not yet in the 24-month window)
- Answer Relevancy on filter-type queries (0.280 for `severity_filtered`) is penalised by RAGAS because strict answers are sometimes too terse relative to the question's expected scope
- A user asking a general cybersecurity question unrelated to the corpus will receive "Insufficient information" rather than a helpful general answer

**Acceptable trade-off:**
The completeness limitation is a feature for a security tool — *"I don't know"* is a safer answer than a confident but wrong one. Users seeking general cybersecurity information should use a general-purpose assistant; `argus-sec` is explicitly scoped to the 912-CVE corpus.

---

## Alternatives Considered

| Option | Rejected because |
|---|---|
| Augmented grounding | Risk of outdated/hallucinated CVSS scores and version ranges in a security context |
| No synthesis (retrieval only) | `/search` endpoint serves this use case; `/query` is explicitly for synthesised answers |
| LLM-extracted citations | Models fabricate CVE IDs; extracting from retrieved chunk IDs is reliable |

---

## References

- RAGAS Faithfulness metric: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
- "Lost in the Middle" (Liu et al., 2023) — LLM tendency to ignore middle context chunks, motivating cross-encoder reranking to surface the most relevant chunks first
