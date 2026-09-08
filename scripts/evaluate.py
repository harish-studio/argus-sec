# scripts/evaluate.py

"""
argus-sec — RAGAS evaluation harness.

Evaluates the RAG pipeline against the golden question set using three
reference-free RAGAS metrics:

  1. Faithfulness       — are all answer claims supported by retrieved context?
     Hallucination risk. Ref: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
  2. Answer Relevancy   — does the answer address the question asked?
     Ref: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevancy/
  3. Context Precision  — are relevant chunks ranked first?
     Ref: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/

WHY REPEAT RUNS
---------------
RAGAS uses an LLM as judge, so every score carries judge variance on top of
sampling variance from a small golden set. Two consecutive runs of this
harness against an unchanged pipeline produced context precision of 0.767
(PASS) and 0.679 (FAIL) — the gate flipped without a code change.

A single run therefore cannot support a published number or a CI gate. This
harness runs the evaluation EVAL_RUNS times and reports mean and sample
standard deviation. A metric whose threshold falls within +/-1 SD of the mean
is reported INCONCLUSIVE rather than PASS or FAIL: the run cannot distinguish
it from noise, and calling it either way is a coin flip.

Per-type sample counts are printed alongside per-type scores. With 25
questions across 6 types, a type average rests on ~4 samples; the count is
shown so the reader can weight it accordingly.

Models:
  Judge     — JUDGE_MODEL (default gpt-4o-mini). Must be a model RAGAS's
              llm_factory can drive: it sends `max_tokens`, which newer
              OpenAI models reject in favour of `max_completion_tokens`.
  Synthesis — OPENAI_MODEL, used by the API, not by this harness.

Output:
  data/eval_results.json    — per-run, per-question scores + aggregates
  data/eval_summary.txt     — human-readable summary

Usage:
    python -m scripts.evaluate
    EVAL_RUNS=3 python -m scripts.evaluate

Cost: ~25 questions x 3 metrics x ~1,500 tokens per run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

import httpx
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    Faithfulness,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GOLDEN_PATH  = Path("data/golden_questions.json")
RESULTS_PATH = Path("data/eval_results.json")
SUMMARY_PATH = Path("data/eval_summary.txt")

API_BASE     = os.environ.get("ARGUS_API_URL", "http://localhost:8001")
OPENAI_KEY   = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano")
JUDGE_MODEL  = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
EVAL_RUNS    = int(os.environ.get("EVAL_RUNS", "3"))

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision")

THRESHOLDS = {
    "faithfulness":      0.85,
    "answer_relevancy":  0.75,
    "context_precision": 0.75,
}

# Samples scoring below this on faithfulness are listed in the summary,
# so a failing question type points at specific questions to inspect.
WORST_SAMPLE_CUTOFF = 0.70
WORST_SAMPLE_LIMIT  = 5


# ---------------------------------------------------------------------------
# Metric setup
# ---------------------------------------------------------------------------

def build_metrics(llm, embeddings) -> dict:
    """
    Instantiate the three reference-free RAGAS metrics.
    AnswerRelevancy needs embeddings for question/answer similarity.
    Ref: https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v03_to_v04/
    """
    return {
        "faithfulness":      Faithfulness(llm=llm),
        "answer_relevancy":  AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithoutReference(llm=llm),
    }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def mean_or_none(values) -> float | None:
    """Mean of non-None values, or None if every value failed.

    Never returns 0.0 for an all-failed metric: a total metric failure and a
    genuine score of zero are different facts and must not be conflated."""
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 4) if valid else None


def stdev_or_none(values) -> float | None:
    valid = [v for v in values if v is not None]
    return round(statistics.stdev(valid), 4) if len(valid) > 1 else None


def fmt(v: float | None, places: int = 3) -> str:
    return "ERROR" if v is None else f"{v:.{places}f}"


def verdict(mean: float | None, sd: float | None, threshold: float) -> str:
    """
    PASS / FAIL only when the threshold sits outside +/-1 SD of the mean.
    Inside that band the run cannot separate the result from judge noise.
    A single run has no SD, so its verdict is always provisional.
    """
    if mean is None:
        return "NOT MEASURED"
    if sd is None:
        return ("PASS" if mean >= threshold else "FAIL") + " (single run — unverified)"
    if abs(mean - threshold) <= sd:
        return "INCONCLUSIVE (threshold within 1 SD)"
    return "PASS" if mean >= threshold else "FAIL"


# ---------------------------------------------------------------------------
# RAG pipeline call
# ---------------------------------------------------------------------------

async def call_query_endpoint(question: dict, client: httpx.AsyncClient) -> dict | None:
    """Call /query with the question and any filter hints. None on failure."""
    payload = {"query": question["question"], "k": 5}

    hint = question.get("filter_hint", {})
    if hint.get("severity"):      payload["severity"]      = hint["severity"]
    if hint.get("package"):       payload["package"]       = hint["package"]
    if "fix_available" in hint:   payload["fix_available"] = hint["fix_available"]
    if hint.get("attack_vector"): payload["attack_vector"] = hint["attack_vector"]

    try:
        resp = await client.post(f"{API_BASE}/query", json=payload, timeout=120.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error("Query failed for Q%s: %s", question["id"], e)
        return None


# ---------------------------------------------------------------------------
# Per-sample scoring
# ---------------------------------------------------------------------------

async def score_sample(question: str, answer: str, contexts: list[str], metrics: dict) -> dict:
    """Score one triple. A failed metric records None, never 0.0."""
    scores: dict[str, float | None] = {}
    for name, metric in metrics.items():
        try:
            if name == "answer_relevancy":
                result = await metric.ascore(user_input=question, response=answer)
            else:
                result = await metric.ascore(
                    user_input=question, response=answer, retrieved_contexts=contexts
                )
            scores[name] = round(float(result.value), 4)
        except Exception as e:
            logger.warning("Metric %s failed: %s", name, e)
            scores[name] = None
    return scores


# ---------------------------------------------------------------------------
# One full pass over the golden set
# ---------------------------------------------------------------------------

async def run_once(questions: list[dict], metrics: dict, run_index: int) -> list[dict]:
    results = []

    async with httpx.AsyncClient() as http_client:
        desc = f"Run {run_index}/{EVAL_RUNS}"
        for i, question in tqdm(
            enumerate(questions, 1), total=len(questions), desc=desc, unit="q"
        ):
            t_query = time.monotonic()
            response = await call_query_endpoint(question, http_client)
            query_ms = int((time.monotonic() - t_query) * 1000)

            base = {
                "run":      run_index,
                "id":       question["id"],
                "type":     question["type"],
                "question": question["question"],
            }

            if response is None:
                results.append({**base, "error": "query_failed", "scores": {}})
                continue

            answer   = response.get("answer", "")
            chunks   = response.get("chunks", [])
            contexts = [c["narrative_text"] for c in chunks if c.get("narrative_text")]

            if response.get("confidence") == "low" or not answer or not contexts:
                results.append({
                    **base,
                    "answer":      answer,
                    "confidence":  response.get("confidence", "low"),
                    "query_ms":    query_ms,
                    "scores":      {},
                    "skipped":     True,
                    "skip_reason": "low_confidence",
                })
                continue

            t_score = time.monotonic()
            scores = await score_sample(question["question"], answer, contexts, metrics)
            score_ms = int((time.monotonic() - t_score) * 1000)

            logger.info(
                "R%d Q%d — F=%s R=%s P=%s (%dms)",
                run_index, i,
                fmt(scores.get("faithfulness")),
                fmt(scores.get("answer_relevancy")),
                fmt(scores.get("context_precision")),
                score_ms,
            )

            results.append({
                **base,
                "answer":     answer,
                "confidence": response.get("confidence"),
                "model_used": response.get("model_used"),
                "fallback":   response.get("fallback", False),
                "citations":  response.get("citations", []),
                "query_ms":   query_ms,
                "score_ms":   score_ms,
                "scores":     scores,
                "skipped":    False,
            })

    return results


def aggregate_run(results: list[dict], n_questions: int) -> dict:
    scored = [r for r in results if not r.get("skipped") and r.get("scores")]
    agg = {m: mean_or_none([r["scores"].get(m) for r in scored]) for m in METRIC_NAMES}
    agg.update({
        "n_scored":        len(scored),
        "n_questions":     n_questions,
        "n_skipped":       len([r for r in results if r.get("skipped")]),
        "n_failed":        len([r for r in results if r.get("error")]),
        "n_fallback":      len([r for r in scored if r.get("fallback")]),
        "n_metric_calls":  len(scored) * len(METRIC_NAMES),
        "n_metric_errors": sum(
            1 for r in scored for m in METRIC_NAMES if r["scores"].get(m) is None
        ),
    })
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"{GOLDEN_PATH} not found — run generate_golden_questions.py first"
        )

    questions = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d golden questions; %d run(s)", len(questions), EVAL_RUNS)

    openai_client = AsyncOpenAI(api_key=OPENAI_KEY)
    judge_llm = llm_factory(JUDGE_MODEL, client=openai_client)
    from ragas.embeddings import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", client=openai_client)
    metrics = build_metrics(judge_llm, embeddings)
    logger.info("RAGAS metrics initialised with judge: %s", JUDGE_MODEL)

    t_start = time.monotonic()
    all_results: list[dict] = []
    run_aggregates: list[dict] = []

    for run_index in range(1, EVAL_RUNS + 1):
        run_results = await run_once(questions, metrics, run_index)
        all_results.extend(run_results)
        run_aggregates.append(aggregate_run(run_results, len(questions)))

    total_elapsed = int(time.monotonic() - t_start)
    logger.info("Evaluation complete in %ds across %d run(s)", total_elapsed, EVAL_RUNS)

    # --- Across-run statistics -------------------------------------------

    stability = {}
    for m in METRIC_NAMES:
        per_run = [a[m] for a in run_aggregates]
        stability[m] = {
            "mean":      mean_or_none(per_run),
            "sd":        stdev_or_none(per_run),
            "per_run":   per_run,
            "threshold": THRESHOLDS[m],
            "verdict":   verdict(mean_or_none(per_run), stdev_or_none(per_run), THRESHOLDS[m]),
        }

    # --- Per-type breakdown, pooled across runs, with sample counts -------

    scored_all = [r for r in all_results if not r.get("skipped") and r.get("scores")]
    type_scores: dict[str, dict[str, list]] = {}
    for r in scored_all:
        bucket = type_scores.setdefault(r["type"], {m: [] for m in METRIC_NAMES})
        for m in METRIC_NAMES:
            v = r["scores"].get(m)
            if v is not None:
                bucket[m].append(v)

    type_aggregate = {
        qtype: {
            **{m: mean_or_none(vals) for m, vals in buckets.items()},
            "n": len({r["id"] for r in scored_all if r["type"] == qtype}),
            "n_samples": max(len(v) for v in buckets.values()) if buckets else 0,
        }
        for qtype, buckets in type_scores.items()
    }

    # --- Worst faithfulness samples, for diagnosis ------------------------

    worst = sorted(
        (r for r in scored_all
         if r["scores"].get("faithfulness") is not None
         and r["scores"]["faithfulness"] < WORST_SAMPLE_CUTOFF),
        key=lambda r: r["scores"]["faithfulness"],
    )[:WORST_SAMPLE_LIMIT]

    total_metric_errors = sum(a["n_metric_errors"] for a in run_aggregates)
    total_metric_calls  = sum(a["n_metric_calls"] for a in run_aggregates)

    output = {
        "runs":            EVAL_RUNS,
        "elapsed_s":       total_elapsed,
        "judge_model":     JUDGE_MODEL,
        "synthesis_model": OPENAI_MODEL,
        "stability":       stability,
        "run_aggregates":  run_aggregates,
        "by_type":         type_aggregate,
        "thresholds":      THRESHOLDS,
        "results":         all_results,
    }

    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results written to %s", RESULTS_PATH)

    # --- Summary ----------------------------------------------------------

    lines = [
        "argus-sec RAG Evaluation Summary",
        "=" * 62,
        f"Runs                : {EVAL_RUNS}",
        f"Questions per run   : {run_aggregates[0]['n_scored']}/{len(questions)} scored",
        f"Skipped (low conf)  : {run_aggregates[0]['n_skipped']} (run 1)",
        f"Metric errors       : {total_metric_errors}/{total_metric_calls}",
        f"Total time          : {total_elapsed}s",
        "",
        "Aggregate Scores (mean +/- SD across runs)",
        "-" * 62,
    ]

    for m, label in (
        ("faithfulness",      "Faithfulness"),
        ("answer_relevancy",  "Answer Relevancy"),
        ("context_precision", "Context Precision"),
    ):
        s = stability[m]
        sd_txt = f" +/- {fmt(s['sd'])}" if s["sd"] is not None else ""
        lines.append(
            f"{label:<20}: {fmt(s['mean'], 4)}{sd_txt}  "
            f"{s['verdict']} (threshold {s['threshold']})"
        )
        if EVAL_RUNS > 1:
            lines.append(f"{'':20}  per run: {[fmt(v) for v in s['per_run']]}")

    if total_metric_errors:
        lines += [
            "",
            f"WARNING: {total_metric_errors} metric call(s) failed and were excluded.",
            "Means are computed from successful calls only — treat as partial.",
        ]

    if EVAL_RUNS == 1:
        lines += [
            "",
            "WARNING: single run. Judge variance alone has moved context precision",
            "across its threshold between identical runs. Do not publish or gate on",
            "this number — set EVAL_RUNS=3 or higher.",
        ]

    lines += [
        "",
        "Scores by Question Type (pooled across runs)",
        "-" * 62,
        f"  {'type':<25} {'n':>3}  {'F':>6}  {'R':>6}  {'P':>6}",
    ]
    for qtype, s in sorted(type_aggregate.items()):
        lines.append(
            f"  {qtype:<25} {s['n']:>3}  "
            f"{fmt(s['faithfulness']):>6}  "
            f"{fmt(s['answer_relevancy']):>6}  "
            f"{fmt(s['context_precision']):>6}"
        )
    lines.append("  n = distinct questions of that type; each scored once per run.")

    if worst:
        lines += [
            "",
            f"Lowest-faithfulness samples (below {WORST_SAMPLE_CUTOFF})",
            "-" * 62,
        ]
        for r in worst:
            lines.append(
                f"  [{r['type']}] Q{r['id']} run {r['run']} "
                f"F={fmt(r['scores']['faithfulness'])}"
            )
            lines.append(f"    {r['question'][:70]}")
        lines.append("  Full answers and contexts are in data/eval_results.json.")

    lines += [
        "",
        "Judge LLM   : " + JUDGE_MODEL,
        "Corpus      : 912 CVEs across 31 packages (NVD, last 24 months)",
        "Retrieval   : Hybrid BM25 + dense (BAAI/bge-small-en-v1.5), RRF fusion, cross-encoder rerank",
        "Synthesis   : " + OPENAI_MODEL + " (strict grounding — no external knowledge)",
    ]

    summary_text = "\n".join(lines)
    SUMMARY_PATH.write_text(summary_text, encoding="utf-8")
    logger.info("Summary written to %s", SUMMARY_PATH)

    print("\n" + summary_text)


if __name__ == "__main__":
    asyncio.run(main())
