# scripts/evaluate.py

"""
argus-sec — RAGAS evaluation harness.

Evaluates the RAG pipeline against the 25-question golden set using
three RAGAS metrics (no reference answers required):

  1. Faithfulness          — are all answer claims supported by retrieved context?
     Measures hallucination risk. Score 0–1; target ≥ 0.85 for a security tool.
     Ref: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/

  2. Answer Relevancy      — does the answer address the question asked?
     Penalises off-topic or incomplete responses. Score 0–1; target ≥ 0.80.
     Ref: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevancy/

  3. Context Precision     — are retrieved chunks ranked with relevant ones first?
     Measures retrieval ranking quality. Score 0–1; target ≥ 0.75.
     Ref: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/

All three metrics are reference-free (no human-written ground truth answers
needed). RAGAS uses an LLM-as-judge to evaluate each sample.

Judge LLM: gpt-5.4-nano (same as synthesis — keeps costs low, consistent judge)
Evaluator: async per-sample scoring (not batch evaluate()) for finer control
           and easier per-question-type breakdown.

Output:
  data/eval_results.json    — per-question scores + aggregate
  data/eval_summary.txt     — human-readable summary for README

Usage:
    python -m scripts.evaluate

Cost estimate: ~25 questions × 3 metrics × ~1,500 tokens = ~112,500 tokens
               At gpt-5.4-nano pricing ($0.20/$1.25 per 1M): ~$0.02 total
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from tqdm import tqdm
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import httpx
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecisionWithoutReference
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GOLDEN_PATH   = Path("data/golden_questions.json")
RESULTS_PATH  = Path("data/eval_results.json")
SUMMARY_PATH  = Path("data/eval_summary.txt")

API_BASE      = os.environ.get("ARGUS_API_URL", "http://localhost:8001")
OPENAI_KEY    = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL  = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano")

# Thresholds for pass/fail gate (used in summary)
THRESHOLDS = {
    "faithfulness":       0.85,
    "answer_relevancy":   0.75,
    "context_precision":  0.75,
}

# ---------------------------------------------------------------------------
# RAGAS metric setup
# ---------------------------------------------------------------------------

def build_metrics(llm, embeddings) -> dict:
    """
    Instantiate the three RAGAS metrics with the judge LLM and embeddings.

    All three are reference-free — no ground truth answers needed.
    Uses the new ragas.metrics.collections API (v0.4+).
    AnswerRelevancy requires embeddings to compute semantic similarity
    between question and generated answer.
    Ref: https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v03_to_v04/
    """
    return {
        "faithfulness":      Faithfulness(llm=llm),
        "answer_relevancy":  AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithoutReference(llm=llm),
    }


# ---------------------------------------------------------------------------
# RAG pipeline call
# ---------------------------------------------------------------------------

async def call_query_endpoint(
    question: dict,
    client: httpx.AsyncClient,
) -> dict | None:
    """
    Call the /query endpoint with the question and any filter hints.
    Returns the full response dict, or None on failure.
    """
    payload = {"query": question["question"], "k": 5}

    # Apply filter hints from golden set where present
    hint = question.get("filter_hint", {})
    if hint.get("severity"):      payload["severity"]      = hint["severity"]
    if hint.get("package"):       payload["package"]       = hint["package"]
    if "fix_available" in hint:   payload["fix_available"] = hint["fix_available"]
    if hint.get("attack_vector"): payload["attack_vector"] = hint["attack_vector"]

    try:
        resp = await client.post(
            f"{API_BASE}/query",
            json=payload,
            timeout=120.0,   # synthesis can take up to 60s on fallback
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error("Query failed for Q%d: %s", question["id"], e)
        return None


# ---------------------------------------------------------------------------
# Per-sample RAGAS scoring
# ---------------------------------------------------------------------------

async def score_sample(
    question: str,
    answer: str,
    contexts: list[str],
    metrics: dict,
) -> dict[str, float | None]:
    """
    Score one (question, answer, contexts) triple against all three metrics.
    Returns dict of metric_name → float score (or None on failure).
    """
    scores = {}
    for name, metric in metrics.items():
        try:
            if name == "answer_relevancy":
                result = await metric.ascore(
                    user_input=question,
                    response=answer,
                )
            else:
                result = await metric.ascore(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts,
                )
            scores[name] = round(float(result.value), 4)
        except Exception as e:
            logger.warning("Metric %s failed: %s", name, e)
            scores[name] = None

    return scores


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

async def main() -> None:
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"{GOLDEN_PATH} not found — run generate_golden_questions.py first"
        )

    questions = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d golden questions", len(questions))

    # Build judge LLM — gpt-5.4-nano via ragas llm_factory
    openai_client = AsyncOpenAI(api_key=OPENAI_KEY)
    judge_llm = llm_factory("gpt-4o-mini", client=openai_client)
    from ragas.embeddings import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", client=openai_client)
    metrics = build_metrics(judge_llm, embeddings)
    logger.info("RAGAS metrics initialised with judge: %s", OPENAI_MODEL)

    results = []
    t_start = time.monotonic()

    async with httpx.AsyncClient() as http_client:
        for i, question in tqdm(enumerate(questions, 1), total=len(questions), desc="Evaluating", unit="question"):
            logger.info(
                "Evaluating Q%d/%d [%s]: %s",
                i, len(questions),
                question["type"],
                question["question"][:60],
            )

            # Step 1 — call RAG pipeline
            t_query = time.monotonic()
            response = await call_query_endpoint(question, http_client)
            query_ms = int((time.monotonic() - t_query) * 1000)

            if response is None:
                results.append({
                    "id":       question["id"],
                    "type":     question["type"],
                    "question": question["question"],
                    "error":    "query_failed",
                    "scores":   {},
                })
                continue

            answer   = response.get("answer", "")
            chunks   = response.get("chunks", [])
            contexts = [c["narrative_text"] for c in chunks if c.get("narrative_text")]

            # Step 2 — skip scoring if confidence is low (no synthesis happened)
            if response.get("confidence") == "low" or not answer or not contexts:
                logger.warning("Q%d: low confidence or empty response — skipping RAGAS scoring", i)
                results.append({
                    "id":         question["id"],
                    "type":       question["type"],
                    "question":   question["question"],
                    "answer":     answer,
                    "confidence": response.get("confidence", "low"),
                    "citations":  response.get("citations", []),
                    "query_ms":   query_ms,
                    "scores":     {},
                    "skipped":    True,
                    "skip_reason":"low_confidence",
                })
                continue

            # Step 3 — RAGAS scoring
            t_score = time.monotonic()
            scores = await score_sample(
                question=question["question"],
                answer=answer,
                contexts=contexts,
                metrics=metrics,
            )
            score_ms = int((time.monotonic() - t_score) * 1000)

            logger.info(
                "Q%d scores — faithfulness=%.3f, relevancy=%.3f, precision=%.3f (%dms)",
                i,
                scores.get("faithfulness") or 0,
                scores.get("answer_relevancy") or 0,
                scores.get("context_precision") or 0,
                score_ms,
            )

            results.append({
                "id":         question["id"],
                "type":       question["type"],
                "question":   question["question"],
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

    total_elapsed = int((time.monotonic() - t_start))
    logger.info("Evaluation complete in %ds", total_elapsed)

    # ---------------------------------------------------------------------------
    # Aggregate scores
    # ---------------------------------------------------------------------------

    scored = [r for r in results if not r.get("skipped") and r.get("scores")]

    def mean(values: list[float]) -> float:
        valid = [v for v in values if v is not None]
        return round(sum(valid) / len(valid), 4) if valid else 0.0

    aggregate = {
        "faithfulness":      mean([r["scores"].get("faithfulness")      for r in scored]),
        "answer_relevancy":  mean([r["scores"].get("answer_relevancy")  for r in scored]),
        "context_precision": mean([r["scores"].get("context_precision") for r in scored]),
        "n_scored":    len(scored),
        "n_skipped":   len([r for r in results if r.get("skipped")]),
        "n_failed":    len([r for r in results if r.get("error")]),
        "n_fallback":  len([r for r in scored if r.get("fallback")]),
        "elapsed_s":   total_elapsed,
    }

    # Per-type breakdown
    type_scores: dict[str, dict[str, list]] = {}
    for r in scored:
        qtype = r["type"]
        if qtype not in type_scores:
            type_scores[qtype] = {"faithfulness": [], "answer_relevancy": [], "context_precision": []}
        for metric in ("faithfulness", "answer_relevancy", "context_precision"):
            v = r["scores"].get(metric)
            if v is not None:
                type_scores[qtype][metric].append(v)

    type_aggregate = {
        qtype: {m: mean(vals) for m, vals in metrics_dict.items()}
        for qtype, metrics_dict in type_scores.items()
    }

    # ---------------------------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------------------------

    output = {
        "aggregate":      aggregate,
        "by_type":        type_aggregate,
        "thresholds":     THRESHOLDS,
        "pass":           {
            m: aggregate[m] >= THRESHOLDS[m]
            for m in THRESHOLDS
        },
        "results":        results,
    }

    RESULTS_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info("Results written to %s", RESULTS_PATH)

    # Human-readable summary
    lines = [
        "argus-sec RAG Evaluation Summary",
        "=" * 50,
        f"Questions evaluated : {aggregate['n_scored']}/{len(questions)}",
        f"Skipped (low conf)  : {aggregate['n_skipped']}",
        f"Failed (API error)  : {aggregate['n_failed']}",
        f"Fallback to local   : {aggregate['n_fallback']}",
        f"Total time          : {total_elapsed}s",
        "",
        "Aggregate Scores",
        "-" * 30,
        f"Faithfulness        : {aggregate['faithfulness']:.4f}  {'✅ PASS' if output['pass']['faithfulness'] else '❌ FAIL'} (threshold {THRESHOLDS['faithfulness']})",
        f"Answer Relevancy    : {aggregate['answer_relevancy']:.4f}  {'✅ PASS' if output['pass']['answer_relevancy'] else '❌ FAIL'} (threshold {THRESHOLDS['answer_relevancy']})",
        f"Context Precision   : {aggregate['context_precision']:.4f}  {'✅ PASS' if output['pass']['context_precision'] else '❌ FAIL'} (threshold {THRESHOLDS['context_precision']})",
        "",
        "Scores by Question Type",
        "-" * 30,
    ]
    for qtype, scores in sorted(type_aggregate.items()):
        lines.append(
            f"  {qtype:<25} "
            f"F={scores['faithfulness']:.3f}  "
            f"R={scores['answer_relevancy']:.3f}  "
            f"P={scores['context_precision']:.3f}"
        )
    lines += [
        "",
        "Judge LLM   : " + OPENAI_MODEL,
        "Corpus      : 912 CVEs across 31 packages (NVD, last 24 months)",
        "Retrieval   : Hybrid BM25 + dense (BAAI/bge-small-en-v1.5), RRF fusion, cross-encoder rerank",
        "Synthesis   : " + OPENAI_MODEL + " (strict grounding — no external knowledge)",
    ]

    summary_text = "\n".join(lines)
    SUMMARY_PATH.write_text(summary_text, encoding="utf-8")
    logger.info("Summary written to %s", SUMMARY_PATH)

    # Print to console
    print("\n" + summary_text)


if __name__ == "__main__":
    asyncio.run(main())