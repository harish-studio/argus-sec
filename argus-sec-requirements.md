# argus-sec — Project Requirements

**CVE intelligence RAG — measured, not asserted**

Status: Built and public. This document is a retrospective specification, capturing what was built and formally recording that this project's scope is closed.

---

## 1. Objective

Answer plain-language questions about known vulnerabilities affecting a given tech stack, grounded in live CVE data, with citations — as a deliberate pivot away from the saturated "chat with a document" RAG pattern.

---

## 2. As-built scope

- Hybrid retrieval: dense (BAAI/bge-small-en-v1.5 via FastEmbed) + sparse (BM25) via Qdrant, combined with reciprocal rank fusion.
- Cross-encoder reranking (ms-marco-MiniLM).
- Confidence gating — low-confidence retrievals are flagged rather than answered with false certainty.
- Local LLM fallback (Ollama) alongside hosted providers.
- RAGAS evaluation (mean ± SD over 3 runs, gpt-4o-mini judge, 25-question golden set): Faithfulness 0.917 ± 0.014, Context Precision 0.784 ± 0.019 — both above threshold; Answer Relevancy 0.783 ± 0.036 — inconclusive, straddles the 0.75 threshold. Single-run scores vary enough to flip a pass/fail gate, so figures are reported with variance.
- Full ADRs and `SCALING.md` documentation published.

---

## 3. Scope decision — closed, not extended

Two capabilities considered and explicitly **not** added to this project, per direct evaluation against its existing features:

- **Human-in-the-loop (HITL):** rejected. The existing confidence-gating mechanism already serves the "flag for review" function HITL would add. Layering HITL on top would be redundant with a feature that already exists, not a new differentiator.
- **Agent memory (short/long-term):** rejected. This is a largely single-shot CVE lookup tool; there is no strong case for conversational memory, and adding it would not reflect a genuine need in the system's actual use pattern.

**Graph-based retrieval** (an earlier candidate for this project) was also reconsidered and moved to `atlas-supply` instead, where it answers a genuine multi-hop question (trade-flow exposure) that argus-sec's single-domain CVE lookup doesn't have an equivalent for.

---

## 4. Non-goals (confirmed, unchanged)

- Not a general-purpose security chatbot — scoped to CVE/stack-relevance questions only.
- Not coupled to any other repo in the portfolio.
- Not a production vulnerability-management tool — the README should state this plainly, same principle as the sanctions-data disclaimer in atlas-supply: describe what it demonstrates, not what it's certified for.

---

## 5. Status

No further scope changes planned. Available for revision only if a specific interview or market signal justifies it — evaluate any future addition against the same test applied above: does the existing system already serve this function, or is this genuinely new capability?
