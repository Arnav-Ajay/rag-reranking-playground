# `rag-reranking-playground`

## Why This Repository Exists

The previous repository, `rag-hybrid-retrieval`, established a critical result:

> **Hybrid retrieval improves evidence surfacing — but does not reliably convert surfaced evidence into Top-K inclusion.**

In other words:

* The *right* chunks are often present in the candidate pool
* But they lose priority to less decisive neighbors
* And therefore never reach the language model

This repository exists to isolate and answer a single, narrow question:

> **Given a fixed candidate pool that already contains the correct evidence, can reranking reliably promote that evidence into Top-K?**

This is not an optimization demo.
It is a **controlled experiment in conflict resolution**.

---

## What Problem This System Solves

This repository introduces an **explicit reranking stage after retrieval** and measures its impact using the **unchanged evaluation harness from `rag-retrieval-eval`**, under a **frozen retrieval contract**.

It answers:

* Whether reranking improves **Top-K dominance**
* Which **question intents** benefit from reranking
* Which failure modes **reranking does not resolve**

The focus is **ranking quality**, not answer quality.

---

## What This System Explicitly Does NOT Do

This repository deliberately avoids:

* Changing the corpus
* Changing chunking strategy
* Changing embeddings
* Changing dense or sparse retrieval
* Prompt engineering
* LLM-based grading
* Agent behavior or tool use
* Any claim that reranking “fixes RAG”

If an improvement cannot be attributed **solely** to reranking, it does not belong here.

---

## System Relationship to Previous Repositories

This repository builds directly on:

* **`rag-minimal-control`**
  A strict, deterministic RAG control system

* **`rag-retrieval-eval`**
  A retrieval observability and evaluation harness

* **`rag-hybrid-retrieval`**
  Dense + sparse retrieval with explicit hybrid merge logic

All components from these repositories remain **frozen and authoritative**, including:

* Corpus
* Chunking
* Embeddings
* Dense similarity function
* Sparse retriever (BM25)
* Hybrid merge logic
* Top-K passed to the LLM (K = 4)
* Evaluation metrics

The **only new system component** is an explicit reranking stage.

---

## System Overview

**Repo Contract:**

* Inputs:

  * Hybrid-retrieved candidate pool (Top-N)
  * Deterministic evaluation questions
* Outputs:

  * Reranked candidate lists
  * Before/after retrieval metrics

No retrieval decisions are altered upstream.

---

## Retrieval + Reranking Pipeline

```
Document → Chunk → Embed
                 ↘
                  Dense Retriever +
Query ───────────→ Sparse Retriever
                     ↓
                Explicit Hybrid Merge
                     ↓
               Candidate Pool (Top-N)
                     ↓
                Reranking Stage
                     ↓
              Top-K → Generator (unchanged)
```

**Critical constraint:**
The candidate pool is **identical before and after reranking**.

Reranking may only **reorder**, never add or remove candidates.

---

## Reranking Strategy

This repository evaluates **explicit, inspectable reranking logic**, applied *only* to the hybrid candidate set.

### Reranker Classes in Scope

1. **Explainable heuristic reranker**

   * Linear combination of interpretable signals
   * Designed for failure analysis and causal clarity

2. **Cross-encoder reranker** *(optional, evaluated second)*

   * Introduced only if justified by heuristic results
   * Compared under the same frozen evaluation contract

No reranker is allowed to:

* Access corpus-level statistics
* Modify candidate membership
* Influence retrieval thresholds
* Leak gold labels

---

## Evaluation Methodology

All evaluation uses the **unchanged retrieval evaluation harness** from `rag-retrieval-eval`.

### Metrics (Locked)

* **Context Recall @ K (K = 4)**
* **Rank of First Relevant Chunk**
* Δ vs hybrid baseline

No new metrics are introduced.


## Expected Outcomes

This system is expected to show:

* Clear reranking gains for:

  * **definition** questions
  * **procedural** questions
  * **scope / inventory** questions
* Limited or inconsistent gains for:

  * **rationale-heavy** questions
* Persistent failures where:

  * candidate pools are noisy
  * decisive evidence is distributed across chunks

Reranking is **not** expected to:

* Improve corpus coverage
* Eliminate hallucinations
* Guarantee answer correctness

---

## Why This Matters

Most RAG systems treat ranking as an implementation detail.

This repository treats it as a **first-class system boundary**.

By isolating reranking from:

* retrieval
* generation
* evaluation

one can say, with precision:

> **When reranking helps, why it helps — and when it does not.**

---

## Empirical Results

This section reports **measured reranking impact**, using the **unchanged evaluation harness** from `rag-retrieval-eval`.

### Overall Impact

Reranking was evaluated over a **fixed candidate pool** (Top-N = 42), with **Top-K = 4** passed to the generator.

**Median Rank of First Relevant Chunk:**

| System           | Median Rank |
| ---------------- | ----------- |
| Hybrid Retrieval | **10.5**    |
| Reranked Hybrid  | **3.5**     |
| **Median Gain**  | **+4.5**    |

**Interpretation:**

* In many cases, the correct chunk was already present in the candidate pool
* Reranking materially improved **priority**, not recall
* Gains were achieved *without* changing retrieval, embeddings, or chunking

This confirms the Week-3 hypothesis:

> **Retrieval often fails by misordering evidence, not by missing it.**

---

## Results Stratified by Question Intent

Reranking benefits are **not uniform** across question types.

### Strong Gains Observed For

* **Definition questions**

  * Clear lexical anchors (“is defined as”, “refers to”)
* **Procedural questions**

  * Step indicators and ordering cues
* **Scope / inventory questions**

  * Enumerations and inclusion language

### Limited or Inconsistent Gains For

* **Rationale questions**

  * Evidence distributed across multiple chunks
  * Explanatory prose with weak local anchors

### Implication

Reranking improves **decisiveness**, not **semantic synthesis**.

When correct evidence is:

* **localized** → reranking helps
* **distributed** → reranking saturates quickly

These failures are **not reranking failures** — they indicate upstream limits in chunking or representation (Week-5).

---

## What Reranking Cannot Fix (Confirmed)

The following failure modes persist after reranking:

* Missing evidence in the candidate pool
* Gold evidence split across multiple chunks
* Queries requiring cross-chunk reasoning
* Generator ignoring provided evidence

These remain out of scope **by design**.

---

## Conclusion

This repository demonstrates that:

> **Reranking is a first-class system boundary that materially improves retrieval quality — but only within its causal limits.**

Reranking:

* improves evidence prioritization
* reduces ranking failures
* does not expand recall
* does not guarantee correct answers

---