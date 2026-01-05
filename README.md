# `rag-reranking-playground`

## Why This Repository Exists

The previous repository, `rag-hybrid-retrieval`, established a critical result:

> **Hybrid retrieval improves evidence surfacing — but does not reliably convert surfaced evidence into Top-K inclusion.**

In other words:

* The *right* chunks are often present in the candidate set
* But they lose priority to less decisive neighbors
* And therefore never reach the language model

This repository exists to isolate and answer a single, narrow question:

> **Given a fixed candidate pool that already contains the correct evidence, can reranking reliably promote that evidence into Top-K?**

This is not an optimization demo.
It is a **controlled experiment in conflict resolution**.

---

## What Problem This System Solves

This repository introduces an **explicit reranking stage** *after retrieval* and measures its impact using the **exact same evaluation harness from `rag-retrieval-eval`**.

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

* **[`rag-minimal-control`](https://github.com/Arnav-Ajay/rag-minimal-control)**
  A strict, deterministic RAG control system
* **[`rag-retrieval-eval`](https://github.com/Arnav-Ajay/rag-retrieval-eval)**
  A retrieval observability and evaluation harness
* **[`rag-hybrid-retrieval`](https://github.com/Arnav-Ajay/rag-hybrid-retrieval)**
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
* Output:

  * Reranked candidate lists
  * Before/after retrieval metrics
* Non-goal:

  * Generating better answers

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

---

## Reranking Strategy (Week-4 Scope)

This repository evaluates **explicit, inspectable reranking logic**, applied *only* to the hybrid candidate set.

Two reranker classes are in scope:

1. **Explainable heuristic reranker**

   * Linear combination of interpretable signals
   * Designed for failure analysis and causal clarity

2. **Cross-encoder reranker** *(optional, evaluated second)*

   * Introduced only if justified by heuristic results
   * Compared under the same evaluation contract

No reranker is allowed to:

* Access corpus-level information
* Modify candidate membership
* Influence retrieval thresholds

---

## Evaluation Methodology (Frozen)

All evaluation uses the **retrieval evaluation harness** from `rag-retrieve-eval`, unchanged.

### Metrics (Locked)

* **Context Recall @ K (K = 4)**
* **Rank of First Relevant Chunk**
* Δ vs hybrid baseline

No new metrics are introduced.

### Stratification (Required)

All results are reported:

* Overall
* Stratified by **question intent**:

  * factual
  * definition
  * procedural
  * rationale
  * scope / inventory

---

## Failure Taxonomy (Locked)

Observed failures are labeled using the existing taxonomy:

### Failure Layers

* `retrieval`
* `reranking`
* `generation`

### Observable Failure Types

* `ranking_failure`
* `evidence_not_prioritized`
* `answer_not_used`
* `hallucination_no_evidence`

No causal claims are made without controlled comparison.

---

## Expected Outcomes

This system is expected to show:

* Clear reranking gains for:

  * definition
  * procedural
  * scope / inventory questions
* Limited or no gains for:

  * rationale-heavy questions
* Persistent failures where:

  * candidate pools are noisy
  * decisive evidence is distributed across chunks

Reranking is not expected to:

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

One can say, with precision:

> *When reranking helps, why it helps — and when it does not.*


---