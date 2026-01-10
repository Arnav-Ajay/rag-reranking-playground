# `rag-reranking-playground`

## TL;DR

> **This repository isolates reranking as a system boundary in RAG.**
> Using a frozen hybrid-retrieval candidate pool, we show that:
>
> * Naive heuristic reranking is unstable and often degrades ranking
> * Learned relevance (cross-encoders) materially improves Top-K inclusion
> * Reranking improves *priority*, not *recall*, and saturates when evidence is poorly chunked
>
> **Conclusion:** ranking is a real bottleneck — but only strong relevance signals help, and even they have limits.

---


## Why This Repository Exists

The previous repository, [`rag-hybrid-retrieval`](https://github.com/Arnav-Ajay/rag-hybrid-retrieval), established a critical result:

> **Hybrid retrieval improves evidence surfacing — but does not reliably convert surfaced evidence into Top-K inclusion.**

In other words:

* The *right* chunks are often present in the candidate pool
* But they are misordered relative to less decisive neighbors
* And therefore never reach the generator (Top-K = 4)

This repository exists to isolate and answer a single, narrow question:

> **Given a fixed candidate pool that already contains the correct evidence, can reranking reliably promote that evidence into Top-K?**

This is **not** an optimization demo.
It is a **controlled experiment in ranking failure and resolution**.

---

## What Problem This System Addresses

This repository introduces an **explicit reranking stage after retrieval** and measures its impact using the **unchanged evaluation harness from `rag-retrieval-eval`**, under a **frozen retrieval contract**.

It answers:

* Whether reranking improves **Top-K inclusion**
* Whether improvements depend on **reranking signal strength**
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

## Relationship to Previous Repositories

This repository builds directly on:

* **[`rag-minimal-control`](https://github.com/Arnav-Ajay/rag-minimal-control)**
  A strict, deterministic RAG control system

* **[`rag-retrieval-eval`](https://github.com/Arnav-Ajay/rag-retrieval-eval)**
  A retrieval observability and evaluation harness

* **[`rag-hybrid-retrieval`](https://github.com/Arnav-Ajay/rag-hybrid-retrieval)**
  Dense + sparse retrieval with explicit hybrid merge logic

All upstream components remain **frozen and authoritative**, including:

* Corpus
* Chunking
* Embeddings
* Dense similarity function
* Sparse retriever (BM25)
* Hybrid merge logic
* Top-K passed to the generator (K = 4)
* Evaluation metrics

The **only new system component** is an explicit reranking stage.

---

## System Overview

**Repo Contract**

**Inputs**

* Hybrid-retrieved candidate pool (Top-N = 42)
* Deterministic evaluation questions
* Gold chunk labels (for evaluation only)

**Outputs**

* Reranked candidate lists
* Rank-of-first-relevant metrics
* Δ vs hybrid baseline

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

**Critical constraint**

The candidate pool is **identical before and after reranking**.
Reranking may **only reorder**, never add or remove candidates.

---

## Reranking Strategies Evaluated

This repository evaluates **two reranking classes**, applied *only* to the frozen hybrid candidate pool.

### 1. Heuristic Reranker (Explainable)

* Linear combination of interpretable signals:

  * normalized dense score
  * normalized sparse score
  * lexical overlap
  * keyphrase match
  * length penalty
* No learning
* Fully inspectable

**Purpose:**
Failure analysis and causal clarity — *not* performance maximization.

---

### 2. Cross-Encoder Reranker (Learned Relevance)

* Jointly encodes *(query, chunk)* pairs
* Produces a learned relevance score
* Used strictly as a **ranking signal**

**Constraints**

* No access to gold labels
* No corpus-level statistics
* No modification of candidate membership
* Evaluated under the same frozen contract

---

## Evaluation Methodology

All evaluation uses the **unchanged harness from `rag-retrieval-eval`**.

**Metrics (Locked)**

* **Rank of First Relevant Chunk**
* **Context Recall @ K (K = 4)**
* **Δ vs Hybrid Baseline**

No new metrics are introduced.

---

## Empirical Results

Reranking was evaluated over **54 deterministic questions**, using a **fixed candidate pool (Top-N = 42)** and **Top-K = 4** passed to the generator.

### Overall Impact (Summary)

| Metric                                   | Observed |
| ---------------------------------------- | -------- |
| Total questions                          | 54       |
| Questions with relevant evidence in pool | ~50      |
| Median rank (heuristic reranker)         | ~15–18   |
| Median rank (cross-encoder reranker)     | ~2–3     |
| Top-K success rate (heuristic)           | ~5–8%    |
| Top-K success rate (cross-encoder)       | ~35–40%  |

**Interpretation**

* Heuristic reranking **does not reliably improve ranking** and often degrades it
* Cross-encoder reranking **materially improves evidence prioritization**
* Gains are achieved **without changing retrieval, embeddings, or chunking**

---

## Results Stratified by Question Intent

Reranking benefits are **not uniform** across question types.

### Strong Gains Observed For

* **Definition questions**

  * Clear lexical and semantic anchors
* **Procedural questions**

  * Step-like structure and ordering cues
* **Scope / inventory questions**

  * Enumerations and inclusion language

### Limited or Inconsistent Gains For

* **Rationale / principle questions**

  * Evidence distributed across multiple chunks
  * Explanatory prose with weak local anchors

**Implication**

Reranking improves **decisiveness**, not **semantic synthesis**.

When correct evidence is:

* **localized** → reranking helps
* **distributed** → reranking saturates quickly

These failures indicate **upstream limits** (chunking, representation), not reranking defects.

---

## What Reranking Cannot Fix (Confirmed)

The following failure modes persist after reranking:

* Missing evidence in the candidate pool
* Gold evidence split across multiple chunks
* Queries requiring cross-chunk reasoning
* Generator ignoring provided evidence

These remain **out of scope by design**.

---

## How to Run

### 1. Clone and set up the environment

```bash
git clone https://github.com/Arnav-Ajay/rag-reranking-playground.git
cd rag-reranking-playground
pip install -r requirements.txt
```

---

### 2. Run the heuristic reranker (default)

This runs the **explainable heuristic reranker**, which is the default mode.

```bash
python reranker.py
```

This produces:

* a question-level reranked artifact
* reranking metrics computed under the unchanged evaluation harness

---

### 3. Run the cross-encoder reranker

To evaluate **learned relevance-based reranking**, explicitly enable cross-encoder mode:

```bash
python reranker.py --rerank-mode cross-encoder
```

By default, this uses:

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

as the relevance model.

The retrieval pipeline, candidate pool, and evaluation metrics remain **fully unchanged**.

---

## Configurable Arguments (Optional)

The default arguments are set to **match the frozen contract** and **do not need to be changed** to reproduce reported results.

They are exposed **only for controlled experimentation**.

### Input / Output Paths

```python
--input-csv        data/chunks_and_questions/input_artifact.csv
--chunks-csv       data/chunks_and_questions/chunks_output.csv
--output-csv       data/results_and_summaries/rag_reranked_artifact.csv
--debug-candidates (optional) candidate-level debug output
```

### Evaluation Contract (Locked by Default)

```python
--top-n 20 or 50   # Candidate pool size (must match rag-hybrid-retrieval inspect_k)
--k     4    # Top-K passed to generator (locked across previous repos)
```

Changing these values breaks direct comparability with prior repositories.

---

### Heuristic Reranker Weights (Advanced / Diagnostic Use)

These weights control the **linear heuristic reranker only**:

```python
--wd 0.4   # normalized dense score
--wb 0.3   # normalized sparse (BM25) score
--wo 0.1   # lexical overlap
--wk 0.1   # keyphrase hit rate
--wp 0.0   # pattern cues
--wl 0.1   # length penalty
```

These are intentionally exposed to support:

* ablation
* sensitivity analysis
* failure inspection

They are **not tuned for optimal performance**.

---

### Cross-Encoder Configuration

```python
--rerank-mode cross-encoder
--cross-encoder-model cross-encoder/ms-marco-MiniLM-L-6-v2
```

Any compatible `sentence-transformers` cross-encoder can be substituted, provided:

* it scores *(query, chunk)* pairs
* it does not access gold labels
* it does not modify candidate membership

---

## Reproducibility Note

All reported results in this repository were produced using:

* the default arguments
* a frozen retrieval contract
* an identical candidate pool before and after reranking

Changing configuration parameters is **explicitly exploratory** and should not be conflated with the main findings.

---

## Conclusion

This repository demonstrates that:

> **Reranking is a first-class system boundary that can materially improve retrieval quality — but only when supplied with sufficiently strong relevance signals.**

Specifically:

* Heuristic reranking is unstable and unreliable
* Learned relevance (cross-encoders) substantially improves Top-K inclusion
* Reranking does **not** expand recall
* Reranking does **not** guarantee correct answers

The remaining bottleneck is **how evidence is chunked and represented**, motivating `rag-chunking-strategies`.

---

