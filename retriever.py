# retriever.py → embeddings + top-K
import re
import math
import numpy as np
from collections import Counter, defaultdict

# Function to embedd chunked text into vector
# NOTE: This is a diagnostic embedding, not a semantic embedding
def get_embedding(chunk):
    # Dummy embedding function: convert each character to its ASCII value and create a fixed-size vector
    embedding_size = 128
    embedding = np.zeros(embedding_size)
    for i, char in enumerate(chunk):
        if i < embedding_size:
            embedding[i] = ord(char)
    return embedding

# Function to create a vector store from document chunks
def create_vector_store(chunks):
    vector_store = []
    for id, chunk_info in chunks.items():
        chunk_text = chunk_info["text"]
        doc_id = chunk_info["doc_id"]
        embedding = get_embedding(chunk_text)
        vector_store.append((id, doc_id, chunk_text, embedding))
    return vector_store


# Function to compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2.T)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# store embeddings in a list
def retrieve_similar_documents(vector_store, query, top_k=4):
    query_embedding = get_embedding(query)
    similarities = []

    for chunk_id, doc_id, chunk_text, embedding in vector_store:
        sim = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk_id, doc_id, chunk_text, sim))

    similarities.sort(key=lambda x: x[3], reverse=True)

    return similarities[:top_k]

# -------------------------
# Sparse (BM25) — Week 3
# -------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str):
    # Deterministic, minimal tokenizer (no stemming)
    return _TOKEN_RE.findall(text.lower())

# BM25 Index Creation
def create_bm25_index(chunks, k1=1.5, b=0.75):
    """
    Returns a dict containing:
      - doc_len, avgdl
      - df, idf
      - tf_per_doc (list of Counters)
      - chunk_ids (list aligned with tf_per_doc)
      - meta mapping: chunk_id -> (doc_id, text)
      - params k1, b
    """
    chunk_ids = []
    tf_per_doc = []
    doc_len = []
    df = defaultdict(int)

    meta = {}
    for chunk_id, info in chunks.items():
        text = info["text"]
        doc_id = info["doc_id"]
        meta[chunk_id] = (doc_id, text)

        toks = tokenize(text)
        tf = Counter(toks)

        chunk_ids.append(chunk_id)
        tf_per_doc.append(tf)
        doc_len.append(len(toks))

        # document frequency
        for term in tf.keys():
            df[term] += 1

    N = len(chunk_ids)
    avgdl = (sum(doc_len) / N) if N else 0.0

    # IDF (BM25-style)
    idf = {}
    for term, dfi in df.items():
        # classic BM25 idf with +1 smoothing
        idf[term] = math.log(1 + (N - dfi + 0.5) / (dfi + 0.5))

    return {
        "k1": k1,
        "b": b,
        "N": N,
        "avgdl": avgdl,
        "chunk_ids": chunk_ids,
        "tf_per_doc": tf_per_doc,
        "doc_len": doc_len,
        "idf": idf,
        "meta": meta
    }


# Sparse Retriever (BM25)
def sparse_retriever(query, bm25_index, top_k=50):
    q_terms = tokenize(query)
    if not q_terms or bm25_index["N"] == 0:
        return []

    k1 = bm25_index["k1"]
    b = bm25_index["b"]
    avgdl = bm25_index["avgdl"] if bm25_index["avgdl"] > 0 else 1.0
    idf = bm25_index["idf"]

    scores = []
    for i, chunk_id in enumerate(bm25_index["chunk_ids"]):
        tf = bm25_index["tf_per_doc"][i]
        dl = bm25_index["doc_len"][i] if bm25_index["doc_len"][i] > 0 else 1

        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            f = tf[term]
            term_idf = idf.get(term, 0.0)
            denom = f + k1 * (1 - b + b * (dl / avgdl))
            score += term_idf * ((f * (k1 + 1)) / denom)

        if score > 0:
            scores.append((chunk_id, float(score)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# -------------------------
# Hybrid merge — explicit + deterministic
# -------------------------
def hybrid_retriever(query, vector_store, bm25_index,
                    top_k=4, dense_top_n=20, sparse_top_n=42,
                    D=20, S=20):
    dense = retrieve_similar_documents(vector_store, query, top_k=dense_top_n)
    sparse = sparse_retriever(query, bm25_index, top_k=sparse_top_n)

    merged = {}

    # Dense annotate
    for r, (chunk_id, doc_id, text, dense_score) in enumerate(dense, start=1):
        merged[chunk_id] = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": text,
            "dense_rank": r,
            "dense_score": dense_score,
            "sparse_rank": None,
            "bm25_score": None
        }

    # Sparse annotate (must attach doc_id/text from same corpus)
    meta = bm25_index["meta"]
    for r, (chunk_id, bm25_score) in enumerate(sparse, start=1):
        if chunk_id in merged:
            merged[chunk_id]["sparse_rank"] = r
            merged[chunk_id]["bm25_score"] = bm25_score
        else:
            doc_id, text = meta.get(chunk_id, ("unknown", ""))
            merged[chunk_id] = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": text,
                "dense_rank": None,
                "dense_score": None,
                "sparse_rank": r,
                "bm25_score": bm25_score
            }

    # Bucket priority rules (inspectable)
    final = []
    for chunk_id, info in merged.items():
        dr = info["dense_rank"]
        sr = info["sparse_rank"]

        if dr is not None and dr <= D and sr is not None and sr <= S:
            priority = 1
        elif sr is not None and sr <= S:
            priority = 2
        elif dr is not None and dr <= D:
            priority = 3
        else:
            # drop (keeps merge logic aligned to your stated rule)
            continue

        # Deterministic tiebreak (not a heuristic, just stable ordering)
        dr_t = dr if dr is not None else 10**9
        sr_t = sr if sr is not None else 10**9
        info["priority"] = priority
        final.append((priority, dr_t, sr_t, chunk_id, info))

    final.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    # Return shape compatible with app: (chunk_id, doc_id, text, score)
    # score here is "priority bucket" (small int) just for printing
    out = []
    for priority, _, _, _, info in final[:top_k]:
        # Hybrid "score" is not a similarity score.
        # We expose retrieval provenance instead.
        hybrid_score = {
            "priority": info["priority"],
            "dense_rank": info["dense_rank"],
            "dense_score": info["dense_score"],
            "sparse_rank": info["sparse_rank"],
            "sparse_score": info["bm25_score"],
        }

        out.append((
            info["chunk_id"],
            info["doc_id"],
            info["text"],
            hybrid_score
        ))

    return out