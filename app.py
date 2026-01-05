# app.py → glue + debug prints
import os
import argparse
from llm import get_llm_response
from ingest import load_pdf, chunk_texts
from retriever import create_bm25_index, create_vector_store, hybrid_retriever, retrieve_similar_documents, sparse_retriever
import csv


# Export chunks to CSV for debugging
def export_chunks_csv(all_chunks, output_path="data/chunks_debug.csv"):

    with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['chunk_id', 'doc_id', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for chunk_id, chunk_info in all_chunks.items():
            writer.writerow({
                'chunk_id': chunk_id,
                'doc_id': chunk_info['doc_id'],
                'text': chunk_info['text']
            })

    print(f"Chunks exported to {output_path}")

# Retrieval evaluation
def run_retrieval_evaluation(args, vector_store, bm25_index, inspect_k=42):
    import pandas as pd
    print("Running retrieval evaluation...\n")

    questions_df = pd.read_csv(args.questions_csv)
    evaluation_results = []

    def eval_one(mode, question_text):
        if mode == "dense":
            results = retrieve_similar_documents(vector_store, question_text, top_k=inspect_k)
        elif mode == "sparse":
            results = sparse_retriever(question_text, bm25_index, top_k=inspect_k)
        elif mode == "hybrid":
            results = hybrid_retriever(question_text, vector_store, bm25_index,
                                      top_k=inspect_k, dense_top_n=inspect_k, sparse_top_n=inspect_k)
        else:
            raise ValueError("Unknown mode")
        def extract_chunk_ids(results, mode):
            if mode == "sparse":
                # results = [(chunk_id, bm25_score)]
                return [chunk_id for chunk_id, _ in results]
            else:
                # dense / hybrid
                return [chunk_id for chunk_id, _, _, _ in results]
        return extract_chunk_ids(results, mode)

    for _, row in questions_df.iterrows():
        question_id = row["question_id"]
        question_text = row["question"]
        gold_chunk_id = int(row["gold_chunk_id"])
        gold_doc_id = row.get("gold_doc_id", "")

        dense_ids = eval_one("dense", question_text)
        sparse_ids = eval_one("sparse", question_text)
        hybrid_ids = eval_one("hybrid", question_text)

        def rank_of_gold(ids):
            try:
                return ids.index(gold_chunk_id) + 1
            except ValueError:
                return ""

        def in_top_k(ids, k=4):
            return gold_chunk_id in ids[:k]

        result = {
            "question_id": question_id,
            "question": question_text,
            "gold_chunk_id": gold_chunk_id,
            "gold_doc_id": gold_doc_id,

            # Dense (Week-2 metrics)
            "retrieved_chunk_ids_dense": "|".join(map(str, dense_ids)),
            "rank_of_first_relevant_dense": rank_of_gold(dense_ids),
            "retrieved_in_top_k_dense": in_top_k(dense_ids, k=4),

            "notes": row.get("notes", "")
        }

        # Sparse (Week-3 comparison)
        if sparse_ids is not None:
            result.update({
                "retrieved_chunk_ids_sparse": "|".join(map(str, sparse_ids)),
                "rank_of_first_relevant_sparse": rank_of_gold(sparse_ids),
                "retrieved_in_top_k_sparse": in_top_k(sparse_ids, k=4),
            })

        # Hybrid (Week-3 comparison)
        if hybrid_ids is not None:
            result.update({
                "retrieved_chunk_ids_hybrid": "|".join(map(str, hybrid_ids)),
                "rank_of_first_relevant_hybrid": rank_of_gold(hybrid_ids),
                "retrieved_in_top_k_hybrid": in_top_k(hybrid_ids, k=4),

                # Deltas (allowed because derived from existing metrics)
                "delta_rank": (
                    (rank_of_gold(dense_ids) - rank_of_gold(hybrid_ids))
                    if rank_of_gold(dense_ids) != "" and rank_of_gold(hybrid_ids) != ""
                    else ""
                )
            })

        evaluation_results.append(result)

    pd.DataFrame(evaluation_results).to_csv(args.eval_output, index=False)
    print(f"Retrieval evaluation results saved to {args.eval_output}\n")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default=r"data/") # Path to directory containing PDFs
    parser.add_argument("--query", default="What downstream actions are suggested after Step-1 completion?") # Query for retrieval
    
    parser.add_argument("--export-chunks", action="store_true") # Export chunks to CSV for debugging
    parser.add_argument("--corpus-diag", action="store_true") # Print corpus diagnostics
    parser.add_argument("--run-retrieval-eval", action="store_true") # Run retrieval evaluation
    
    parser.add_argument("--questions-csv", default=r"data/retrieval_eval.csv") # Path to questions csv
    parser.add_argument("--eval-output", default=r"data/retrieval_evaluation_results.csv") # output to eval results

    parser.add_argument("--hybrid-retrieval", action="store_true") # enable hybrid retrieval
    parser.add_argument("--hybrid-output", default=r"data/hybrid_evaluation_results.csv") # output to hybrid retrieval

    args = parser.parse_args()

    pdf_path = args.pdf_dir
    query = args.query
    all_chunks = {}
    global_chunk_id = 0
    corpus_diagnostics = {}
    top_k = 4
    results = None
    
    # Ingest PDFs and create chunks
    for filename in os.listdir(pdf_path):
        if filename.endswith(".pdf"):
            pdf_text = load_pdf(os.path.join(pdf_path, filename))
            chunks = chunk_texts(pdf_text)
            corpus_diagnostics[filename] = len(chunks)
            for _, chunk_text in chunks.items():
                # Preserve document boundary via prefix (no new data structures)
                all_chunks[global_chunk_id] = {
                    "doc_id": filename,
                    "text": chunk_text
                }

                global_chunk_id += 1

                if global_chunk_id >= 1000:
                    print("⚠️ Chunk limit reached. Document truncated for control-system execution.\n")
                    break
    
    # Create vector store and BM25 index
    vector_store = create_vector_store(all_chunks)
    bm25_index = create_bm25_index(all_chunks)


    ############# Retrieval Observability & Debugging #############
    # Export chunks to CSV for debugging
    if args.export_chunks:
        print("\n")
        export_chunks_csv(all_chunks)

    # Corpus diagnostics
    if args.corpus_diag:
        print("\nCorpus Diagnostics:\n")

        for doc, chunk_count in corpus_diagnostics.items():
            print(f"Document: {doc} | Chunks: {chunk_count}")
            
        print(f"\nTotal chunks across corpus: {len(all_chunks)}\n")
        print("Chunk ID → Document ID mapping:")
        for chunk_id, chunk_info in all_chunks.items():
            print(f"Chunk ID: {chunk_id} | Document ID: {chunk_info['doc_id']}")
        
        print("\nCorpus Diagnostics Complete.\n")

    # Retrieval evaluation
    # updated to run hybrid
    if args.run_retrieval_eval:
        run_retrieval_evaluation(args, vector_store, bm25_index=bm25_index)
        return

    ############# End of Retrieval Observability & Debugging #############

    # Hybrid retrieval
    if args.hybrid_retrieval:
        results = hybrid_retriever(query, vector_store, bm25_index, top_k=top_k)
        print(f"\nTop {top_k} hybrid ranked chunks retrieved:")
    
    # Standard retrieval
    else:
        results = retrieve_similar_documents(vector_store, query, top_k=top_k)
        print(f"Top {top_k} similar chunks retrieved:")
    
    # Prepare context for LLM
    context = ""

    for chunk_id, doc_id, chunk_text, score in results:
        print(f"Chunk {chunk_id} | doc={doc_id} | similarity={score:.4f}")
        context += f"\n[Chunk {chunk_id} | Source: {doc_id}]\n{chunk_text}"

    prompt = f"""
You are answering a question using ONLY the information provided below.

If the information is insufficient, respond exactly with:
"I don’t have enough information in the provided documents."

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question:
{query}
"""
    
    # Get LLM response
    response = get_llm_response(prompt)
    print("LLM Response:")
    print(response)

if __name__ == "__main__":
    main()
