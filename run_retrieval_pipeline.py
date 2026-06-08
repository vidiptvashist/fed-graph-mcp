import os
import torch
import json
from src.retrieval_pipeline import (
    RetrievalPipeline,
    retrieve_dense,
    retrieve_gnn,
    fuse_rrf,
    graph_expand,
    train_reranker,
    LearnedReranker
)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    reranker_weights_path = os.path.join(base_dir, "mcp_reranker.pt")
    
    print("====================================================")
    print("Starting Retrieval, Fusion, and Reranking Pipeline")
    print("====================================================")
    
    # 1. Initialize the retrieval pipeline (loads graph, ST embeddings, FAISS indices)
    pipeline = RetrievalPipeline()
    
    # 2. Train or load the Learned Reranker model
    print("\n[Step 2] Training LearnedReranker MLP on API-Bank dataset...")
    # Train for 15 epochs with learning rate 0.001 and pairwise margin loss
    reranker = train_reranker(base_dir=base_dir, epochs=15, lr=0.001)
    
    # 3. Define a query to test retrieval
    # Let's query for sqlite table creation and row insertion
    query = "Create some tables in the sqlite database and insert data into them"
    k = 5
    
    print(f"\n[Step 3] Querying Retrieval Spaces...")
    print(f"Query: '{query}'")
    
    # 4. Dense Retrieval (raw SentenceTransformer embeddings)
    print("\n--- 1. Dense Retrieval (Raw description space) ---")
    dense_results = retrieve_dense(query, k, pipeline)
    for i, res in enumerate(dense_results):
        print(f"{i+1}. {res['name']} (Score: {res['score']:.4f})")
        print(f"   Desc: {res['description']}")
        
    # 5. GNN Retrieval (GNN-enriched embeddings H)
    print("\n--- 2. GNN Retrieval (GNN-enriched space) ---")
    gnn_results = retrieve_gnn(query, k, pipeline)
    for i, res in enumerate(gnn_results):
        print(f"{i+1}. {res['name']} (Score: {res['score']:.4f})")
        print(f"   Desc: {res['description']}")
        
    # 6. Reciprocal Rank Fusion (RRF)
    print("\n--- 3. Reciprocal Rank Fusion (RRF Baseline) ---")
    fused_results = fuse_rrf(dense_results, gnn_results, k=5, c=60)
    for i, res in enumerate(fused_results):
        print(f"{i+1}. {res['name']} (RRF Score: {res['score']:.4f})")
        print(f"   Desc: {res['description']}")
        
    # 7. Learned Reranker (MLP)
    print("\n--- 4. Learned Reranker MLP scoring and ranking ---")
    # Embed the query
    query_emb = torch.tensor(pipeline.model_st.encode([query]).astype("float32"))
    
    # Rerank the fused candidate tools
    candidates = fused_results.copy()
    candidate_embs = []
    for c in candidates:
        idx = pipeline.node_to_idx[c["name"]]
        # We can use the enriched GNN embedding or raw embedding of the candidate.
        # Let's use the enriched embedding.
        candidate_embs.append(torch.tensor(pipeline.H[idx]))
    candidate_embs = torch.stack(candidate_embs)
    
    # Compute scores from the 2-layer MLP
    with torch.no_grad():
        rerank_scores = reranker(query_emb, candidate_embs).cpu().numpy()
        
    for c, score in zip(candidates, rerank_scores):
        c["rerank_score"] = float(score)
        
    reranked_results = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    for i, res in enumerate(reranked_results):
        print(f"{i+1}. {res['name']} (Rerank Score: {res['rerank_score']:.4f})")
        print(f"   Desc: {res['description']}")
        
    # 8. Graph Expansion (1-hop)
    print("\n--- 5. Graph Expansion (1-hop neighbors) ---")
    # Take the top 2 reranked tools and expand them
    top_seeds = reranked_results[:2]
    expanded_results = graph_expand(top_seeds, pipeline.G, hop=1)
    
    print(f"Seed Nodes (top 2): {[s['name'] for s in top_seeds]}")
    print(f"Expanded Set (including neighbors along co_invoke / compose_dep):")
    for i, res in enumerate(expanded_results):
        print(f"{i+1}. {res['name']}")
        print(f"   Desc: {res['description']}")
        
    print("\n====================================================")
    print("Retrieval Pipeline Verification Finished Successfully")
    print("====================================================")

if __name__ == "__main__":
    main()
