import os
import json
import numpy as np
import torch
import faiss
from src.graph_builder import GraphBuilder
from src.rgcn_trainer import train_rgcn, build_faiss_index

def run_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Setup paths matching Module 1 config
    manifest_path = os.path.join(base_dir, "mcp_zero_repo/MCP-tools/mcp_tools_with_embedding.json")
    trajectory_dir = os.path.join(base_dir, "damo_convai_repo/api-bank/lv1-lv2-samples")
    cache_path = os.path.join(base_dir, "mcp_zero_repo/MCP-tools/.compose_dep_cache.json")
    
    # Output file paths
    embeddings_out_path = os.path.join(base_dir, "mcp_enriched_embeddings.npy")
    mapping_out_path = os.path.join(base_dir, "mcp_node_mapping.json")
    faiss_index_out_path = os.path.join(base_dir, "mcp_hnsw_index.faiss")
    
    print("====================================================")
    # Check model device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        
    print(f"Starting R-GCN Training Pipeline on device: {device.upper()}")
    print(f"Manifest: {manifest_path}")
    print("====================================================")
    
    # 2. Build the MultiDiGraph G
    print("\n[Pipeline] Building multi-relational graph G...")
    builder = GraphBuilder(
        trajectory_dir=trajectory_dir,
        compose_cache_path=cache_path,
        openai_api_key=None  # Fallback to heuristic
    )
    G = builder.build_graph(manifest_path)
    
    # 3. Train R-GCN and generate enriched embeddings
    print("\n[Pipeline] Initializing and training R-GCN Model...")
    # 50 epochs is sufficient for convergence and fast execution
    H, node_to_idx, model = train_rgcn(
        G,
        epochs=2,
        lr=0.01,
        val_ratio=0.1,
        test_ratio=0.1,
        device=device
    )
    
    # 4. Save numpy embeddings & node to index mapping
    print("\n[Pipeline] Saving node embeddings and index mapping...")
    np.save(embeddings_out_path, H)
    print(f"[Pipeline] Saved enriched embeddings shape {H.shape} to: {embeddings_out_path}")
    
    model_out_path = os.path.join(base_dir, "mcp_rgcn_model.pt")
    torch.save(model.state_dict(), model_out_path)
    print(f"[Pipeline] Saved trained GNN model weights to: {model_out_path}")
    
    with open(mapping_out_path, "w", encoding="utf-8") as f:
        json.dump(node_to_idx, f, indent=2)
    print(f"[Pipeline] Saved node mapping dict to: {mapping_out_path}")
    
    # 5. Build FAISS IndexHNSWFlat and save it
    print("\n[Pipeline] Building FAISS IndexHNSWFlat index...")
    index = build_faiss_index(H)
    
    print(f"[Pipeline] Saving FAISS index to: {faiss_index_out_path}")
    faiss.write_index(index, faiss_index_out_path)
    print("[Pipeline] FAISS index saved successfully.")
    
    # 6. Run Nearest-Neighbor Validation Query
    print("\n=================== VALIDATION QUERY ===================")
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    # Choose a sample query node (prefer a well-connected one like Sqlite/create_table if available)
    query_node = "Sqlite/create_table"
    if query_node not in node_to_idx:
        query_node = idx_to_node[0]
        
    query_idx = node_to_idx[query_node]
    query_vector = H[query_idx].reshape(1, -1).astype("float32")
    
    # Query FAISS index for top 5 neighbors
    k = 5
    distances, indices = index.search(query_vector, k)
    
    print(f"Query Node: {query_node}")
    print(f"Description: {G.nodes[query_node].get('description')}\n")
    print(f"Top {k} Nearest Neighbors in Enriched Space:")
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        neighbor_node = idx_to_node[idx]
        neighbor_data = G.nodes[neighbor_node]
        print(f"{rank + 1}. {neighbor_node} (Distance: {dist:.4f})")
        print(f"   Description: {neighbor_data.get('description')}")
    print("========================================================\n")

if __name__ == "__main__":
    run_pipeline()
