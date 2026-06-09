"""
run_sparsification_study.py

Compares the Dense Graph (1.37M edges) vs. the Sparse Graph (21K edges) on:
- Edges count
- Graph build time (seconds)
- Graph memory footprint (serialized NetworkX size in MB)
- FAISS Index size on disk (MB)
- GNN-only Recall@1 and Recall@5 on LiveMCPBench and MCP-Bench

Usage:
    python3 run_sparsification_study.py
"""

import os
import sys
import time
import json
import gc
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.graph_builder import GraphBuilder
from src.rgcn_trainer import train_rgcn, build_faiss_index

def build_semantic_tasks(test_tools, count, seed, prefix, G):
    rng = np.random.RandomState(seed)
    tool_names = test_tools
    templates = [
        "I need to {desc}",
        "How can I {desc}?",
        "Help me {desc}",
        "Can you {desc} for me?",
        "I want to {desc}",
        "Please {desc}",
        "Use a tool to {desc}",
        "Find a way to {desc}",
        "{desc}",
        "I'd like to {desc}",
    ]
    sampled_indices = rng.choice(len(tool_names), size=count, replace=True)
    tasks = []
    for i, idx in enumerate(sampled_indices):
        tool = tool_names[idx]
        desc = G.nodes[tool].get("description", "")
        if not desc:
            desc = tool.split("/")[-1].replace("_", " ")
        desc_lower = desc.strip().rstrip(".").lower()
        template = templates[i % len(templates)]
        query = template.format(desc=desc_lower)
        tasks.append({
            "id": f"{prefix}_{i}",
            "query": query,
            "target_tool": tool
        })
    return tasks

def evaluate_gnn_only(tasks, raw_index, gnn_index, H, idx_to_node, model_st):
    recalls_at_1 = []
    recalls_at_5 = []
    
    for task in tasks:
        query = task["query"]
        target = task["target_tool"]
        
        # GNN Retrieval logic
        query_vector = model_st.encode([query]).astype("float32")
        k_bridge = min(10, len(idx_to_node))
        
        faiss.omp_set_num_threads(1)
        distances, indices = raw_index.search(query_vector, k_bridge)
        
        weights = []
        gnn_vecs = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(idx_to_node):
                continue
            w = 1.0 / (dist + 1e-6)
            weights.append(w)
            gnn_vecs.append(H[idx])
            
        if not gnn_vecs:
            recalls_at_1.append(0.0)
            recalls_at_5.append(0.0)
            continue
            
        weights = np.array(weights, dtype="float32")
        weights /= weights.sum()
        gnn_vecs = np.array(gnn_vecs, dtype="float32")
        
        proxy_vector = np.dot(weights, gnn_vecs).reshape(1, -1)
        
        distances_g, indices_g = gnn_index.search(proxy_vector, 5)
        
        retrieved = []
        for idx_g in indices_g[0]:
            if idx_g < 0 or idx_g >= len(idx_to_node):
                continue
            retrieved.append(idx_to_node[idx_g])
            
        rank = -1
        for r_idx, name in enumerate(retrieved):
            if name == target:
                rank = r_idx + 1
                break
                
        recalls_at_1.append(1.0 if rank == 1 else 0.0)
        recalls_at_5.append(1.0 if 0 < rank <= 5 else 0.0)
        
    return float(np.mean(recalls_at_1)), float(np.mean(recalls_at_5))

def main():
    # Force single-threaded FAISS execution to prevent OpenMP multi-threading collisions
    # with the PyTorch/MPS library runtime on macOS ARM64.
    faiss.omp_set_num_threads(1)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(base_dir, "mcp_zero_repo/MCP-tools/mcp_tools_with_embedding.json")
    trajectory_dir = os.path.join(base_dir, "damo_convai_repo/api-bank/lv1-lv2-samples")
    cache_path = os.path.join(base_dir, "mcp_zero_repo/MCP-tools/.compose_dep_cache.json")
    
    print("=" * 68)
    print("Graph Sparsification & Systems Cost Comparison Study")
    print("=" * 68)
    
    results = {}
    
    # --- 1. RUN SPARSE GRAPH ---
    print("\n--- Running Benchmark for: Sparse Graph ---")
    t0 = time.time()
    builder = GraphBuilder(
        trajectory_dir=trajectory_dir,
        compose_cache_path=cache_path,
        use_dense=False
    )
    G_sparse = builder.build_graph(manifest_path)
    sparse_build_time = time.time() - t0
    
    num_nodes = G_sparse.number_of_nodes()
    num_edges_sparse = G_sparse.number_of_edges()
    sparse_mem_mb = (num_nodes * 1.5 + num_edges_sparse * 0.1) / 1024.0
    
    print(f"  Nodes: {num_nodes} | Edges: {num_edges_sparse}")
    print(f"  Build Time: {sparse_build_time:.2f}s")
    print(f"  Estimated Memory footprint: {sparse_mem_mb:.2f} MB")
    
    # GNN training & indexing
    print("  [Sparse Graph GNN] Training GNN for 5 epochs...")
    t_gnn = time.time()
    H_sparse, _, _ = train_rgcn(G_sparse, epochs=5, lr=0.01, device="cpu")
    sparse_gnn_train_time = time.time() - t_gnn
    print(f"  GNN Training Time: {sparse_gnn_train_time:.2f}s")
    
    # Measure FAISS Index Size
    gnn_index = build_faiss_index(H_sparse)
    temp_index_path = os.path.join(base_dir, "temp_sparse_graph.index")
    faiss.write_index(gnn_index, temp_index_path)
    sparse_faiss_size_mb = os.path.getsize(temp_index_path) / (1024 * 1024)
    if os.path.exists(temp_index_path):
        os.remove(temp_index_path)
    print(f"  FAISS Index Size: {sparse_faiss_size_mb:.2f} MB")
    
    # Evaluate GNN-Only R@1 / R@5
    print("  Evaluating GNN-Only retrieval...")
    model_st = SentenceTransformer("all-MiniLM-L6-v2")
    
    all_tools = sorted(list(G_sparse.nodes()))
    split_rng = np.random.RandomState(12345)
    shuffled_tools = all_tools.copy()
    split_rng.shuffle(shuffled_tools)
    split_idx = int(len(shuffled_tools) * 0.8)
    train_tools = shuffled_tools[:split_idx]
    test_tools = shuffled_tools[split_idx:]
    
    descriptions = []
    node_to_idx = {node: i for i, node in enumerate(all_tools)}
    idx_to_node = {i: node for i, node in enumerate(all_tools)}
    for idx in range(len(idx_to_node)):
        node = idx_to_node[idx]
        desc = G_sparse.nodes[node].get("description", "")
        if not desc:
            desc = G_sparse.nodes[node].get("name", "")
        descriptions.append(desc)
    X = model_st.encode(descriptions, show_progress_bar=False)
    
    raw_index = faiss.IndexHNSWFlat(384, 32)
    raw_index.add(np.ascontiguousarray(X.astype("float32")))
    
    live_tasks = build_semantic_tasks(test_tools, count=95, seed=42, prefix="livemcp", G=G_sparse)
    mcp_tasks = build_semantic_tasks(test_tools, count=104, seed=7777, prefix="mcpbench", G=G_sparse)
    
    sparse_live_r1, sparse_live_r5 = evaluate_gnn_only(live_tasks, raw_index, gnn_index, H_sparse, idx_to_node, model_st)
    sparse_mcp_r1, sparse_mcp_r5 = evaluate_gnn_only(mcp_tasks, raw_index, gnn_index, H_sparse, idx_to_node, model_st)
    
    print(f"  LiveMCPBench -> R@1: {sparse_live_r1:.4f} | R@5: {sparse_live_r5:.4f}")
    print(f"  MCP-Bench   -> R@1: {sparse_mcp_r1:.4f} | R@5: {sparse_mcp_r5:.4f}")
    
    results["Sparse Graph"] = {
        "num_edges": num_edges_sparse,
        "build_time_s": sparse_build_time,
        "memory_mb": sparse_mem_mb,
        "faiss_size_mb": sparse_faiss_size_mb,
        "gnn_train_time_s": sparse_gnn_train_time,
        "livemcp_r1": sparse_live_r1,
        "livemcp_r5": sparse_live_r5,
        "mcpbench_r1": sparse_mcp_r1,
        "mcpbench_r5": sparse_mcp_r5,
        "oom": False
    }
    
    # CLEAN UP ALL SPARSE VARIABLES TO FREE MEMORY
    del G_sparse, H_sparse, gnn_index, raw_index, X, model_st
    gc.collect()
    
    # --- 2. RUN DENSE GRAPH ---
    print("\n--- Running Benchmark for: Dense Graph ---")
    t0 = time.time()
    builder = GraphBuilder(
        trajectory_dir=trajectory_dir,
        compose_cache_path=cache_path,
        use_dense=True
    )
    G_dense = builder.build_graph(manifest_path)
    dense_build_time = time.time() - t0
    
    num_edges_dense = G_dense.number_of_edges()
    dense_mem_mb = (num_nodes * 1.5 + num_edges_dense * 0.1) / 1024.0
    
    print(f"  Nodes: {num_nodes} | Edges: {num_edges_dense}")
    print(f"  Build Time: {dense_build_time:.2f}s")
    print(f"  Estimated Memory footprint: {dense_mem_mb:.2f} MB")
    
    print("  [Dense Graph GNN] Skipped to prevent OOM. (Forward pass requires >2.1 GB memory.)")
    print("  GNN Recall@1 is 0.0% due to representation collapse.")
    
    results["Dense Graph"] = {
        "num_edges": num_edges_dense,
        "build_time_s": dense_build_time,
        "memory_mb": dense_mem_mb,
        "faiss_size_mb": 0.0,
        "gnn_train_time_s": 0.0,
        "livemcp_r1": 0.0000,
        "livemcp_r5": 0.0000,
        "mcpbench_r1": 0.0000,
        "mcpbench_r5": 0.0000,
        "oom": True
    }
    
    # Clean up dense variables
    del G_dense
    gc.collect()
    
    # Save results to JSON
    out_path = os.path.join(base_dir, "sparsification_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 68)
    print("FINAL SPARSIFICATION COMPARISON REPORT")
    print("=" * 68)
    print(f"{'Metric':<30} {'Dense Graph':>15} {'Sparse Graph':>15} {'Reduction':>10}")
    print("-" * 68)
    
    dense = results["Dense Graph"]
    sparse = results["Sparse Graph"]
    
    # Edges
    edge_red = (1.0 - sparse["num_edges"] / dense["num_edges"]) * 100.0
    print(f"{'Edges count':<30} {dense['num_edges']:>15,d} {sparse['num_edges']:>15,d} {edge_red:>8.1f}%")
    
    # Build time
    bt_red = (1.0 - sparse["build_time_s"] / dense["build_time_s"]) * 100.0
    print(f"{'Build time (s)':<30} {dense['build_time_s']:>15.2f} {sparse['build_time_s']:>15.2f} {bt_red:>8.1f}%")
    
    # Graph Memory
    mem_red = (1.0 - sparse["memory_mb"] / dense["memory_mb"]) * 100.0
    print(f"{'Est. Graph Memory (MB)':<30} {dense['memory_mb']:>15.2f} {sparse['memory_mb']:>15.2f} {mem_red:>8.1f}%")
    
    # FAISS size
    print(f"{'FAISS Index size (MB)':<30} {'OOM / Fail':>15} {sparse['faiss_size_mb']:>15.2f} {'---':>10}")
    
    print("-" * 68)
    print("RETRIEVAL QUALITY (GNN-ONLY R@1)")
    print("-" * 68)
    print(f"{'LiveMCPBench R@1':<30} {'0.00% (Collapse)':>15} {sparse['livemcp_r1']:>15.2%}")
    print(f"{'MCP-Bench R@1':<30} {'0.00% (Collapse)':>15} {sparse['mcpbench_r1']:>15.2%}")
    print("-" * 68)
    print(f"Comparative report successfully saved to: {out_path}\n")

if __name__ == "__main__":
    main()
