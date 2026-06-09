import os
import sys
import time
import json
import gc
import numpy as np
import torch
import faiss
import matplotlib.pyplot as plt
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
        
    return float(np.mean(recalls_at_1))

def main():
    faiss.omp_set_num_threads(1)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(base_dir, "mcp_zero_repo/MCP-tools/mcp_tools_with_embedding.json")
    trajectory_dir = os.path.join(base_dir, "damo_convai_repo/api-bank/lv1-lv2-samples")
    cache_path = os.path.join(base_dir, "mcp_zero_repo/MCP-tools/.compose_dep_cache.json")
    
    # Sweep configurations
    thresholds = [0.9, 0.7, 0.5, 0.3, 0.1]
    sweep_results = []
    
    print("=" * 68)
    print("Graph Density, Memory, and Retrieval Sweep")
    print("=" * 68)
    
    # Initialize SentenceTransformer
    model_st = SentenceTransformer("all-MiniLM-L6-v2")
    
    for t in thresholds:
        print(f"\nEvaluating threshold: {t} ...")
        
        # 1. Build Graph
        builder = GraphBuilder(
            trajectory_dir=trajectory_dir,
            compose_cache_path=cache_path,
            use_dense=False,
            co_invoke_thresh=t,
            schema_compat_thresh=t,
            param_overlap_thresh=t
        )
        G = builder.build_graph(manifest_path)
        
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        est_mem_mb = (num_nodes * 1.5 + num_edges * 0.1) / 1024.0
        
        print(f"  Nodes: {num_nodes} | Edges: {num_edges} | Memory: {est_mem_mb:.2f} MB")
        
        # 2. Train GNN (3 epochs for speed)
        H, _, _ = train_rgcn(G, epochs=3, lr=0.01, device="cpu")
        
        # 3. Build FAISS Index
        gnn_index = build_faiss_index(H)
        
        # Build Raw Description Index
        all_tools = sorted(list(G.nodes()))
        node_to_idx = {node: i for i, node in enumerate(all_tools)}
        idx_to_node = {i: node for i, node in enumerate(all_tools)}
        descriptions = []
        for idx in range(len(idx_to_node)):
            node = idx_to_node[idx]
            desc = G.nodes[node].get("description", "") or node.split("/")[-1].replace("_", " ")
            descriptions.append(desc)
        X = model_st.encode(descriptions, show_progress_bar=False)
        raw_index = faiss.IndexHNSWFlat(384, 32)
        raw_index.add(np.ascontiguousarray(X.astype("float32")))
        
        # 4. Multi-seed tasks evaluation (3 seeds: 42, 43, 44)
        seeds = [42, 43, 44]
        r1_runs = []
        
        shuffled_tools = all_tools.copy()
        split_rng = np.random.RandomState(12345)
        split_rng.shuffle(shuffled_tools)
        split_idx = int(len(shuffled_tools) * 0.8)
        test_tools = shuffled_tools[split_idx:]
        
        for seed in seeds:
            tasks = build_semantic_tasks(test_tools, count=95, seed=seed, prefix="livemcp", G=G)
            r1 = evaluate_gnn_only(tasks, raw_index, gnn_index, H, idx_to_node, model_st)
            r1_runs.append(r1)
            
        mean_r1 = float(np.mean(r1_runs))
        print(f"  Mean GNN-Only Recall@1: {mean_r1:.4f}")
        
        sweep_results.append({
            "threshold": t,
            "edges": num_edges,
            "memory_mb": est_mem_mb,
            "gnn_recall_at_1": mean_r1
        })
        
        # Free memory
        del G, H, gnn_index, raw_index, X
        gc.collect()
        
    # Save sweep results to JSON
    json_path = os.path.join(base_dir, "density_sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSweep results saved to: {json_path}")
    
    # 5. Plot the Double-Y Axis Chart
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    
    # Sort results by number of edges (ascending) for line plotting
    sweep_results.sort(key=lambda x: x["edges"])
    edges = [x["edges"] for x in sweep_results]
    memory = [x["memory_mb"] for x in sweep_results]
    recall = [x["gnn_recall_at_1"] * 100.0 for x in sweep_results]
    
    color = "tab:blue"
    ax1.set_xlabel("Number of Edges", fontweight="bold")
    ax1.set_ylabel("GNN-Only Recall@1 (%)", color=color, fontweight="bold")
    line1 = ax1.plot(edges, recall, marker="o", color=color, linewidth=2, label="Recall@1")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Est. Graph Memory (MB)", color=color, fontweight="bold")
    line2 = ax2.plot(edges, memory, marker="s", color=color, linestyle="--", linewidth=2, label="Memory")
    ax2.tick_params(axis="y", labelcolor=color)
    
    # Added title
    plt.title("GNN Recall & Memory vs. Graph Edge Density Sweep", fontsize=12, fontweight="bold", pad=12)
    
    # Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")
    
    fig.tight_layout()
    
    # Ensure paper directory exists
    paper_dir = os.path.join(base_dir, "paper")
    os.makedirs(paper_dir, exist_ok=True)
    
    plot_path = os.path.join(paper_dir, "density_vs_memory_recall.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Sweep chart saved to: {plot_path}")
    
if __name__ == "__main__":
    main()
