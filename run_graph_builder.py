import os
import sys
import networkx as nx

# Add current workspace to path to import src modules
sys.path.append("/Users/vidiptvashist/Desktop/Projects/FED-GRAPH")

from src.graph_builder import GraphBuilder

def main():
    manifest_path = "/Users/vidiptvashist/Desktop/Projects/FED-GRAPH/mcp_zero_repo/MCP-tools/mcp_tools_with_embedding.json"
    trajectory_dir = "/Users/vidiptvashist/Desktop/Projects/FED-GRAPH/damo_convai_repo/api-bank/lv1-lv2-samples"
    cache_path = "/Users/vidiptvashist/Desktop/Projects/FED-GRAPH/mcp_zero_repo/MCP-tools/.compose_dep_cache.json"
    
    print("====================================================")
    print("Building FED-GRAPH-MCP on full dataset...")
    print(f"Manifest Path: {manifest_path}")
    print(f"Trajectory Dir: {trajectory_dir}")
    print("====================================================")
    
    builder = GraphBuilder(
        trajectory_dir=trajectory_dir,
        compose_cache_path=cache_path,
        openai_api_key=None  # Use rule-based fallback heuristic
    )
    
    G = builder.build_graph(manifest_path)
    
    # Analyze graph structure
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    print("\n--- General Statistics ---")
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")
    
    if num_nodes > 0:
        density = num_edges / (num_nodes * (num_nodes - 1))
        print(f"Graph Density: {density:.6f}")
        
    # Analyze edge type counts and average weights
    type_counts = {}
    type_weights = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        etype = d["type"]
        weight = d["weight"]
        type_counts[etype] = type_counts.get(etype, 0) + 1
        type_weights[etype] = type_weights.get(etype, 0.0) + weight
        
    print("\n--- Edge Statistics by Type ---")
    for etype in sorted(type_counts.keys()):
        count = type_counts[etype]
        avg_weight = type_weights[etype] / count
        print(f"Type '{etype}': {count} edges (average weight: {avg_weight:.4f})")
        
    # Self-loops check
    self_loops = len(list(nx.selfloop_edges(G)))
    print(f"\nSelf-loops: {self_loops}")
    
    # Weight bounds check
    invalid_weights = 0
    for u, v, k, d in G.edges(keys=True, data=True):
        if not (0.0 <= d["weight"] <= 1.0):
            invalid_weights += 1
    print(f"Edges with weights outside [0, 1]: {invalid_weights}")
    
    # Weakly connected components check
    is_weakly_connected = nx.is_weakly_connected(G)
    num_components = nx.number_weakly_connected_components(G)
    print(f"\nIs Weakly Connected: {is_weakly_connected}")
    print(f"Number of Weakly Connected Components: {num_components}")
    
    if num_components > 1:
        sizes = [len(c) for c in nx.weakly_connected_components(G)]
        sizes.sort(reverse=True)
        print(f"Component size distribution (top 10): {sizes[:10]}")
        
if __name__ == "__main__":
    main()
