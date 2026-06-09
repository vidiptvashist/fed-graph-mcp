import networkx as nx
from typing import List, Dict, Any, Tuple

from src.utils import load_mcp_tools
from src.edge_miners.co_invoke import CoInvocationMiner
from src.edge_miners.schema_compat import SchemaCompatMiner
from src.edge_miners.param_overlap import ParamOverlapMiner
from src.edge_miners.compose_dep import ComposeDepMiner

class GraphBuilder:
    """
    Orchestrates the construction of a FED-GRAPH-MCP multi-typed tool graph.
    """
    
    def __init__(self, trajectory_dir: str = None, openai_api_key: str = None, compose_cache_path: str = ".compose_dep_cache.json", use_dense: bool = False):
        self.trajectory_dir = trajectory_dir
        self.openai_api_key = openai_api_key
        self.compose_cache_path = compose_cache_path
        self.use_dense = use_dense
        
    def build_graph(self, manifest_path: str) -> nx.MultiDiGraph:
        """
        Loads tool manifests, constructs nodes, runs miners, filters self-loops,
        and constructs the final NetworkX MultiDiGraph G.
        """
        # 1. Load and normalize tools
        tools = load_mcp_tools(manifest_path)
        print(f"[GraphBuilder] Loaded {len(tools)} tools from manifest: {manifest_path}")
        
        # 2. Initialize MultiDiGraph
        G = nx.MultiDiGraph()
        
        # 3. Add nodes with metadata attributes
        for t in tools:
            G.add_node(
                t["id"],
                name=t["name"],
                server=t["server"],
                description=t["description"],
                parameter_names=t["parameter_names"],
                parameter_types=t["parameter_types"],
                server_summary=t.get("server_summary", ""),
                description_embedding=t.get("description_embedding", [])
            )
            
        # 4. Instantiate miners and mine edges
        miners = {
            "co_invoke": CoInvocationMiner(self.trajectory_dir, use_dense=self.use_dense),
            "schema_compat": SchemaCompatMiner(use_dense=self.use_dense),
            "param_overlap": ParamOverlapMiner(use_dense=self.use_dense),
            "compose_dep": ComposeDepMiner(cache_path=self.compose_cache_path, api_key=self.openai_api_key)
        }
        
        for edge_type, miner in miners.items():
            print(f"[GraphBuilder] Running edge miner: {edge_type}...")
            mined_edges = miner.mine_edges(tools)
            
            for u, v, w in mined_edges:
                # Enforce no self-loops
                if u == v:
                    continue
                    
                # Ensure nodes exist in the graph before adding edge
                if not G.has_node(u) or not G.has_node(v):
                    continue
                    
                # Enforce edge weight range [0, 1]
                w_clipped = min(1.0, max(0.0, float(w)))
                
                # Add to MultiDiGraph. key=edge_type prevents duplicate edges of same type between u and v
                G.add_edge(u, v, key=edge_type, type=edge_type, weight=w_clipped)
                
        print(f"[GraphBuilder] Constructed graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G
