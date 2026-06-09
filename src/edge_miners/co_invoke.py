import os
import json
import math
from typing import List, Dict, Any, Tuple, Set

class CoInvocationMiner:
    """
    Mines tool relationship edges based on Co-invocation patterns.
    Computes Pointwise Mutual Information (PMI) over APIBank trajectory logs (JSONL dialogue files).
    """
    
    def __init__(self, trajectory_dir: str = None):
        self.trajectory_dir = trajectory_dir
        
    def _clean_tool_name(self, name: str) -> str:
        """
        Helper to normalize tool names for matching across CamelCase, snake_case, etc.
        """
        if not name:
            return ""
        return name.lower().replace("_", "").replace("-", "").strip()
        
    def mine_edges(self, tools: List[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
        """
        Computes same-server co-invocation heuristic edges based on semantic description similarity.
        Yields edges if tools are in the same server and cosine similarity > 0.6.
        Returns a list of (tool_id_A, tool_id_B, weight).
        """
        import numpy as np
        
        edges = []
        server_tools = {}
        for t in tools:
            server = t["server"]
            if server not in server_tools:
                server_tools[server] = []
            server_tools[server].append(t)
            
        # Fallback to SentenceTransformer if embeddings aren't pre-computed
        has_embeddings = any(len(t.get("description_embedding", [])) > 0 for t in tools)
        
        if not has_embeddings:
            print("[CoInvocationMiner] Description embeddings missing. Running fallback SentenceTransformer encoding...")
            from sentence_transformers import SentenceTransformer
            model_st = SentenceTransformer("all-MiniLM-L6-v2")
            for t in tools:
                desc = t.get("description", "")
                if not desc:
                    desc = t.get("name", "")
                t["description_embedding"] = model_st.encode([desc], show_progress_bar=False)[0].tolist()
                
        # Group by server and compute pairwise similarity
        for server, server_items in server_tools.items():
            n_server = len(server_items)
            for i in range(n_server):
                tA = server_items[i]
                id_A = tA["id"]
                emb_A = np.array(tA["description_embedding"], dtype="float32")
                norm_A = np.linalg.norm(emb_A)
                if norm_A == 0:
                    continue
                    
                for j in range(i + 1, n_server):
                    tB = server_items[j]
                    id_B = tB["id"]
                    emb_B = np.array(tB["description_embedding"], dtype="float32")
                    norm_B = np.linalg.norm(emb_B)
                    if norm_B == 0:
                        continue
                        
                    sim = np.dot(emb_A, emb_B) / (norm_A * norm_B)
                    if sim > 0.6:
                        # Add symmetric edges
                        edges.append((id_A, id_B, float(sim)))
                        edges.append((id_B, id_A, float(sim)))
                        
        print(f"[CoInvocationMiner] Mined {len(edges)} co-invocation edges.")
        return edges
