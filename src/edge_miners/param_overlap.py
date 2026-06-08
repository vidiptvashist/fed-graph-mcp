from typing import List, Dict, Any, Tuple

class ParamOverlapMiner:
    """
    Mines tool relationship edges based on Parameter overlap.
    Computes Jaccard similarity over parameter type sets:
    jaccard = |T_A intersect T_B| / |T_A union T_B|.
    """
    
    def mine_edges(self, tools: List[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
        """
        Computes Jaccard similarity for all pairs of tools and yields edges if similarity > 0.4.
        Returns a list of (tool_id_A, tool_id_B, weight).
        """
        edges = []
        n_tools = len(tools)
        
        # Pre-compute parameter type sets to avoid O(V^2) redundant set() conversions
        tool_sets = []
        for t in tools:
            param_types = t.get("parameter_types", [])
            tool_sets.append(set(param_types) if param_types else set())
            
        for i in range(n_tools):
            id_A = tools[i]["id"]
            types_A = tool_sets[i]
            
            # Skip tools with no parameters to avoid artificial fully-connected components
            if not types_A:
                continue
                
            for j in range(i + 1, n_tools):
                types_B = tool_sets[j]
                
                # Skip tools with no parameters
                if not types_B:
                    continue
                    
                # Compute Jaccard Similarity: |A intersect B| / |A union B|
                intersection = types_A.intersection(types_B)
                union = types_A.union(types_B)
                jaccard = len(intersection) / len(union) if union else 0.0
                
                if jaccard > 0.4:
                    id_B = tools[j]["id"]
                    # Add directed edges in both directions (symmetric)
                    edges.append((id_A, id_B, jaccard))
                    edges.append((id_B, id_A, jaccard))
                    
        print(f"[ParamOverlapMiner] Mined {len(edges)} parameter overlap edges.")
        return edges
