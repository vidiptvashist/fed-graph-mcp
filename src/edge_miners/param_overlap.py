from typing import List, Dict, Any, Tuple

class ParamOverlapMiner:
    """
    Mines tool relationship edges based on Parameter overlap.
    Computes Jaccard similarity over parameter type sets:
    jaccard = |T_A intersect T_B| / |T_A union T_B|.
    """
    
    def mine_edges(self, tools: List[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
        """
        Computes Jaccard similarity over parameter names for all pairs of tools.
        Yields edges if similarity >= 0.3.
        Excludes generic parameter names.
        Returns a list of (tool_id_A, tool_id_B, weight).
        """
        edges = []
        n_tools = len(tools)
        
        generic_names = {'id', 'name', 'type', 'query', 'text', 'desc', 'description', 'key', 'value', 'data', 'params', 'args'}
        
        # Pre-compute parameter name sets
        tool_sets = []
        for t in tools:
            param_names = t.get("parameter_names", [])
            names_set = set(param_names) - generic_names if param_names else set()
            tool_sets.append(names_set)
            
        for i in range(n_tools):
            id_A = tools[i]["id"]
            names_A = tool_sets[i]
            
            # Skip tools with no non-generic parameters
            if not names_A:
                continue
                
            for j in range(i + 1, n_tools):
                names_B = tool_sets[j]
                
                # Skip tools with no non-generic parameters
                if not names_B:
                    continue
                    
                # Compute Jaccard Similarity: |A intersect B| / |A union B|
                intersection = names_A.intersection(names_B)
                if not intersection:
                    continue
                    
                union = names_A.union(names_B)
                jaccard = len(intersection) / len(union) if union else 0.0
                
                if jaccard >= 0.3:
                    id_B = tools[j]["id"]
                    # Add directed edges in both directions (symmetric)
                    edges.append((id_A, id_B, jaccard))
                    edges.append((id_B, id_A, jaccard))
                    
        print(f"[ParamOverlapMiner] Mined {len(edges)} parameter overlap edges.")
        return edges
