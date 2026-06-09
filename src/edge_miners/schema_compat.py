from typing import List, Dict, Any, Tuple

class SchemaCompatMiner:
    """
    Mines tool relationship edges based on Schema compatibility.
    Computes type intersection over parameter type sets:
    overlap = |T_A intersect T_B| / min(|T_A|, |T_B|).
    """
    
    def mine_edges(self, tools: List[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
        """
        Computes schema compatibility overlap for all pairs of tools based on parameter names.
        Yields edges if overlap >= 0.7.
        Excludes generic parameter names to avoid false positive hubs.
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
                    
                # Compute overlap coefficient: |A intersect B| / min(|A|, |B|)
                intersection = names_A.intersection(names_B)
                if not intersection:
                    continue
                    
                overlap = len(intersection) / min(len(names_A), len(names_B))
                
                if overlap >= 0.7:
                    id_B = tools[j]["id"]
                    # Add directed edges in both directions (symmetric)
                    edges.append((id_A, id_B, overlap))
                    edges.append((id_B, id_A, overlap))
                    
        print(f"[SchemaCompatMiner] Mined {len(edges)} schema compatibility edges.")
        return edges
