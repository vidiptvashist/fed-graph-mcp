from typing import List, Dict, Any, Tuple

class SchemaCompatMiner:
    """
    Mines tool relationship edges based on Schema compatibility.
    Computes type intersection over parameter type sets:
    overlap = |T_A intersect T_B| / min(|T_A|, |T_B|).
    """
    
    def mine_edges(self, tools: List[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
        """
        Computes schema compatibility overlap for all pairs of tools and yields edges if overlap > 0.7.
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
            
            # Skip tools with no parameters
            if not types_A:
                continue
                
            for j in range(i + 1, n_tools):
                types_B = tool_sets[j]
                
                # Skip tools with no parameters
                if not types_B:
                    continue
                    
                # Compute overlap coefficient: |A intersect B| / min(|A|, |B|)
                intersection = types_A.intersection(types_B)
                overlap = len(intersection) / min(len(types_A), len(types_B))
                
                if overlap > 0.7:
                    id_B = tools[j]["id"]
                    # Add directed edges in both directions (symmetric)
                    edges.append((id_A, id_B, overlap))
                    edges.append((id_B, id_A, overlap))
                    
        print(f"[SchemaCompatMiner] Mined {len(edges)} schema compatibility edges.")
        return edges
