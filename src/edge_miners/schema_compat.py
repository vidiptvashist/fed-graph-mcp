from typing import List, Dict, Any, Tuple

class SchemaCompatMiner:
    """
    Mines tool relationship edges based on Schema compatibility.
    Computes type intersection over parameter type sets:
    overlap = |T_A intersect T_B| / min(|T_A|, |T_B|).
    """
    
    def __init__(self, use_dense: bool = False, threshold: float = None):
        self.use_dense = use_dense
        self.threshold = threshold
        
    def mine_edges(self, tools: List[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
        """
        If use_dense=True, computes schema compatibility overlap based on parameter types.
        If use_dense=False, computes schema compatibility overlap based on parameter names.
        Yields edges if overlap >= 0.7.
        """
        edges = []
        n_tools = len(tools)
        
        if self.use_dense:
            # Pre-compute parameter type sets
            tool_sets = []
            for t in tools:
                param_types = t.get("parameter_types", [])
                tool_sets.append(set(param_types) if param_types else set())
                
            for i in range(n_tools):
                id_A = tools[i]["id"]
                types_A = tool_sets[i]
                if not types_A:
                    continue
                    
                for j in range(i + 1, n_tools):
                    types_B = tool_sets[j]
                    if not types_B:
                        continue
                        
                    intersection = types_A.intersection(types_B)
                    overlap = len(intersection) / min(len(types_A), len(types_B)) if min(len(types_A), len(types_B)) > 0 else 0.0
                    
                    thresh_val = self.threshold if self.threshold is not None else 0.7
                    if overlap >= thresh_val:
                        id_B = tools[j]["id"]
                        edges.append((id_A, id_B, overlap))
                        edges.append((id_B, id_A, overlap))
                        
            print(f"[SchemaCompatMiner] Mined {len(edges)} schema compatibility edges (Dense mode).")
            return edges
            
        else:
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
                if not names_A:
                    continue
                    
                for j in range(i + 1, n_tools):
                    names_B = tool_sets[j]
                    if not names_B:
                        continue
                        
                    intersection = names_A.intersection(names_B)
                    if not intersection:
                        continue
                        
                    overlap = len(intersection) / min(len(names_A), len(names_B))
                    thresh_val = self.threshold if self.threshold is not None else 0.7
                    if overlap >= thresh_val:
                        id_B = tools[j]["id"]
                        edges.append((id_A, id_B, overlap))
                        edges.append((id_B, id_A, overlap))
                        
            print(f"[SchemaCompatMiner] Mined {len(edges)} schema compatibility edges.")
            return edges
