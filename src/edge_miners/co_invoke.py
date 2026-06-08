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
        Computes PMI over all dialogue logs and yields edges where PMI > 0.5.
        Returns a list of (tool_id_A, tool_id_B, normalized_weight).
        """
        if not self.trajectory_dir or not os.path.isdir(self.trajectory_dir):
            print(f"[CoInvocationMiner] Trajectory directory '{self.trajectory_dir}' not found. Skipping co-invocation mining.")
            return []
            
        # 1. Collect all dialogue files
        dialogue_files = []
        for root, _, files in os.walk(self.trajectory_dir):
            for file in files:
                if file.endswith('.jsonl'):
                    dialogue_files.append(os.path.join(root, file))
                    
        N = len(dialogue_files)
        if N == 0:
            print("[CoInvocationMiner] No dialogue files found in trajectory directory.")
            return []
            
        # Map cleaned tool names from the manifest to their full node IDs
        cleaned_to_ids = {}
        for tool in tools:
            cleaned = self._clean_tool_name(tool["name"])
            if cleaned not in cleaned_to_ids:
                cleaned_to_ids[cleaned] = []
            cleaned_to_ids[cleaned].append(tool["id"])
            
        # 2. Extract API invocation sets from each dialogue log
        # Each dialogue has a set of unique tools called
        dialogue_invocations = []
        tool_counts = {}  # Count of dialogues containing tool A (keyed by cleaned name)
        
        for file_path in dialogue_files:
            invoked_in_dialogue = set()
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        turn = json.loads(line)
                        role = turn.get("role", "").upper()
                        # Extract api_name if the turn represents an API call
                        if role == "API":
                            api_name = turn.get("api_name") or turn.get("api")
                            if api_name:
                                invoked_in_dialogue.add(self._clean_tool_name(api_name))
            except Exception as e:
                print(f"[CoInvocationMiner] Error reading dialogue log {file_path}: {e}")
                continue
                
            dialogue_invocations.append(invoked_in_dialogue)
            for api in invoked_in_dialogue:
                tool_counts[api] = tool_counts.get(api, 0) + 1
                
        # 3. Compute joint counts for pairs of tools that both exist in our manifest
        joint_counts = {}
        cleaned_manifest_tools = list(cleaned_to_ids.keys())
        
        for invoked_set in dialogue_invocations:
            # Filter invoked tools to only those present in the manifest
            invoked_manifest_tools = [t for t in invoked_set if t in cleaned_to_ids]
            # Add pairwise occurrences
            for i in range(len(invoked_manifest_tools)):
                for j in range(i + 1, len(invoked_manifest_tools)):
                    t1, t2 = invoked_manifest_tools[i], invoked_manifest_tools[j]
                    pair = tuple(sorted([t1, t2]))
                    joint_counts[pair] = joint_counts.get(pair, 0) + 1
                    
        # 4. Compute PMI and build edges
        edges = []
        max_possible_pmi = math.log2(N) if N > 1 else 1.0
        
        for (t1, t2), count_joint in joint_counts.items():
            count_t1 = tool_counts.get(t1, 0)
            count_t2 = tool_counts.get(t2, 0)
            
            if count_t1 == 0 or count_t2 == 0:
                continue
                
            # PMI Formula: log2( P(t1, t2) / (P(t1)*P(t2)) )
            # P(t1, t2) = count_joint / N
            # P(t1) = count_t1 / N, P(t2) = count_t2 / N
            # PMI = log2( (count_joint * N) / (count_t1 * count_t2) )
            pmi = math.log2((count_joint * N) / (count_t1 * count_t2))
            
            if pmi > 0.5:
                # Normalize weight to [0, 1] using max possible PMI (log2(N))
                weight = min(1.0, max(0.0, pmi / max_possible_pmi))
                
                # Add edges between all matched tool IDs (handles duplicates/overloaded tool names)
                for id_t1 in cleaned_to_ids[t1]:
                    for id_t2 in cleaned_to_ids[t2]:
                        # Since co-invocation is undirected, add directed edges in both directions
                        edges.append((id_t1, id_t2, weight))
                        edges.append((id_t2, id_t1, weight))
                        
        print(f"[CoInvocationMiner] Mined {len(edges)} co-invocation edges.")
        return edges
