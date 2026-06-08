import os
import json
import requests
import re
from typing import List, Dict, Any, Tuple

class ComposeDepMiner:
    """
    Mines tool relationship edges based on Compositional dependencies (asymmetric).
    Prompts GPT-4o-mini or uses a rule-based heuristic fallback if OPENAI_API_KEY is not available.
    Uses a local JSON file cache to persist LLM query results.
    """
    
    def __init__(self, cache_path: str = ".compose_dep_cache.json", api_key: str = None, model_provider: str = None):
        self.cache_path = cache_path
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_provider = model_provider or os.environ.get("MODEL_PROVIDER")
        self.cache = self._load_cache()
        
        # Auto-detect provider if not explicitly specified
        if not self.model_provider:
            if self.api_key:
                self.model_provider = "openai"
            elif os.environ.get("OLLAMA_MODEL") or os.environ.get("OLLAMA_HOST"):
                self.model_provider = "ollama"
            else:
                self.model_provider = "heuristic"
                
    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ComposeDepMiner] Failed to load cache: {e}")
        return {}
        
    def _save_cache(self) -> None:
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"[ComposeDepMiner] Failed to save cache: {e}")
            
    def _get_cache_key(self, tool_A: Dict[str, Any], tool_B: Dict[str, Any]) -> str:
        # Sort keys to ensure direction-specific keying
        return f"{tool_A['id']}->{tool_B['id']}"
        
    def _query_heuristic(self, tool_A: Dict[str, Any], tool_B: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Rule-based precursor heuristic fallback.
        Matches read-before-write flows for the same entities or auth requirements within the same server.
        """
        name_A = tool_A["name"].lower()
        name_B = tool_B["name"].lower()
        server_A = tool_A.get("server", "")
        server_B = tool_B.get("server", "")
        
        # 1. Auth dependency (A is auth/token tool on the same server, B is not)
        auth_words = ["token", "login", "auth", "session"]
        is_A_auth = any(w in name_A for w in auth_words)
        is_B_auth = any(w in name_B for w in auth_words)
        if is_A_auth and not is_B_auth and server_A == server_B:
            return True, 0.9, f"Tool {tool_A['name']} provides authentication for server {server_A} tools."
            
        # 2. Entity flow dependency: A reads/gets, B writes/updates/deletes on same entity
        read_words = ["get", "read", "search", "query", "find", "list", "show", "retrieve", "fetch", "check"]
        write_words = ["update", "delete", "remove", "modify", "cancel", "add", "create", "write", "set", "post", "record", "book", "send", "play", "timed"]
        
        # Check if A matches a read action
        A_action = None
        for r in read_words:
            if name_A.startswith(r) or f"_{r}" in name_A:
                A_action = r
                break
                
        # Check if B matches a write action
        B_action = None
        for w in write_words:
            if name_B.startswith(w) or f"_{w}" in name_B:
                B_action = w
                break
                
        if A_action and B_action:
            # Strip the action prefix to get the core noun / entity
            # e.g., QueryHealthData -> healthdata
            core_A = name_A.replace(A_action, "").replace("_", "").strip()
            core_B = name_B.replace(B_action, "").replace("_", "").strip()
            
            # If they operate on the same entity (or A is a substring of B's entity, or vice-versa)
            if core_A and core_B and (core_A == core_B or core_A in core_B or core_B in core_A):
                # Also ensure they belong to the same server if we have server context
                if server_A == server_B:
                    return True, 0.85, f"Entity read ({tool_A['name']}) matches entity write ({tool_B['name']}) flow."
                    
        return False, 0.0, "No compositional dependency detected via heuristics."
        
    def _query_llm(self, tool_A: Dict[str, Any], tool_B: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Queries GPT-4o-mini to check prerequisite relation.
        """
        prompt = f"""You are an expert AI system analyzing tool dependencies.
Analyze the following two tools:

Tool A (Potential Prerequisite):
Name: {tool_A['name']}
Description: {tool_A.get('description', '')}
Parameters: {tool_A.get('parameter_names', [])} (Types: {tool_A.get('parameter_types', [])})

Tool B (Dependent Tool):
Name: {tool_B['name']}
Description: {tool_B.get('description', '')}
Parameters: {tool_B.get('parameter_names', [])} (Types: {tool_B.get('parameter_types', [])})

Is Tool A a prerequisite, compositional dependency, or typical precursor for Tool B?
For example, you need to call a search/get tool (Tool A) to obtain an ID or data before calling an update/delete/modify tool (Tool B) on that same entity, or Tool A performs authentication needed for Tool B.

Respond strictly in JSON format:
{{
  "prerequisite": true/false,
  "confidence": float between 0.0 and 1.0,
  "reason": "explanation"
}}"""

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            res_json = response.json()
            content_str = res_json["choices"][0]["message"]["content"]
            result = json.loads(content_str)
            return (
                result.get("prerequisite", False),
                float(result.get("confidence", 0.0)),
                result.get("reason", "")
            )
        except Exception as e:
            print(f"[ComposeDepMiner] LLM call failed, falling back to heuristic: {e}")
            return self._query_heuristic(tool_A, tool_B)

    def _query_ollama(self, tool_A: Dict[str, Any], tool_B: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Queries a local Ollama model to check prerequisite relation.
        """
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        model = os.environ.get("OLLAMA_MODEL", "qwen2.5-instruct")
        url = f"{host}/api/chat"
        
        prompt = f"""You are an expert AI system analyzing tool dependencies.
Analyze the following two tools:

Tool A (Potential Prerequisite):
Name: {tool_A['name']}
Description: {tool_A.get('description', '')}
Parameters: {tool_A.get('parameter_names', [])} (Types: {tool_A.get('parameter_types', [])})

Tool B (Dependent Tool):
Name: {tool_B['name']}
Description: {tool_B.get('description', '')}
Parameters: {tool_B.get('parameter_names', [])} (Types: {tool_B.get('parameter_types', [])})

Is Tool A a prerequisite, compositional dependency, or typical precursor for Tool B?
For example, you need to call a search/get tool (Tool A) to obtain an ID or data before calling an update/delete/modify tool (Tool B) on that same entity, or Tool A performs authentication needed for Tool B.

Respond strictly in JSON format:
{{
  "prerequisite": true/false,
  "confidence": float between 0.0 and 1.0,
  "reason": "explanation"
}}"""

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=20)
            response.raise_for_status()
            res_json = response.json()
            content_str = res_json["message"]["content"]
            result = json.loads(content_str)
            return (
                result.get("prerequisite", False),
                float(result.get("confidence", 0.0)),
                result.get("reason", "")
            )
        except Exception as e:
            print(f"[ComposeDepMiner] Ollama call failed for model {model}, falling back to heuristic: {e}")
            return self._query_heuristic(tool_A, tool_B)

    def evaluate_dependency(self, tool_A: Dict[str, Any], tool_B: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Evaluates if Tool A is a prerequisite for Tool B.
        Checks the cache first.
        """
        cache_key = self._get_cache_key(tool_A, tool_B)
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            return entry["prerequisite"], entry["confidence"]
            
        if self.model_provider == "openai":
            prereq, conf, reason = self._query_llm(tool_A, tool_B)
        elif self.model_provider == "ollama":
            prereq, conf, reason = self._query_ollama(tool_A, tool_B)
        else:
            prereq, conf, reason = self._query_heuristic(tool_A, tool_B)
            
        # Write to cache
        self.cache[cache_key] = {
            "prerequisite": prereq,
            "confidence": conf,
            "reason": reason
        }
        
        return prereq, conf

    def mine_edges(self, tools: List[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
        """
        Computes prerequisite relationships.
        Scales computationally:
        - If <= 200 total tools: pairwise evaluation of all tools.
        - If > 200 total tools: only evaluates tools on the same server to prevent O(V^2) blowup.
        """
        edges = []
        n_tools = len(tools)
        
        if n_tools <= 200:
            # Pairwise evaluation for small tool sets (like APIBank)
            for i in range(n_tools):
                tool_A = tools[i]
                for j in range(n_tools):
                    if i == j:
                        continue
                    tool_B = tools[j]
                    prereq, confidence = self.evaluate_dependency(tool_A, tool_B)
                    if prereq and confidence > 0.8:
                        edges.append((tool_A["id"], tool_B["id"], confidence))
        else:
            # Scaled evaluation by grouping by server for large ecosystems (like MCP-tools)
            server_groups = {}
            for t in tools:
                server = t.get("server", "unknown")
                if server not in server_groups:
                    server_groups[server] = []
                server_groups[server].append(t)
                
            for server, group_tools in server_groups.items():
                m = len(group_tools)
                for i in range(m):
                    tool_A = group_tools[i]
                    for j in range(m):
                        if i == j:
                            continue
                        tool_B = group_tools[j]
                        prereq, confidence = self.evaluate_dependency(tool_A, tool_B)
                        if prereq and confidence > 0.8:
                            edges.append((tool_A["id"], tool_B["id"], confidence))
                            
        # Save cache once at the end of execution to avoid O(V^2) I/O bottleneck
        self._save_cache()
        print(f"[ComposeDepMiner] Mined {len(edges)} compositional dependency edges.")
        return edges
