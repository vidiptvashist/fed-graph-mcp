import re
import json
from typing import List, Dict, Any, Tuple

def parse_parameter_type_from_string(type_desc: str) -> str:
    """
    Parses a parameter description string like '(Optional, string) description'
    or '(integer) count' and returns the standard type name.
    """
    if not isinstance(type_desc, str):
        return "any"
    # Match content inside first parenthesis
    match = re.search(r'\(([^)]+)\)', type_desc)
    if not match:
        return "any"
    content = match.group(1).lower()
    # Split by comma in case of "Optional, string"
    parts = [p.strip() for p in content.split(',')]
    for p in parts:
        if p != 'optional':
            return p
    return "any"

def extract_parameter_types_from_tool(tool: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Extracts lists of parameter names and parameter types.
    Handles both '(type) description' dict and JSON Schema 'properties' dict.
    """
    param_names = []
    param_types = []
    
    # 1. MCP-tools dataset format: {"parameter": {"param1": "(string) desc1"}}
    if "parameter" in tool and isinstance(tool["parameter"], dict):
        for name, type_desc in tool["parameter"].items():
            param_names.append(name)
            param_types.append(parse_parameter_type_from_string(type_desc))
            
    # 2. Standard JSON Schema format: {"inputSchema": {"properties": {"param1": {"type": "string"}}}}
    elif "inputSchema" in tool and isinstance(tool["inputSchema"], dict):
        properties = tool["inputSchema"].get("properties", {})
        if isinstance(properties, dict):
            for name, prop_info in properties.items():
                param_names.append(name)
                val_type = "any"
                if isinstance(prop_info, dict):
                    val_type = prop_info.get("type", "any")
                param_types.append(val_type)
    
    # 3. Alternative direct properties field if any
    elif "properties" in tool and isinstance(tool["properties"], dict):
        for name, prop_info in tool["properties"].items():
            param_names.append(name)
            val_type = "any"
            if isinstance(prop_info, dict):
                val_type = prop_info.get("type", "any")
            param_types.append(val_type)
            
    return param_names, param_types

def load_mcp_tools(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads tool manifests from a JSON file and normalizes them.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    normalized_tools = []
    # If the top level is a list of servers
    if isinstance(data, list):
        for server in data:
            server_name = server.get("name") or server.get("server_name") or "unknown_server"
            server_desc = server.get("description") or server.get("server_description") or ""
            tools = server.get("tools", [])
            for tool in tools:
                tool_name = tool.get("name")
                desc = tool.get("description", "")
                param_names, param_types = extract_parameter_types_from_tool(tool)
                
                # Combine server name and tool name for unique ID
                node_id = f"{server_name}/{tool_name}"
                normalized_tools.append({
                    "id": node_id,
                    "name": tool_name,
                    "server": server_name,
                    "description": desc,
                    "parameter_names": param_names,
                    "parameter_types": param_types,
                    "raw": tool,
                    "server_summary": server.get("summary", ""),
                    "description_embedding": tool.get("description_embedding", [])
                })
    return normalized_tools
