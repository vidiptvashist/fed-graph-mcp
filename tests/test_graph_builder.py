import os
import json
import pytest
import tempfile
import shutil
import networkx as nx

from src.utils import load_mcp_tools, extract_parameter_types_from_tool
from src.edge_miners.co_invoke import CoInvocationMiner
from src.edge_miners.schema_compat import SchemaCompatMiner
from src.edge_miners.param_overlap import ParamOverlapMiner
from src.edge_miners.compose_dep import ComposeDepMiner
from src.graph_builder import GraphBuilder

@pytest.fixture
def mock_manifest_file():
    manifest_data = [
        {
            "name": "ServerA",
            "description": "Mock Server A",
            "summary": "Summary A",
            "tools": [
                {
                    "name": "get_user_token",
                    "description": "Get authentication token",
                    "parameter": {
                        "username": "(string) The username"
                    }
                },
                {
                    "name": "query_alarm",
                    "description": "Query all set alarms",
                    "parameter": {
                        "user_id": "(string) The user identity"
                    }
                },
                {
                    "name": "cancel_alarm",
                    "description": "Cancel a specific alarm",
                    "parameter": {
                        "alarm_id": "(integer) The ID of alarm to cancel",
                        "user_id": "(string) The user identity"
                    }
                }
            ]
        },
        {
            "name": "ServerB",
            "description": "Mock Server B",
            "summary": "Summary B",
            "tools": [
                {
                    "name": "fetch_weather",
                    "description": "Get current weather info",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "lat": {"type": "number", "description": "Latitude"},
                            "lon": {"type": "number", "description": "Longitude"}
                        }
                    }
                }
            ]
        }
    ]
    
    temp_dir = tempfile.mkdtemp()
    manifest_path = os.path.join(temp_dir, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f)
        
    yield manifest_path
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_trajectory_dir():
    # Setup some mock dialogue files
    temp_dir = tempfile.mkdtemp()
    
    # Dialogue 1: calls query_alarm and cancel_alarm
    d1 = [
        {"role": "API", "api_name": "query_alarm"},
        {"role": "API", "api_name": "cancel_alarm"}
    ]
    
    # Dialogue 2: calls query_alarm and cancel_alarm
    d2 = [
        {"role": "API", "api_name": "query_alarm"},
        {"role": "API", "api_name": "cancel_alarm"}
    ]
    
    # Dialogue 3: calls fetch_weather only
    d3 = [
        {"role": "API", "api_name": "fetch_weather"}
    ]
    
    # Dialogue 4: calls fetch_weather only
    d4 = [
        {"role": "API", "api_name": "fetch_weather"}
    ]
    
    for idx, d in enumerate([d1, d2, d3, d4]):
        file_path = os.path.join(temp_dir, f"dialogue_{idx}.jsonl")
        with open(file_path, 'w', encoding='utf-8') as f:
            for turn in d:
                f.write(json.dumps(turn) + "\n")
                
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_manifest_parsing(mock_manifest_file):
    tools = load_mcp_tools(mock_manifest_file)
    assert len(tools) == 4
    
    # Check ServerA/get_user_token
    token_tool = next(t for t in tools if t["name"] == "get_user_token")
    assert token_tool["server"] == "ServerA"
    assert token_tool["parameter_names"] == ["username"]
    assert token_tool["parameter_types"] == ["string"]
    
    # Check ServerB/fetch_weather (JSON Schema properties)
    weather_tool = next(t for t in tools if t["name"] == "fetch_weather")
    assert weather_tool["server"] == "ServerB"
    assert set(weather_tool["parameter_names"]) == {"lat", "lon"}
    assert set(weather_tool["parameter_types"]) == {"number"}

def test_co_invocation_miner(mock_manifest_file, mock_trajectory_dir):
    tools = load_mcp_tools(mock_manifest_file)
    miner = CoInvocationMiner(mock_trajectory_dir)
    edges = miner.mine_edges(tools)
    
    # Check that edges exist. There should be edges between:
    # - query_alarm and cancel_alarm (co-occur in d1)
    # - get_user_token and query_alarm (co-occur in d4)
    assert len(edges) > 0
    
    # Check symmetric property (if A->B exists, B->A exists with same weight)
    for u, v, w in edges:
        reversed_edge = next((x for x in edges if x[0] == v and x[1] == u), None)
        assert reversed_edge is not None
        assert reversed_edge[2] == pytest.approx(w)

def test_schema_compat_miner():
    # Tool A parameters: ['string', 'integer']
    # Tool B parameters: ['string'] -> overlap is 1.0 (intersection is {'string'}, min size is 1)
    # Tool C parameters: ['number'] -> overlap is 0.0
    tools = [
        {"id": "A", "parameter_types": ["string", "integer"]},
        {"id": "B", "parameter_types": ["string"]},
        {"id": "C", "parameter_types": ["number"]}
    ]
    miner = SchemaCompatMiner()
    edges = miner.mine_edges(tools)
    
    # Check edges between A and B (overlap = 1.0 > 0.7)
    ab_edge = next((e for e in edges if e[0] == "A" and e[1] == "B"), None)
    assert ab_edge is not None
    assert ab_edge[2] == 1.0
    
    # No edges to C
    c_edges = [e for e in edges if e[0] == "C" or e[1] == "C"]
    assert len(c_edges) == 0

def test_param_overlap_miner():
    # Tool A: ['string', 'integer']
    # Tool B: ['string'] -> union is {'string', 'integer'}, intersection {'string'} -> Jaccard = 1/2 = 0.5 (> 0.4)
    # Tool C: ['string', 'number'] -> Jaccard A-C is {'string'} / {'string', 'integer', 'number'} = 1/3 = 0.33 (<= 0.4)
    tools = [
        {"id": "A", "parameter_types": ["string", "integer"]},
        {"id": "B", "parameter_types": ["string"]},
        {"id": "C", "parameter_types": ["string", "number"]}
    ]
    miner = ParamOverlapMiner()
    edges = miner.mine_edges(tools)
    
    # Edge A-B exists (Jaccard = 0.5)
    ab_edge = next((e for e in edges if e[0] == "A" and e[1] == "B"), None)
    assert ab_edge is not None
    assert ab_edge[2] == 0.5
    
    # Edge A-C does not exist
    ac_edge = next((e for e in edges if e[0] == "A" and e[1] == "C"), None)
    assert ac_edge is None

def test_compose_dep_miner_heuristic():
    # Heuristic composition test (no API Key)
    tools = [
        {"id": "A", "name": "get_user_token", "server": "ServerA"},
        {"id": "B", "name": "query_alarm", "server": "ServerA"},
        {"id": "C", "name": "cancel_alarm", "server": "ServerA"},
        {"id": "D", "name": "fetch_weather", "server": "ServerB"}
    ]
    
    miner = ComposeDepMiner(api_key=None)
    edges = miner.mine_edges(tools)
    
    # Expect:
    # - get_user_token -> query_alarm (Auth dependency, weight 0.9)
    # - get_user_token -> cancel_alarm (Auth dependency, weight 0.9)
    # - query_alarm -> cancel_alarm (Query/Cancel entity dependency, weight 0.85)
    # No dependencies between ServerA and ServerB tools (different servers)
    
    token_query = next((e for e in edges if e[0] == "A" and e[1] == "B"), None)
    assert token_query is not None
    assert token_query[2] == 0.9
    
    query_cancel = next((e for e in edges if e[0] == "B" and e[1] == "C"), None)
    assert query_cancel is not None
    assert query_cancel[2] == 0.85
    
    # Asymmetry test: cancel_alarm should NOT be a prerequisite for query_alarm
    cancel_query = next((e for e in edges if e[0] == "C" and e[1] == "B"), None)
    assert cancel_query is None

def test_graph_builder_integration(mock_manifest_file, mock_trajectory_dir):
    builder = GraphBuilder(
        trajectory_dir=mock_trajectory_dir,
        compose_cache_path=os.path.join(os.path.dirname(mock_manifest_file), "compose_cache.json")
    )
    G = builder.build_graph(mock_manifest_file)
    
    # Check node count
    assert G.number_of_nodes() == 4
    
    # Verify no self-loops
    assert len(list(nx.selfloop_edges(G))) == 0
    
    # Verify all edge weights are in [0, 1]
    for u, v, k, d in G.edges(keys=True, data=True):
        assert 0.0 <= d["weight"] <= 1.0
        
    # Verify we have multiple edge types populated
    edge_types = {d["type"] for u, v, d in G.edges(data=True)}
    assert len(edge_types) > 0
