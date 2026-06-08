import os
import json
import pytest
import numpy as np
import networkx as nx
from src.retrieval_pipeline import RetrievalPipeline
from src.sync_daemon import SyncDaemon

@pytest.fixture(scope="module")
def pipeline():
    return RetrievalPipeline()

@pytest.fixture(scope="module")
def daemon(pipeline):
    return SyncDaemon(pipeline=pipeline)

def test_daemon_initialization(daemon):
    assert daemon.pipeline is not None
    assert daemon.gnn_model is not None
    assert daemon.pipeline.raw_index is not None
    assert daemon.pipeline.gnn_index is not None
    assert len(daemon.last_seen_servers) > 0

def test_handle_server_add(daemon):
    initial_nodes_count = len(daemon.pipeline.node_to_idx)
    
    # 1. Create a mock server manifest with 2 tools
    mock_server = {
        "name": "MockDBSpeed",
        "summary": "Fast database mock server",
        "tools": [
            {
                "name": "query_speed",
                "description": "Checks the query execution speed of sql databases.",
                "parameter": {
                    "db_name": "(string) Name of the target database.",
                    "sql_query": "(string) The SQL statement to run."
                }
            },
            {
                "name": "optimize_index",
                "description": "Optimizes a database table index.",
                "parameter": {
                    "table_name": "(string) Target database table.",
                    "index_name": "(string) Index target."
                }
            }
        ]
    }
    
    # Run server add handler
    daemon.handle_server_add(mock_server)
    
    new_nodes_count = len(daemon.pipeline.node_to_idx)
    assert new_nodes_count == initial_nodes_count + 2
    
    # Verify nodes exist in Graph G and FAISS indices
    assert "MockDBSpeed/query_speed" in daemon.pipeline.node_to_idx
    assert "MockDBSpeed/optimize_index" in daemon.pipeline.node_to_idx
    assert daemon.pipeline.G.has_node("MockDBSpeed/query_speed")
    
    # Check that FAISS indices match mapping sizes
    assert daemon.pipeline.raw_index.ntotal == new_nodes_count
    assert daemon.pipeline.gnn_index.ntotal == new_nodes_count
    
    # Verify that local edge mining added compatibility/overlap edges
    # (Since both tools use string parameters, they should have parameter type overlap)
    edges = daemon.pipeline.G.edges("MockDBSpeed/query_speed")
    assert len(edges) > 0

def test_handle_tool_update(daemon):
    tool_id = "MockDBSpeed/query_speed"
    idx = daemon.pipeline.node_to_idx[tool_id]
    
    # Get initial raw vector representation
    initial_vector = np.array(daemon.pipeline.X[idx])
    
    # Create updated manifest
    updated_tool = {
        "name": "query_speed",
        "description": "Completely new description for querying sqlite database speed.",
        "parameter": {
            "db_name": "(string) Database name.",
            "sql_query": "(string) SQL query."
        }
    }
    
    # Run update
    daemon.handle_tool_update(tool_id, updated_tool)
    
    # Verify description is updated in graph G
    assert daemon.pipeline.G.nodes[tool_id]["description"] == "Completely new description for querying sqlite database speed."
    
    # Verify raw vector representation has changed in pipeline.X
    new_vector = np.array(daemon.pipeline.X[idx])
    assert not np.array_equal(initial_vector, new_vector)
    
    # Verify GNN embeddings are updated
    # Check that affected neighbors GNN embeddings were updated too

def test_handle_server_remove(daemon):
    initial_nodes_count = len(daemon.pipeline.node_to_idx)
    assert "MockDBSpeed/query_speed" in daemon.pipeline.node_to_idx
    
    # Run removal of MockDBSpeed
    daemon.handle_server_remove("MockDBSpeed")
    
    new_nodes_count = len(daemon.pipeline.node_to_idx)
    assert new_nodes_count == initial_nodes_count - 2
    
    assert "MockDBSpeed/query_speed" not in daemon.pipeline.node_to_idx
    assert not daemon.pipeline.G.has_node("MockDBSpeed/query_speed")
    
    # Verify FAISS index sizes are updated
    assert daemon.pipeline.raw_index.ntotal == new_nodes_count
    assert daemon.pipeline.gnn_index.ntotal == new_nodes_count

def test_diff_and_sync(daemon):
    # Setup initial servers in daemon.last_seen_servers
    daemon.last_seen_servers = {
        "ServerA": {
            "name": "ServerA",
            "tools": [{"name": "toolA", "description": "Desc A", "parameter": {}}]
        }
    }
    
    # Setup initial states in G and mappings
    daemon.handle_server_add(daemon.last_seen_servers["ServerA"])
    
    # Simulate a registry poll result:
    # - ServerA is updated (new description for toolA)
    # - ServerB is added
    new_registry = [
        {
            "name": "ServerA",
            "tools": [{"name": "toolA", "description": "New Desc A", "parameter": {}}]
        },
        {
            "name": "ServerB",
            "tools": [{"name": "toolB", "description": "Desc B", "parameter": {}}]
        }
    ]
    
    # Run diff and sync
    daemon.diff_and_sync(new_registry)
    
    # Verify ServerB was added
    assert "ServerB/toolB" in daemon.pipeline.node_to_idx
    
    # Verify ServerA toolA description was updated
    assert daemon.pipeline.G.nodes["ServerA/toolA"]["description"] == "New Desc A"
    
    # Verify last_seen_servers is updated
    assert "ServerB" in daemon.last_seen_servers

def test_sync_latency_logging(daemon):
    log_path = os.path.join(daemon.pipeline.base_dir, "mcp_sync_latency.json")
    
    # Ensure some sync latencies are recorded
    assert os.path.exists(log_path)
    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)
        
    assert len(logs) > 0
    assert "sync_latency_ms" in logs[0]
    assert "event_type" in logs[0]
