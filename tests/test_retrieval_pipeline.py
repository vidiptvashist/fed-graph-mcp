import os
import pytest
import numpy as np
import torch
import networkx as nx
from src.retrieval_pipeline import (
    RetrievalPipeline,
    retrieve_dense,
    retrieve_gnn,
    fuse_rrf,
    graph_expand,
    LearnedReranker,
    train_reranker
)

@pytest.fixture(scope="module")
def sample_graph():
    G = nx.MultiDiGraph()
    G.add_node("Sqlite/create_table", name="create_table", server="Sqlite", description="Create a database table.")
    G.add_node("Sqlite/insert_row", name="insert_row", server="Sqlite", description="Insert a row into a table.")
    G.add_node("Stripe/charge_create", name="charge_create", server="Stripe", description="Create a Stripe credit card charge.")
    G.add_node("Stripe/customer_create", name="customer_create", server="Stripe", description="Create a Stripe customer profile.")
    
    # Add relations
    G.add_edge("Sqlite/create_table", "Sqlite/insert_row", key="compose_dep", type="compose_dep", weight=0.9)
    G.add_edge("Stripe/customer_create", "Stripe/charge_create", key="co_invoke", type="co_invoke", weight=0.8)
    return G

def test_learned_reranker_mlp():
    reranker = LearnedReranker(input_dim=768, hidden_dim=64)
    query_emb = torch.randn(1, 384)
    tool_emb = torch.randn(5, 384)
    
    # Forward pass
    scores = reranker(query_emb, tool_emb)
    assert scores.shape == (5,)
    
    # 1D input test
    single_q = torch.randn(384)
    single_t = torch.randn(384)
    single_score = reranker(single_q, single_t)
    assert single_score.shape == (1,)

def test_fuse_rrf():
    dense_res = [
        {"name": "Sqlite/create_table", "description": "Create table", "score": -0.1},
        {"name": "Stripe/customer_create", "description": "Create customer", "score": -0.5},
        {"name": "Sqlite/insert_row", "description": "Insert row", "score": -0.9}
    ]
    gnn_res = [
        {"name": "Stripe/customer_create", "description": "Create customer", "score": -0.2},
        {"name": "Sqlite/create_table", "description": "Create table", "score": -0.4},
        {"name": "Stripe/charge_create", "description": "Create charge", "score": -0.8}
    ]
    
    # Combine with c=60, k=3
    fused = fuse_rrf(dense_res, gnn_res, k=3, c=60)
    assert len(fused) <= 3
    # Check that highest ranked items are the most relevant ones
    assert fused[0]["name"] in ["Sqlite/create_table", "Stripe/customer_create"]
    assert "score" in fused[0]

def test_graph_expand(sample_graph):
    # Test expansion starting from create_table
    seeds = [{"name": "Sqlite/create_table", "description": "Create table"}]
    
    # Expanded by 1 hop
    expanded = graph_expand(seeds, sample_graph, hop=1)
    expanded_names = [item["name"] for item in expanded]
    
    assert "Sqlite/create_table" in expanded_names
    assert "Sqlite/insert_row" in expanded_names
    assert "Stripe/charge_create" not in expanded_names

    # Test string list support
    seeds_str = ["Stripe/customer_create"]
    expanded_str = graph_expand(seeds_str, sample_graph, hop=1)
    assert "Stripe/customer_create" in expanded_str
    assert "Stripe/charge_create" in expanded_str
    assert "Sqlite/create_table" not in expanded_str

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_retrieval_pipeline_instantiation():
    # Verify that RetrievalPipeline loads and constructs successfully
    pipeline = RetrievalPipeline()
    assert pipeline.G is not None
    assert pipeline.H is not None
    assert pipeline.raw_index is not None
    assert pipeline.gnn_index is not None
    
    # Test dense retrieval
    dense_res = pipeline.retrieve_dense("sql database table", k=3)
    assert len(dense_res) == 3
    assert "name" in dense_res[0]
    assert "description" in dense_res[0]
    
    # Test GNN retrieval
    gnn_res = pipeline.retrieve_gnn("sql database table", k=3)
    assert len(gnn_res) == 3

def test_train_reranker_mock():
    # Run reranker training for a tiny number of epochs on real trajectories
    reranker = train_reranker(epochs=2, lr=0.01)
    assert isinstance(reranker, LearnedReranker)
