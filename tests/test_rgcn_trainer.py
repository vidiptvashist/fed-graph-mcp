import pytest
import numpy as np
import torch
import faiss
import networkx as nx
from src.rgcn_trainer import RGCNNet, RGCNLinkPredictor, sample_negative_edges, train_rgcn, build_faiss_index

def test_rgcn_net_shapes():
    num_nodes = 10
    in_dim = 384
    hidden_dim = 256
    out_dim = 384
    num_relations = 4

    # Dummy features and edges
    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
    ], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 2, 3, 1, 2, 3, 0, 1, 2], dtype=torch.long)

    # 1. Test encoder network shapes
    encoder = RGCNNet(in_dim, hidden_dim, out_dim, num_relations)
    h = encoder(x, edge_index, edge_type)
    assert h.shape == (num_nodes, out_dim)

    # 2. Test link predictor wrapper
    predictor = RGCNLinkPredictor(in_dim, hidden_dim, out_dim, num_relations)
    h_pred = predictor.encode(x, edge_index, edge_type)
    assert h_pred.shape == (num_nodes, out_dim)

    # Decode scores for edges
    scores = predictor.decode(h_pred, edge_index)
    assert scores.shape == (edge_index.shape[1],)
    assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0)

def test_negative_sampling():
    num_nodes = 5
    num_samples = 4
    pos_edges_set = {
        (0, 1, 0),
        (1, 2, 0),
        (2, 3, 1),
        (3, 4, 2)
    }

    # Sample for relation 0
    neg_src, neg_dst = sample_negative_edges(num_nodes, num_samples, pos_edges_set, relation_id=0)
    assert len(neg_src) == num_samples
    assert len(neg_dst) == num_samples

    for u, v in zip(neg_src, neg_dst):
        assert u != v  # No self-loops
        assert (u, v, 0) not in pos_edges_set  # Not in positive edges

def test_faiss_hnsw_flat():
    num_nodes = 20
    d = 384
    np.random.seed(42)
    H = np.random.randn(num_nodes, d).astype("float32")

    # Build index
    index = build_faiss_index(H)
    assert index.ntotal == num_nodes

    # Query nearest neighbors for first vector
    query = H[0].reshape(1, -1)
    distances, indices = index.search(query, k=3)
    
    assert indices.shape == (1, 3)
    assert distances.shape == (1, 3)
    assert indices[0][0] == 0  # Nearest neighbor should be itself (distance ~0)
    assert distances[0][0] == pytest.approx(0.0, abs=1e-4)

def test_train_rgcn_mock_graph():
    # Build a tiny mock MultiDiGraph G
    G = nx.MultiDiGraph()
    G.add_node("A", name="Tool A", description="Retrieve user authentication details.")
    G.add_node("B", name="Tool B", description="Query user active status.")
    G.add_node("C", name="Tool C", description="Modify user settings.")
    G.add_node("D", name="Tool D", description="Get weather information.")
    G.add_node("E", name="Tool E", description="Compute location geometry.")

    # Add edges of different relation types
    G.add_edge("A", "B", key="schema_compat", type="schema_compat", weight=0.85)
    G.add_edge("A", "C", key="param_overlap", type="param_overlap", weight=0.90)
    G.add_edge("B", "C", key="compose_dep", type="compose_dep", weight=0.80)
    G.add_edge("D", "E", key="schema_compat", type="schema_compat", weight=0.95)

    # Run tiny training loop for 5 epochs
    H, node_to_idx, predictor = train_rgcn(
        G,
        epochs=5,
        lr=0.01,
        val_ratio=0.2,
        test_ratio=0.2,
        device="cpu"
    )

    assert H.shape == (5, 384)
    assert len(node_to_idx) == 5
    assert set(node_to_idx.keys()) == {"A", "B", "C", "D", "E"}
    assert isinstance(predictor, RGCNLinkPredictor)
