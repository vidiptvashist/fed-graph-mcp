import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List, Set
import networkx as nx
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import RGCNConv
import faiss
from sklearn.metrics import roc_auc_score, average_precision_score

class RGCNNet(torch.nn.Module):
    """
    2-layer R-GCN architecture with LeakyReLU and skip connections to prevent over-smoothing.
    - hidden_dim = 384
    - out_dim = 384
    - num_relations = 5 (including self-loops)
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_relations: int):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations)
        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        # Layer 1 + Residual
        x1 = self.conv1(x, edge_index, edge_type)
        if x1.shape == x.shape:
            x1 = x1 + x
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        
        # Layer 2 + Residual
        x2 = self.conv2(x1, edge_index, edge_type)
        if x2.shape == x1.shape:
            x2 = x2 + x1
        return x2

class RGCNLinkPredictor(torch.nn.Module):
    """
    R-GCN net coupled with a link prediction decoder.
    Predicts edge existence using inner product of node representations: sigmoid(h_u^T h_v).
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_relations: int):
        super().__init__()
        self.encoder = RGCNNet(in_dim, hidden_dim, out_dim, num_relations)
        
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_type)

    def decode(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # edge_index has shape [2, E]
        src, dst = edge_index[0], edge_index[1]
        scores = torch.sum(h[src] * h[dst], dim=-1)
        return torch.sigmoid(scores)

def sample_negative_edges(
    num_nodes: int, 
    num_samples: int, 
    pos_edges_set: Set[Tuple[int, int, int]], 
    relation_id: int
) -> Tuple[List[int], List[int]]:
    """
    Samples unique negative directed edges (u, v) for a specific relation_id
    that do not exist in the positive edges set.
    """
    neg_src = []
    neg_dst = []
    attempts = 0
    max_attempts = num_samples * 10
    
    while len(neg_src) < num_samples and attempts < max_attempts:
        attempts += 1
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u == v:
            continue
        if (u, v, relation_id) not in pos_edges_set:
            neg_src.append(u)
            neg_dst.append(v)
            
    # If not enough unique negatives could be found, fall back to simple random generation
    while len(neg_src) < num_samples:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v:
            neg_src.append(u)
            neg_dst.append(v)
            
    return neg_src, neg_dst

def train_rgcn(
    G: nx.MultiDiGraph,
    epochs: int = 50,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    device: str = "cpu"
) -> Tuple[np.ndarray, Dict[str, int], RGCNLinkPredictor]:
    """
    Loads descriptions, computes ST embeddings, sets up R-GCN link prediction dataset,
    trains model, and outputs enriched node features H + node mappings.
    """
    # 1. Map node IDs to indices
    nodes_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    num_nodes = len(nodes_list)
    
    # 2. Extract edge list
    rel_map = {
        "co_invoke": 0,
        "schema_compat": 1,
        "param_overlap": 2,
        "compose_dep": 3
    }
    
    edges_list = []
    pos_edges_set = set()
    for u, v, key, data in G.edges(keys=True, data=True):
        if key in rel_map:
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            r_idx = rel_map[key]
            edges_list.append((u_idx, v_idx, r_idx))
            pos_edges_set.add((u_idx, v_idx, r_idx))
            
    print(f"[RGCNTrainer] Extracted {len(edges_list)} relational edges from NetworkX graph.")
    
    # 3. Create initial node features using Sentence-Transformers (all-MiniLM-L6-v2)
    print("[RGCNTrainer] Generating initial description embeddings via SentenceTransformer...")
    model_st = SentenceTransformer("all-MiniLM-L6-v2")
    descriptions = []
    for idx in range(num_nodes):
        node = nodes_list[idx]
        desc = G.nodes[node].get("description", "")
        if not desc:
            desc = G.nodes[node].get("name", "")
        descriptions.append(desc)
        
    X_numpy = model_st.encode(descriptions, show_progress_bar=False)
    X = torch.tensor(X_numpy, dtype=torch.float32).to(device)
    print(f"[RGCNTrainer] Initial embedding matrix shape: {X.shape}")
    
    # 4. Shuffle and split edges (80/10/10)
    random.seed(42)
    random.shuffle(edges_list)
    
    num_edges = len(edges_list)
    num_val = int(num_edges * val_ratio)
    num_test = int(num_edges * test_ratio)
    num_train = num_edges - num_val - num_test
    
    train_edges = edges_list[:num_train]
    val_edges = edges_list[num_train : num_train + num_val]
    test_edges = edges_list[num_train + num_val :]
    
    print(f"[RGCNTrainer] Split dataset into Train: {len(train_edges)}, Val: {len(val_edges)}, Test: {len(test_edges)} edges.")
    
    # 5. Prepare tensors for training (Message Passing Graph)
    # We only pass training edges through R-GCN layers to prevent validation/test leakage.
    train_src = [e[0] for e in train_edges]
    train_dst = [e[1] for e in train_edges]
    train_type = [e[2] for e in train_edges]
    
    # Add self-loops to training edge list to avoid representation collapse of isolated nodes
    for idx in range(num_nodes):
        train_src.append(idx)
        train_dst.append(idx)
        train_type.append(4) # Relation ID 4 represents self-loops
        
    train_edge_index = torch.tensor([train_src, train_dst], dtype=torch.long).to(device)
    train_edge_type = torch.tensor(train_type, dtype=torch.long).to(device)
    
    # We also prepare full graph tensors for transductive final embedding generation
    full_src = [e[0] for e in edges_list]
    full_dst = [e[1] for e in edges_list]
    full_type = [e[2] for e in edges_list]
    
    # Add self-loops to full graph edge list
    for idx in range(num_nodes):
        full_src.append(idx)
        full_dst.append(idx)
        full_type.append(4)
        
    full_edge_index = torch.tensor([full_src, full_dst], dtype=torch.long).to(device)
    full_edge_type = torch.tensor(full_type, dtype=torch.long).to(device)
    
    # 6. Initialize model, optimizer, and loss function
    predictor = RGCNLinkPredictor(in_dim=384, hidden_dim=384, out_dim=384, num_relations=5).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    # Pre-split validation/test sets for quick evaluation
    val_pos_src = [e[0] for e in val_edges]
    val_pos_dst = [e[1] for e in val_edges]
    val_pos_edge_index = torch.tensor([val_pos_src, val_pos_dst], dtype=torch.long).to(device)
    
    test_pos_src = [e[0] for e in test_edges]
    test_pos_dst = [e[1] for e in test_edges]
    test_pos_edge_index = torch.tensor([test_pos_src, test_pos_dst], dtype=torch.long).to(device)
    
    # Sample static val/test negative edges to keep metrics consistent across epochs
    val_neg_src, val_neg_dst = [], []
    for e in val_edges:
        neg_u, neg_v = sample_negative_edges(num_nodes, 1, pos_edges_set, e[2])
        val_neg_src.append(neg_u[0])
        val_neg_dst.append(neg_v[0])
    val_neg_edge_index = torch.tensor([val_neg_src, val_neg_dst], dtype=torch.long).to(device)
    
    test_neg_src, test_neg_dst = [], []
    for e in test_edges:
        neg_u, neg_v = sample_negative_edges(num_nodes, 1, pos_edges_set, e[2])
        test_neg_src.append(neg_u[0])
        test_neg_dst.append(neg_v[0])
    test_neg_edge_index = torch.tensor([test_neg_src, test_neg_dst], dtype=torch.long).to(device)
    
    # 7. Training Loop
    print("[RGCNTrainer] Commencing training...")
    for epoch in range(1, epochs + 1):
        predictor.train()
        optimizer.zero_grad()
        
        # Forward pass on train graph
        h = predictor.encode(X, train_edge_index, train_edge_type)
        
        # Dynamic negative sampling for training positive edges
        # Generates 1 negative edge for each positive training edge
        train_neg_src = []
        train_neg_dst = []
        for e in train_edges:
            neg_u, neg_v = sample_negative_edges(num_nodes, 1, pos_edges_set, e[2])
            train_neg_src.append(neg_u[0])
            train_neg_dst.append(neg_v[0])
            
        train_pos_edge_index = train_edge_index
        train_neg_edge_index = torch.tensor([train_neg_src, train_neg_dst], dtype=torch.long).to(device)
        
        # Predict positive & negative edges
        pos_scores = predictor.decode(h, train_pos_edge_index)
        neg_scores = predictor.decode(h, train_neg_edge_index)
        
        pos_loss = criterion(pos_scores, torch.ones_like(pos_scores))
        neg_loss = criterion(neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        
        # Evaluation
        if epoch % 5 == 0 or epoch == 1:
            predictor.eval()
            with torch.no_grad():
                h_eval = predictor.encode(X, train_edge_index, train_edge_type)
                
                # Validation scores
                if len(val_edges) > 0:
                    val_pos_scores = predictor.decode(h_eval, val_pos_edge_index).cpu().numpy()
                    val_neg_scores = predictor.decode(h_eval, val_neg_edge_index).cpu().numpy()
                    
                    y_true = np.concatenate([np.ones_like(val_pos_scores), np.zeros_like(val_neg_scores)])
                    y_scores = np.concatenate([val_pos_scores, val_neg_scores])
                    
                    val_auc = roc_auc_score(y_true, y_scores)
                    val_ap = average_precision_score(y_true, y_scores)
                else:
                    val_auc = 0.0
                    val_ap = 0.0
                
            print(f"Epoch {epoch:02d} | Train Loss: {loss.item():.4f} | Val ROC-AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")
            
    # 8. Final Evaluation on Held-out Test Set
    predictor.eval()
    with torch.no_grad():
        h_final = predictor.encode(X, train_edge_index, train_edge_type)
        
        if len(test_edges) > 0:
            test_pos_scores = predictor.decode(h_final, test_pos_edge_index).cpu().numpy()
            test_neg_scores = predictor.decode(h_final, test_neg_edge_index).cpu().numpy()
            
            test_true = np.concatenate([np.ones_like(test_pos_scores), np.zeros_like(test_neg_scores)])
            test_scores = np.concatenate([test_pos_scores, test_neg_scores])
            
            test_auc = roc_auc_score(test_true, test_scores)
            test_ap = average_precision_score(test_true, test_scores)
        else:
            test_auc = 0.0
            test_ap = 0.0
        
        print("\n========================= TEST SET METRICS =========================")
        print(f"Test ROC-AUC: {test_auc:.4f}")
        print(f"Test Average Precision (AP): {test_ap:.4f}")
        print("====================================================================\n")
        
        # Enriched Embeddings H generated using the entire graph transductively
        H_tensor = predictor.encode(X, full_edge_index, full_edge_type)
        H_numpy = H_tensor.cpu().numpy()
        
    return H_numpy, node_to_idx, predictor

def build_faiss_index(H: np.ndarray) -> faiss.IndexHNSWFlat:
    """
    Builds a FAISS IndexHNSWFlat over the enriched embeddings H.
    - H shape: [N, 384]
    - Dimension d: 384
    - Connections M: 32
    """
    # Force single-threaded FAISS execution to prevent OpenMP multi-threading collisions
    # with the PyTorch/MPS library runtime on macOS ARM64.
    faiss.omp_set_num_threads(1)
    
    d = H.shape[1]
    # Build index using L2 distance metric
    index = faiss.IndexHNSWFlat(d, 32)
    
    # FAISS requires float32 contiguous array
    H_contiguous = np.ascontiguousarray(H.astype("float32"))
    index.add(H_contiguous)
    
    print(f"[FAISS] Built IndexHNSWFlat index with {index.ntotal} vectors of dimension {d}.")
    return index
