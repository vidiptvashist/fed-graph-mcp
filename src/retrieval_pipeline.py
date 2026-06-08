import os
import json
import ast
import random
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import faiss
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

from src.graph_builder import GraphBuilder
from src.rgcn_trainer import build_faiss_index

class LearnedReranker(nn.Module):
    """
    2-layer MLP Learned Reranker.
    Takes concatenated query embedding and tool embedding as input (size 768)
    and outputs a ranking score.
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, query_emb: torch.Tensor, tool_emb: torch.Tensor) -> torch.Tensor:
        # If query_emb is [384] and tool_emb is [N, 384], expand query_emb to [N, 384]
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        if tool_emb.dim() == 1:
            tool_emb = tool_emb.unsqueeze(0)
            
        if query_emb.size(0) != tool_emb.size(0):
            query_emb = query_emb.expand(tool_emb.size(0), -1)
            
        # Concatenate query and tool embeddings
        x = torch.cat([query_emb, tool_emb], dim=-1)
        return self.mlp(x).squeeze(-1)

class RetrievalPipeline:
    """
    Retrieval and Reranking Pipeline for FED-GRAPH-MCP.
    """
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.base_dir = base_dir
            
        self.manifest_path = os.path.join(self.base_dir, "mcp_zero_repo/MCP-tools/mcp_tools_with_embedding.json")
        self.trajectory_dir = os.path.join(self.base_dir, "damo_convai_repo/api-bank/lv1-lv2-samples")
        self.cache_path = os.path.join(self.base_dir, "mcp_zero_repo/MCP-tools/.compose_dep_cache.json")
        
        # Load graph G
        print("[RetrievalPipeline] Loading multi-relational graph G...")
        builder = GraphBuilder(
            trajectory_dir=self.trajectory_dir,
            compose_cache_path=self.cache_path
        )
        self.G = builder.build_graph(self.manifest_path)
        
        # Mapping and embeddings paths
        self.mapping_path = os.path.join(self.base_dir, "mcp_node_mapping.json")
        self.enriched_emb_path = os.path.join(self.base_dir, "mcp_enriched_embeddings.npy")
        self.gnn_index_path = os.path.join(self.base_dir, "mcp_hnsw_index.faiss")
        
        # Load node mapping
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                self.node_to_idx = json.load(f)
        else:
            self.node_to_idx = {node: i for i, node in enumerate(self.G.nodes())}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # Load GNN enriched embeddings
        if os.path.exists(self.enriched_emb_path):
            self.H = np.load(self.enriched_emb_path)
        else:
            # Fallback to random features if not trained yet
            print("[RetrievalPipeline] Warning: enriched embeddings not found. Fallback to random initialization.")
            self.H = np.random.randn(len(self.node_to_idx), 384).astype("float32")
            
        # Initialize SentenceTransformer
        print("[RetrievalPipeline] Loading SentenceTransformer 'all-MiniLM-L6-v2'...")
        self.model_st = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Generate or load raw description embeddings
        self.raw_emb_path = os.path.join(self.base_dir, "mcp_raw_embeddings.npy")
        if os.path.exists(self.raw_emb_path):
            self.X = np.load(self.raw_emb_path)
        else:
            print("[RetrievalPipeline] Generating raw description embeddings...")
            descriptions = []
            for idx in range(len(self.idx_to_node)):
                node = self.idx_to_node[idx]
                desc = self.G.nodes[node].get("description", "")
                if not desc:
                    desc = self.G.nodes[node].get("name", "")
                descriptions.append(desc)
            self.X = self.model_st.encode(descriptions, show_progress_bar=False)
            np.save(self.raw_emb_path, self.X)
            
        # Build FAISS indices
        print("[RetrievalPipeline] Setting up FAISS indices...")
        # Force single-threading for FAISS to prevent OpenMP crashes with PyTorch MPS
        faiss.omp_set_num_threads(1)
        
        # Raw index
        self.raw_index = build_faiss_index(self.X)
        
        # GNN index
        if os.path.exists(self.gnn_index_path):
            self.gnn_index = faiss.read_index(self.gnn_index_path)
        else:
            self.gnn_index = build_faiss_index(self.H)
            
    def retrieve_dense(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Retrieves top-k tools using FAISS search over raw description embeddings.
        """
        # Embed query
        query_vector = self.model_st.encode([query]).astype("float32")
        
        # Search index
        faiss.omp_set_num_threads(1)
        distances, indices = self.raw_index.search(query_vector, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.idx_to_node):
                continue
            node_id = self.idx_to_node[idx]
            results.append({
                "name": node_id,
                "description": self.G.nodes[node_id].get("description", ""),
                "score": float(-dist)  # convert L2 distance to score (smaller distance = larger score)
            })
        return results

    def retrieve_gnn(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Retrieves top-k tools using GNN-enriched embeddings H.
        
        Because H lives in the R-GCN output space (different from raw ST space),
        we first find the closest raw-space neighbors and use their GNN embeddings
        as a weighted proxy query vector in GNN space.
        """
        query_vector = self.model_st.encode([query]).astype("float32")
        
        # Step 1: Find top-k_bridge neighbors in raw embedding space
        k_bridge = min(10, len(self.idx_to_node))
        faiss.omp_set_num_threads(1)
        distances, indices = self.raw_index.search(query_vector, k_bridge)
        
        # Step 2: Compute distance-weighted average of their GNN embeddings
        weights = []
        gnn_vecs = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.idx_to_node):
                continue
            # Convert L2 distance to similarity weight (inverse distance)
            w = 1.0 / (dist + 1e-6)
            weights.append(w)
            gnn_vecs.append(self.H[idx])
        
        if not gnn_vecs:
            return []
        
        weights = np.array(weights, dtype="float32")
        weights /= weights.sum()
        gnn_vecs = np.array(gnn_vecs, dtype="float32")
        proxy_query = np.average(gnn_vecs, axis=0, weights=weights).reshape(1, -1).astype("float32")
        
        # Step 3: Search GNN FAISS index with the projected proxy vector
        distances_gnn, indices_gnn = self.gnn_index.search(proxy_query, k)
        
        results = []
        for dist, idx in zip(distances_gnn[0], indices_gnn[0]):
            if idx < 0 or idx >= len(self.idx_to_node):
                continue
            node_id = self.idx_to_node[idx]
            results.append({
                "name": node_id,
                "description": self.G.nodes[node_id].get("description", ""),
                "score": float(-dist)
            })
        return results

def retrieve_dense(query: str, k: int, pipeline: RetrievalPipeline = None) -> List[Dict[str, Any]]:
    """
    Standalone function to retrieve top-k tools using raw description embeddings.
    """
    p = pipeline or _get_global_pipeline()
    return p.retrieve_dense(query, k)

def retrieve_gnn(query: str, k: int, pipeline: RetrievalPipeline = None) -> List[Dict[str, Any]]:
    """
    Standalone function to retrieve top-k tools using GNN-enriched embeddings.
    """
    p = pipeline or _get_global_pipeline()
    return p.retrieve_gnn(query, k)

def fuse_rrf(dense_results: List[Any], gnn_results: List[Any], k: int = 60, c: int = 60) -> List[Dict[str, Any]]:
    """
    Fuses ranking lists from retrieve_dense and retrieve_gnn using Reciprocal Rank Fusion (RRF).
    Score(d) = sum_{m in M} 1 / (c + rank_m(d))
    Returns top-k fused results.
    """
    def get_tool_id(item):
        if isinstance(item, dict):
            return item.get("name") or item.get("id")
        elif isinstance(item, tuple):
            return item[0]
        return item

    # Extract tool descriptions to preserve metadata in the final list
    descriptions = {}
    for res in dense_results + gnn_results:
        tid = get_tool_id(res)
        if isinstance(res, dict) and "description" in res:
            descriptions[tid] = res["description"]
        elif isinstance(res, tuple) and len(res) > 1:
            descriptions[tid] = res[1]

    # Compute RRF score
    rrf_scores = {}
    
    # Process dense rank list
    for rank, res in enumerate(dense_results):
        tid = get_tool_id(res)
        if tid:
            rrf_scores[tid] = rrf_scores.get(tid, 0.0) + 1.0 / (c + (rank + 1))
            
    # Process GNN rank list
    for rank, res in enumerate(gnn_results):
        tid = get_tool_id(res)
        if tid:
            rrf_scores[tid] = rrf_scores.get(tid, 0.0) + 1.0 / (c + (rank + 1))
            
    # Sort tools by fused RRF score descending
    sorted_tools = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Build result dictionaries
    fused_results = []
    for tid, score in sorted_tools[:k]:
        fused_results.append({
            "name": tid,
            "description": descriptions.get(tid, ""),
            "score": score
        })
    return fused_results

def graph_expand(top_k_tools: List[Any], G: nx.MultiDiGraph, hop: int = 1) -> List[Any]:
    """
    Expands the top-k retrieved tools by adding their neighbors in the graph G
    along 'co_invoke' and 'compose_dep' edge relations.
    Preserves original seed nodes' order, and appends neighbors.
    """
    def get_tool_id(item):
        if isinstance(item, dict):
            return item.get("name") or item.get("id")
        elif isinstance(item, tuple):
            return item[0]
        return item

    seed_nodes = [get_tool_id(item) for item in top_k_tools]
    current_nodes = set(seed_nodes)
    
    # Cache input descriptions
    descriptions = {}
    for item in top_k_tools:
        tid = get_tool_id(item)
        if isinstance(item, dict) and "description" in item:
            descriptions[tid] = item["description"]
        elif isinstance(item, tuple) and len(item) > 1:
            descriptions[tid] = item[1]
            
    allowed_relations = {"co_invoke", "compose_dep"}
    
    for _ in range(hop):
        new_nodes = set()
        for u in current_nodes:
            if u not in G:
                continue
            # Check outgoing edges
            for _, v, key, data in G.out_edges(u, keys=True, data=True):
                if key in allowed_relations or data.get("type") in allowed_relations:
                    new_nodes.add(v)
            # Check incoming edges
            for v, _, key, data in G.in_edges(u, keys=True, data=True):
                if key in allowed_relations or data.get("type") in allowed_relations:
                    new_nodes.add(v)
        current_nodes.update(new_nodes)
        
    # Order: first original seed nodes, then newly expanded neighbors (sorted)
    expanded_nodes = []
    for node in seed_nodes:
        if node not in expanded_nodes:
            expanded_nodes.append(node)
            
    for node in sorted(current_nodes):
        if node not in expanded_nodes:
            expanded_nodes.append(node)
            
    # Format return value to match the input format (dictionaries or strings)
    if len(top_k_tools) > 0 and isinstance(top_k_tools[0], dict):
        result = []
        for node in expanded_nodes:
            desc = descriptions.get(node, "")
            if not desc and node in G:
                desc = G.nodes[node].get("description", "")
            result.append({"name": node, "description": desc})
        return result
    else:
        return expanded_nodes

# Helper to statically parse API-Bank tools and descriptions
def parse_apibank_tools(apis_dir: str) -> Dict[str, str]:
    """
    Statically parses API-Bank python files to extract API names and descriptions.
    """
    tools = {}
    if not os.path.exists(apis_dir):
        return tools
        
    for filename in os.listdir(apis_dir):
        if filename.endswith(".py") and filename not in ["__init__.py", "api.py"]:
            filepath = os.path.join(apis_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=filepath)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        description = ""
                        for item in node.body:
                            if isinstance(item, ast.Assign):
                                for target in item.targets:
                                    if isinstance(target, ast.Name) and target.id == "description":
                                        if isinstance(item.value, ast.Constant):
                                            description = item.value.value
                                        elif isinstance(item.value, ast.Str):
                                            description = item.value.s
                        if not description:
                            doc = ast.get_docstring(node)
                            if doc:
                                description = doc.strip().split("\n")[0]
                        if node.name != "API":
                            tools[node.name] = description
            except Exception as e:
                print(f"[RerankerPreprocess] Warning: Failed to parse {filename}: {e}")
    return tools

def train_reranker(
    base_dir: str = None,
    epochs: int = 15,
    lr: float = 0.001,
    margin: float = 1.0,
    device: str = "cpu"
) -> LearnedReranker:
    """
    Loads API-Bank trajectories, extracts query-tool pairs, builds positive/negative pairs,
    and trains a LearnedReranker MLP model with pairwise margin loss.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    apis_dir = os.path.join(base_dir, "damo_convai_repo/api-bank/apis")
    trajectory_dir = os.path.join(base_dir, "damo_convai_repo/api-bank/lv1-lv2-samples")
    
    # 1. Parse API-Bank tool descriptions
    apibank_tools = parse_apibank_tools(apis_dir)
    print(f"[RerankerTrain] Mined {len(apibank_tools)} API-Bank tools.")
    
    if len(apibank_tools) == 0:
        print("[RerankerTrain] Warning: No API-Bank tools found. Returning default Reranker.")
        return LearnedReranker()
        
    # 2. Extract trajectory dialogues
    dialogue_files = []
    for root, _, files in os.walk(trajectory_dir):
        for file in files:
            if file.endswith(".jsonl"):
                dialogue_files.append(os.path.join(root, file))
                
    # Parse queries and positive APIs called
    dataset_pairs = [] # List of (query_text, pos_tool_name)
    
    # Normalize tool name keys
    clean_to_exact_name = {k.lower().replace("_", "").replace("-", ""): k for k in apibank_tools.keys()}
    
    for file_path in dialogue_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                continue
                
            # First User turn is the query
            query_text = ""
            for line in lines:
                turn = json.loads(line)
                if turn.get("role", "").upper() == "USER":
                    query_text = turn.get("text", "")
                    break
                    
            if not query_text:
                continue
                
            # Collect invoked APIs
            invoked_apis = set()
            for line in lines:
                turn = json.loads(line)
                if turn.get("role", "").upper() == "API":
                    api_name = turn.get("api_name") or turn.get("api")
                    if api_name:
                        cleaned = api_name.lower().replace("_", "").replace("-", "")
                        if cleaned in clean_to_exact_name:
                            invoked_apis.add(clean_to_exact_name[cleaned])
                            
            # Record dataset pairs
            for api in invoked_apis:
                dataset_pairs.append((query_text, api))
        except Exception as e:
            print(f"[RerankerTrain] Error reading {file_path}: {e}")
            continue
            
    print(f"[RerankerTrain] Extracted {len(dataset_pairs)} query-tool positive pairs.")
    if len(dataset_pairs) == 0:
        return LearnedReranker()
        
    # 3. Embed queries and tool descriptions using SentenceTransformer
    print("[RerankerTrain] Generating embeddings for Reranker training...")
    model_st = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Distinct queries and tools
    unique_queries = list(set([p[0] for p in dataset_pairs]))
    unique_tools = list(apibank_tools.keys())
    
    query_embs = model_st.encode(unique_queries, show_progress_bar=False)
    query_emb_map = {q: emb for q, emb in zip(unique_queries, query_embs)}
    
    tool_descriptions = [apibank_tools[t] if apibank_tools[t] else t for t in unique_tools]
    tool_embs = model_st.encode(tool_descriptions, show_progress_bar=False)
    tool_emb_map = {t: emb for t, emb in zip(unique_tools, tool_embs)}
    
    # 4. Construct triplets (query, positive, negative)
    triplets = []
    for q, pos_tool in dataset_pairs:
        # Find all APIs invoked in any query associated with this dialogue context
        # (For simplicity, negative tool is any tool not equal to pos_tool)
        other_tools = [t for t in unique_tools if t != pos_tool]
        if not other_tools:
            continue
        neg_tool = random.choice(other_tools)
        
        q_emb = query_emb_map[q]
        pos_emb = tool_emb_map[pos_tool]
        neg_emb = tool_emb_map[neg_tool]
        
        triplets.append((
            torch.tensor(q_emb, dtype=torch.float32),
            torch.tensor(pos_emb, dtype=torch.float32),
            torch.tensor(neg_emb, dtype=torch.float32)
        ))
        
    # 5. Training loop
    reranker = LearnedReranker(input_dim=768, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(reranker.parameters(), lr=lr)
    loss_fn = nn.MarginRankingLoss(margin=margin)
    
    print(f"[RerankerTrain] Commencing training over {len(triplets)} triplets...")
    reranker.train()
    for epoch in range(1, epochs + 1):
        random.shuffle(triplets)
        epoch_loss = 0.0
        
        # Batching (simple batch size = 32)
        batch_size = 32
        for idx in range(0, len(triplets), batch_size):
            batch = triplets[idx : idx + batch_size]
            if not batch:
                continue
                
            q_batch = torch.stack([item[0] for item in batch]).to(device)
            pos_batch = torch.stack([item[1] for item in batch]).to(device)
            neg_batch = torch.stack([item[2] for item in batch]).to(device)
            
            pos_scores = reranker(q_batch, pos_batch)
            neg_scores = reranker(q_batch, neg_batch)
            
            # Target y = 1 (pos_score should be larger than neg_score by margin)
            loss = loss_fn(pos_scores, neg_scores, torch.ones_like(pos_scores))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch)
            
        epoch_loss /= len(triplets)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Margin Loss: {epoch_loss:.4f}")
            
    # Save trained model weights
    save_path = os.path.join(base_dir, "mcp_reranker.pt")
    torch.save(reranker.state_dict(), save_path)
    print(f"[RerankerTrain] Saved trained model weights to {save_path}")
    
    reranker.eval()
    return reranker

# Global lazy-loaded pipeline instance for stand-alone calls
_global_pipeline = None

def _get_global_pipeline() -> RetrievalPipeline:
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = RetrievalPipeline()
    return _global_pipeline
