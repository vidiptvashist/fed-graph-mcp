import os
import json
import time
import asyncio
import numpy as np
import torch
import faiss
import aiohttp
import networkx as nx
from typing import Dict, Any, List, Tuple, Set

from src.retrieval_pipeline import RetrievalPipeline
from src.rgcn_trainer import RGCNLinkPredictor
from src.utils import extract_parameter_types_from_tool

class SyncDaemon:
    """
    Sync Daemon for real-time synchronization of the FED-GRAPH MCP tool registry.
    Handles dynamic server/tool additions, updates, and removals.
    Tracks sync latency metrics.
    """
    def __init__(self, pipeline: RetrievalPipeline, gnn_model_path: str = None, device: str = "cpu"):
        self.pipeline = pipeline
        self.device = device
        self.last_seen_servers = {}
        
        # Load GNN model weights
        if gnn_model_path is None:
            gnn_model_path = os.path.join(pipeline.base_dir, "mcp_rgcn_model.pt")
            
        self.gnn_model = RGCNLinkPredictor(in_dim=384, hidden_dim=384, out_dim=384, num_relations=5).to(device)
        if os.path.exists(gnn_model_path):
            self.gnn_model.load_state_dict(torch.load(gnn_model_path, map_location=device))
            print(f"[SyncDaemon] Loaded trained GNN model from {gnn_model_path}")
        else:
            print("[SyncDaemon] Warning: Trained GNN model weights not found. Running GNN updates on initial model weights.")
        self.gnn_model.eval()
        
        # Re-initialize FAISS indices in pipeline to support remove_ids (using IndexIDMap2(IndexFlatL2))
        d = 384
        print("[SyncDaemon] Initializing dynamic FAISS indices (IndexIDMap2)...")
        faiss.omp_set_num_threads(1)
        
        self.pipeline.raw_index = faiss.IndexIDMap2(faiss.IndexFlatL2(d))
        self.pipeline.gnn_index = faiss.IndexIDMap2(faiss.IndexFlatL2(d))
        
        # Populate initial embeddings with consecutive IDs matching index in mappings
        ids = np.array(range(len(self.pipeline.X)), dtype=np.int64)
        self.pipeline.raw_index.add_with_ids(self.pipeline.X.astype("float32"), ids)
        self.pipeline.gnn_index.add_with_ids(self.pipeline.H.astype("float32"), ids)
        
        # Initialize last_seen_servers from the initial manifest
        self._load_initial_servers()
        
    def _load_initial_servers(self):
        """
        Builds last_seen_servers state from the initial manifest.
        """
        try:
            if os.path.exists(self.pipeline.manifest_path):
                with open(self.pipeline.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for server in data:
                        server_name = server.get("name") or server.get("server_name") or "unknown"
                        self.last_seen_servers[server_name] = server
        except Exception as e:
            print(f"[SyncDaemon] Error loading initial servers manifest: {e}")

    def log_latency(self, event_type: str, details: str, latency_ms: float):
        """
        Appends sync latency metrics to the evaluation log file.
        """
        log_path = os.path.join(self.pipeline.base_dir, "mcp_sync_latency.json")
        log_entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details,
            "sync_latency_ms": latency_ms
        }
        
        logs = []
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except Exception:
                pass
                
        logs.append(log_entry)
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"[SyncDaemon] Error logging latency metric: {e}")

    async def poll_registry(self, registry_url: str):
        """
        Asynchronously polls the registry API, diffs against last-seen state,
        and triggers handlers for server additions, removals, and tool updates.
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(registry_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.diff_and_sync(data)
            except Exception as e:
                # Handle connection issues gracefully
                print(f"[SyncDaemon] Registry polling error: {e}")

    def diff_and_sync(self, current_servers_list: List[Dict[str, Any]]):
        """
        Diffs a new registry response list against the last-seen state.
        Executes handle_* operations.
        """
        current_servers = {}
        for server in current_servers_list:
            name = server.get("name") or server.get("server_name")
            if name:
                current_servers[name] = server
                
        # Removals
        removed_servers = [s for s in self.last_seen_servers if s not in current_servers]
        for server_id in removed_servers:
            self.handle_server_remove(server_id)
            del self.last_seen_servers[server_id]
            
        # Additions
        added_servers = [s for s in current_servers if s not in self.last_seen_servers]
        for server_id in added_servers:
            self.handle_server_add(current_servers[server_id])
            self.last_seen_servers[server_id] = current_servers[server_id]
            
        # Updates
        common_servers = [s for s in current_servers if s in self.last_seen_servers]
        for server_id in common_servers:
            # Check if JSON manifest content has changed
            old_manifest = self.last_seen_servers[server_id]
            new_manifest = current_servers[server_id]
            if json.dumps(old_manifest, sort_keys=True) != json.dumps(new_manifest, sort_keys=True):
                print(f"[SyncDaemon] Detected updates in server: {server_id}")
                # Analyze individual tools
                old_tools = {t["name"]: t for t in old_manifest.get("tools", [])}
                new_tools = {t["name"]: t for t in new_manifest.get("tools", [])}
                
                # Check for tool changes/additions within this server
                for name, new_t in new_tools.items():
                    tool_id = f"{server_id}/{name}"
                    if name not in old_tools:
                        # Dynamic tool addition inside existing server
                        self.handle_server_add({
                            "name": server_id,
                            "summary": new_manifest.get("summary", ""),
                            "tools": [new_t]
                        })
                    elif json.dumps(old_tools[name], sort_keys=True) != json.dumps(new_t, sort_keys=True):
                        # Dynamic tool update
                        self.handle_tool_update(tool_id, new_t)
                        
                # Check for tools removed from the server
                for name in old_tools:
                    if name not in new_tools:
                        # Remove tool individual node
                        tool_id = f"{server_id}/{name}"
                        self.handle_tool_remove(tool_id)
                        
                self.last_seen_servers[server_id] = new_manifest

    def handle_server_add(self, server_manifest: Dict[str, Any]):
        """
        Registers new tools, runs schema compat and param overlap miners locally,
        adds vectors to FAISS indices, and updates graph representation.
        """
        start_time = time.perf_counter()
        server_name = server_manifest.get("name") or server_manifest.get("server_name")
        server_summary = server_manifest.get("summary", "")
        tools = server_manifest.get("tools", [])
        
        new_tools_normalized = []
        for tool in tools:
            tool_name = tool.get("name")
            desc = tool.get("description", "")
            param_names, param_types = extract_parameter_types_from_tool(tool)
            node_id = f"{server_name}/{tool_name}"
            
            new_tools_normalized.append({
                "id": node_id,
                "name": tool_name,
                "server": server_name,
                "description": desc,
                "parameter_names": param_names,
                "parameter_types": param_types,
                "raw": tool,
                "server_summary": server_summary,
                "description_embedding": tool.get("description_embedding", [])
            })
            
        for t in new_tools_normalized:
            node_id = t["id"]
            if node_id in self.pipeline.node_to_idx:
                print(f"[SyncDaemon] Tool {node_id} already exists. Skipping addition.")
                continue
                
            print(f"[SyncDaemon] Adding new tool: {node_id}")
            
            # 1. Update mappings
            new_idx = len(self.pipeline.node_to_idx)
            self.pipeline.node_to_idx[node_id] = new_idx
            self.pipeline.idx_to_node[new_idx] = node_id
            
            # 2. Add to NetworkX Graph
            self.pipeline.G.add_node(
                node_id,
                name=t["name"],
                server=t["server"],
                description=t["description"],
                parameter_names=t["parameter_names"],
                parameter_types=t["parameter_types"],
                server_summary=t["server_summary"]
            )
            
            # 3. Mine local heuristic relationships (schema_compat, param_overlap)
            self._mine_edges_for_node(node_id, t)
            
            # 4. Generate SentenceTransformer raw description embedding
            raw_desc = t["description"] if t["description"] else t["name"]
            raw_emb = self.pipeline.model_st.encode([raw_desc], show_progress_bar=False)[0]
            
            # Append to pipeline.X
            self.pipeline.X = np.append(self.pipeline.X, [raw_emb], axis=0)
            
            # 5. Initialize node GNN embedding (use raw_emb placeholder initially)
            self.pipeline.H = np.append(self.pipeline.H, [raw_emb], axis=0)
            
            # 6. Add to FAISS index
            id_arr = np.array([new_idx], dtype=np.int64)
            self.pipeline.raw_index.add_with_ids(raw_emb.reshape(1, -1).astype("float32"), id_arr)
            self.pipeline.gnn_index.add_with_ids(raw_emb.reshape(1, -1).astype("float32"), id_arr)
            
        # Recompute GNN embeddings for the entire graph to align representations
        if len(new_tools_normalized) > 0:
            self._recompute_gnn_embeddings()
            
            # Update GNN FAISS index with recomputed embeddings
            for t in new_tools_normalized:
                node_id = t["id"]
                idx = self.pipeline.node_to_idx[node_id]
                id_arr = np.array([idx], dtype=np.int64)
                self.pipeline.gnn_index.remove_ids(id_arr)
                self.pipeline.gnn_index.add_with_ids(self.pipeline.H[idx].reshape(1, -1).astype("float32"), id_arr)
                
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        self.log_latency("server_add", f"Server {server_name} with {len(tools)} tools", latency_ms)
        print(f"[SyncDaemon] Finished adding server {server_name}. Latency: {latency_ms:.2f} ms")

    def _mine_edges_for_node(self, node_id: str, tool_data: Dict[str, Any]):
        """
        Compares a single new tool against all existing tools in G and adds mined edges.
        """
        types_A = set(tool_data["parameter_types"])
        if not types_A:
            return
            
        for other_node in self.pipeline.G.nodes():
            if other_node == node_id:
                continue
                
            other_data = self.pipeline.G.nodes[other_node]
            types_B = set(other_data.get("parameter_types", []))
            if not types_B:
                continue
                
            # Intersect
            intersection = types_A.intersection(types_B)
            if not intersection:
                continue
                
            # Schema compatibility: |A intersect B| / min(|A|, |B|)
            overlap = len(intersection) / min(len(types_A), len(types_B))
            if overlap > 0.7:
                self.pipeline.G.add_edge(node_id, other_node, key="schema_compat", type="schema_compat", weight=overlap)
                self.pipeline.G.add_edge(other_node, node_id, key="schema_compat", type="schema_compat", weight=overlap)
                
            # Parameter overlap: |A intersect B| / |A union B| (Jaccard)
            union = types_A.union(types_B)
            jaccard = len(intersection) / len(union) if union else 0.0
            if jaccard > 0.4:
                self.pipeline.G.add_edge(node_id, other_node, key="param_overlap", type="param_overlap", weight=jaccard)
                self.pipeline.G.add_edge(other_node, node_id, key="param_overlap", type="param_overlap", weight=jaccard)

    def handle_tool_update(self, tool_id: str, new_tool_manifest: Dict[str, Any]):
        """
        Handles tools whose manifests/descriptions have changed.
        Recomputes incident edges, updates FAISS vector, and recomputes GNN embeddings.
        """
        start_time = time.perf_counter()
        if tool_id not in self.pipeline.node_to_idx:
            print(f"[SyncDaemon] Tool {tool_id} not found for updates.")
            return
            
        print(f"[SyncDaemon] Updating tool: {tool_id}")
        
        idx = self.pipeline.node_to_idx[tool_id]
        
        # 1. Update Graph Attributes
        param_names, param_types = extract_parameter_types_from_tool(new_tool_manifest)
        self.pipeline.G.nodes[tool_id]["description"] = new_tool_manifest.get("description", "")
        self.pipeline.G.nodes[tool_id]["parameter_names"] = param_names
        self.pipeline.G.nodes[tool_id]["parameter_types"] = param_types
        
        # 2. Recompute Incident Edges (Remove old, mine new)
        old_edges = list(self.pipeline.G.edges(tool_id, keys=True)) + list(self.pipeline.G.in_edges(tool_id, keys=True))
        for u, v, k in old_edges:
            try:
                self.pipeline.G.remove_edge(u, v, key=k)
            except Exception:
                pass
                
        # Re-run local miner
        tool_data = {
            "parameter_types": param_types
        }
        self._mine_edges_for_node(tool_id, tool_data)
        
        # 3. Update FAISS raw vector
        new_desc = new_tool_manifest.get("description", "") or new_tool_manifest.get("name", "")
        new_raw_emb = self.pipeline.model_st.encode([new_desc], show_progress_bar=False)[0]
        
        # Update raw embedding matrix
        self.pipeline.X[idx] = new_raw_emb
        
        id_arr = np.array([idx], dtype=np.int64)
        self.pipeline.raw_index.remove_ids(id_arr)
        self.pipeline.raw_index.add_with_ids(new_raw_emb.reshape(1, -1).astype("float32"), id_arr)
        
        # 4. Find affected nodes within 2 hops in G
        affected_nodes = self._find_k_hop_neighbors(tool_id, hops=2)
        
        # 5. Recompute GNN embeddings for the graph
        self._recompute_gnn_embeddings()
        
        # 6. Update FAISS GNN vectors for affected nodes
        for node in affected_nodes:
            if node in self.pipeline.node_to_idx:
                a_idx = self.pipeline.node_to_idx[node]
                a_id_arr = np.array([a_idx], dtype=np.int64)
                self.pipeline.gnn_index.remove_ids(a_id_arr)
                self.pipeline.gnn_index.add_with_ids(self.pipeline.H[a_idx].reshape(1, -1).astype("float32"), a_id_arr)
                
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        self.log_latency("tool_update", f"Tool {tool_id}", latency_ms)
        print(f"[SyncDaemon] Finished updating tool {tool_id}. Latency: {latency_ms:.2f} ms")

    def _find_k_hop_neighbors(self, node: str, hops: int = 2) -> Set[str]:
        """
        Finds all neighbors within k-hops of a node (undirected sense).
        """
        current_set = {node}
        for _ in range(hops):
            new_nodes = set()
            for u in current_set:
                if u not in self.pipeline.G:
                    continue
                new_nodes.update(self.pipeline.G.successors(u))
                new_nodes.update(self.pipeline.G.predecessors(u))
            current_set.update(new_nodes)
        return current_set

    def handle_tool_remove(self, tool_id: str):
        """
        Utility handler to remove a single tool node.
        """
        if tool_id not in self.pipeline.node_to_idx:
            return
            
        print(f"[SyncDaemon] Removing tool: {tool_id}")
        
        idx = self.pipeline.node_to_idx[tool_id]
        
        # 1. Remove from NetworkX graph
        if self.pipeline.G.has_node(tool_id):
            self.pipeline.G.remove_node(tool_id)
            
        # 2. Call FAISS remove_ids
        id_arr = np.array([idx], dtype=np.int64)
        self.pipeline.raw_index.remove_ids(id_arr)
        self.pipeline.gnn_index.remove_ids(id_arr)
        
        # 3. Delete from numpy arrays X and H, and shift indices in mapping
        self.pipeline.X = np.delete(self.pipeline.X, idx, axis=0)
        self.pipeline.H = np.delete(self.pipeline.H, idx, axis=0)
        
        del self.pipeline.node_to_idx[tool_id]
        del self.pipeline.idx_to_node[idx]
        
        # Shift all mappings
        new_node_to_idx = {}
        for node, old_idx in self.pipeline.node_to_idx.items():
            if old_idx > idx:
                new_node_to_idx[node] = old_idx - 1
            else:
                new_node_to_idx[node] = old_idx
        self.pipeline.node_to_idx = new_node_to_idx
        
        new_idx_to_node = {}
        for old_idx, node in self.pipeline.idx_to_node.items():
            if old_idx > idx:
                new_idx_to_node[old_idx - 1] = node
            else:
                new_idx_to_node[old_idx] = node
        self.pipeline.idx_to_node = new_idx_to_node
        
        # Re-initialize/rebuild FAISS indices to keep explicit IDs in sync with shifted row index
        d = 384
        self.pipeline.raw_index = faiss.IndexIDMap2(faiss.IndexFlatL2(d))
        self.pipeline.gnn_index = faiss.IndexIDMap2(faiss.IndexFlatL2(d))
        
        new_ids = np.array(range(len(self.pipeline.X)), dtype=np.int64)
        self.pipeline.raw_index.add_with_ids(self.pipeline.X.astype("float32"), new_ids)
        self.pipeline.gnn_index.add_with_ids(self.pipeline.H.astype("float32"), new_ids)

    def handle_server_remove(self, server_id: str):
        """
        Removes all tools and incident edges associated with server_id.
        Updates FAISS indices via remove_ids.
        """
        start_time = time.perf_counter()
        print(f"[SyncDaemon] Removing server: {server_id}")
        
        # Find all tools belonging to this server
        tools_to_remove = [node for node in self.pipeline.G.nodes() if node.startswith(server_id + "/")]
        
        for tool_id in tools_to_remove:
            self.handle_tool_remove(tool_id)
            
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        self.log_latency("server_remove", f"Server {server_id} with {len(tools_to_remove)} tools", latency_ms)
        print(f"[SyncDaemon] Finished removing server {server_id}. Latency: {latency_ms:.2f} ms")

    def _recompute_gnn_embeddings(self):
        """
        Runs R-GCN link predictor model over the entire updated graph structure
        and updates self.pipeline.H.
        """
        rel_map = {
            "co_invoke": 0,
            "schema_compat": 1,
            "param_overlap": 2,
            "compose_dep": 3
        }
        
        # Generate edge_index and edge_type tensors
        edges_list = []
        for u, v, key, data in self.pipeline.G.edges(keys=True, data=True):
            if key in rel_map:
                u_idx = self.pipeline.node_to_idx[u]
                v_idx = self.pipeline.node_to_idx[v]
                r_idx = rel_map[key]
                edges_list.append((u_idx, v_idx, r_idx))
                
        if len(edges_list) > 0:
            src = [e[0] for e in edges_list]
            dst = [e[1] for e in edges_list]
            types = [e[2] for e in edges_list]
            edge_index = torch.tensor([src, dst], dtype=torch.long).to(self.device)
            edge_type = torch.tensor(types, dtype=torch.long).to(self.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
            edge_type = torch.empty((0,), dtype=torch.long).to(self.device)
            
        # Run forward pass
        X_tensor = torch.tensor(self.pipeline.X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            H_tensor = self.gnn_model.encode(X_tensor, edge_index, edge_type)
            H_numpy = H_tensor.cpu().numpy()
            
        self.pipeline.H = H_numpy

    def measure_full_rebuild_latency(self) -> Dict[str, float]:
        """
        Measures the latency of a complete from-scratch rebuild of the graph,
        raw FAISS index, and GNN FAISS index. This provides the baseline
        denominator for computing the percentage reduction that incremental
        sync operations achieve.

        Returns dict with full_rebuild_ms, graph_build_ms, embedding_ms,
        faiss_build_ms, and gnn_forward_ms.
        """
        import copy
        from src.graph_builder import GraphBuilder
        from src.rgcn_trainer import build_faiss_index

        print("\n[FullRebuild] Measuring full-rebuild baseline latency...")

        # ── Phase 1: Graph construction from manifest ──────────────────
        t0 = time.perf_counter()
        builder = GraphBuilder(
            trajectory_dir=self.pipeline.trajectory_dir,
            compose_cache_path=self.pipeline.cache_path
        )
        G_new = builder.build_graph(self.pipeline.manifest_path)
        graph_build_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[FullRebuild]   Graph build:       {graph_build_ms:>10.2f} ms  ({len(G_new.nodes())} nodes, {len(G_new.edges())} edges)")

        # ── Phase 2: Embedding generation ──────────────────────────────
        t0 = time.perf_counter()
        node_to_idx = {node: i for i, node in enumerate(G_new.nodes())}
        idx_to_node = {i: node for node, i in node_to_idx.items()}
        descriptions = []
        for i in range(len(idx_to_node)):
            node = idx_to_node[i]
            desc = G_new.nodes[node].get("description", "")
            if not desc:
                desc = G_new.nodes[node].get("name", "")
            descriptions.append(desc)
        X_new = self.pipeline.model_st.encode(descriptions, show_progress_bar=False)
        embedding_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[FullRebuild]   Embedding gen:     {embedding_ms:>10.2f} ms  ({len(descriptions)} tools)")

        # ── Phase 3: GNN forward pass ──────────────────────────────────
        t0 = time.perf_counter()
        rel_map = {"co_invoke": 0, "schema_compat": 1, "param_overlap": 2, "compose_dep": 3}
        edges_list = []
        for u, v, key, data in G_new.edges(keys=True, data=True):
            if key in rel_map and u in node_to_idx and v in node_to_idx:
                edges_list.append((node_to_idx[u], node_to_idx[v], rel_map[key]))

        if edges_list:
            src = [e[0] for e in edges_list]
            dst = [e[1] for e in edges_list]
            types = [e[2] for e in edges_list]
            edge_index = torch.tensor([src, dst], dtype=torch.long).to(self.device)
            edge_type = torch.tensor(types, dtype=torch.long).to(self.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
            edge_type = torch.empty((0,), dtype=torch.long).to(self.device)

        X_tensor = torch.tensor(X_new, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            H_new = self.gnn_model.encode(X_tensor, edge_index, edge_type).cpu().numpy()
        gnn_forward_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[FullRebuild]   GNN forward:       {gnn_forward_ms:>10.2f} ms")

        # ── Phase 4: FAISS index construction ──────────────────────────
        t0 = time.perf_counter()
        faiss.omp_set_num_threads(1)
        raw_idx = build_faiss_index(X_new)
        gnn_idx = build_faiss_index(H_new)
        faiss_build_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[FullRebuild]   FAISS build:       {faiss_build_ms:>10.2f} ms  (2 indices × {len(X_new)} vectors)")

        full_rebuild_ms = graph_build_ms + embedding_ms + gnn_forward_ms + faiss_build_ms
        print(f"[FullRebuild]   ─────────────────────────────")
        print(f"[FullRebuild]   TOTAL rebuild:     {full_rebuild_ms:>10.2f} ms")

        result = {
            "full_rebuild_ms": full_rebuild_ms,
            "graph_build_ms": graph_build_ms,
            "embedding_ms": embedding_ms,
            "gnn_forward_ms": gnn_forward_ms,
            "faiss_build_ms": faiss_build_ms
        }

        self.log_latency("full_rebuild", f"Complete rebuild ({len(G_new.nodes())} tools)", full_rebuild_ms)
        return result

