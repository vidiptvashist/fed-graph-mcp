# FED-GRAPH-MCP

**Functionally-Enriched Dynamic Tool Graphs with Incremental Index Synchronization for Large-Scale MCP Ecosystems**

> Vidipt Vashist · Independent Researcher · Bengaluru, India

---

## Abstract

The Model Context Protocol (MCP) ecosystem has grown to over 10,000 registered servers as of mid-2026, creating a severe tool-discovery bottleneck. FED-GRAPH-MCP addresses this with three core contributions:

1. **Multi-typed functional tool graph** — four categories of inter-tool edges (CO_INVOKE, SCHEMA_COMPAT, PARAM_TYPE_OVERLAP, COMPOSE_DEP) beyond ownership hierarchies
2. **Dual-branch retrieval pipeline** — R-GCN structural embeddings fused with dense semantic embeddings via a learned late-fusion reranker
3. **CRUD-triggered incremental sync daemon** — propagates registry changes with 83.9–99.9% lower latency than full recomputation

Under simulated benchmark conditions, FED-GRAPH-MCP achieves **Recall@5 of 0.853 / 0.856** on LiveMCPBench and MCP-Bench task distributions (+10.6pp over MCP-Zero baseline).

## Architecture

```
Tool Manifests → Edge Miners (4 types) → Graph G (NetworkX)
                                              ↓
                                         R-GCN Trainer → GNN FAISS Index
                                              ↓
User Query → Dense Retrieval ──┐
             GNN Retrieval   ──┤→ RRF Fusion → Learned Reranker → Graph Expand → Top-k Tools
                               │
Registry ──→ Sync Daemon ──────┘ (live CRUD updates)
```

## Project Structure

```
FED-GRAPH/
├── src/
│   ├── graph_builder.py          # Multi-relational graph construction
│   ├── edge_miners/              # CO_INVOKE, SCHEMA_COMPAT, PARAM_OVERLAP, COMPOSE_DEP
│   ├── rgcn_trainer.py           # R-GCN link prediction training + FAISS index
│   ├── retrieval_pipeline.py     # Dense + GNN retrieval, RRF fusion, reranker, graph expand
│   ├── sync_daemon.py            # Incremental CRUD sync with latency tracking
│   ├── evaluation_harness.py     # Ablation grid + LaTeX table generation
│   └── utils.py                  # Shared utilities
├── tests/                        # 15 unit tests (pytest)
├── run_graph_builder.py          # Build multi-relational graph
├── run_rgcn_trainer.py           # Train R-GCN + build FAISS indices
├── run_retrieval_pipeline.py     # End-to-end retrieval demo
├── run_sync_daemon.py            # Sync daemon + full-rebuild baseline
├── run_evaluation.py             # Ablation grid (7 conditions × 2 benchmarks)
├── run_visualization.py          # Interactive graph visualization
└── paper/                        # LaTeX paper source
    └── main.tex
```

## Quick Start

### Prerequisites

```bash
pip install torch torch-geometric sentence-transformers faiss-cpu networkx aiohttp
```

### External Data (clone separately)

```bash
# MCP-Zero tool manifest (2,797 tools from 308 servers)
git clone https://github.com/your-org/MCP-zero.git mcp_zero_repo

# API-Bank trajectories (for co-invocation mining + reranker training)
git clone https://github.com/AlibabaResearch/DAMO-ConvAI.git damo_convai_repo
```

### Run Pipeline

```bash
# 1. Build graph
python3 run_graph_builder.py

# 2. Train R-GCN + build FAISS indices
python3 run_rgcn_trainer.py

# 3. Train reranker + test retrieval
python3 run_retrieval_pipeline.py

# 4. Test sync daemon + measure rebuild baseline
python3 run_sync_daemon.py

# 5. Run full ablation evaluation
python3 run_evaluation.py

# 6. Run tests
python3 -m pytest tests/ -v
```

## Key Results

### Retrieval Ablation (Simulated Benchmark Conditions)

| Condition | R@1 | R@5 | nDCG@5 | Success |
|---|---|---|---|---|
| Baseline (MCP-Zero) | 0.427 | 0.749 | 0.565 | 63.1% |
| Dense Only | 0.437 | 0.759 | 0.577 | 64.4% |
| GNN Only | 0.437 | 0.759 | 0.573 | 64.1% |
| RRF Fusion | 0.447 | 0.779 | 0.593 | 66.4% |
| RRF + Reranker | 0.492 | 0.804 | 0.630 | 70.0% |
| RRF + Expansion | 0.437 | 0.794 | 0.598 | 68.0% |
| **FED-GRAPH-MCP** | **0.512** | **0.855** | **0.661** | **73.9%** |

### Sync Daemon Latency

| Operation | Incremental | Full Rebuild | Reduction |
|---|---|---|---|
| server_add | 1,173 ms | 7,908 ms | **85.2%** |
| tool_update | 1,270 ms | 7,908 ms | **83.9%** |
| server_remove | 4 ms | 7,908 ms | **99.9%** |

## Citation

```bibtex
@article{vashist2026fedgraph,
  title={Functionally-Enriched Dynamic Tool Graphs with Incremental Index
         Synchronization for Large-Scale MCP Ecosystems},
  author={Vashist, Vidipt},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT
