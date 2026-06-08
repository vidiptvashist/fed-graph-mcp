"""
run_evaluation.py — Module 5: Full Evaluation Harness

Runs the complete ablation study grid (7 conditions × 2 benchmarks = 14 runs)
and generates paper-ready LaTeX tables from evaluation output JSON.

Usage:
    python3 run_evaluation.py
"""

import os
import sys
import time
import json

# Ensure project root is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrieval_pipeline import RetrievalPipeline
from src.evaluation_harness import EvaluationHarness


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 68)
    print("Module 5: FED-GRAPH-MCP Evaluation Harness")
    print("  NOTE: Simulated benchmark conditions (see paper §Evaluation)")
    print("=" * 68)
    print(f"Base directory: {base_dir}")
    print()

    # ── Step 1: Initialize retrieval pipeline ──────────────────────────
    print("[Step 1] Initializing RetrievalPipeline (graph, embeddings, FAISS)...")
    t0 = time.time()
    pipeline = RetrievalPipeline()
    print(f"  Pipeline ready — {len(pipeline.node_to_idx)} tools loaded in {time.time()-t0:.2f}s")
    print()

    # ── Step 2: Initialize evaluation harness ──────────────────────────
    print("[Step 2] Building EvaluationHarness (tasks + reranker)...")
    t0 = time.time()
    harness = EvaluationHarness(pipeline=pipeline)
    print(f"  LiveMCPBench tasks : {len(harness.livemcpbench_tasks)}")
    print(f"  MCP-Bench tasks   : {len(harness.mcpbench_tasks)}")
    print(f"  Harness ready in {time.time()-t0:.2f}s")
    print()

    # ── Step 3: Run full ablation grid ─────────────────────────────────
    print("[Step 3] Running ablation grid (7 conditions × 2 benchmarks = 14 runs)...")
    print("-" * 68)
    t0 = time.time()
    results = harness.run_ablation_grid()
    elapsed_total = time.time() - t0
    print(f"\n  Ablation grid completed in {elapsed_total:.2f}s")
    print()

    # ── Step 4: Generate LaTeX tables ──────────────────────────────────
    print("[Step 4] Generating paper-ready LaTeX tables...")
    latex_str = harness.generate_latex_tables(results)
    print()

    # ── Step 5: Print summary ──────────────────────────────────────────
    print("=" * 68)
    print("EVALUATION SUMMARY — SIMULATED BENCHMARK CONDITIONS")
    print("  Tasks derived from MCP manifest via paraphrase templates.")
    print("  These results isolate retrieval quality; they are not from")
    print("  the official LiveMCPBench / MCP-Bench Docker harnesses.")
    print("=" * 68)

    for bench_name, bench_results in results.items():
        print(f"\n{'─'*40}")
        print(f"  Benchmark: {bench_name}")
        print(f"{'─'*40}")
        print(f"  {'Condition':<20} {'R@1':>6} {'R@5':>6} {'nDCG@5':>8} {'Success':>8} {'Tokens':>7}")
        print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*7}")
        for cond in ["baseline", "dense_only", "gnn_only", "rrf_only",
                      "rrf_rerank", "rrf_expand", "full_pipeline"]:
            m = bench_results[cond]
            print(f"  {cond:<20} {m['Recall@1']:>6.4f} {m['Recall@5']:>6.4f} "
                  f"{m['nDCG@5']:>8.4f} {m['SuccessRate']:>7.2%} {int(m['TokenCount']):>7}")

    # ── Step 6: Print LaTeX output ─────────────────────────────────────
    print()
    print("=" * 68)
    print("GENERATED LATEX TABLES")
    print("=" * 68)
    print(latex_str)

    # ── Done ───────────────────────────────────────────────────────────
    eval_json = os.path.join(base_dir, "evaluation_results.json")
    eval_tex  = os.path.join(base_dir, "evaluation_tables.tex")
    print()
    print("=" * 68)
    print("All outputs saved:")
    print(f"  JSON  → {eval_json}")
    print(f"  LaTeX → {eval_tex}")
    print("=" * 68)


if __name__ == "__main__":
    main()
