import os
import json
import time
from src.retrieval_pipeline import RetrievalPipeline
from src.sync_daemon import SyncDaemon

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    latency_log_path = os.path.join(base_dir, "mcp_sync_latency.json")
    
    # Clean up previous latency logs to have a clean slate
    if os.path.exists(latency_log_path):
        try:
            os.remove(latency_log_path)
        except Exception:
            pass
            
    print("====================================================")
    print("Starting Sync Daemon Live Verification Run")
    print("====================================================")
    
    # 1. Initialize pipeline and daemon
    pipeline = RetrievalPipeline()
    daemon = SyncDaemon(pipeline=pipeline)
    
    # Let's perform a query before we add our custom tools
    query = "Optimize database index and query speed"
    print(f"\n[Step 1] Baseline retrieval query: '{query}'")
    baseline_res = pipeline.retrieve_dense(query, k=3)
    for i, r in enumerate(baseline_res):
        print(f"  {i+1}. {r['name']} (Score: {r['score']:.4f})")
        
    # 2. Add a new server manifest dynamically
    print("\n[Step 2] Dynamically adding a new server 'MockOptimizer'...")
    new_server = {
        "name": "MockOptimizer",
        "summary": "Specialized SQL performance optimization utilities",
        "tools": [
            {
                "name": "sqlite_optimize_index",
                "description": "Optimizes sqlite database index and improves query speed.",
                "parameter": {
                    "table_name": "(string) Name of the sqlite table.",
                    "index_name": "(string) Name of the index."
                }
            }
        ]
    }
    
    daemon.handle_server_add(new_server)
    
    # Verify the tool is now retrieved
    print(f"\n[Step 3] Querying after server addition: '{query}'")
    post_add_res = pipeline.retrieve_dense(query, k=3)
    for i, r in enumerate(post_add_res):
        print(f"  {i+1}. {r['name']} (Score: {r['score']:.4f})")
        
    # Check that our newly added tool is retrieved in top results
    assert any(r["name"] == "MockOptimizer/sqlite_optimize_index" for r in post_add_res)
    
    # 3. Update the tool dynamically (e.g. description is changed to something irrelevant)
    print("\n[Step 4] Dynamically updating 'MockOptimizer/sqlite_optimize_index' to be unrelated...")
    updated_tool = {
        "name": "sqlite_optimize_index",
        "description": "Brew a cup of hot espresso coffee.",
        "parameter": {
            "espresso_type": "(string) The type of coffee bean."
        }
    }
    
    daemon.handle_tool_update("MockOptimizer/sqlite_optimize_index", updated_tool)
    
    # Query again - it should no longer be in the top results for database queries
    print(f"\n[Step 5] Querying after tool update (now coffee brew): '{query}'")
    post_update_res = pipeline.retrieve_dense(query, k=3)
    for i, r in enumerate(post_update_res):
        print(f"  {i+1}. {r['name']} (Score: {r['score']:.4f})")
        
    assert not any(r["name"] == "MockOptimizer/sqlite_optimize_index" for r in post_update_res)
    
    # 4. Remove the server dynamically
    print("\n[Step 6] Dynamically removing server 'MockOptimizer'...")
    daemon.handle_server_remove("MockOptimizer")
    
    # Verify it is no longer in the pipeline mappings
    assert "MockOptimizer/sqlite_optimize_index" not in pipeline.node_to_idx
    print("Verification passed: Tool successfully removed from pipeline mappings.")
    
    # 5. Display logged sync latencies
    print("\n=================== INCREMENTAL SYNC LATENCIES ===================")
    incremental_latencies = {}
    if os.path.exists(latency_log_path):
        with open(latency_log_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
        for log in logs:
            event = log['event_type']
            lat = log['sync_latency_ms']
            incremental_latencies[event] = lat
            print(f"Event: {event:<15} | Details: {log['details']:<40} | Latency: {lat:.2f} ms")
    else:
        print("No latency metrics logged.")
    print("=" * 62)
    
    # 6. Measure full-rebuild baseline latency
    print("\n=================== FULL-REBUILD BASELINE =======================")
    rebuild_result = daemon.measure_full_rebuild_latency()
    full_rebuild_ms = rebuild_result["full_rebuild_ms"]
    print("=" * 62)
    
    # 7. Compute and report reduction ratios
    print("\n=================== LATENCY REDUCTION ANALYSIS ==================")
    print(f"{'Operation':<20} {'Incremental':>12} {'Full Rebuild':>14} {'Reduction':>10}")
    print(f"{'─'*20} {'─'*12} {'─'*14} {'─'*10}")
    
    reduction_report = {}
    for event_type, inc_ms in incremental_latencies.items():
        reduction_pct = (1.0 - inc_ms / full_rebuild_ms) * 100.0
        reduction_report[event_type] = {
            "incremental_ms": inc_ms,
            "full_rebuild_ms": full_rebuild_ms,
            "reduction_pct": reduction_pct
        }
        print(f"{event_type:<20} {inc_ms:>10.2f}ms {full_rebuild_ms:>12.2f}ms {reduction_pct:>8.1f}%")
    
    print(f"{'─'*20} {'─'*12} {'─'*14} {'─'*10}")
    
    # Save combined latency report with reduction ratios
    combined_report = {
        "incremental_latencies": incremental_latencies,
        "full_rebuild": rebuild_result,
        "reduction_ratios": reduction_report
    }
    report_path = os.path.join(base_dir, "mcp_sync_latency_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2)
    print(f"\nSaved full latency report to: {report_path}")
    print("=" * 62 + "\n")
    
    print("====================================================")
    print("Sync Daemon Verification Finished Successfully")
    print("====================================================")

if __name__ == "__main__":
    main()

