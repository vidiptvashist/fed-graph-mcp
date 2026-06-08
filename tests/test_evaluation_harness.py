import os
import json
import pytest
from src.retrieval_pipeline import RetrievalPipeline
from src.evaluation_harness import EvaluationHarness

@pytest.fixture(scope="module")
def pipeline():
    return RetrievalPipeline()

@pytest.fixture(scope="module")
def harness(pipeline):
    return EvaluationHarness(pipeline=pipeline)

def test_harness_initialization(harness):
    assert harness.pipeline is not None
    assert len(harness.livemcpbench_tasks) == 95
    assert len(harness.mcpbench_tasks) == 104

def test_run_evaluation_single_condition(harness):
    # Test a small subset of 3 tasks to run quickly
    small_tasks = harness.livemcpbench_tasks[:3]
    
    # Run evaluation on dense_only condition
    metrics = harness.run_evaluation(small_tasks, "dense_only")
    
    assert "Recall@1" in metrics
    assert "Recall@5" in metrics
    assert "nDCG@5" in metrics
    assert "SuccessRate" in metrics
    assert "TokenCount" in metrics
    
    assert 0.0 <= metrics["Recall@1"] <= 1.0
    assert 0.0 <= metrics["Recall@5"] <= 1.0
    assert metrics["TokenCount"] > 500

def test_generate_latex_tables(harness):
    # Make a tiny mock results dictionary
    mock_results = {
        "LiveMCPBench (Simulated)": {
            "baseline": {"Recall@1": 0.5, "Recall@5": 0.8, "nDCG@5": 0.7, "SuccessRate": 0.75, "TokenCount": 1000},
            "dense_only": {"Recall@1": 0.55, "Recall@5": 0.82, "nDCG@5": 0.72, "SuccessRate": 0.77, "TokenCount": 1250},
            "gnn_only": {"Recall@1": 0.56, "Recall@5": 0.83, "nDCG@5": 0.73, "SuccessRate": 0.78, "TokenCount": 1250},
            "rrf_only": {"Recall@1": 0.6, "Recall@5": 0.85, "nDCG@5": 0.75, "SuccessRate": 0.80, "TokenCount": 1250},
            "rrf_rerank": {"Recall@1": 0.65, "Recall@5": 0.88, "nDCG@5": 0.78, "SuccessRate": 0.83, "TokenCount": 1250},
            "rrf_expand": {"Recall@1": 0.66, "Recall@5": 0.89, "nDCG@5": 0.79, "SuccessRate": 0.84, "TokenCount": 1250},
            "full_pipeline": {"Recall@1": 0.72, "Recall@5": 0.92, "nDCG@5": 0.85, "SuccessRate": 0.88, "TokenCount": 1250}
        },
        "MCP-Bench (Simulated)": {
            "baseline": {"Recall@1": 0.48, "Recall@5": 0.78, "nDCG@5": 0.68, "SuccessRate": 0.73, "TokenCount": 1000},
            "dense_only": {"Recall@1": 0.52, "Recall@5": 0.80, "nDCG@5": 0.70, "SuccessRate": 0.75, "TokenCount": 1250},
            "gnn_only": {"Recall@1": 0.53, "Recall@5": 0.81, "nDCG@5": 0.71, "SuccessRate": 0.76, "TokenCount": 1250},
            "rrf_only": {"Recall@1": 0.58, "Recall@5": 0.84, "nDCG@5": 0.74, "SuccessRate": 0.79, "TokenCount": 1250},
            "rrf_rerank": {"Recall@1": 0.62, "Recall@5": 0.86, "nDCG@5": 0.76, "SuccessRate": 0.81, "TokenCount": 1250},
            "rrf_expand": {"Recall@1": 0.63, "Recall@5": 0.87, "nDCG@5": 0.77, "SuccessRate": 0.82, "TokenCount": 1250},
            "full_pipeline": {"Recall@1": 0.70, "Recall@5": 0.90, "nDCG@5": 0.83, "SuccessRate": 0.86, "TokenCount": 1250}
        }
    }
    
    latex_str = harness.generate_latex_tables(mock_results)
    assert "\\begin{table}" in latex_str
    assert "Recall@1" in latex_str
    assert "LiveMCPBench" in latex_str
    assert "MCP-Bench" in latex_str
    assert "simulated benchmark conditions" in latex_str
    
    # Check that it saves the file successfully
    latex_file = os.path.join(harness.base_dir, "evaluation_tables.tex")
    assert os.path.exists(latex_file)

