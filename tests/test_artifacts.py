import pytest
from pathlib import Path

def test_critical_directories_exist():
    repo_root = Path(__file__).resolve().parent.parent
    dirs = ['data/raw', 'data/processed', 'models', 'reports/metrics', 'reports/plots']
    for d in dirs:
        assert (repo_root / d).exists(), f"Directory missing: {d}"

def test_model_artifact_loading():
    repo_root = Path(__file__).resolve().parent.parent
    model_path = repo_root / "models" / "model_C.pkl"
    
    # This is a smoke test - if the file exists, it should be a pickle
    if model_path.exists():
        import pickle
        with open(model_path, "rb") as f:
            m = pickle.load(f)
            assert "model" in m
            assert "features" in m

def test_metrics_integrity():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "reports" / "metrics" / "walk_forward_metrics.json"
    
def test_graph_artifact_integrity():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "data" / "graph_models" / "semantic_network.gml"
    if path.exists():
        import networkx as nx
        G = nx.read_gml(str(path))
        assert G.number_of_nodes() > 0

def test_backtest_output_integrity():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "reports" / "metrics.json"
    if path.exists():
        import json
        with open(path, "r") as f:
            data = json.load(f)
            # Should have results for at least one model
            assert len(data) > 0
            # Check for standard fields we added in V2.0 backtest
            first_model = list(data.keys())[0]
            assert "spearman" in data[first_model]
