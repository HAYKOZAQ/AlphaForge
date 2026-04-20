import pytest
import pandas as pd
import numpy as np
from src.graph_builder import build_multi_layer_graph

def test_graph_generation_logic():
    # Mock embedding data
    data = {
        "ticker": ["AAPL", "MSFT", "GOOG"],
        "vec_0": [1.0, 0.9, 0.0],
        "vec_1": [0.0, 0.0, 1.0]
    }
    df = pd.DataFrame(data)
    
    # Threshold 0.8: AAPL-MSFT should link, GOOG isolated
    # Use alpha=1.0 to focus on semantic similarity for this test
    G = build_multi_layer_graph(df, pd.DataFrame(), pd.DataFrame(), threshold=0.8, alpha=1.0, beta=0.0, gamma=0.0)
    
    assert G.number_of_nodes() == 3
    assert G.has_edge("AAPL", "MSFT")
    assert not G.has_edge("AAPL", "GOOG")

def test_target_labeling_logic():
    from src.target_builder import compute_targets
    
    # Mock feature data
    dates = pd.date_range("2020-01-01", periods=10, freq="M")
    data = []
    for d in dates:
        for t in ["T1", "T2", "T3"]:
            data.append({
                "date": d,
                "ticker": t,
                "close": 100.0,
                "volume_zscore": 0.0,
                "mom_63d": 0.0,
                "vol_63d": 0.1,
                "theme_increase": 0.0
            })
    df = pd.DataFrame(data)
    
    # Force a winner in T1 for a specific date
    # Row 15 is date[5], T1
    df.loc[15, "close"] = 150.0 # Huge forward return if price stays same later... 
    # Actually shift(-3) means it looks forward.
    # Let's just mock the fwd_ret directly for the test if compute_targets allows it? 
    # No, compute_targets calculates it.
    
    # T1 at date index 2 will look at date index 5.
    df.loc[6, "close"] = 200.0 # T1, Date 2 -> Return = (200/100) - 1 = 100%
    
    df_res = compute_targets(df)
    assert "target_class" in df_res.columns
    # We shouldn't have targets in the final 3 months
    assert df_res[df_res["date"] == dates[-1]]["target_class"].isna().all()
