import pytest
import pandas as pd
from pathlib import Path

def test_config_loading():
    from src.utils.config_loader import load_config
    config = load_config()
    assert 'universe' in config
    assert 'modeling' in config

def test_raw_data_exists():
    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "data" / "raw"
    # Basic check for existence (doesn't require files to be there yet, but directory should)
    assert raw_dir.exists()

def test_processed_data_schema():
    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = repo_root / "data" / "processed"
    master_path = processed_dir / "master_dataset.parquet"
    
    if master_path.exists():
        df = pd.read_parquet(master_path)
        assert 'date' in df.columns
        assert 'ticker' in df.columns
        assert 'target_class' in df.columns

def test_feature_engineering_logic():
    from src.feature_builder import engineer_quant_features
    # Create fake data
    dates = pd.date_range("2020-01-01", periods=200)
    df = pd.DataFrame({
        'date': dates,
        'ticker': ['TEST'] * 200,
        'close': [100.0 + i for i in range(200)],
        'volume': [1000] * 200
    })
    
    windows = [21, 63]
    result = engineer_quant_features(df, windows)
    
    assert not result.empty
    assert 'vol_21d' in result.columns
    assert 'mom_63d' in result.columns
