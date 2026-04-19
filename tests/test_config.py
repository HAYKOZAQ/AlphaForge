import pytest
from src.utils.config_loader import load_config

def test_config_structure():
    config = load_config()
    required_keys = ['seed', 'dates', 'universe', 'modeling', 'graph', 'features']
    for key in required_keys:
        assert key in config, f"Missing critical key: {key}"

def test_config_date_formats():
    config = load_config()
    import pandas as pd
    try:
        pd.to_datetime(config['dates']['sector_start'])
        pd.to_datetime(config['dates']['sp500_start'])
    except Exception as e:
        pytest.fail(f"Invalid date format in config: {e}")

def test_modeling_params():
    config = load_config()
    params = config['modeling']['xgb_params']
    assert 'n_estimators' in params
    assert 'max_depth' in params
    assert params['n_estimators'] > 0
