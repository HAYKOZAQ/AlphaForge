import pandas as pd
import logging
from pydantic import BaseModel, Field, validator
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OHLCVSchema(BaseModel):
    date: object
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class FeatureSchema(BaseModel):
    date: object
    ticker: str
    # Flexible as feature set grows
    class Config:
        extra = 'allow'

class TargetSchema(BaseModel):
    date: object
    ticker: str
    target_class: float
    class Config:
        extra = 'allow'

class MacroSchema(BaseModel):
    date: object
    # FRED series names vary, so allow extra
    class Config:
        extra = 'allow'

class TextSignalSchema(BaseModel):
    date: object
    ticker: str
    sentiment: float
    class Config:
        extra = 'allow'

class EmbeddingSchema(BaseModel):
    ticker: str
    class Config:
        extra = 'allow'

class CentralitySchema(BaseModel):
    ticker: str
    network_centrality: float
    class Config:
        extra = 'allow'

class RegimeSchema(BaseModel):
    date: object
    macro_regime: int
    regime_prob_0: float
    regime_prob_1: float
    class Config:
        extra = 'allow'

class InsiderSchema(BaseModel):
    date: object
    ticker: str
    insider_net_intensity: float
    insider_conviction_count: float
    class Config:
        extra = 'allow'

class LaborSchema(BaseModel):
    date: object
    class Config:
        extra = 'allow'

def validate_dataframe(df: pd.DataFrame, schema_class, node_name: str):
    """
    Validates a dataframe against a Pydantic schema.
    Performs basic sanity checks (nulls, duplicates).
    """
    logging.info(f"Running validation for [{node_name}]...")
    
    # 1. Null Check
    null_counts = df.isnull().sum()
    if null_counts.any():
        logging.warning(f"[{node_name}] Null values detected: \n{null_counts[null_counts > 0]}")
    
    # 2. Duplicate Check
    if 'date' in df.columns and 'ticker' in df.columns:
        dupes = df.duplicated(subset=['date', 'ticker']).sum()
        if dupes > 0:
            logging.error(f"[{node_name}] {dupes} duplicate (date, ticker) rows found!")
            
    # 3. Column Presence
    required_cols = list(schema_class.__fields__.keys())
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[{node_name}] Missing required columns: {missing_cols}")
        
    logging.info(f"[{node_name}] Validation passed for {len(df)} rows.")
    return True

def data_sanity_decorator(schema_class, node_name):
    """Decorator to automatically validate the returned dataframe of a node."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            df = func(*args, **kwargs)
            if isinstance(df, pd.DataFrame):
                validate_dataframe(df, schema_class, node_name)
            return df
        return wrapper
    return decorator
