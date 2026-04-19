import pandas as pd
import numpy as np
from pathlib import Path
import logging
from src.utils.validators import validate_dataframe, TargetSchema

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_features(group):
    # Ensure correct time ordering
    group = group.sort_values('date')
    
    # Advanced Momentum and Risk
    group['returns'] = group['close'].pct_change()
    group['mom_1m'] = group['close'].pct_change(20)
    group['vol_20d'] = group['returns'].rolling(20).std()
    
    # Future target labeling (20-day forward return)
    group['fwd_return_20d'] = group['close'].shift(-20) / group['close'] - 1.0
    return group

def main():
    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "data" / "raw"
    proc_dir = repo_root / "data" / "processed"
    
    ohlcv_path = raw_dir / "sp500_ohlcv.parquet"
    sec_path = proc_dir / "sec_vector_embeddings.parquet"
    graph_path = proc_dir / "graph_centrality.parquet"
    
    if not ohlcv_path.exists():
        logging.error("Missing Firm OHLCV data.")
        return
        
    logging.info("Ingesting huge OHLCV pool...")
    df = pd.read_parquet(ohlcv_path)
    
    # Calculate Base Quant Features
    logging.info("Calculating Firm-level rolling ML features...")
    df = df.groupby('ticker', group_keys=False).apply(build_features).dropna(subset=['fwd_return_20d', 'mom_1m', 'vol_20d'])
    
    # Merge Rigid NLP Themes
    if sec_path.exists():
        sec_df = pd.read_parquet(sec_path)
        if 'theme_cluster' in sec_df.columns:
            sec_mapped = sec_df[['ticker', 'theme_cluster']]
            df = df.merge(sec_mapped, on='ticker', how='left')
            # One-hot encode thematic clusters strictly mapping textual structure
            df = pd.get_dummies(df, columns=['theme_cluster'], dummy_na=False)
            logging.info("Firm Thematic taxonomy mapped into One-Hot continuous array.")
            
    # Merge Topological Graph Weights
    if graph_path.exists():
        graph_df = pd.read_parquet(graph_path)
        df = df.merge(graph_df, on='ticker', how='left')
        for col in ['network_pagerank', 'network_eigenvector']:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        logging.info("Strategic Network Centrality (PageRank/Eigen) injected to time-series nodes.")
        
    # Strict Firm-Level Target Definition
    # Identifies firms breaking strictly into the Top 20% of cross-sectional returns over the next 20 days.
    logging.info("Computing structural firm-level Target Labels (Top 20% Cross-Sectional Accelerators)")
    
    def compute_quantile(group):
        group['target_class'] = (group['fwd_return_20d'] > group['fwd_return_20d'].quantile(0.80)).astype(int)
        return group
        
    final_df = df.groupby('date', group_keys=False).apply(compute_quantile)
    
    # Drop NAs safely generated from merging
    final_df = final_df.dropna()
    
    out_path = proc_dir / "master_firm_dataset.parquet"
    validate_dataframe(final_df, TargetSchema, "Firm Target Generation")
    final_df.to_parquet(out_path, index=False)
    logging.info(f"Rigorous ML Target logic globally compiled -> {out_path} (Shape: {final_df.shape})")

if __name__ == "__main__":
    main()
