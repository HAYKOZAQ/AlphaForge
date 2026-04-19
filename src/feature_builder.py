import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from src.utils.config_loader import load_config
from src.utils.validators import data_sanity_decorator, FeatureSchema

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@data_sanity_decorator(FeatureSchema, "Feature Engineering")
def engineer_quant_features(df, windows):
    logging.info(f"Engineering quantitative features for windows: {windows}")
    df = df.sort_values(['ticker', 'date']).copy()
    
    # Calculate daily returns
    df['ret_1d'] = df.groupby('ticker')['close'].pct_change()
    
    # Volatility and Momentum
    for w in windows:
        # Volatility (Annualized)
        df[f'vol_{w}d'] = df.groupby('ticker')['ret_1d'].transform(lambda x: x.rolling(w).std() * np.sqrt(252))
        
        # Momentum (Log returns Sum)
        df['log_ret'] = np.log(df['close'] / df.groupby('ticker')['close'].shift(1))
        df[f'mom_{w}d'] = df.groupby('ticker')['log_ret'].transform(lambda x: x.rolling(w).sum())
    
    # Volume Z-Score
    df['vol_ma_20'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(21).mean())
    df['volume_zscore'] = (df['volume'] - df['vol_ma_20']) / (df.groupby('ticker')['volume'].transform(lambda x: x.rolling(21).std()) + 1e-6)
    
    # Drop temp
    df = df.drop(columns=['log_ret', 'vol_ma_20'])
    
    # Monthly Resampling
    df['month'] = df['date'].dt.to_period('M')
    
    agg_dict = {
        'close': 'last',
        'volume': 'sum',
        'volume_zscore': 'last'
    }
    for w in windows:
        agg_dict[f'vol_{w}d'] = 'last'
        agg_dict[f'mom_{w}d'] = 'last'
        
    monthly_features = df.groupby(['ticker', 'month']).agg(agg_dict).reset_index()
    monthly_features['date'] = monthly_features['month'].dt.to_timestamp(how='end').dt.normalize()
    monthly_features = monthly_features.drop(columns=['month'])
    return monthly_features

def engineer_options_features(df_opt):
    """Calculates relative option skewness z-scores for sector ETFs."""
    logging.info("Engineering options skew features...")
    df_opt = df_opt.copy()
    
    # Calculate z-scores across all sectors to identify cross-sectional fear
    for col in ['put_call_oi_ratio', 'put_call_vol_ratio']:
        mean_val = df_opt[col].mean()
        std_val = df_opt[col].std()
        df_opt[f'opt_{col}_z'] = (df_opt[col] - mean_val) / (std_val + 1e-6)
        
    return df_opt[['ticker', 'date', 'opt_put_call_oi_ratio_z', 'opt_put_call_vol_ratio_z']]

def engineer_transcript_features(df_trans):
    """Processes raw transcript text into numeric sentiment scores."""
    logging.info("Engineering transcript sentiment signals...")
    df_trans = df_trans.copy()
    
    def calculate_score(text):
        pos_keywords = ['growth', 'increase', 'strong', 'adoption', 'momentum', 'upside']
        neg_keywords = ['weak', 'challenge', 'uncertainty', 'risk', 'decline', 'headwind']
        
        lower_text = text.lower()
        pos_count = sum(lower_text.count(k) for k in pos_keywords)
        neg_count = sum(lower_text.count(k) for k in neg_keywords)
        
        return (pos_count - neg_count) / (pos_count + neg_count + 1e-6)

    df_trans['transcript_sentiment'] = df_trans['transcript_text'].apply(calculate_score)
    return df_trans[['ticker', 'date', 'transcript_sentiment']]

def engineer_labor_features(df_macro):
    """Calculates relative labor demand momentum for industries."""
    logging.info("Engineering labor market demand features...")
    df = df_macro.copy()
    
    # Ensure sequential dates
    df = df.sort_values('date')
    
    # 1. Calculate momentum for all job series (3-month window)
    job_cols = [c for c in df.columns if 'job_openings' in c]
    for col in job_cols:
        # We use percent change over 3 months to capture structural trends
        df[f'{col}_3m_mom'] = df[col].pct_change(3)
        
    # 2. Calculate relative hiring strength vs the total economy
    base_col = 'job_openings_total_3m_mom'
    for col in [c for c in job_cols if c != 'job_openings_total']:
        df[f'labor_rel_strength_{col}'] = df[f'{col}_3m_mom'] - df[base_col]
        
    # Keep only the relative strength and raw momentum features
    keep_cols = ['date'] + [c for c in df.columns if 'labor_rel_strength' in c]
    return df[keep_cols]

def engineer_macro_regimes(df_macro):
    """Fits an HMM to macro data to detect hidden market regimes."""
    logging.info("Engineering Macroeconomic Regime Switching features (HMM)...")
    df = df_macro.copy()
    
    feature_cols = [c for c in ['fed_funds_rate', 'cpi', 'unemployment_rate', 'wti_oil', 'industrial_production'] if c in df.columns]
    if not feature_cols:
        logging.warning("No macro features found for HMM. Returning empty regime features.")
        return pd.DataFrame(columns=['date', 'macro_regime', 'regime_prob_0', 'regime_prob_1'])
        
    df = df.dropna(subset=feature_cols).sort_values('date')
    X = df[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit 2-state HMM
    model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X_scaled)
    
    df['macro_regime'] = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)
    df['regime_prob_0'] = probs[:, 0]
    df['regime_prob_1'] = probs[:, 1]
    
    return df[['date', 'macro_regime', 'regime_prob_0', 'regime_prob_1']]

def engineer_insider_features(df_insider, df_meta=None):
    """Aggregates raw Form 4 transactions into monthly conviction scores."""
    logging.info("Engineering insider trading conviction features...")
    df = df_insider.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Focus on high-signal 'Open Market' Purchase (P) and Sale (S)
    df = df[df['code'].isin(['P', 'S'])]
    
    if df.empty:
        return pd.DataFrame(columns=['ticker', 'date', 'insider_net_intensity', 'insider_conviction_count'])

    df['value'] = df['shares'] * df['price']
    
    # Resample to monthly start for alignment
    df['date'] = df['date'].dt.to_period('M').dt.to_timestamp('M')

    # Mapping to Sector ETFs if metadata is provided
    if df_meta is not None and not df_meta.empty:
        sector_map = {
            'Information Technology': 'XLK',
            'Health Care': 'XLV',
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Communication Services': 'XLC',
            'Industrials': 'XLI',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Materials': 'XLB'
        }
        
        # Merge GICS Sector
        df = pd.merge(df, df_meta[['Symbol', 'GICS Sector']], left_on='ticker', right_on='Symbol', how='left')
        
        # Map to ETF
        df['ticker'] = df['GICS Sector'].map(sector_map)
        
        # Drop rows where mapping failed or sector is unknown
        df = df.dropna(subset=['ticker'])

    summary = []
    for (ticker, month), group in df.groupby(['ticker', 'date']):
        buys = group[group['code'] == 'P']['value'].sum()
        sells = group[group['code'] == 'S']['value'].sum()
        unique_buyers = group[group['code'] == 'P']['owner'].nunique()
        
        # Net intensity: 1 (pure buying), -1 (pure selling)
        net_intensity = (buys - sells) / (buys + sells + 1e-6)
        
        summary.append({
            'ticker': ticker,
            'date': month,
            'insider_net_intensity': net_intensity,
            'insider_conviction_count': unique_buyers
        })
        
    return pd.DataFrame(summary)

def main():
    config = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "data" / "raw"
    processed_dir = repo_root / "data" / "processed"
    
    # 1. Process OHLCV
    ohlcv_path = raw_dir / "sectors_ohlcv.parquet"
    if ohlcv_path.exists():
        df_ohlcv = pd.read_parquet(ohlcv_path)
        monthly_features = engineer_quant_features(df_ohlcv, config['features']['quant_windows'])
        monthly_features.to_parquet(processed_dir / "engineered_features_monthly.parquet", index=False)
        logging.info("Standard quant features engineered.")
    
    # 2. Process Options
    options_path = raw_dir / "options_signals.parquet"
    if options_path.exists():
        df_opt = pd.read_parquet(options_path)
        opt_features = engineer_options_features(df_opt)
        opt_features.to_parquet(processed_dir / "options_features.parquet", index=False)
        logging.info("Options skew features engineered.")

    # 3. Process Transcripts
    trans_path = raw_dir / "transcripts.parquet"
    if trans_path.exists():
        df_trans = pd.read_parquet(trans_path)
        trans_features = engineer_transcript_features(df_trans)
        trans_features.to_parquet(processed_dir / "transcript_features.parquet", index=False)
        logging.info("Transcript sentiment features engineered.")

    # 4. Process Labor Trends and Macro Regimes
    macro_path = raw_dir / "macro_data.parquet"
    if macro_path.exists():
        df_macro = pd.read_parquet(macro_path)
        labor_features = engineer_labor_features(df_macro)
        labor_features.to_parquet(processed_dir / "labor_features.parquet", index=False)
        logging.info("Labor market demand features engineered.")
        
        regime_features = engineer_macro_regimes(df_macro)
        regime_features.to_parquet(processed_dir / "regime_features.parquet", index=False)
        logging.info("Macro regime switching features engineered.")

    # 5. Process Insider Trading
    insider_raw_path = raw_dir / "insider_transactions.parquet"
    meta_path = raw_dir / "sp500_metadata.csv"
    
    if insider_raw_path.exists():
        df_insider = pd.read_parquet(insider_raw_path)
        df_meta = pd.read_csv(meta_path) if meta_path.exists() else None
        
        insider_features = engineer_insider_features(df_insider, df_meta)
        insider_features.to_parquet(processed_dir / "insider_features.parquet", index=False)
        logging.info("Insider trading features engineered (aggregated to Sector ETFs).")

if __name__ == "__main__":
    main()
