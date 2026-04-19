import pandas as pd
import numpy as np
from pathlib import Path
import logging
from src.utils.validators import data_sanity_decorator, TargetSchema
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_master_dataset(df_feat, df_macro, df_nlp, df_opt, df_trans, df_labor, df_insider, df_regime=None):
    # Align dates exactly to month start/end for consistency
    df_feat['date'] = df_feat['date'].dt.to_period('M').dt.to_timestamp('M')

    # 1. Merge Macro
    if not df_macro.empty:
        df_macro['date'] = pd.to_datetime(df_macro['date']).dt.to_period('M').dt.to_timestamp('M')
        df_feat = pd.merge_asof(
            df_feat.sort_values('date'),
            df_macro.sort_values('date'),
            on='date', direction='backward'
        )

    # 7. Merge Macro Regimes
    if df_regime is not None and not df_regime.empty:
        df_regime['date'] = pd.to_datetime(df_regime['date']).dt.to_period('M').dt.to_timestamp('M')
        df_regime = df_regime.groupby('date').last().reset_index()
        df_feat = pd.merge(df_feat, df_regime, on='date', how='left')
        for col in ['macro_regime', 'regime_prob_0', 'regime_prob_1']:
            if col in df_feat.columns:
                df_feat[col] = df_feat[col].ffill().fillna(0)    
    # 2. Merge Sector NLP (Themes)
    if not df_nlp.empty:
        df_nlp['date'] = df_nlp['date'].dt.to_period('M').dt.to_timestamp('M')
        df_feat = pd.merge(df_feat, df_nlp, on=['ticker', 'date'], how='left')
        df_feat.fillna(0, inplace=True)
        
        df_feat['theme_strength'] = df_feat['ai_adoption'] + df_feat['policy_support'] + df_feat['capex_intent']
        df_feat['theme_increase'] = df_feat.groupby('ticker')['theme_strength'].diff().fillna(0)
    else:
        df_feat['theme_increase'] = 0.0

    # 3. Merge Options Skew
    if not df_opt.empty:
        df_opt['date'] = pd.to_datetime(df_opt['date']).dt.to_period('M').dt.to_timestamp('M')
        df_feat = pd.merge(df_feat, df_opt, on=['ticker', 'date'], how='left')

    # 4. Merge Transcript Sentiment
    if not df_trans.empty:
        df_trans['date'] = pd.to_datetime(df_trans['date']).dt.to_period('M').dt.to_timestamp('M')
        df_feat = pd.merge(df_feat, df_trans, on=['ticker', 'date'], how='left')

    # 5. Merge Labor Market Demand
    if df_labor is not None and not df_labor.empty:
        df_labor['date'] = pd.to_datetime(df_labor['date']).dt.to_period('M').dt.to_timestamp('M')
        
        # Resample to monthly to avoid join explosion with daily macro/labor data
        df_labor = df_labor.groupby('date').last().reset_index()
        
        # Sector to Labor Column Mapping
        mapping = {
            'XLK': 'labor_rel_strength_job_openings_info',
            'XLC': 'labor_rel_strength_job_openings_info',
            'XLF': 'labor_rel_strength_job_openings_fin',
            'XLRE': 'labor_rel_strength_job_openings_fin',
            'XLI': 'labor_rel_strength_job_openings_prof_services',
            'XLV': 'labor_rel_strength_job_openings_health',
            'XLB': 'labor_rel_strength_job_openings_mfg',
            'XLY': 'labor_rel_strength_job_openings_retail',
            'XLP': 'labor_rel_strength_job_openings_retail'
        }
        
        # Create a tidy long-form labor dataframe for merging
        labor_melted = []
        for ticker, col in mapping.items():
            if col in df_labor.columns:
                temp = df_labor[['date', col]].copy()
                temp['ticker'] = ticker
                temp = temp.rename(columns={col: 'labor_hiring_momentum'})
                labor_melted.append(temp)
        
        if labor_melted:
            df_labor_final = pd.concat(labor_melted)
            df_feat = pd.merge(df_feat, df_labor_final, on=['ticker', 'date'], how='left')

    # 6. Merge Insider Conviction
    if df_insider is not None and not df_insider.empty:
        df_insider['date'] = pd.to_datetime(df_insider['date']).dt.to_period('M').dt.to_timestamp('M')
        # We merge directly on ticker/date for the firms that have insider data
        df_feat = pd.merge(df_feat, df_insider, on=['ticker', 'date'], how='left')

    # Improved Imputation: Forward-fill lagging indicators (Labor/Insider) before zero-filling
    # JOLTS and Insider data often lag by 1-2 months.
    df_feat = df_feat.sort_values(['ticker', 'date'])
    for col in ['labor_hiring_momentum', 'insider_net_intensity']:
        if col in df_feat.columns:
            df_feat[col] = df_feat.groupby('ticker')[col].ffill(limit=2)

    # Global Imputation for Alt Data
    alt_cols = [c for c in df_feat.columns if any(p in c for p in ['opt_', 'transcript_', 'labor_', 'insider_'])]
    df_feat[alt_cols] = df_feat[alt_cols].fillna(0)
        
    return df_feat

@data_sanity_decorator(TargetSchema, "Target Generation")
def compute_targets(df):
    config = load_config()
    t_cfg = config['targets']
    h = t_cfg['horizon_months']
    
    df = df.sort_values(['ticker', 'date'])
    
    # Targets derived from forward price action
    df['fwd_ret'] = df.groupby('ticker')['close'].shift(-h) / df['close'] - 1
    df['cross_mean_ret'] = df.groupby('date')['fwd_ret'].transform('mean')
    df['rel_ret'] = df['fwd_ret'] - df['cross_mean_ret']
    
    df['fwd_vol_z'] = df.groupby('ticker')['volume_zscore'].shift(-h)
    df['turnover_improvement'] = df['fwd_vol_z'] - df['volume_zscore']
    
    # Target 3: volatility-adjusted momentum
    df['vol_adj_mom'] = df['mom_63d'] / (df['vol_63d'] + 1e-6)
    
    w = t_cfg['weights']
    df['target_score'] = (
        w['rel_ret'] * df['rel_ret'].fillna(0) +
        w['turnover'] * df['turnover_improvement'].fillna(0) +
        w['momentum'] * df['vol_adj_mom'].fillna(0) +
        w['themes'] * df['theme_increase'].fillna(0)
    )
    
    def top_n_pct(series):
        return (series.rank(pct=True, ascending=False) <= t_cfg['top_n_percentile']).astype(int)
        
    df['target_class'] = df.groupby('date')['target_score'].transform(top_n_pct)
    df.loc[df['fwd_ret'].isna(), ['target_score', 'target_class']] = np.nan
    
    return df

def main():
    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "data" / "raw"
    processed_dir = repo_root / "data" / "processed"
    
    features_path = processed_dir / "engineered_features_monthly.parquet"
    macro_path = raw_dir / "macro_data.parquet"
    nlp_path = processed_dir / "sector_nlp_monthly.parquet"
    opt_path = processed_dir / "options_features.parquet"
    trans_path = processed_dir / "transcript_features.parquet"
    labor_path = processed_dir / "labor_features.parquet"
    insider_path = processed_dir / "insider_features.parquet"
    regime_path = processed_dir / "regime_features.parquet"
    out_path = processed_dir / "master_dataset.parquet"

    try:
        df_feat = pd.read_parquet(features_path)
    except:
        logging.error("Engineered features missing.")
        return

    df_macro = pd.read_parquet(macro_path) if macro_path.exists() else pd.DataFrame()
    df_nlp = pd.read_parquet(nlp_path) if nlp_path.exists() else pd.DataFrame()
    df_opt = pd.read_parquet(opt_path) if opt_path.exists() else pd.DataFrame()
    df_trans = pd.read_parquet(trans_path) if trans_path.exists() else pd.DataFrame()
    df_labor = pd.read_parquet(labor_path) if labor_path.exists() else pd.DataFrame()
    df_insider = pd.read_parquet(insider_path) if insider_path.exists() else pd.DataFrame()     
    df_regime = pd.read_parquet(regime_path) if regime_path.exists() else pd.DataFrame()

    master_df = build_master_dataset(df_feat, df_macro, df_nlp, df_opt, df_trans, df_labor, df_insider, df_regime)
    master_df = compute_targets(master_df)
    
    master_df.to_parquet(out_path, index=False)
    logging.info(f"Saved master dataset to {out_path} (Shape: {master_df.shape})")

if __name__ == "__main__":
    main()
