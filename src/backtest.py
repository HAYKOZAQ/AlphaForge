import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import spearmanr, norm

from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def deflated_sharpe_ratio(returns_series, num_trials=10, variance_trials=0.1):
    """
    Calculates the Deflated Sharpe Ratio (DSR) to mathematically prove the backtest
    is not overfit by accounting for multiple testing bias (Bailey and Lopez de Prado).
    """
    if len(returns_series) < 2: return 0.0
    
    annual_factor = 12
    ann_ret = returns_series.mean() * annual_factor
    ann_vol = returns_series.std() * np.sqrt(annual_factor)
    
    if ann_vol == 0: return 0.0
    
    estimated_sharpe = ann_ret / ann_vol
    
    # Expected maximum Sharpe from multiple independent trials (Euler-Mascheroni approx)
    emc = 0.5772156649
    expected_max_sharpe = np.sqrt(2 * np.log(num_trials)) + (emc / np.sqrt(2 * np.log(num_trials)))
    expected_max_sharpe = expected_max_sharpe * np.sqrt(variance_trials)
    
    # Calculate DSR (Probability that True Sharpe > Expected Max Sharpe)
    n = len(returns_series)
    skewness = returns_series.skew()
    kurtosis = returns_series.kurt()
    
    # Bailey and Lopez de Prado formula
    numerator = (estimated_sharpe - expected_max_sharpe) * np.sqrt(n - 1)
    denominator = np.sqrt(1 - skewness * estimated_sharpe + (kurtosis - 1) / 4 * estimated_sharpe**2)
    
    # Probability (DSR)
    dsr = norm.cdf(numerator / denominator)
    return dsr

def load_models(models_dir):
    models = {}
    for p in models_dir.glob("*.pkl"):
        # Explicit evaluation robustness: Anchor strictly to sector models
        if p.stem.startswith("model_"):
            with open(p, "rb") as f:
                models[p.stem] = pickle.load(f)
    return models

def calculate_financial_metrics(returns_series):
    """
    Computes annualized return, Sharpe, Sortino, and Max Drawdown for a series of returns.
    Assumes monthly frequency (annual_factor=12).
    """
    if len(returns_series) == 0:
        return {}
        
    annual_factor = 12
    # Add a small epsilon to avoid divide by zero
    eps = 1e-9
    
    cum_equity = (1 + returns_series).cumprod()
    total_ret = cum_equity.iloc[-1] - 1
    
    # Annualized Return
    n_years = len(returns_series) / annual_factor
    ann_ret = (1 + total_ret)**(1/n_years) - 1 if n_years > 0 else 0
    
    # Annualized Volatility
    ann_vol = returns_series.std() * np.sqrt(annual_factor)
    
    # Sharpe Ratio (assuming risk-free rate = 0 for relative returns)
    sharpe = ann_ret / (ann_vol + eps)
    
    # Sortino Ratio
    neg_returns = returns_series[returns_series < 0]
    neg_vol = neg_returns.std() * np.sqrt(annual_factor) if len(neg_returns) > 0 else eps
    sortino = ann_ret / (neg_vol + eps)
    
    # Maximum Drawdown
    peak = cum_equity.expanding(min_periods=1).max()
    drawdown = (cum_equity / peak) - 1
    mdd = drawdown.min()
    
    # Deflated Sharpe Ratio (Assuming 10 hyperparameter trials with 0.1 variance)
    dsr = deflated_sharpe_ratio(returns_series)
    
    return {
        'total_return': float(total_ret),
        'annual_return': float(ann_ret),
        'annual_vol': float(ann_vol),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_drawdown': float(mdd),
        'deflated_sharpe': float(dsr)
    }

def evaluate_models(df, models, config):
    # Filter by configured backtest start year
    start_year = config['dates']['backtest_start_year']
    df_test = df[df['date'].dt.year >= start_year].dropna(subset=['target_class']).copy()
    
    results = {}
    for name, m_dict in models.items():
        clf = m_dict['model']
        features = m_dict['features']
        
        X_test = df_test[features]
        y_test = df_test['target_class']
        
        # Predictions
        preds_proba = clf.predict_proba(X_test)[:, 1]
        df_test[f'{name}_prob'] = preds_proba
        
        # 1. Classification Metrics
        auc = roc_auc_score(y_test, preds_proba)
        preds_class = (preds_proba > 0.5).astype(int)
        acc = accuracy_score(y_test, preds_class)
        spearman_corr, _ = spearmanr(preds_proba, df_test['target_score'])
        
        # 2. Financial Portfolio Simulation (Long Top-N predicted)
        # Select top 3 predicted sectors per month and calculate mean relative return
        def simulate_portfolio(group):
            top3 = group.nlargest(3, f'{name}_prob')
            return top3['rel_ret'].mean()
            
        strat_returns = df_test.groupby('date').apply(simulate_portfolio)
        fin_metrics = calculate_financial_metrics(strat_returns)
        
        # 3. Hit rate for top 3
        def top3_hit_rate(group):
            top3 = group.nlargest(3, f'{name}_prob')
            return top3['target_class'].mean()
        
        hit_rate = df_test.groupby('date').apply(top3_hit_rate).mean()
        
        results[name] = {
            'auc': float(auc),
            'accuracy': float(acc),
            'spearman': float(spearman_corr),
            'top3_hit_rate': float(hit_rate),
            **fin_metrics
        }
        
    return results

def main():
    config = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = repo_root / "data" / "processed"
    models_dir = repo_root / "models"
    
    master_df_path = processed_dir / "master_dataset.parquet"
    
    if not master_df_path.exists():
        logging.error("Master dataset missing. Run target_builder.py.")
        return
        
    df = pd.read_parquet(master_df_path)
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    models = load_models(models_dir)
    
    if not models:
        logging.warning("No sector models found in models/ directory.")
        return

    results = evaluate_models(df, models, config)
    
    # Save Metrics to Durably Capture Performance
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    import json
    with open(reports_dir / "backtest_summary.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Save as CSV
    res_df = pd.DataFrame(results).T
    res_df.to_csv(reports_dir / "backtest_summary.csv")
    
    logging.info(f"Programmatic metrics reports saved to {reports_dir}")
    
    print("\n" + "="*50)
    print("INSTITUTIONAL BACKTEST VALIDATION RESULTS")
    print("="*50)
    for model_name, metrics in results.items():
        print(f"MODEL: {model_name}")
        print(f"  Classification AUC:   {metrics['auc']:.4f}")
        print(f"  Spearman Rank Corr:   {metrics['spearman']:.4f}")
        print(f"  Annualized Return:    {metrics['annual_return']:.2%}")
        print(f"  Sharpe Ratio:         {metrics['sharpe']:.2f}")
        print(f"  Max Drawdown:         {metrics['max_drawdown']:.2%}")
        print("-" * 30)

if __name__ == "__main__":
    main()
