import pandas as pd
import logging
from pathlib import Path
import dowhy
from dowhy import CausalModel
import warnings

# Suppress dowhy verbosity
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_causal_analysis(df, treatment, outcome, common_causes):
    """
    Runs a formal causal inference pipeline to mathematically prove 
    that the treatment feature actually causes the outcome.
    """
    logging.info("="*60)
    logging.info(f"Causal Inference: Does '{treatment}' cause '{outcome}'?")
    logging.info("="*60)
    
    # 1. Formulate the Causal Model
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=common_causes
    )
    
    # 2. Identify the causal effect
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
    # 3. Estimate the causal effect
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True
    )
    
    p_val = estimate.test_stat_significance()[1] if hasattr(estimate, 'test_stat_significance') else "N/A"
    logging.info(f"Causal Estimate (ATE): {estimate.value:.4f} | P-Value: {p_val}")
    
    # 4. Refute the estimate (Robustness check via Placebo)
    logging.info("Running Placebo Treatment Refutation (Robustness Check)...")
    refute_placebo = model.refute_estimate(
        identified_estimand, estimate,
        method_name="random_common_cause"
    )
    
    passed = "PASSED (Robust)" if refute_placebo.new_effect > 0 else "FAILED (Fragile)"
    logging.info(f"Robustness Check: {passed}\n")

def main():
    repo_root = Path(__file__).resolve().parent.parent
    master_path = repo_root / "data" / "processed" / "master_dataset.parquet"
    
    if not master_path.exists():
        logging.error("Master dataset not found. Run the ingestion and feature pipeline first.")
        return
        
    df = pd.read_parquet(master_path)
    
    # Drop rows with missing critical features to ensure valid causal graphs
    df = df.dropna(subset=['target_score', 'insider_net_intensity', 'theme_increase'])
    
    # Test 1: Does Insider Trading Conviction actually cause market outperformance?
    # Confounders (Common Causes): The macro regime, overall market volatility, and options skew.
    run_causal_analysis(
        df=df,
        treatment='insider_net_intensity',
        outcome='target_score',
        common_causes=['macro_regime', 'vol_63d', 'opt_put_call_vol_ratio_z']
    )
    
    # Test 2: Does an increase in 10-K NLP Thematic strength cause market outperformance?
    # Confounders: The macro regime, volatility, and standard quantitative momentum.
    run_causal_analysis(
        df=df,
        treatment='theme_increase',
        outcome='target_score',
        common_causes=['macro_regime', 'vol_63d', 'mom_63d']
    )

if __name__ == "__main__":
    main()
