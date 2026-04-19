import pandas as pd
import numpy as np
from pathlib import Path
import logging
import xgboost as xgb
import pickle
import shap
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PurgedKFoldCV:
    """
    Implements Purged K-Fold Cross-Validation (Lopez de Prado).
    Minimizes temporal leakage by purging train samples that overlap with test, 
    and embargoing train samples immediately following the test set.
    """
    def __init__(self, df, min_train_months=None, n_splits=5, purge_months=1, embargo_months=1):
        self.df = df.sort_values('date').copy()
        self.n_splits = n_splits
        self.purge_months = purge_months
        self.embargo_months = embargo_months
        self.unique_months = sorted(self.df['date'].dt.to_period('M').unique())
        
    def split(self):
        """Yields train and test indices based on purged k-fold."""
        fold_size = len(self.unique_months) // self.n_splits
        
        for i in range(self.n_splits):
            test_start_idx = i * fold_size
            test_end_idx = (i + 1) * fold_size if i < self.n_splits - 1 else len(self.unique_months)
            
            test_months = self.unique_months[test_start_idx:test_end_idx]
            if not test_months:
                continue
                
            train_months = []
            for m in self.unique_months:
                if m < test_months[0] - self.purge_months:
                    train_months.append(m)
                elif m > test_months[-1] + self.purge_months + self.embargo_months:
                    train_months.append(m)
                    
            train_idx = self.df[self.df['date'].dt.to_period('M').isin(train_months)].index
            test_idx = self.df[self.df['date'].dt.to_period('M').isin(test_months)].index
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
                
            yield train_idx, test_idx

def define_feature_sets():
    base_quant = ['vol_21d', 'vol_63d', 'mom_21d', 'mom_63d', 'mom_126d', 'volume_zscore']
    macro = ['fed_funds_rate', 'cpi', 'industrial_production', 'unemployment_rate', 'wti_oil']
    nlp = ['ai_adoption', 'policy_support', 'capex_intent', 'uncertainty', 'sentiment', 'news_volume', 'theme_increase']
    
    # New Alt Data & Regime Features (Forward Looking)
    alt_data = [
        'opt_put_call_oi_ratio_z', 'opt_put_call_vol_ratio_z', 
        'transcript_sentiment', 'labor_hiring_momentum',
        'insider_net_intensity', 'insider_conviction_count',
        'macro_regime', 'regime_prob_1'
    ]
    
    return {
        'model_A': base_quant,
        'model_B': base_quant + macro,
        'model_C': base_quant + macro + nlp + alt_data
    }

def run_walk_forward_backtest(df, features_dict, config):
    df_valid = df.dropna(subset=['target_class']).reset_index(drop=True)
    
    # Global Seed Management
    np.random.seed(config['seed'])
    
    cv = PurgedKFoldCV(
        df_valid,
        n_splits=5,
        purge_months=1,
        embargo_months=1
    )    
    xgb_params = config['modeling']['xgb_params']
    results = {name: [] for name in features_dict.keys()}
    final_models = {}
    
    logging.info("Starting Rolling Walk-Forward Backtest...")
    
    for train_idx, test_idx in cv.split():
        train_df, test_df = df_valid.loc[train_idx], df_valid.loc[test_idx]
        test_date = test_df['date'].iloc[0]
        
        for name, features in features_dict.items():
            avail_features = [f for f in features if f in train_df.columns]
            
            X_train, y_train = train_df[avail_features], train_df['target_class']
            X_test, y_test = test_df[avail_features], test_df['target_class']
            
            # Use a slightly smaller validation set (prior to test)
            v_size = config['modeling']['walk_forward']['val_size']
            val_months = train_df['date'].dt.to_period('M').unique()[-v_size:]
            val_idx = train_df[train_df['date'].dt.to_period('M').isin(val_months)].index
            X_val, y_val = train_df.loc[val_idx, avail_features], train_df.loc[val_idx, 'target_class']
            
            clf = xgb.XGBClassifier(
                **xgb_params,
                random_state=config['seed']
            )
            
            clf.fit(
                X_train, y_train, 
                eval_set=[(X_val, y_val)], 
                verbose=False
            )
            
            probs = clf.predict_proba(X_test)[:, 1]
            preds = (probs > 0.5).astype(int)
            
            results[name].append({
                'date': test_date,
                'probs': probs,
                'actuals': y_test.values,
                'preds': preds
            })
            
            # Keep the most recent model for final persistence
            final_models[name] = {'model': clf, 'features': avail_features}
            
    return results, final_models

import mlflow
import mlflow.xgboost

def main():
    config = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = repo_root / "data" / "processed"
    models_dir = repo_root / "models"
    reports_dir = repo_root / "reports"
    
    for d in [models_dir, reports_dir / "metrics", reports_dir / "plots"]:
        d.mkdir(parents=True, exist_ok=True)
    
    master_df_path = processed_dir / "master_dataset.parquet"
    if not master_df_path.exists():
        logging.error("Master dataset missing.")
        return
        
    df = pd.read_parquet(master_df_path)
    features_dict = define_feature_sets()
    
    # --- MLflow Initialization ---
    mlflow.set_experiment("Sector_Intelligence_Research")
    
    with mlflow.start_run(run_name=f"WalkForward_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"):
        # Log Hyperparameters
        mlflow.log_params(config['modeling']['xgb_params'])
        mlflow.log_param("min_train_months", config['modeling']['walk_forward']['min_train_size'])
        mlflow.log_params({"seed": config['seed'], "prod_model": config['modeling']['production_model']})

        backtest_results, models = run_walk_forward_backtest(df, features_dict, config)
        
        # 1. Save Models and Log to MLflow
        for name, m_dict in models.items():
            model_path = models_dir / f"{name}.pkl"
            pickle.dump(m_dict, open(model_path, "wb"))
            # Optionally log model as artifact or mlflow model
            # mlflow.xgboost.log_model(m_dict['model'], artifact_path=f"models/{name}")
            
        # 2. Process and Save Detailed Metrics
        metrics_summary = {}
        for name, month_data in backtest_results.items():
            all_probs = np.concatenate([d['probs'] for d in month_data])
            all_actuals = np.concatenate([d['actuals'] for d in month_data])
            all_preds = np.concatenate([d['preds'] for d in month_data])
            
            metrics = {
                'auc': float(roc_auc_score(all_actuals, all_probs)),
                'accuracy': float(accuracy_score(all_actuals, all_preds)),
                'f1': float(f1_score(all_actuals, all_preds)),
                'precision': float(precision_score(all_actuals, all_preds)),
                'recall': float(recall_score(all_actuals, all_preds)),
                'conf_matrix': confusion_matrix(all_actuals, all_preds).tolist()
            }
            metrics_summary[name] = metrics
            
            # Log metrics to MLflow
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items() if k != 'conf_matrix'})
            
        json.dump(metrics_summary, open(reports_dir / "metrics" / "walk_forward_metrics.json", "w"), indent=4)
        logging.info(f"Walk-forward metrics saved to {reports_dir / 'metrics'}")
        
        # 3. Generate SHAP explainability for primary model
        prod_model_name = config['modeling']['production_model']
        if prod_model_name in models:
            m_dict = models[prod_model_name]
            clf = m_dict['model']
            feats = m_dict['features']
            
            # Explain the most recent month of data
            last_month = df['date'].dt.to_period('M').max()
            X_latest = df[df['date'].dt.to_period('M') == last_month][feats]
            
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_latest)
            
            # Export raw SHAP data for Plotly visualization in dashboard
            shap_df = pd.DataFrame(shap_values, columns=feats)
            shap_df.to_parquet(reports_dir / "shap_values.parquet", index=False)
            X_latest.to_parquet(reports_dir / "shap_base_features.parquet", index=False)
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_latest, show=False)
            plt.tight_layout()
            shap_plot_path = reports_dir / "plots" / "shap_summary.png"
            plt.savefig(shap_plot_path)
            logging.info("SHAP Summary plot and raw data exported.")
            
            # Log plots to MLflow
            mlflow.log_artifact(str(shap_plot_path))
            
        # Log the metrics.json as well
        mlflow.log_artifact(str(reports_dir / "metrics" / "walk_forward_metrics.json"))

if __name__ == "__main__":
    main()
