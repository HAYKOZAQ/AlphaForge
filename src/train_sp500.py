import json
import logging
import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score


import numpy as np
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    config = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    proc_dir = repo_root / "data" / "processed"
    models_dir = repo_root / "models"
    reports_dir = repo_root / "reports"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    data_path = proc_dir / "master_firm_dataset.parquet"
    if not data_path.exists():
        logging.error("Missing master firm dataset. Run target_builder_sp500.py first.")
        return

    df = pd.read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # Global Seed Management
    np.random.seed(config['seed'])

    metadata = [
        "date", "ticker", "open", "high", "low", "close", "adj close", 
        "volume", "returns", "fwd_return_20d", "target_class",
    ]
    features = [c for c in df.columns if c not in metadata]

    logging.info("Firm-level feature count: %s", len(features))

    # Static 3-Way Split for Firm-Level Proof-of-Concept
    val_date = pd.to_datetime(config['dates']['firm_val_start'])
    test_date = pd.to_datetime(config['dates']['firm_test_start'])

    train_df = df[df["date"] < val_date]
    val_df = df[(df["date"] >= val_date) & (df["date"] < test_date)]
    test_df = df[df["date"] >= test_date]

    if train_df.empty or val_df.empty or test_df.empty:
        logging.error("Insufficient data for train/validation/test splits.")
        return

    X_train, y_train = train_df[features], train_df["target_class"]
    X_val, y_val = val_df[features], val_df["target_class"]
    X_test, y_test = test_df[features], test_df["target_class"]

    logging.info(
        "Training firm-level proof-of-concept model... train=%s, val=%s, test=%s",
        len(X_train), len(X_val), len(X_test)
    )

    clf = xgb.XGBClassifier(
        **config['modeling']['xgb_params'],
        random_state=config['seed']
    )

    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs) if len(y_test.unique()) > 1 else None
    
    logging.info("Firm out-of-sample accuracy: %.4f", accuracy)
    
    out_dict = {"model": clf, "features": features}
    out_model = models_dir / "firm_xgboost_core.pkl"
    with open(out_model, "wb") as f:
        pickle.dump(out_dict, f)
    logging.info("Saved firm-level model to %s", out_model)

    metrics = {
        "auc": float(auc) if auc is not None else None,
        "accuracy": float(accuracy),
        "test_start": str(test_df["date"].min().date())
    }

    metrics_path = reports_dir / "firm_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Saved firm metrics to %s", metrics_path)


if __name__ == "__main__":
    main()