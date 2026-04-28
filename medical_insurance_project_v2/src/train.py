"""
train.py
--------
Trains and compares multiple regressors with cross-validation, logs to
MLflow, registers the best model in our local registry.

Run:  python -m src.train --dataset medical_insurance.csv
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
from .paths import MODELS_DIR, ROOT
from .data_loader import load_dataset
from .preprocessing import build_preprocessor, split_xy

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def candidate_models() -> dict:
    models = {
        "linear":     LinearRegression(),
        "ridge":      Ridge(alpha=1.0, random_state=42),
        "lasso":      Lasso(alpha=0.1, random_state=42),
        "rf":         RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "gbr":        GradientBoostingRegressor(n_estimators=300, random_state=42),
    }
    if HAS_XGB:
        models["xgb"] = XGBRegressor(n_estimators=400, learning_rate=0.05,
                                     max_depth=4, random_state=42, n_jobs=-1)
    return models


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "r2":   r2_score(y_test, pred),
        "mae":  mean_absolute_error(y_test, pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }


def train_all(dataset: str | None = None, register_best: bool = True):
    from . import registry
    df = load_dataset(dataset)
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = build_preprocessor()

    mlflow.set_tracking_uri(f"file:{ROOT/'mlruns'}")
    mlflow.set_experiment("medical_insurance")

    rows = []
    best = {"r2": -np.inf}
    for name, est in candidate_models().items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        with mlflow.start_run(run_name=name):
            cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="r2", n_jobs=-1)
            pipe.fit(X_tr, y_tr)
            metrics = evaluate(pipe, X_te, y_te)
            metrics["cv_r2_mean"] = float(cv.mean())
            mlflow.log_params({"algo": name, **est.get_params()})
            mlflow.log_metrics(metrics)
            row = {"model": name, **metrics}; rows.append(row)
            print(f"  {name:8s} -> R2={metrics['r2']:.4f}  MAE={metrics['mae']:.0f}")
            if metrics["r2"] > best["r2"]:
                best = {**metrics, "name": name, "pipe": pipe, "params": est.get_params()}

    comp = pd.DataFrame(rows).sort_values("r2", ascending=False)
    comp.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    print("\nBest model:", best["name"], "R2=", best["r2"])

    if register_best:
        card = registry.register(
            best["pipe"], name=best["name"], algorithm=best["name"],
            metrics={k: best[k] for k in ("r2", "mae", "rmse", "cv_r2_mean")},
            params=best["params"], feature_count=X.shape[1],
            rows=len(df), dataset=dataset or "medical_insurance.csv",
        )
        print(f"Registered as {card.name} v{card.version}")
    return comp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()
    train_all(args.dataset)
