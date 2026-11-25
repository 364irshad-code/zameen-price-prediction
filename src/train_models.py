"""PERSON B - Train, tune, evaluate and compare 4 models. Saves the best one.

Models : Ridge Regression (baseline) | Random Forest | Gradient Boosting (XGBoost) | Neural Network (MLP)
Target : log(1 + price). Metrics are reported after converting back to PKR.

Run:  python -m src.train_models            (full tuning)
      python -m src.train_models --fast     (quick test)
"""
from __future__ import annotations
import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src import config
from src.preprocessing import split_data

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:                       # xgboost not installed -> sklearn fallback
    HAS_XGB = False

sns.set_theme(style="whitegrid")


# ------------------------------------------------------------- pipeline
def make_preprocessor():
    """Scaling + one-hot encoding + target encoding, all inside the pipeline
    (so test data never leaks into training)."""
    return ColumnTransformer([
        ("num", StandardScaler(), config.NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CAT_FEATURES),
        ("loc", TargetEncoder(target_type="continuous", smooth="auto"), config.HIGH_CARD_FEATURES),
    ])


def get_models():
    rs = config.RANDOM_STATE
    models = {
        "Ridge Regression": (Ridge(), {"model__alpha": [0.01, 0.1, 1, 10, 100]}),
        "Random Forest": (
            RandomForestRegressor(n_jobs=-1, random_state=rs),
            {"model__n_estimators": [150, 250], "model__max_depth": [12, 20, None],
             "model__min_samples_leaf": [1, 2, 4], "model__max_features": [0.6, 0.8, 1.0]}),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            XGBRegressor(tree_method="hist", n_jobs=-1, random_state=rs),
            {"model__n_estimators": [300, 500, 800], "model__learning_rate": [0.03, 0.05, 0.1],
             "model__max_depth": [4, 6, 8], "model__subsample": [0.7, 0.9, 1.0],
             "model__colsample_bytree": [0.7, 0.9, 1.0]})
    else:
        models["Gradient Boosting"] = (
            HistGradientBoostingRegressor(random_state=rs),
            {"model__max_iter": [200, 400, 600], "model__learning_rate": [0.03, 0.05, 0.1],
             "model__max_depth": [None, 6, 10], "model__l2_regularization": [0.0, 0.1, 1.0]})
    models["Neural Network (MLP)"] = (
        MLPRegressor(early_stopping=True, max_iter=300, random_state=rs),
        {"model__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
         "model__alpha": [1e-4, 1e-3, 1e-2], "model__learning_rate_init": [1e-3, 3e-3]})
    return models


# -------------------------------------------------------------- metrics
def evaluate(y_true_log, pred_log) -> dict:
    y, p = np.expm1(y_true_log), np.expm1(pred_log)
    ape = np.abs(y - p) / y * 100
    return {
        "MAE (PKR)": mean_absolute_error(y, p),
        "RMSE (PKR)": float(np.sqrt(mean_squared_error(y, p))),
        "R2 (PKR)": r2_score(y, p),
        "R2 (log)": r2_score(y_true_log, pred_log),
        "RMSE (log)": float(np.sqrt(mean_squared_error(y_true_log, pred_log))),
        "MedAPE (%)": float(np.median(ape)),
        "MAPE (%)": float(np.mean(ape)),
    }


# ----------------------------------------------------------------- main
def main(n_iter: int, tune_rows: int, purpose: str = "sale"):
    out = config.paths(purpose)
    tag = out["tag"]
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out["clean_csv"])
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Purpose: {purpose.upper()}  Train: {len(X_train):,}  Test: {len(X_test):,}  XGBoost available: {HAS_XGB}\n")

    tune_idx = X_train.sample(min(tune_rows, len(X_train)), random_state=config.RANDOM_STATE).index
    results, fitted, preds = [], {}, {}

    for name, (estimator, grid) in get_models().items():
        print(f"=== {name}")
        pipe = Pipeline([("prep", make_preprocessor()), ("model", estimator)])
        n = min(n_iter, int(np.prod([len(v) for v in grid.values()])))
        search = RandomizedSearchCV(pipe, grid, n_iter=n, cv=3, scoring="neg_root_mean_squared_error",
                                    random_state=config.RANDOM_STATE, n_jobs=1)
        search.fit(X_train.loc[tune_idx], y_train.loc[tune_idx])        # tune on a sample (fast)
        best = clone(pipe).set_params(**search.best_params_)
        start = time.time()
        best.fit(X_train, y_train)                                       # refit on ALL training data
        fit_time = time.time() - start
        pred = best.predict(X_test)
        row = {"Model": name, **evaluate(y_test, pred),
               "CV RMSE (log)": -search.best_score_, "Train time (s)": fit_time}
        results.append(row); fitted[name] = best; preds[name] = pred
        print(f"best params: {search.best_params_}")
        print({k: round(v, 4) for k, v in row.items() if k != 'Model'}, "\n")

    table = pd.DataFrame(results).set_index("Model").sort_values("RMSE (log)")
    table.to_csv(config.REPORTS_DIR / (tag + "model_comparison.csv"))
    best_name = table.index[0]
    best_model, best_pred = fitted[best_name], preds[best_name]
    print("MODEL COMPARISON (sorted best -> worst)\n", table.round(3).to_string())
    print(f"\nBEST MODEL: {best_name}")

    # ---- figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, col, color in zip(ax, ["R2 (log)", "MedAPE (%)", "RMSE (PKR)"], ["#4c72b0", "#dd8452", "#55a868"]):
        sns.barplot(x=table.index, y=table[col], ax=a, color=color)
        a.set_title(col); a.set_xlabel(""); a.tick_params(axis="x", rotation=20)
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / (tag + "10_model_comparison.png"), dpi=130); plt.close()

    idx = np.random.default_rng(1).choice(len(y_test), min(3000, len(y_test)), replace=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(np.expm1(y_test.values[idx]) / 1e6, np.expm1(best_pred[idx]) / 1e6, s=6, alpha=0.4)
    lim = [0.5, np.expm1(y_test.values[idx]).max() / 1e6]
    ax[0].plot(lim, lim, "r--"); ax[0].set(xscale="log", yscale="log", xlabel="Actual (million PKR)",
                                            ylabel="Predicted (million PKR)", title=f"{best_name}: actual vs predicted")
    resid = y_test.values - best_pred
    sns.histplot(resid, bins=60, ax=ax[1], color="#8172b2")
    ax[1].set(title="Residuals (log scale) - centred at 0 = unbiased", xlabel="actual - predicted")
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / (tag + "11_best_model_diagnostics.png"), dpi=130); plt.close()

    # ---- permutation importance (which inputs matter most)
    sub = X_test.sample(min(3000, len(X_test)), random_state=1)
    imp = permutation_importance(best_model, sub, y_test.loc[sub.index], n_repeats=5,
                                 random_state=1, scoring="neg_root_mean_squared_error", n_jobs=1)
    importance = pd.Series(imp.importances_mean, index=config.FEATURES).sort_values()
    plt.figure(figsize=(7, 5)); importance.plot.barh(color="#4c72b0")
    plt.title(f"Feature importance ({best_name})"); plt.xlabel("increase in RMSE when feature is shuffled")
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / (tag + "12_feature_importance.png"), dpi=130); plt.close()
    importance.sort_values(ascending=False).to_csv(config.REPORTS_DIR / (tag + "feature_importance.csv"), header=["importance"])

    # ---- error by city
    test_city = df.loc[X_test.index, "city"]
    ape = np.abs(np.expm1(y_test) - np.expm1(best_pred)) / np.expm1(y_test) * 100
    by_city = ape.groupby(test_city).median().sort_values()
    by_city.to_csv(config.REPORTS_DIR / (tag + "error_by_city.csv"), header=["MedAPE (%)"])
    print("\nMedian error by city (%):\n", by_city.round(1).to_string())

    # ---- save artefacts for the web app
    lookup = (df.groupby(["city", "location", "locality"])
                .agg(latitude=("latitude", "median"), longitude=("longitude", "median"),
                     listings=("price", "size"), median_psf=("price_per_sqft", "median"))
                .reset_index())
    lookup = lookup[lookup["listings"] >= 5].sort_values(["city", "location"])
    lookup.to_csv(out["lookup"], index=False)
    joblib.dump(best_model, out["model"])
    meta = {
        "best_model": best_name,
        "metrics": {k: float(v) for k, v in table.loc[best_name].items()},
        "residual_q10": float(np.quantile(resid, 0.10)),
        "residual_q90": float(np.quantile(resid, 0.90)),
        "property_types": sorted(df["property_type"].unique().tolist()),
        "city_centers": json.loads(out["centers"].read_text()),
        "trained_on_rows": int(len(X_train)),
        "xgboost_used": HAS_XGB,
        "purpose": purpose,
    }
    (out["meta"]).write_text(json.dumps(meta, indent=2))
    print(f"\nSaved model + metadata in {config.MODELS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="quick run for testing")
    parser.add_argument("--n-iter", type=int, default=8, help="random-search trials per model")
    parser.add_argument("--tune-rows", type=int, default=25000, help="rows used while tuning")
    parser.add_argument("--purpose", choices=["sale", "rent"], default="sale")
    a = parser.parse_args()
    main(3 if a.fast else a.n_iter, 6000 if a.fast else a.tune_rows, a.purpose)
