"""Training pipeline for Second Bell.

The MVP uses interpretable, lightweight ML rather than a heavyweight cloud model:
- RandomForestClassifier for high ghost-component risk.
- GradientBoostingRegressor with quantile loss for unopened-return intervals.
- RandomForestRegressor for after-school snack demand.
- IsolationForest for anomaly flags.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path
import sys
from typing import Dict, List
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES: List[str] = [
    "day_of_week",
    "lunch_period",
    "menu_item",
    "item_category",
    "component_type",
    "sealed_or_whole",
    "cold_chain_required",
    "planned_count",
    "staged_count",
    "expected_attendance",
    "expected_lunch_period_attendance",
    "weather_tag",
    "event_tag",
    "entree",
    "entree_popularity_score",
    "line_position",
    "share_table_monitor",
    "cooler_capacity",
    "afterschool_activity_count",
    "kg_per_item",
    "co2e_kg_per_item",
    "cost_per_item",
]

CATEGORICAL = [
    "day_of_week",
    "menu_item",
    "item_category",
    "component_type",
    "weather_tag",
    "event_tag",
    "entree",
    "line_position",
]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
MODEL_FILES = [
    "return_q10.joblib",
    "return_q50.joblib",
    "return_q90.joblib",
    "risk_classifier.joblib",
    "afterschool_model.joblib",
    "anomaly_preprocessor.joblib",
    "anomaly_model.joblib",
    "conformal_padding.joblib",
]


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ],
        remainder="drop",
    )


def _quantile_model(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("model", GradientBoostingRegressor(
                loss="quantile",
                alpha=alpha,
                n_estimators=240,
                learning_rate=0.04,
                max_depth=3,
                random_state=42,
            )),
        ]
    )


def train_all(data_path: Path, model_dir: Path) -> Dict[str, float]:
    df = pd.read_csv(data_path)
    X = df[FEATURES]
    y_return = df["unopened_return_count"]
    y_risk = df["high_ghost_risk"]
    y_after = df["afterschool_pickup_count"]

    X_train, X_test, yr_train, yr_test = train_test_split(X, y_return, test_size=0.22, shuffle=False)
    _, _, yrisk_train, yrisk_test = train_test_split(X, y_risk, test_size=0.22, shuffle=False)
    _, _, ya_train, ya_test = train_test_split(X, y_after, test_size=0.22, shuffle=False)

    models = {
        "return_q10": _quantile_model(0.10),
        "return_q50": _quantile_model(0.50),
        "return_q90": _quantile_model(0.90),
        "risk_classifier": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=260,
                max_depth=9,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=42,
            )),
        ]),
        "afterschool_model": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=220,
                max_depth=9,
                min_samples_leaf=3,
                random_state=42,
            )),
        ]),
    }

    for name in ["return_q10", "return_q50", "return_q90"]:
        models[name].fit(X_train, yr_train)
    models["risk_classifier"].fit(X_train, yrisk_train)
    models["afterschool_model"].fit(X_train, ya_train)

    # IsolationForest uses transformed features from the same preprocessor.
    anomaly_preprocessor = make_preprocessor()
    Xt = anomaly_preprocessor.fit_transform(X_train)
    anomaly_model = IsolationForest(n_estimators=180, contamination=0.06, random_state=42)
    anomaly_model.fit(Xt)
    models["anomaly_preprocessor"] = anomaly_preprocessor
    models["anomaly_model"] = anomaly_model

    # Simple conformal padding based on median model residuals.
    q50_pred = models["return_q50"].predict(X_test)
    residual_padding = float(np.quantile(np.abs(yr_test - q50_pred), 0.80))
    models["conformal_padding"] = residual_padding

    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, model_dir / f"{name}.joblib")

    risk_pred = models["risk_classifier"].predict(X_test)
    risk_proba = models["risk_classifier"].predict_proba(X_test)[:, 1]
    after_pred = models["afterschool_model"].predict(X_test)
    metrics = {
        "return_mae": round(float(mean_absolute_error(yr_test, q50_pred)), 3),
        "afterschool_mae": round(float(mean_absolute_error(ya_test, after_pred)), 3),
        "risk_accuracy": round(float(accuracy_score(yrisk_test, risk_pred)), 3),
        "risk_auc": round(float(roc_auc_score(yrisk_test, risk_proba)), 3) if len(set(yrisk_test)) > 1 else np.nan,
        "conformal_padding": round(residual_padding, 3),
        "training_rows": int(len(df)),
        "training_school_days": int(df["date"].nunique()) if "date" in df else None,
    }
    pd.Series(metrics).to_json(model_dir / "metrics.json", indent=2)
    manifest = {
        "schema_version": 1,
        "project": "Second Bell",
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": {
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "features": FEATURES,
        "categorical_features": CATEGORICAL,
        "numeric_features": NUMERIC,
        "model_files": MODEL_FILES,
        "training_rows": int(len(df)),
        "training_school_days": int(df["date"].nunique()) if "date" in df else None,
    }
    (model_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "second_bell_synthetic_cafeteria.csv"
    model_dir = root / "models"
    metrics = train_all(data_path, model_dir)
    print(metrics)


if __name__ == "__main__":
    main()
