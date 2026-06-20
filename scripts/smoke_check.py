from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from scenario import build_plan
from simulate_data import generate_cafeteria_history
from train_models import FEATURES, MODEL_FILES, train_all
from recommend import confidence_label, recommend_for_plan
from impact import summarize_impact

DATA_PATH = ROOT / "data" / "second_bell_synthetic_cafeteria.csv"
MODEL_DIR = ROOT / "models"
MANIFEST_PATH = MODEL_DIR / "model_manifest.json"
TARGET_SCHOOL_DAYS = 540


def ensure_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
            if df["date"].nunique() >= TARGET_SCHOOL_DAYS and all(feature in df.columns for feature in FEATURES):
                return df
        except Exception:
            pass
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_cafeteria_history(n_days=TARGET_SCHOOL_DAYS, seed=42)
    df.to_csv(DATA_PATH, index=False)
    return df


def runtime_package_versions() -> dict:
    return {
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def load_models() -> dict:
    df = ensure_data()
    should_train = True
    if MANIFEST_PATH.exists() and all((MODEL_DIR / name).exists() for name in MODEL_FILES):
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            should_train = not (
                manifest.get("features") == FEATURES
                and manifest.get("package_versions") == runtime_package_versions()
                and int(manifest.get("training_rows", -1)) == len(df)
            )
        except Exception:
            should_train = True
    if should_train:
        train_all(DATA_PATH, MODEL_DIR)

    try:
        return {
            "q10": joblib.load(MODEL_DIR / "return_q10.joblib"),
            "q50": joblib.load(MODEL_DIR / "return_q50.joblib"),
            "q90": joblib.load(MODEL_DIR / "return_q90.joblib"),
            "risk": joblib.load(MODEL_DIR / "risk_classifier.joblib"),
            "after": joblib.load(MODEL_DIR / "afterschool_model.joblib"),
            "anomaly_pre": joblib.load(MODEL_DIR / "anomaly_preprocessor.joblib"),
            "anomaly": joblib.load(MODEL_DIR / "anomaly_model.joblib"),
            "padding": float(joblib.load(MODEL_DIR / "conformal_padding.joblib")),
        }
    except Exception:
        train_all(DATA_PATH, MODEL_DIR)
        return {
            "q10": joblib.load(MODEL_DIR / "return_q10.joblib"),
            "q50": joblib.load(MODEL_DIR / "return_q50.joblib"),
            "q90": joblib.load(MODEL_DIR / "return_q90.joblib"),
            "risk": joblib.load(MODEL_DIR / "risk_classifier.joblib"),
            "after": joblib.load(MODEL_DIR / "afterschool_model.joblib"),
            "anomaly_pre": joblib.load(MODEL_DIR / "anomaly_preprocessor.joblib"),
            "anomaly": joblib.load(MODEL_DIR / "anomaly_model.joblib"),
            "padding": float(joblib.load(MODEL_DIR / "conformal_padding.joblib")),
        }


def predict(plan: pd.DataFrame, models: dict) -> pd.DataFrame:
    x = plan[FEATURES]
    out = plan.copy()
    pad = models["padding"]
    out["predicted_return_q10"] = pd.Series(models["q10"].predict(x), index=out.index).sub(pad * 0.25).clip(lower=0).round(1)
    out["predicted_return_q50"] = pd.Series(models["q50"].predict(x), index=out.index).clip(lower=0).round(1)
    out["predicted_return_q90"] = pd.Series(models["q90"].predict(x), index=out.index).add(pad * 0.35).clip(lower=0).round(1)
    out["ghost_risk_probability"] = pd.Series(models["risk"].predict_proba(x)[:, 1], index=out.index).round(3)
    out["predicted_afterschool_demand"] = pd.Series(models["after"].predict(x), index=out.index).clip(lower=0).round(1)
    transformed = models["anomaly_pre"].transform(x)
    out["anomaly_flag"] = models["anomaly"].predict(transformed)
    out["confidence"] = [
        confidence_label(q10, q90, q50)
        for q10, q90, q50 in zip(out["predicted_return_q10"], out["predicted_return_q90"], out["predicted_return_q50"])
    ]
    return out


def assert_scenarios_include_entrees() -> None:
    entrees = ["pasta bowl", "veggie wrap", "chicken sandwich", "cheese pizza", "rice bowl", "taco tray"]
    for entree in entrees:
        plan = build_plan("Tuesday", "normal", "normal", entree, 900, 1, 90, 150)
        assert entree in set(plan["menu_item"]), f"Missing selectable entree from scenario: {entree}"


def assert_no_overcount(actions: pd.DataFrame, pred: pd.DataFrame) -> None:
    counted = actions[actions["count_in_impact"]]
    merged = counted.merge(
        pred[["menu_item", "lunch_period", "predicted_return_q50"]],
        on=["menu_item", "lunch_period"],
        how="left",
    )
    for _, row in merged.iterrows():
        assert row["marginal_items_recovered"] <= row["predicted_return_q50"] + 0.01, row.to_dict()
    assert summarize_impact(actions)["items_recovered_mid"] <= float(pred["predicted_return_q50"].sum()) + 0.01


def assert_cold_chain_blocks(models: dict) -> None:
    plan = build_plan("Tuesday", "rainy", "field_trip", "pasta bowl", 910, 0, 0, 175)
    pred = predict(plan, models)
    actions = recommend_for_plan(pred)
    cold_items = set(pred[pred["cold_chain_required"] == 1]["menu_item"])
    blocked = actions[
        actions["menu_item"].isin(cold_items)
        & actions["recommended_action"].str.contains("share-table|after-school", case=False, regex=True)
    ]
    assert blocked.empty, blocked[["menu_item", "recommended_action"]].to_dict("records")


def main() -> None:
    models = load_models()
    assert_scenarios_include_entrees()

    plan = build_plan("Tuesday", "rainy", "field_trip", "pasta bowl", 910, 1, 90, 175)
    pred = predict(plan, models)
    assert "anomaly_flag" in pred.columns
    assert set(pred["anomaly_flag"].unique()).issubset({-1, 1})
    actions = recommend_for_plan(pred)
    assert not actions.empty
    assert_no_overcount(actions, pred)
    assert_cold_chain_blocks(models)
    print("Second Bell smoke check passed.")


if __name__ == "__main__":
    main()
