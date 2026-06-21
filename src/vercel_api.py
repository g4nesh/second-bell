"""Vercel API helpers for the static Second Bell demo."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd

from afterschool import forecast_after_school_count
from digital_twin import build_digital_twin
from impact import summarize_impact
from recommend import confidence_label, recommend_for_plan
from scenario import ITEM_META, build_plan
from train_models import train_all
from vision import simulate_reduction_detection


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "second_bell_synthetic_cafeteria.csv"
MODEL_DIR = ROOT / "models"

MODEL_NAMES = {
    "q10": "return_q10.joblib",
    "q50": "return_q50.joblib",
    "q90": "return_q90.joblib",
    "risk": "risk_classifier.joblib",
    "after": "afterschool_model.joblib",
    "anomaly_pre": "anomaly_preprocessor.joblib",
    "anomaly": "anomaly_model.joblib",
    "padding": "conformal_padding.joblib",
}


@lru_cache(maxsize=1)
def load_history() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def _load_models_from(model_dir: Path) -> dict[str, Any]:
    models = {key: joblib.load(model_dir / filename) for key, filename in MODEL_NAMES.items()}
    models["padding"] = float(models["padding"])
    return models


@lru_cache(maxsize=1)
def load_models() -> dict[str, Any]:
    try:
        return _load_models_from(MODEL_DIR)
    except Exception:
        fallback_dir = Path(tempfile.gettempdir()) / "second_bell_models"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        train_all(DATA_PATH, fallback_dir)
        return _load_models_from(fallback_dir)


def predict_plan(plan: pd.DataFrame, models: dict[str, Any]) -> pd.DataFrame:
    from train_models import FEATURES

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
        for q10, q90, q50 in zip(
            out["predicted_return_q10"],
            out["predicted_return_q90"],
            out["predicted_return_q50"],
        )
    ]
    out["rescue_pressure"] = (out["predicted_return_q50"] * (0.6 + out["ghost_risk_probability"])).round(1)
    out["anomaly_review"] = out["anomaly_flag"].map({-1: "Manager review", 1: "Normal"})
    out["driver_explanation"] = out.apply(explain_drivers, axis=1)
    return out


def explain_drivers(row: pd.Series) -> str:
    reasons = []
    if row["event_tag"] in ["field_trip", "early_release", "exam_day"]:
        reasons.append(f"{row['event_tag'].replace('_', ' ')} changes attendance and line behavior")
    if row["weather_tag"] in ["rainy", "cold", "hot"]:
        reasons.append(f"{row['weather_tag']} weather shifts demand")
    if row["entree"] in ["pasta bowl", "cheese pizza"] and row["component_type"] == "milk":
        reasons.append(f"{row['entree']} historically creates more unopened milk returns")
    if row["line_position"] == "before_cashier" and row["component_type"] in ["fruit", "vegetable", "milk"]:
        reasons.append("item is placed before students finish deciding what they actually want")
    if row["lunch_period"] == 2:
        reasons.append("second lunch has more end-of-service surplus risk")
    if row["anomaly_flag"] == -1:
        reasons.append("pattern is unusual enough to require manager review")
    if not reasons:
        reasons.append("similar historical item-period patterns in the synthetic cafeteria data")
    return "; ".join(reasons[:3])


def build_run_response(payload: dict[str, Any]) -> dict[str, Any]:
    history = load_history()
    models = load_models()

    setup = {
        "day_of_week": str(payload.get("day_of_week") or "Tuesday"),
        "weather_tag": str(payload.get("weather_tag") or "rainy"),
        "event_tag": str(payload.get("event_tag") or "field_trip"),
        "entree": str(payload.get("entree") or "pasta bowl"),
        "expected_attendance": int(payload.get("expected_attendance") or 910),
        "share_table_monitor": bool(payload.get("share_table_monitor", True)),
        "cooler_capacity": int(payload.get("cooler_capacity") or 90),
        "line_mode": str(payload.get("line_mode") or "default"),
        "approved_action_ids": list(payload.get("approved_action_ids") or []),
    }
    if not setup["share_table_monitor"]:
        setup["cooler_capacity"] = 0

    afterschool = forecast_after_school_count(
        history,
        day_of_week=setup["day_of_week"],
        weather_tag=setup["weather_tag"],
        event_tag=setup["event_tag"],
        expected_attendance=setup["expected_attendance"],
    )
    plan = build_plan(
        day_of_week=setup["day_of_week"],
        weather_tag=setup["weather_tag"],
        event_tag=setup["event_tag"],
        entree=setup["entree"],
        expected_attendance=setup["expected_attendance"],
        share_table_monitor=int(setup["share_table_monitor"]),
        cooler_capacity=setup["cooler_capacity"],
        afterschool_activity_count=afterschool.predicted_count,
        line_mode=setup["line_mode"],
    )
    pred = predict_plan(plan, models)
    twin = build_digital_twin(
        pred,
        afterschool_predicted_count=afterschool.predicted_count,
        afterschool_low=afterschool.low,
        afterschool_high=afterschool.high,
        share_table_monitor=setup["share_table_monitor"],
    )
    pred = pred.merge(
        twin.item_estimates,
        on=["menu_item", "lunch_period", "component_type"],
        how="left",
    )
    pred = pred.sort_values("rescue_pressure", ascending=False)
    actions = recommend_for_plan(pred)
    vision = simulate_reduction_detection(pred, twin.item_estimates)

    return jsonable(
        {
            "setup": setup,
            "afterschool": {
                "predicted_count": afterschool.predicted_count,
                "low": afterschool.low,
                "high": afterschool.high,
                "model_type": afterschool.model_type,
                "mean_absolute_error": afterschool.mean_absolute_error,
                "r2": afterschool.r2,
                "history_window": records(afterschool.history_window),
            },
            "predictions": records(pred),
            "actions": records(actions),
            "approved_action_ids": setup["approved_action_ids"],
            "impact": summarize_impact(actions),
            "digital_twin": {
                "sensor_inventory": records(twin.sensor_inventory),
                "observations": records(twin.observations),
                "item_estimates": records(twin.item_estimates),
                "world_state": records(twin.world_state),
                "bridge_notes": records(twin.bridge_notes),
            },
            "vision": records(vision),
        }
    )


def build_scale_response(payload: dict[str, Any]) -> dict[str, Any]:
    menu_item = str(payload.get("menu_item") or "white milk")
    meta = ITEM_META.get(menu_item, ITEM_META["white milk"])
    unit_weight_g = round(float(meta[4]) * 1000.0, 1)

    gross = float(payload.get("gross_weight_g", payload.get("weight_g", 0)) or 0)
    tare = float(payload.get("tare_weight_g", 0) or 0)
    baseline = float(payload.get("baseline_weight_g", 0) or 0)
    stable_samples = int(payload.get("stable_samples", 0) or 0)
    finger_contact = bool(payload.get("finger_contact", gross > 0))
    net = max(0.0, gross - tare - baseline)
    count = round(net / unit_weight_g, 1) if unit_weight_g > 0 else 0.0

    if not finger_contact or net <= 0:
        confidence = "Low"
    elif stable_samples >= 6:
        confidence = "High"
    elif stable_samples >= 3:
        confidence = "Medium"
    else:
        confidence = "Low"

    return jsonable(
        {
            "source": str(payload.get("source") or "browser/manual scale"),
            "menu_item": menu_item,
            "gross_weight_g": round(gross, 1),
            "tare_weight_g": round(tare, 1),
            "baseline_weight_g": round(baseline, 1),
            "net_weight_g": round(net, 1),
            "unit_weight_g": unit_weight_g,
            "estimated_item_count": count,
            "confidence": confidence,
            "sensor_observation": {
                "sensor": "Mac Force Touch trackpad scale",
                "metric": "scaled_recoverable_item_count",
                "observed_count": count,
                "sigma": 1.0 if confidence == "High" else 1.8 if confidence == "Medium" else 2.6,
                "used_in_estimate": bool(net > 0),
            },
            "bridge": {
                "compatible_repo": "https://github.com/KrishKrosh/TrackWeight",
                "support_library": "https://github.com/KrishKrosh/OpenMultitouchSupport",
                "endpoint": "/api/scale/reading",
                "payload_fields": [
                    "menu_item",
                    "gross_weight_g",
                    "tare_weight_g",
                    "baseline_weight_g",
                    "stable_samples",
                    "finger_contact",
                ],
            },
        }
    )


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    return jsonable(df.replace({np.nan: None}).to_dict(orient="records"))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
