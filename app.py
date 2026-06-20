from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

from scenario import build_plan
from recommend import recommend_for_plan, confidence_label
from impact import summarize_impact
from simulate_data import generate_cafeteria_history
from train_models import FEATURES, MODEL_FILES, train_all

st.set_page_config(page_title="Second Bell", layout="wide")

DATA_PATH = ROOT / "data" / "second_bell_synthetic_cafeteria.csv"
MODEL_DIR = ROOT / "models"
MANIFEST_PATH = MODEL_DIR / "model_manifest.json"
TARGET_SCHOOL_DAYS = 540


def _dataset_needs_refresh() -> bool:
    if not DATA_PATH.exists():
        return True
    try:
        df = pd.read_csv(DATA_PATH, usecols=lambda col: col in set(FEATURES + ["date", "unopened_return_count"]))
    except Exception:
        return True
    if "date" not in df or df["date"].nunique() < TARGET_SCHOOL_DAYS:
        return True
    return any(feature not in df.columns for feature in FEATURES)


def _write_dataset() -> pd.DataFrame:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_cafeteria_history(n_days=TARGET_SCHOOL_DAYS, seed=42)
    df.to_csv(DATA_PATH, index=False)
    return df


def _load_artifacts() -> dict:
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


def _runtime_package_versions() -> dict:
    return {
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _manifest_is_current(row_count: int) -> bool:
    if not MANIFEST_PATH.exists():
        return False
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        manifest.get("schema_version") == 1
        and manifest.get("features") == FEATURES
        and manifest.get("package_versions") == _runtime_package_versions()
        and int(manifest.get("training_rows", -1)) == int(row_count)
        and all((MODEL_DIR / name).exists() for name in MODEL_FILES)
    )


def ensure_assets() -> str:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    status = "Loaded validated local model artifacts."

    if _dataset_needs_refresh():
        _write_dataset()
        status = "Regenerated synthetic cafeteria data."

    history = pd.read_csv(DATA_PATH)
    should_train = not _manifest_is_current(len(history))
    if not should_train:
        try:
            _load_artifacts()
        except Exception:
            should_train = True
            status = "Detected incompatible model artifacts and retrained them."

    if should_train:
        train_all(DATA_PATH, MODEL_DIR)
        _load_artifacts()
        if status == "Loaded validated local model artifacts.":
            status = "Trained model artifacts from current synthetic data."
    return status


@st.cache_resource(show_spinner="Training or loading Second Bell models...")
def load_models():
    asset_status = ensure_assets()
    models = _load_artifacts()
    models["_asset_status"] = asset_status
    return models


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    ensure_assets()
    return pd.read_csv(DATA_PATH)


def predict_plan(plan: pd.DataFrame, models: dict) -> pd.DataFrame:
    X = plan[FEATURES]
    out = plan.copy()
    pad = models["padding"]
    out["predicted_return_q10"] = pd.Series(models["q10"].predict(X), index=out.index).sub(pad * 0.25).clip(lower=0).round(1)
    out["predicted_return_q50"] = pd.Series(models["q50"].predict(X), index=out.index).clip(lower=0).round(1)
    out["predicted_return_q90"] = pd.Series(models["q90"].predict(X), index=out.index).add(pad * 0.35).clip(lower=0).round(1)
    out["ghost_risk_probability"] = pd.Series(models["risk"].predict_proba(X)[:, 1], index=out.index).round(3)
    out["predicted_afterschool_demand"] = pd.Series(models["after"].predict(X), index=out.index).clip(lower=0).round(1)
    transformed = models["anomaly_pre"].transform(X)
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


def _range_text(low: float, high: float, suffix: str = "") -> str:
    return f"{low:.0f}-{high:.0f}{suffix}"


history = load_history()
models = load_models()

st.title("Second Bell")
st.caption("AI rescue clock for ghost components: unopened cafeteria food before the safe recovery window closes.")

with st.sidebar:
    st.header("Morning setup")
    day_of_week = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], index=1)
    weather_tag = st.selectbox("Weather", ["normal", "rainy", "hot", "cold", "windy"], index=1)
    event_tag = st.selectbox("School event", ["normal", "field_trip", "exam_day", "assembly", "sports_away", "club_fair", "early_release"], index=1)
    entree = st.selectbox("Main entree", ["pasta bowl", "veggie wrap", "chicken sandwich", "cheese pizza", "rice bowl", "taco tray"], index=0)
    expected_attendance = st.slider("Expected attendance", 700, 1120, 910, 10)
    afterschool_activity_count = st.slider("After-school students needing snacks", 20, 270, 175, 5)
    share_table_monitor = st.toggle("Share-table monitor available", value=True)
    cooler_capacity = st.slider("Cooler capacity for cold items", 0, 120, 90, 5, disabled=not share_table_monitor)
    line_mode = st.radio(
        "Line setup",
        ["default", "after_entree"],
        format_func=lambda value: "Default line placement" if value == "default" else "Move components after entree choice",
    )

plan = build_plan(
    day_of_week=day_of_week,
    weather_tag=weather_tag,
    event_tag=event_tag,
    entree=entree,
    expected_attendance=expected_attendance,
    share_table_monitor=int(share_table_monitor),
    cooler_capacity=int(cooler_capacity if share_table_monitor else 0),
    afterschool_activity_count=afterschool_activity_count,
    line_mode=line_mode,
)

pred = predict_plan(plan, models)
pred["drivers"] = pred.apply(explain_drivers, axis=1)
actions = recommend_for_plan(pred)
baseline_items = float(pred["predicted_return_q50"].sum())
primary_actions = actions[actions["count_in_impact"]] if not actions.empty else actions
anomaly_count = int((pred["anomaly_flag"] == -1).sum())

st.subheader("1. Morning setup")
setup = pd.DataFrame(
    [
        ["User", "Ms. Rivera, cafeteria manager with a student eco-club monitor"],
        ["School", "Cedar Grove High School synthetic demo"],
        ["Lunch context", f"{day_of_week}, {weather_tag.replace('_', ' ')}, {event_tag.replace('_', ' ')}, {entree}"],
        ["Operations", f"{expected_attendance} expected students, cooler capacity {int(cooler_capacity if share_table_monitor else 0)}, monitor {'yes' if share_table_monitor else 'no'}"],
        ["After school", f"{afterschool_activity_count} students across tutoring, robotics, track, and clubs"],
    ],
    columns=["Input", "Value"],
)
st.dataframe(setup, use_container_width=True, hide_index=True)

st.subheader("2. Ghost risk forecast")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted unopened returns", f"{baseline_items:.0f}")
col2.metric("High-risk ghost items", int((pred["ghost_risk_probability"] > 0.55).sum()))
col3.metric("Primary action cards", int(len(primary_actions)))
col4.metric("Anomaly review flags", anomaly_count)

if anomaly_count:
    st.warning("Unusual operating day detected. Second Bell can recommend actions, but Ms. Rivera should review assumptions before approving recovery.")

left, right = st.columns([1.25, 1])
with left:
    heat = pred.pivot_table(index="menu_item", columns="lunch_period", values="rescue_pressure", aggfunc="mean").fillna(0)
    fig = px.imshow(
        heat,
        labels=dict(x="Lunch period", y="Menu item", color="Rescue pressure"),
        aspect="auto",
        text_auto=True,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    rescue_clock = pd.DataFrame(
        [
            ["7:30 AM", "Plan", "Menu, attendance, event, weather, cooler, and monitor inputs"],
            ["10:55 AM", "First action point", "Deploy or hold cold items before first-lunch returns"],
            ["12:12 PM", "Second Bell alert", "Act before second-lunch tray returns close the rescue window"],
            ["12:35 PM", "After-school match", "Route approved sealed surplus to on-campus demand"],
            ["2:45 PM", "Impact receipt", "Eco-club logs recovered, blocked, and discarded items"],
        ],
        columns=["Time", "Stage", "Decision"],
    )
    st.dataframe(rescue_clock, use_container_width=True, hide_index=True)

st.subheader("3. Why this is happening")
show_cols = [
    "menu_item",
    "lunch_period",
    "component_type",
    "line_position",
    "predicted_return_q10",
    "predicted_return_q50",
    "predicted_return_q90",
    "ghost_risk_probability",
    "predicted_afterschool_demand",
    "confidence",
    "anomaly_review",
    "drivers",
]
st.dataframe(
    pred.sort_values("rescue_pressure", ascending=False)[show_cols].head(12),
    use_container_width=True,
    hide_index=True,
)

st.subheader("4. Action plan and human approval")
approved_action_ids: list[str] = []
if actions.empty:
    st.info("No major rescue action recommended for this scenario. Log actual returns and monitor the next lunch period.")
else:
    visible_actions = pd.concat(
        [
            actions[actions["count_in_impact"]],
            actions[~actions["count_in_impact"]].head(4),
        ],
        ignore_index=True,
    ).drop_duplicates("action_id")
    for _, action in visible_actions.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([1.05, 4])
            c1.metric("Score", action["score"])
            c1.write(f"**{action['card_role']}**")
            c1.write(f"**{action['priority']}**")
            c2.markdown(f"#### {action['recommended_action']}")
            c2.write(
                f"**Item:** {action['menu_item']} | **Lunch:** {action['lunch_period']} | "
                f"**Confidence:** {action['confidence']} | **State:** `{action['rescue_state']}`"
            )
            c2.write(action["why"])
            c2.write(
                f"Expected recovery if selected: **{action['expected_items_recovered']} items** | "
                f"Labor: **{action['labor_minutes']} min** | Stockout risk: **{action['stockout_risk_pct']}%**"
            )
            c2.write(
                f"Deploy by **{action['deploy_by']}** | Window **{action['rescue_window_start']} to {action['rescue_window_end']}** | "
                f"Cold-chain deadline **{action['cold_chain_deadline']}**"
            )
            c2.info(f"Safety gate: {action['safety_gate']}")

            if action["count_in_impact"]:
                label = "Ms. Rivera approves this action for today's receipt" if action["requires_manager_approval"] else "Include this prevention action in today's receipt"
                if c2.toggle(label, value=True, key=f"approve_{action['action_id']}"):
                    approved_action_ids.append(action["action_id"])
            elif action["card_role"] == "Guardrail":
                c2.warning("Guardrail card: this action is intentionally blocked and never counted as recovered food.")
            else:
                c2.caption("Alternative shown for transparency. It is not counted in the impact receipt unless promoted to the primary plan.")

impact = summarize_impact(actions, approved_action_ids)
remaining_items = max(0.0, baseline_items - impact["items_recovered_mid"])
baseline_kg = float((pred["predicted_return_q50"] * pred["kg_per_item"]).sum())
remaining_kg = max(0.0, baseline_kg - impact["kg_diverted_mid"])

st.subheader("5. Impact receipt")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Approved actions", impact["selected_action_count"])
col2.metric("Recovered range", _range_text(impact["items_recovered_low"], impact["items_recovered_high"], " items"))
col3.metric("Food diverted", f"{impact['kg_diverted_low']:.1f}-{impact['kg_diverted_high']:.1f} kg")
col4.metric("Food value protected", f"USD {impact['cost_low']:.2f}-{impact['cost_high']:.2f}")

baseline_vs_action = pd.DataFrame(
    [
        ["Unopened items likely without action", f"{baseline_items:.0f}", "Forecast from item-period model"],
        ["Items recovered by approved plan", f"{impact['items_recovered_low']:.0f}-{impact['items_recovered_high']:.0f}", "Capped to primary approved actions only"],
        ["Items still at risk", f"{remaining_items:.0f}", "Requires compost, disposal, or later logging"],
        ["Food mass at risk before action", f"{baseline_kg:.1f} kg", "Predicted returns multiplied by item weights"],
        ["Food mass at risk after action", f"{remaining_kg:.1f} kg", "Baseline minus approved recovery"],
        ["Estimated CO2e avoided", f"{impact['co2e_low']:.1f}-{impact['co2e_high']:.1f} kg CO2e", "Conservative range, not exact climate accounting"],
        ["Operational load", f"{impact['labor_minutes']:.0f} labor minutes", "Estimated extra cafeteria/eco-club time"],
        ["Highest stockout risk", f"{impact['stockout_risk_pct']:.1f}%", "Shown so prevention does not create under-service"],
    ],
    columns=["Metric", "Value", "Judge-facing explanation"],
)
st.dataframe(baseline_vs_action, use_container_width=True, hide_index=True)

with st.expander("Responsible AI and human-in-the-loop design", expanded=True):
    st.markdown(
        """
**Realistic risk:** unsafe redistribution or over-trusting an estimate. A model can predict that sealed food may be recoverable, but it cannot verify temperature history, packaging integrity, local health rules, or school policy.

**Concrete mitigation:** Second Bell blocks automatic redistribution. Cold-chain items require sealed packaging, a monitored cooler, a 41 F-or-colder safety gate, confidence ranges, and manager approval before they count in the impact receipt.

**Decision the AI does not make:** the AI does not decide whether food is safe or policy-approved to share, donate, or route to after-school programs. That decision stays with cafeteria staff.

**Privacy:** the system uses aggregate counts, menu plans, event tags, and synthetic cafeteria history. It does not use student names, IDs, demographics, free/reduced-lunch status, or individual eating behavior.
"""
    )

with st.expander("Model card, data disclosure, and source grounding"):
    try:
        metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    except Exception:
        metrics = {}
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        manifest = {}
    st.write("**Demo health:** " + str(models.get("_asset_status", "Loaded.")))
    st.write(
        "**Models used:** RandomForestClassifier for ghost-component risk, quantile GradientBoostingRegressor for unopened-return intervals, "
        "RandomForestRegressor for after-school demand, and IsolationForest for unusual operating days."
    )
    st.json(metrics)
    st.write(f"Synthetic training rows: {len(history):,}; synthetic school days: {history['date'].nunique():,}")
    st.write("USDA share-table guidance supports whole or unopened items and unopened milk with cold storage; EPA and ReFED ground the prevention/rescue/recycling hierarchy.")
    st.write(
        "[USDA share tables](https://www.usda.gov/sites/default/files/guidance-documents/fns.sp41cacfp13sfsp15-2016-shareTables.pdf) | "
        "[EPA Wasted Food Scale](https://www.epa.gov/sustainable-management-food/wasted-food-scale) | "
        "[ReFED solutions](https://refed.org/food-waste/the-solutions/)"
    )
    if manifest:
        st.write("**Model manifest:**")
        st.json(manifest)
    st.dataframe(history.sample(min(12, len(history)), random_state=7), use_container_width=True, hide_index=True)
