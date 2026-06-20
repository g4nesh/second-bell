"""Action recommender for Second Bell."""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from state_machine import classify_rescue_state


ACTION_COLUMNS = [
    "action_id",
    "card_role",
    "priority",
    "menu_item",
    "lunch_period",
    "recommended_action",
    "why",
    "expected_items_recovered",
    "marginal_items_recovered",
    "count_in_impact",
    "estimated_kg_diverted",
    "estimated_co2e_avoided",
    "estimated_cost_protected",
    "labor_minutes",
    "stockout_risk_pct",
    "confidence",
    "rescue_state",
    "rescue_window_start",
    "rescue_window_end",
    "deploy_by",
    "cold_chain_deadline",
    "safety_gate",
    "requires_manager_approval",
    "human_approval_required",
    "score",
]


def confidence_label(q10: float, q90: float, q50: float) -> str:
    spread = max(0.0, q90 - q10)
    if q50 <= 1:
        return "Low"
    ratio = spread / max(q50, 1)
    if ratio < 0.55:
        return "High"
    if ratio < 1.05:
        return "Medium"
    return "Low"


def action_score(
    expected_recovered: float,
    co2e: float,
    cost: float,
    labor: float,
    safety_penalty: float,
    stockout_risk: float,
    confidence: str,
) -> float:
    conf_penalty = {"High": 0.0, "Medium": 7.0, "Low": 18.0}[confidence]
    return round(
        float(expected_recovered * 1.2 + co2e * 0.45 + cost * 0.35 - labor - safety_penalty - stockout_risk * 3.0 - conf_penalty),
        2,
    )


def _slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("/", "_")


def _timing(action_key: str, lunch_period: int, cold_chain_required: bool) -> Tuple[str, str, str, str, str]:
    if lunch_period == 1:
        share_start, share_end, deploy_by = "10:58 AM", "11:25 AM", "10:55 AM"
        staging_deploy = "10:40 AM"
    else:
        share_start, share_end, deploy_by = "12:12 PM", "12:35 PM", "12:12 PM"
        staging_deploy = "12:05 PM"

    if action_key == "after_school_route":
        return "12:35 PM", "2:45 PM", "12:35 PM", "2:45 PM" if cold_chain_required else "None", "match sealed items before clubs dismiss"
    if action_key == "two_wave_staging":
        return staging_deploy, share_end, staging_deploy, share_end if cold_chain_required else "None", "hold second wave until real demand is visible"
    if action_key == "line_placement":
        return "Before service", share_end, staging_deploy, share_end if cold_chain_required else "None", "change line flow before students start selecting defaults"
    return share_start, share_end, deploy_by, share_end if cold_chain_required else "None", "deploy before tray returns spike"


def _safety_gate(row: pd.Series, action_key: str) -> str:
    if action_key == "line_placement":
        return "No redistribution; line-flow prevention only"
    if row.get("cold_chain_required"):
        return "Sealed package + monitored cooler at 41 F or colder + manager approval"
    if row.get("sealed_or_whole"):
        return "Whole or sealed item + monitored share table + manager approval"
    return "Blocked: opened or prepared food is not eligible for this MVP"


def _stockout_risk_pct(q10: float, q50: float, action_key: str) -> float:
    if action_key == "two_wave_staging":
        return 4.0 if q10 >= 4 else 9.0
    if action_key == "line_placement":
        return 2.0
    if q50 <= 0:
        return 0.0
    return 3.0 if q10 / max(q50, 1.0) >= 0.45 else 6.0


def _candidate(
    row: pd.Series,
    action_key: str,
    action: str,
    why: str,
    recovered: float,
    labor_minutes: int,
    safety_penalty: float,
    confidence: str,
    state: str,
    requires_manager_approval: bool,
    q10: float,
    q50: float,
) -> Dict:
    item = str(row["menu_item"])
    lunch_period = int(row["lunch_period"])
    recovered = round(max(0.0, min(float(recovered), max(q50, 0.0))), 1)
    kg = float(row.get("kg_per_item", 0.1))
    co2e_item = float(row.get("co2e_kg_per_item", 0.1))
    cost_item = float(row.get("cost_per_item", 0.25))
    stockout_risk_pct = _stockout_risk_pct(q10, q50, action_key)
    window_start, window_end, deploy_by, cold_deadline, why_now = _timing(
        action_key,
        lunch_period,
        bool(row.get("cold_chain_required", 0)),
    )
    score = action_score(
        recovered,
        recovered * co2e_item,
        recovered * cost_item,
        labor=labor_minutes,
        safety_penalty=safety_penalty,
        stockout_risk=stockout_risk_pct / 5.0,
        confidence=confidence,
    )
    return {
        "action_id": f"l{lunch_period}-{_slug(item)}-{action_key}",
        "card_role": "Alternative",
        "priority": "High" if recovered >= 18 else "Medium",
        "menu_item": item,
        "lunch_period": lunch_period,
        "recommended_action": action,
        "why": f"{why} Timing: {why_now}.",
        "expected_items_recovered": recovered,
        "marginal_items_recovered": recovered,
        "count_in_impact": False,
        "estimated_kg_diverted": round(recovered * kg, 2),
        "estimated_co2e_avoided": round(recovered * co2e_item, 2),
        "estimated_cost_protected": round(recovered * cost_item, 2),
        "labor_minutes": labor_minutes,
        "stockout_risk_pct": stockout_risk_pct,
        "confidence": confidence,
        "rescue_state": state,
        "rescue_window_start": window_start,
        "rescue_window_end": window_end,
        "deploy_by": deploy_by,
        "cold_chain_deadline": cold_deadline,
        "safety_gate": _safety_gate(row, action_key),
        "requires_manager_approval": requires_manager_approval,
        "human_approval_required": requires_manager_approval,
        "score": score,
    }


def _guardrail_card(row: pd.Series, state, confidence: str) -> Dict:
    lunch_period = int(row["lunch_period"])
    item = str(row["menu_item"])
    window_start, window_end, deploy_by, cold_deadline, _ = _timing(
        "share_table",
        lunch_period,
        bool(row.get("cold_chain_required", 0)),
    )
    return {
        "action_id": f"l{lunch_period}-{_slug(item)}-guardrail",
        "card_role": "Guardrail",
        "priority": "Guardrail",
        "menu_item": item,
        "lunch_period": lunch_period,
        "recommended_action": "Block unsafe automatic redistribution",
        "why": state.explanation,
        "expected_items_recovered": 0.0,
        "marginal_items_recovered": 0.0,
        "count_in_impact": False,
        "estimated_kg_diverted": 0.0,
        "estimated_co2e_avoided": 0.0,
        "estimated_cost_protected": 0.0,
        "labor_minutes": 0,
        "stockout_risk_pct": 0.0,
        "confidence": confidence,
        "rescue_state": state.state,
        "rescue_window_start": window_start,
        "rescue_window_end": window_end,
        "deploy_by": deploy_by,
        "cold_chain_deadline": cold_deadline,
        "safety_gate": state.explanation,
        "requires_manager_approval": state.requires_human_approval,
        "human_approval_required": state.requires_human_approval,
        "score": -1.0,
    }


def recommend_for_plan(pred_df: pd.DataFrame) -> pd.DataFrame:
    actions: List[Dict] = []

    for _, row in pred_df.iterrows():
        state = classify_rescue_state(row.to_dict())
        q10 = float(row["predicted_return_q10"])
        q50 = float(row["predicted_return_q50"])
        q90 = float(row["predicted_return_q90"])
        conf = confidence_label(q10, q90, q50)
        item = str(row["menu_item"])
        recovered_base = max(0.0, q50)

        if "deploy share-table cooler" in state.allowed_actions:
            recovered = min(recovered_base * 0.58, float(row.get("cooler_capacity", 0)) * 0.75)
            actions.append(
                _candidate(
                    row,
                    "share_table",
                    "Deploy monitored share-table cooler before tray-return spike",
                    f"{item} is predicted to return unopened in a recoverable window; share-table reuse keeps food feeding students.",
                    recovered,
                    labor_minutes=8,
                    safety_penalty=5 if row.get("cold_chain_required") else 2,
                    confidence=conf,
                    state=state.state,
                    requires_manager_approval=state.requires_human_approval,
                    q10=q10,
                    q50=q50,
                )
            )

        if "two-wave staging" in state.allowed_actions and bool(row.get("cold_chain_required", 0)):
            actions.append(
                _candidate(
                    row,
                    "two_wave_staging",
                    "Stage cold items in two waves instead of exposing full supply early",
                    "Smaller staging lets the cafeteria observe real demand before the second batch leaves cold storage.",
                    recovered_base * 0.28,
                    labor_minutes=5,
                    safety_penalty=1,
                    confidence=conf,
                    state=state.state,
                    requires_manager_approval=True,
                    q10=q10,
                    q50=q50,
                )
            )

        if "line placement adjustment" in state.allowed_actions and row.get("component_type") in ["fruit", "vegetable", "milk"]:
            actions.append(
                _candidate(
                    row,
                    "line_placement",
                    "Move default components after entree choice to reduce accidental selection",
                    "This targets ghost components: items taken because of line flow, not because students intended to consume them.",
                    recovered_base * 0.24,
                    labor_minutes=3,
                    safety_penalty=0,
                    confidence=conf,
                    state=state.state,
                    requires_manager_approval=False,
                    q10=q10,
                    q50=q50,
                )
            )

        if "after-school demand match" in state.allowed_actions and float(row.get("predicted_afterschool_demand", 0)) > 8:
            recovered = min(recovered_base * 0.36, float(row.get("predicted_afterschool_demand", 0)))
            actions.append(
                _candidate(
                    row,
                    "after_school_route",
                    "Route approved sealed surplus to after-school snack demand",
                    "After-school programs create same-day demand for safe sealed items, reducing landfill without outside logistics.",
                    recovered,
                    labor_minutes=10,
                    safety_penalty=7 if row.get("cold_chain_required") else 3,
                    confidence=conf,
                    state=state.state,
                    requires_manager_approval=True,
                    q10=q10,
                    q50=q50,
                )
            )

        if len(state.blocked_actions) and q50 >= 12:
            actions.append(_guardrail_card(row, state, conf))

    if not actions:
        return pd.DataFrame(columns=ACTION_COLUMNS)

    out = pd.DataFrame(actions)
    positive = out[(out["expected_items_recovered"] > 0) & (out["score"] > 0)].sort_values("score", ascending=False)
    for _, group in positive.groupby(["menu_item", "lunch_period"], sort=False):
        primary_idx = group.index[0]
        out.loc[primary_idx, "count_in_impact"] = True
        out.loc[primary_idx, "card_role"] = "Primary plan"

    out["sort_primary"] = out["count_in_impact"].astype(int)
    out = out.sort_values(
        ["sort_primary", "score", "expected_items_recovered"],
        ascending=[False, False, False],
    ).drop(columns=["sort_primary"])
    return out[ACTION_COLUMNS].reset_index(drop=True)
