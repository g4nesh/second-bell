"""Impact calculations for Second Bell.

Impact outputs are deliberately estimated as ranges instead of false precision.
Only approved primary actions are counted, so alternatives never inflate the
receipt.
"""
from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd


def _empty_impact() -> Dict[str, float]:
    return {
        "items_recovered_low": 0,
        "items_recovered_mid": 0,
        "items_recovered_high": 0,
        "kg_diverted_low": 0,
        "kg_diverted_mid": 0,
        "kg_diverted_high": 0,
        "co2e_low": 0,
        "co2e_mid": 0,
        "co2e_high": 0,
        "cost_low": 0,
        "cost_mid": 0,
        "cost_high": 0,
        "labor_minutes": 0,
        "stockout_risk_pct": 0,
        "selected_action_count": 0,
    }


def summarize_impact(actions: pd.DataFrame, approved_action_ids: Iterable[str] | None = None) -> Dict[str, float]:
    if actions.empty:
        return _empty_impact()

    selected = actions.copy()
    if "count_in_impact" in selected:
        selected = selected[selected["count_in_impact"]]
    if approved_action_ids is not None:
        approved = set(approved_action_ids)
        selected = selected[selected["action_id"].isin(approved)]

    if selected.empty:
        return _empty_impact()

    item_col = "marginal_items_recovered" if "marginal_items_recovered" in selected else "expected_items_recovered"
    items = float(selected[item_col].sum())
    kg = float(selected["estimated_kg_diverted"].sum())
    co2e = float(selected["estimated_co2e_avoided"].sum())
    cost = float(selected["estimated_cost_protected"].sum())
    labor = float(selected["labor_minutes"].sum()) if "labor_minutes" in selected else 0.0
    stockout = float(selected["stockout_risk_pct"].max()) if "stockout_risk_pct" in selected else 0.0

    return {
        "items_recovered_low": round(items * 0.78, 1),
        "items_recovered_mid": round(items, 1),
        "items_recovered_high": round(items * 1.18, 1),
        "kg_diverted_low": round(kg * 0.78, 1),
        "kg_diverted_mid": round(kg, 1),
        "kg_diverted_high": round(kg * 1.18, 1),
        "co2e_low": round(co2e * 0.78, 1),
        "co2e_mid": round(co2e, 1),
        "co2e_high": round(co2e * 1.18, 1),
        "cost_low": round(cost * 0.78, 2),
        "cost_mid": round(cost, 2),
        "cost_high": round(cost * 1.18, 2),
        "labor_minutes": round(labor, 1),
        "stockout_risk_pct": round(stockout, 1),
        "selected_action_count": int(len(selected)),
    }
