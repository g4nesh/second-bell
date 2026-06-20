"""Build demo-day scenarios from user inputs."""
from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

ITEM_META = {
    "white milk": ("cold beverage", "milk", 1, 1, 0.24, 0.42, 0.33, 0.50),
    "chocolate milk": ("cold beverage", "milk", 1, 1, 0.24, 0.44, 0.34, 0.38),
    "apple": ("whole fruit", "fruit", 1, 0, 0.18, 0.36, 0.08, 0.42),
    "banana": ("whole fruit", "fruit", 1, 0, 0.14, 0.34, 0.07, 0.32),
    "fruit cup": ("sealed side", "fruit", 1, 0, 0.12, 0.55, 0.10, 0.30),
    "baby carrots": ("sealed side", "vegetable", 1, 0, 0.09, 0.38, 0.06, 0.28),
    "yogurt cup": ("sealed side", "protein side", 1, 1, 0.17, 0.70, 0.25, 0.18),
    "granola pack": ("sealed side", "grain", 1, 0, 0.04, 0.32, 0.06, 0.20),
    "pasta bowl": ("hot entree", "entree", 0, 0, 0.38, 1.50, 0.76, 0.36),
    "veggie wrap": ("cold entree", "entree", 1, 1, 0.30, 1.75, 0.55, 0.18),
    "chicken sandwich": ("hot entree", "entree", 0, 0, 0.34, 1.85, 0.90, 0.42),
    "cheese pizza": ("hot entree", "entree", 0, 0, 0.28, 1.25, 0.70, 0.48),
    "rice bowl": ("hot entree", "entree", 0, 0, 0.36, 1.62, 0.62, 0.34),
    "taco tray": ("hot entree", "entree", 0, 0, 0.33, 1.58, 0.68, 0.35),
}


def build_plan(day_of_week: str, weather_tag: str, event_tag: str, entree: str, expected_attendance: int, share_table_monitor: int, cooler_capacity: int, afterschool_activity_count: int, line_mode: str = "default") -> pd.DataFrame:
    rows: List[dict] = []
    for lunch_period in [1, 2]:
        period_share = 0.52 if lunch_period == 1 else 0.48
        if event_tag == "field_trip" and lunch_period == 2:
            period_share -= 0.08
        expected_lp = int(expected_attendance * period_share)
        for menu_item, meta in ITEM_META.items():
            item_category, component_type, sealed_or_whole, cold_chain_required, kg, cost, co2e, take_rate = meta
            if component_type == "entree" and menu_item not in [entree, "veggie wrap"]:
                continue
            line_position = "entree_station" if component_type == "entree" else "before_cashier"
            if line_mode == "after_entree" and component_type in ["fruit", "vegetable", "milk"]:
                line_position = "after_entree"
            elif menu_item in ["fruit cup", "yogurt cup", "granola pack"]:
                line_position = "after_entree"

            # Planning count approximates what a manager would stage from attendance and historical take rate.
            adjusted_take_rate = take_rate
            if entree in ["pasta bowl", "cheese pizza"] and component_type == "milk":
                adjusted_take_rate *= 1.08
            if weather_tag == "hot" and component_type == "milk":
                adjusted_take_rate *= 1.04
            if event_tag in ["field_trip", "early_release"] and component_type in ["fruit", "milk"]:
                adjusted_take_rate *= 0.92
            planned_count = int(max(8, expected_lp * adjusted_take_rate * 1.08))
            staged_count = planned_count if not cold_chain_required else int(planned_count * 0.94)

            rows.append({
                "day_of_week": day_of_week,
                "lunch_period": lunch_period,
                "menu_item": menu_item,
                "item_category": item_category,
                "component_type": component_type,
                "sealed_or_whole": sealed_or_whole,
                "cold_chain_required": cold_chain_required,
                "planned_count": planned_count,
                "staged_count": staged_count,
                "expected_attendance": expected_attendance,
                "expected_lunch_period_attendance": expected_lp,
                "weather_tag": weather_tag,
                "event_tag": event_tag,
                "entree": entree,
                "entree_popularity_score": round(adjusted_take_rate, 3),
                "line_position": line_position,
                "share_table_monitor": share_table_monitor,
                "cooler_capacity": cooler_capacity,
                "afterschool_activity_count": afterschool_activity_count,
                "kg_per_item": kg,
                "co2e_kg_per_item": co2e,
                "cost_per_item": cost,
            })
    return pd.DataFrame(rows)
