"""Synthetic cafeteria data generator for Second Bell.

Second Bell models a very specific food-waste pathway in high schools:
"ghost components" — unopened milk, fruit, and sealed sides that students take
but do not actually consume. The simulator creates realistic item-by-lunch-period
records without using any student-level or private school data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

RNG_SEED = 42


@dataclass(frozen=True)
class ItemConfig:
    menu_item: str
    item_category: str
    component_type: str
    sealed_or_whole: int
    cold_chain_required: int
    kg_per_item: float
    cost_per_item: float
    co2e_kg_per_item: float
    base_take_rate: float
    base_return_rate: float
    line_position_default: str


ITEMS: List[ItemConfig] = [
    ItemConfig("white milk", "cold beverage", "milk", 1, 1, 0.24, 0.42, 0.33, 0.50, 0.17, "before_cashier"),
    ItemConfig("chocolate milk", "cold beverage", "milk", 1, 1, 0.24, 0.44, 0.34, 0.38, 0.11, "before_cashier"),
    ItemConfig("apple", "whole fruit", "fruit", 1, 0, 0.18, 0.36, 0.08, 0.42, 0.22, "before_cashier"),
    ItemConfig("banana", "whole fruit", "fruit", 1, 0, 0.14, 0.34, 0.07, 0.32, 0.18, "before_cashier"),
    ItemConfig("fruit cup", "sealed side", "fruit", 1, 0, 0.12, 0.55, 0.10, 0.30, 0.12, "after_entree"),
    ItemConfig("baby carrots", "sealed side", "vegetable", 1, 0, 0.09, 0.38, 0.06, 0.28, 0.19, "before_cashier"),
    ItemConfig("yogurt cup", "sealed side", "protein side", 1, 1, 0.17, 0.70, 0.25, 0.18, 0.10, "after_entree"),
    ItemConfig("granola pack", "sealed side", "grain", 1, 0, 0.04, 0.32, 0.06, 0.20, 0.06, "after_entree"),
    ItemConfig("pasta bowl", "hot entree", "entree", 0, 0, 0.38, 1.50, 0.76, 0.36, 0.07, "entree_station"),
    ItemConfig("veggie wrap", "cold entree", "entree", 1, 1, 0.30, 1.75, 0.55, 0.18, 0.14, "entree_station"),
    ItemConfig("chicken sandwich", "hot entree", "entree", 0, 0, 0.34, 1.85, 0.90, 0.42, 0.06, "entree_station"),
    ItemConfig("cheese pizza", "hot entree", "entree", 0, 0, 0.28, 1.25, 0.70, 0.48, 0.04, "entree_station"),
    ItemConfig("rice bowl", "hot entree", "entree", 0, 0, 0.36, 1.62, 0.62, 0.34, 0.07, "entree_station"),
    ItemConfig("taco tray", "hot entree", "entree", 0, 0, 0.33, 1.58, 0.68, 0.35, 0.08, "entree_station"),
]

ENTREES = ["pasta bowl", "veggie wrap", "chicken sandwich", "cheese pizza", "rice bowl", "taco tray"]
WEATHERS = ["normal", "rainy", "hot", "cold", "windy"]
EVENTS = ["normal", "field_trip", "exam_day", "assembly", "sports_away", "club_fair", "early_release"]
INTERVENTIONS = ["none", "share_cooler", "two_wave_staging", "line_reposition", "after_school_route", "combo"]


def _school_days(start: str = "2025-08-12", n_days: int = 240) -> pd.DatetimeIndex:
    days = pd.bdate_range(start=start, periods=int(n_days * 1.35) + 30)
    # remove a few synthetic breaks by masking fixed windows
    breaks = []
    for d in days:
        if (d.month == 11 and 24 <= d.day <= 28) or (d.month == 12 and d.day >= 22) or (d.month == 1 and d.day <= 3):
            breaks.append(False)
        else:
            breaks.append(True)
    days = days[breaks]
    return days[:n_days]


def generate_cafeteria_history(n_days: int = 240, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    days = _school_days(n_days=n_days)

    # Persistent item-level popularity drift makes the simulator less toy-like.
    item_popularity_noise = {item.menu_item: rng.normal(1.0, 0.05) for item in ITEMS}

    for day_idx, date in enumerate(days):
        dow = date.day_name()
        week_phase = np.sin(day_idx / 17.0) * 0.03
        event_tag = rng.choice(EVENTS, p=[0.62, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04])
        weather_tag = rng.choice(WEATHERS, p=[0.55, 0.16, 0.12, 0.10, 0.07])
        entree = rng.choice(ENTREES, p=[0.20, 0.12, 0.22, 0.25, 0.11, 0.10])

        base_att = 1010
        dow_effect = {"Monday": -0.015, "Tuesday": 0.01, "Wednesday": 0.0, "Thursday": 0.005, "Friday": -0.035}.get(dow, 0.0)
        weather_att_effect = {"normal": 0.0, "rainy": -0.045, "hot": -0.015, "cold": -0.02, "windy": -0.012}[weather_tag]
        event_att_effect = {"normal": 0.0, "field_trip": -0.12, "exam_day": -0.055, "assembly": -0.025, "sports_away": -0.04, "club_fair": 0.02, "early_release": -0.10}[event_tag]
        expected_attendance = int(np.clip(base_att * (1 + dow_effect + weather_att_effect + event_att_effect + week_phase) + rng.normal(0, 28), 700, 1120))
        actual_attendance = int(np.clip(expected_attendance + rng.normal(0, 25), 650, 1140))

        # After-school demand is highest on Tue-Thu, lower on early release and bad weather.
        afterschool_activity_count = int(np.clip(
            135
            + {"Monday": 5, "Tuesday": 45, "Wednesday": 35, "Thursday": 40, "Friday": -35}.get(dow, 0)
            + {"field_trip": -15, "exam_day": -25, "assembly": 10, "sports_away": 25, "club_fair": 70, "early_release": -80, "normal": 0}[event_tag]
            + {"rainy": -12, "hot": 2, "cold": -8, "windy": -4, "normal": 0}[weather_tag]
            + rng.normal(0, 16),
            20, 270
        ))
        share_table_monitor = int(rng.random() < (0.68 if dow in ["Tuesday", "Wednesday", "Thursday"] else 0.48))
        cooler_capacity = int(rng.choice([45, 60, 75, 90, 110], p=[0.15, 0.25, 0.25, 0.25, 0.10]) if share_table_monitor else 0)

        for lunch_period in [1, 2]:
            period_share = 0.52 if lunch_period == 1 else 0.48
            if event_tag == "field_trip" and lunch_period == 2:
                period_share -= 0.08
            if event_tag == "assembly" and lunch_period == 1:
                period_share += 0.04
            actual_lp_att = int(np.clip(actual_attendance * period_share + rng.normal(0, 18), 280, 620))
            expected_lp_att = int(np.clip(expected_attendance * period_share + rng.normal(0, 10), 280, 620))

            # Interventions are not randomized uniformly; they respond to known bad conditions.
            if share_table_monitor and event_tag in ["field_trip", "exam_day"] and weather_tag in ["rainy", "cold", "normal"]:
                intervention_used = rng.choice(["share_cooler", "combo", "two_wave_staging"], p=[0.45, 0.35, 0.20])
            elif share_table_monitor and dow in ["Tuesday", "Thursday"]:
                intervention_used = rng.choice(INTERVENTIONS, p=[0.32, 0.25, 0.16, 0.12, 0.07, 0.08])
            else:
                intervention_used = rng.choice(INTERVENTIONS, p=[0.62, 0.07, 0.10, 0.11, 0.04, 0.06])

            for item in ITEMS:
                if item.component_type == "entree" and item.menu_item not in [entree, "veggie wrap"]:
                    # Only include the main entree and vegetarian alternative as prepared options.
                    continue

                line_position = item.line_position_default
                if intervention_used in ["line_reposition", "combo"] and item.component_type in ["fruit", "vegetable", "milk"]:
                    line_position = "after_entree"

                item_factor = item_popularity_noise[item.menu_item]
                take_rate = item.base_take_rate * item_factor
                # Entree interactions create the "ghost component" pattern.
                if entree == "pasta bowl" and item.component_type == "milk":
                    take_rate *= 1.07
                if entree == "cheese pizza" and item.component_type == "milk":
                    take_rate *= 1.11
                if entree == "veggie wrap" and item.menu_item in ["apple", "baby carrots"]:
                    take_rate *= 0.90
                if event_tag == "early_release" and item.component_type in ["fruit", "milk"]:
                    take_rate *= 0.92
                if weather_tag == "hot" and item.component_type == "milk":
                    take_rate *= 1.05
                if lunch_period == 2 and item.menu_item in ["white milk", "apple", "fruit cup"]:
                    take_rate *= 1.03

                # Planned/staged counts are based on expected attendance, with institutional overage.
                planned_count = int(np.clip(expected_lp_att * take_rate * rng.normal(1.08, 0.05), 5, 520))
                staged_count = planned_count
                if intervention_used in ["two_wave_staging", "combo"] and item.cold_chain_required:
                    staged_count = int(planned_count * rng.uniform(0.78, 0.90))

                true_demand = int(np.clip(actual_lp_att * take_rate * rng.normal(1.0, 0.08), 0, 560))
                selected_count = int(min(staged_count, max(0, true_demand + rng.normal(0, 6))))
                stockout_occurred = int(true_demand > staged_count + 5)

                rr = item.base_return_rate
                if item.component_type == "milk" and entree in ["pasta bowl", "cheese pizza"]:
                    rr += 0.045
                if item.menu_item == "apple" and line_position == "before_cashier":
                    rr += 0.055
                if line_position == "after_entree":
                    rr -= 0.035
                if lunch_period == 2:
                    rr += 0.025
                if weather_tag == "rainy":
                    rr += 0.018
                if event_tag in ["field_trip", "exam_day", "early_release"]:
                    rr += 0.025
                if intervention_used in ["share_cooler", "combo"]:
                    # Share table does not reduce returns, but it tends to improve logging/visibility.
                    rr += 0.005
                if item.component_type == "entree" and item.menu_item == entree:
                    rr *= 0.55

                rr = float(np.clip(rr + rng.normal(0, 0.015), 0.01, 0.42))
                unopened_return_count = int(rng.binomial(max(0, selected_count), rr)) if item.sealed_or_whole else 0
                tray_waste_kg = float(max(0, rng.normal(selected_count * item.kg_per_item * (0.025 if item.component_type == "entree" else 0.006), 1.2)))

                pickup_rate = 0.0
                if item.sealed_or_whole:
                    pickup_rate = 0.12
                    if share_table_monitor and intervention_used in ["share_cooler", "combo"]:
                        pickup_rate += 0.42
                    elif share_table_monitor:
                        pickup_rate += 0.22
                    if item.cold_chain_required and cooler_capacity <= 0:
                        pickup_rate *= 0.25
                    if item.component_type in ["fruit", "grain"]:
                        pickup_rate += 0.07
                share_table_pickup_count = int(min(unopened_return_count, rng.binomial(unopened_return_count, np.clip(pickup_rate, 0, 0.82))))

                afterschool_rate = 0.0
                if item.sealed_or_whole and afterschool_activity_count > 80:
                    afterschool_rate = min(0.42, afterschool_activity_count / 450.0)
                    if intervention_used in ["after_school_route", "combo"]:
                        afterschool_rate += 0.16
                    if item.cold_chain_required and cooler_capacity <= 0:
                        afterschool_rate *= 0.1
                remaining_after_share = max(0, unopened_return_count - share_table_pickup_count)
                afterschool_pickup_count = int(min(remaining_after_share, rng.binomial(remaining_after_share, np.clip(afterschool_rate, 0, 0.66))))
                discarded_count = max(0, unopened_return_count - share_table_pickup_count - afterschool_pickup_count)

                estimated_kg_diverted = (share_table_pickup_count + afterschool_pickup_count) * item.kg_per_item
                estimated_co2e_avoided = (share_table_pickup_count + afterschool_pickup_count) * item.co2e_kg_per_item
                cost_protected = (share_table_pickup_count + afterschool_pickup_count) * item.cost_per_item

                ghost_component_rate = unopened_return_count / selected_count if selected_count else 0.0
                high_ghost_risk = int(item.sealed_or_whole and unopened_return_count >= 20 and ghost_component_rate >= 0.12)

                rows.append({
                    "date": date.date().isoformat(),
                    "day_index": day_idx,
                    "day_of_week": dow,
                    "lunch_period": lunch_period,
                    "menu_item": item.menu_item,
                    "item_category": item.item_category,
                    "component_type": item.component_type,
                    "sealed_or_whole": item.sealed_or_whole,
                    "cold_chain_required": item.cold_chain_required,
                    "planned_count": planned_count,
                    "staged_count": staged_count,
                    "selected_count": selected_count,
                    "unopened_return_count": unopened_return_count,
                    "share_table_pickup_count": share_table_pickup_count,
                    "afterschool_pickup_count": afterschool_pickup_count,
                    "discarded_count": discarded_count,
                    "tray_waste_kg": round(tray_waste_kg, 2),
                    "expected_attendance": expected_attendance,
                    "actual_attendance": actual_attendance,
                    "expected_lunch_period_attendance": expected_lp_att,
                    "actual_lunch_period_attendance": actual_lp_att,
                    "weather_tag": weather_tag,
                    "event_tag": event_tag,
                    "entree": entree,
                    "entree_popularity_score": round(float(np.clip(take_rate, 0.02, 1.0)), 3),
                    "line_position": line_position,
                    "share_table_monitor": share_table_monitor,
                    "cooler_capacity": cooler_capacity,
                    "afterschool_activity_count": afterschool_activity_count,
                    "stockout_occurred": stockout_occurred,
                    "intervention_used": intervention_used,
                    "estimated_kg_diverted": round(estimated_kg_diverted, 3),
                    "estimated_co2e_avoided": round(estimated_co2e_avoided, 3),
                    "estimated_cost_protected": round(cost_protected, 2),
                    "kg_per_item": item.kg_per_item,
                    "cost_per_item": item.cost_per_item,
                    "co2e_kg_per_item": item.co2e_kg_per_item,
                    "ghost_component_rate": round(ghost_component_rate, 4),
                    "high_ghost_risk": high_ghost_risk,
                })

    return pd.DataFrame(rows)


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "data" / "second_bell_synthetic_cafeteria.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_cafeteria_history(n_days=540, seed=RNG_SEED)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df):,} synthetic item-period rows to {output}")
    print(df.head())


if __name__ == "__main__":
    main()
