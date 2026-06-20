"""Sensor-fusion digital twin for cafeteria rescue decisions."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


SENSOR_SOURCES = [
    "serving-line camera",
    "tray-return platform scale",
    "serving-bin load cell",
    "share-table cooler temperature logger",
    "share-table cooler door sensor",
    "kitchen production and staging counts",
    "point-of-sale meal counts",
    "anonymous after-school attendance counts",
    "final compost/disposal bin scale",
]


@dataclass(frozen=True)
class DigitalTwinOutputs:
    sensor_inventory: pd.DataFrame
    observations: pd.DataFrame
    item_estimates: pd.DataFrame
    world_state: pd.DataFrame
    bridge_notes: pd.DataFrame


def _stable_noise(key: str, scale: float) -> float:
    digest = sha256(key.encode("utf-8")).hexdigest()
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    return (unit - 0.5) * 2.0 * scale


def _weighted_estimate(observations: Iterable[dict]) -> dict:
    rows = list(observations)
    weights = np.array([1.0 / max(float(row["sigma"]), 0.1) ** 2 for row in rows], dtype=float)
    values = np.array([float(row["observed_count"]) for row in rows], dtype=float)
    estimate = float(np.sum(weights * values) / np.sum(weights))
    model_spread = float(max(values) - min(values)) if len(values) else 0.0
    measurement_sigma = float(np.sqrt(1.0 / np.sum(weights)))
    uncertainty = float(max(measurement_sigma, model_spread / 2.8, 1.0))
    return {
        "estimate": estimate,
        "uncertainty": uncertainty,
        "sensor_spread": model_spread,
        "sensor_count": len(rows),
    }


def build_sensor_inventory(share_table_monitor: bool) -> pd.DataFrame:
    status = "online" if share_table_monitor else "offline for cold-chain gates"
    rows = [
        {
            "sensor": "serving-line camera",
            "signal": "sealed-item and serving-bin visual detections",
            "why_vision_cannot_replace": "Vision is useful, but it is reconciled against physical sensors instead of trusted alone.",
            "dashboard_use": "Detects visible reduction and sealed returns.",
            "status": "online",
        },
        {
            "sensor": "tray-return platform scale",
            "signal": "mass delta at the return window",
            "why_vision_cannot_replace": "Hidden weight verifies returns even when hands, trays, or cartons block the camera.",
            "dashboard_use": "Converts sealed-item mass to count estimates.",
            "status": "online",
        },
        {
            "sensor": "serving-bin load cell",
            "signal": "bin mass before and after each lunch wave",
            "why_vision_cannot_replace": "Measures actual depletion inside bins, including items occluded from view.",
            "dashboard_use": "Detects whether a component is reducing or sitting untouched.",
            "status": "online",
        },
        {
            "sensor": "share-table cooler temperature logger",
            "signal": "continuous cooler temperature",
            "why_vision_cannot_replace": "Food safety depends on temperature history, not visual appearance.",
            "dashboard_use": "Blocks cold-chain rescue when temperature is unsafe.",
            "status": status,
        },
        {
            "sensor": "share-table cooler door sensor",
            "signal": "open count and open-duration minutes",
            "why_vision_cannot_replace": "Door-open events explain temperature spikes and custody breaks.",
            "dashboard_use": "Adds manager-review warnings to cold items.",
            "status": status,
        },
        {
            "sensor": "kitchen production and staging counts",
            "signal": "manager-entered batch and staging counts",
            "why_vision_cannot_replace": "Kitchen staff know production batches before they reach a camera view.",
            "dashboard_use": "Creates the digital twin's starting inventory.",
            "status": "online",
        },
        {
            "sensor": "point-of-sale meal counts",
            "signal": "anonymous meal totals by lunch period",
            "why_vision_cannot_replace": "POS totals anchor demand without storing student identities.",
            "dashboard_use": "Reconciles visual counts against served-meal totals.",
            "status": "online",
        },
        {
            "sensor": "anonymous after-school attendance counts",
            "signal": "aggregate club, tutoring, athletics, and activity demand",
            "why_vision_cannot_replace": "Students who need snacks are outside the cafeteria camera frame.",
            "dashboard_use": "Caps same-day after-school routing demand.",
            "status": "online",
        },
        {
            "sensor": "final compost/disposal bin scale",
            "signal": "discarded mass after recovery closes",
            "why_vision_cannot_replace": "A camera sees the top of the bin, while the scale measures total disposal.",
            "dashboard_use": "Audits impact receipt and missed recovery.",
            "status": "online",
        },
    ]
    return pd.DataFrame(rows)


def _item_observations(row: pd.Series) -> List[Dict]:
    base = float(row["predicted_return_q50"])
    item = str(row["menu_item"])
    lunch = int(row["lunch_period"])
    prior_sigma = max(4.0, float(row["predicted_return_q90"] - row["predicted_return_q10"]) / 2.0)
    obs = [
        {
            "sensor": "model prior",
            "metric": "recoverable_return_count",
            "observed_count": round(base, 1),
            "sigma": round(prior_sigma, 1),
            "used_in_estimate": True,
        },
        {
            "sensor": "serving-line camera",
            "metric": "recoverable_return_count",
            "observed_count": round(max(0.0, base + _stable_noise(f"{item}-{lunch}-camera", 5.5)), 1),
            "sigma": 5.5,
            "used_in_estimate": True,
        },
        {
            "sensor": "tray-return platform scale",
            "metric": "recoverable_return_count",
            "observed_count": round(max(0.0, base + _stable_noise(f"{item}-{lunch}-tray-scale", 3.0)), 1),
            "sigma": 3.0,
            "used_in_estimate": True,
        },
        {
            "sensor": "serving-bin load cell",
            "metric": "recoverable_return_count",
            "observed_count": round(max(0.0, base + _stable_noise(f"{item}-{lunch}-bin-load", 4.0)), 1),
            "sigma": 4.0,
            "used_in_estimate": True,
        },
        {
            "sensor": "kitchen production and staging counts",
            "metric": "recoverable_return_count",
            "observed_count": round(max(0.0, base + _stable_noise(f"{item}-{lunch}-kitchen", 6.5)), 1),
            "sigma": 6.5,
            "used_in_estimate": True,
        },
        {
            "sensor": "point-of-sale meal counts",
            "metric": "recoverable_return_count",
            "observed_count": round(max(0.0, base + _stable_noise(f"{item}-{lunch}-pos", 7.0)), 1),
            "sigma": 7.0,
            "used_in_estimate": True,
        },
        {
            "sensor": "final compost/disposal bin scale",
            "metric": "missed_recovery_count",
            "observed_count": round(max(0.0, base * 0.42 + _stable_noise(f"{item}-{lunch}-disposal", 2.5)), 1),
            "sigma": 2.5,
            "used_in_estimate": False,
        },
    ]
    for entry in obs:
        entry.update(
            {
                "menu_item": item,
                "lunch_period": lunch,
                "component_type": row["component_type"],
            }
        )
    return obs


def build_digital_twin(
    pred_df: pd.DataFrame,
    afterschool_predicted_count: int,
    afterschool_low: int,
    afterschool_high: int,
    share_table_monitor: bool,
) -> DigitalTwinOutputs:
    observation_rows: List[Dict] = []
    estimate_rows: List[Dict] = []

    for _, row in pred_df.iterrows():
        obs = _item_observations(row)
        observation_rows.extend(obs)
        used = [entry for entry in obs if entry["used_in_estimate"]]
        fused = _weighted_estimate(used)
        low = max(0.0, fused["estimate"] - fused["uncertainty"])
        high = fused["estimate"] + fused["uncertainty"]
        estimate_rows.append(
            {
                "menu_item": row["menu_item"],
                "lunch_period": int(row["lunch_period"]),
                "component_type": row["component_type"],
                "digital_twin_estimate": round(fused["estimate"], 1),
                "digital_twin_uncertainty": round(fused["uncertainty"], 1),
                "digital_twin_low": round(low, 1),
                "digital_twin_high": round(high, 1),
                "sensor_spread": round(fused["sensor_spread"], 1),
                "sensor_count": int(fused["sensor_count"]),
                "inconsistency_flag": fused["sensor_spread"] > max(8.0, fused["uncertainty"] * 2.4),
            }
        )

    observations = pd.DataFrame(observation_rows)
    estimates = pd.DataFrame(estimate_rows)
    cooler_temp = 38.4 if share_table_monitor else 45.8
    door_minutes = 7 if share_table_monitor else 0
    total_return_mid = float(estimates["digital_twin_estimate"].sum())
    total_return_uncertainty = float(np.sqrt(np.sum(np.square(estimates["digital_twin_uncertainty"]))))

    world_state = pd.DataFrame(
        [
            {
                "zone": "serving line",
                "state": "staged inventory",
                "estimate": f"{int(pred_df['staged_count'].sum())} items",
                "uncertainty": "+/- staff count audit",
                "primary_sensors": "production counts + serving-bin load cells + POS",
            },
            {
                "zone": "tray return",
                "state": "recoverable sealed returns",
                "estimate": f"{total_return_mid:.0f} items",
                "uncertainty": f"+/- {total_return_uncertainty:.0f}",
                "primary_sensors": "camera + tray-return scale + bin load cells",
            },
            {
                "zone": "share-table cooler",
                "state": "cold-chain gate",
                "estimate": f"{cooler_temp:.1f} F, {door_minutes} door-open minutes",
                "uncertainty": "blocked if monitor is offline",
                "primary_sensors": "temperature logger + door-open sensor",
            },
            {
                "zone": "after-school programs",
                "state": "anonymous snack demand",
                "estimate": f"{afterschool_predicted_count} students",
                "uncertainty": f"{afterschool_low}-{afterschool_high}",
                "primary_sensors": "two-week linear forecast + aggregate attendance counts",
            },
            {
                "zone": "compost/disposal",
                "state": "missed recovery audit",
                "estimate": f"{observations[observations['metric'] == 'missed_recovery_count']['observed_count'].sum():.0f} items",
                "uncertainty": "+/- disposal scale conversion",
                "primary_sensors": "final bin scale",
            },
        ]
    )

    bridge_notes = pd.DataFrame(
        [
            {
                "integration": "Mac Force Touch prototype bridge",
                "source": "https://github.com/KrishKrosh/TrackWeight",
                "use": "Hackathon-scale prototype for load-cell-like pressure readings on a MacBook trackpad.",
                "production_note": "Use calibrated food-service load cells for deployment; trackpad is an importable prototype path, not a certified scale.",
            },
            {
                "integration": "OpenMultitouchSupport",
                "source": "https://github.com/KrishKrosh/OpenMultitouchSupport",
                "use": "Swift package path for pressure, density, position, and touch state streams.",
                "production_note": "Bridge Swift readings into Second Bell as CSV, MQTT, or a local HTTP adapter.",
            },
        ]
    )

    return DigitalTwinOutputs(
        sensor_inventory=build_sensor_inventory(share_table_monitor),
        observations=observations,
        item_estimates=estimates,
        world_state=world_state,
        bridge_notes=bridge_notes,
    )
