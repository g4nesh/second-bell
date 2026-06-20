"""Computer-vision deployment scaffold for cafeteria reduction detection."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image


VISION_CLASSES = [
    "unopened_white_milk",
    "unopened_chocolate_milk",
    "whole_fruit",
    "sealed_side",
    "serving_bin_remaining",
    "tray_return_sealed_item",
    "opened_tray_waste",
    "empty_slot",
]


class CafeteriaVisionModel:
    """Optional YOLO-compatible wrapper for live camera inference.

    The Streamlit demo works without a heavy vision dependency. If a trained
    Ultralytics-compatible model is placed at ``weights_path`` and the optional
    ``ultralytics`` package is installed, this wrapper can run image, video, or
    webcam inference and return raw detection objects.
    """

    def __init__(self, weights_path: str = "models/cafeteria_yolo.pt", confidence: float = 0.35):
        self.weights_path = Path(weights_path)
        self.confidence = confidence
        self._model = None

    @property
    def available(self) -> bool:
        return self.weights_path.exists()

    def load(self) -> None:
        if not self.available:
            raise FileNotFoundError(f"Vision weights not found: {self.weights_path}")
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install optional package 'ultralytics' to run live CV inference.") from exc
        self._model = YOLO(str(self.weights_path))

    def predict(self, source: str | int):
        if self._model is None:
            self.load()
        return self._model.predict(source=source, conf=self.confidence, stream=True)


class ImageReductionVisionModel:
    """Small deployable baseline that compares before/after serving-bin images."""

    def __init__(self, foreground_cutoff: int = 244, min_channel_spread: int = 12):
        self.foreground_cutoff = foreground_cutoff
        self.min_channel_spread = min_channel_spread

    def compare(self, before_source, after_source) -> Dict[str, float | str | int]:
        before = _to_rgb_array(before_source)
        after = _to_rgb_array(after_source)
        if before.shape != after.shape:
            after = np.asarray(
                Image.fromarray(after).resize((before.shape[1], before.shape[0])),
                dtype=np.uint8,
            )

        before_mask = self._foreground_mask(before)
        after_mask = self._foreground_mask(after)
        before_area = int(before_mask.sum())
        after_area = int(after_mask.sum())
        if before_area <= 0:
            reduction_pct = 0.0
            confidence = 0.0
            status = "no foreground detected"
        else:
            reduction_pct = max(0.0, min(1.0, (before_area - after_area) / before_area))
            changed_area = int(np.logical_xor(before_mask, after_mask).sum())
            confidence = min(0.98, max(0.15, changed_area / max(before_area, 1)))
            if reduction_pct >= 0.18:
                status = "reduced"
            elif reduction_pct <= 0.05:
                status = "not reduced"
            else:
                status = "minor change"

        return {
            "before_foreground_pixels": before_area,
            "after_foreground_pixels": after_area,
            "visual_reduction_pct": round(reduction_pct * 100, 1),
            "confidence": round(float(confidence), 2),
            "status": status,
        }

    def _foreground_mask(self, image: np.ndarray) -> np.ndarray:
        channel_max = image.max(axis=2).astype(np.int16)
        channel_min = image.min(axis=2).astype(np.int16)
        mean = image.mean(axis=2)
        colorful = (channel_max - channel_min) >= self.min_channel_spread
        not_background = mean <= self.foreground_cutoff
        return colorful | not_background


def _to_rgb_array(source) -> np.ndarray:
    if isinstance(source, np.ndarray):
        arr = source
    else:
        arr = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr.astype(np.uint8)


def vision_deployment_card() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layer": "labels",
                "implementation": ", ".join(VISION_CLASSES),
                "why_it_matters": "Separates sealed recoverable food from opened tray waste and remaining serving-bin inventory.",
            },
            {
                "layer": "model",
                "implementation": "Baseline before/after reduction model plus YOLO detector or segmentation weights when labeled frames exist",
                "why_it_matters": "Detects what is reducing, what is untouched, and what returns sealed.",
            },
            {
                "layer": "export",
                "implementation": "ONNX/CoreML/TensorRT export from trained weights",
                "why_it_matters": "Makes the model deployable on a laptop, Apple device, or edge GPU without changing dashboard code.",
            },
            {
                "layer": "fusion",
                "implementation": "CV counts are reconciled with scales, POS, cooler sensors, and kitchen counts",
                "why_it_matters": "The dashboard shows uncertainty instead of pretending the camera is always correct.",
            },
        ]
    )


def export_commands(weights_path: str = "models/cafeteria_yolo.pt") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target": "ONNX laptop/edge runtime",
                "command": f"yolo export model={weights_path} format=onnx dynamic=True",
            },
            {
                "target": "Apple CoreML package",
                "command": f"yolo export model={weights_path} format=coreml half=True",
            },
            {
                "target": "NVIDIA TensorRT engine",
                "command": f"yolo export model={weights_path} format=engine half=True",
            },
        ]
    )


def simulate_reduction_detection(pred_df: pd.DataFrame, twin_estimates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    twin_lookup: Dict[tuple, float] = {}
    if twin_estimates is not None and not twin_estimates.empty:
        for _, row in twin_estimates.iterrows():
            twin_lookup[(row["menu_item"], int(row["lunch_period"]))] = float(row["digital_twin_estimate"])

    rows: List[Dict] = []
    for _, row in pred_df.iterrows():
        staged = float(row["staged_count"])
        if staged <= 0:
            continue
        key = (row["menu_item"], int(row["lunch_period"]))
        return_estimate = twin_lookup.get(key, float(row["predicted_return_q50"]))
        expected_selected = min(
            staged,
            max(0.0, float(row["expected_lunch_period_attendance"]) * float(row["entree_popularity_score"])),
        )
        remaining_estimate = max(0.0, staged - expected_selected + return_estimate * 0.18)
        reduction_pct = max(0.0, min(1.0, (staged - remaining_estimate) / staged))
        if reduction_pct >= 0.68:
            status = "reducing fast"
        elif reduction_pct <= 0.28:
            status = "not reducing"
        else:
            status = "normal drawdown"
        rows.append(
            {
                "menu_item": row["menu_item"],
                "lunch_period": int(row["lunch_period"]),
                "vision_class": _class_for_item(str(row["menu_item"]), str(row["component_type"])),
                "before_count": round(staged, 1),
                "current_detected_count": round(remaining_estimate, 1),
                "visual_reduction_pct": round(reduction_pct * 100, 1),
                "status": status,
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "visual_reduction_pct"], ascending=[True, True])


def _class_for_item(menu_item: str, component_type: str) -> str:
    if menu_item == "white milk":
        return "unopened_white_milk"
    if menu_item == "chocolate milk":
        return "unopened_chocolate_milk"
    if component_type == "fruit":
        return "whole_fruit" if menu_item in {"apple", "banana"} else "sealed_side"
    if component_type in {"vegetable", "protein side", "grain"}:
        return "sealed_side"
    return "serving_bin_remaining"
