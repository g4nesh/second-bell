"""Two-week after-school demand forecasting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


TWO_WEEK_SCHOOL_DAYS = 10
MIN_AFTER_SCHOOL_COUNT = 20
MAX_AFTER_SCHOOL_COUNT = 270


@dataclass(frozen=True)
class AfterschoolForecast:
    predicted_count: int
    low: int
    high: int
    model_type: str
    training_days: int
    mean_absolute_error: float
    r2: float
    coefficients: List[Dict[str, float]]
    history_window: pd.DataFrame


def daily_after_school_counts(history: pd.DataFrame) -> pd.DataFrame:
    """Return one anonymous after-school attendance count per synthetic school day."""
    required = {
        "date",
        "day_of_week",
        "weather_tag",
        "event_tag",
        "expected_attendance",
        "afterschool_activity_count",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Missing after-school forecast columns: {sorted(missing)}")

    daily = (
        history.sort_values(["date"])
        .drop_duplicates("date")
        .loc[
            :,
            [
                "date",
                "day_of_week",
                "weather_tag",
                "event_tag",
                "expected_attendance",
                "afterschool_activity_count",
            ],
        ]
        .reset_index(drop=True)
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.rename(columns={"afterschool_activity_count": "actual_after_school_students"})
    return daily


def _encoded_features(window: pd.DataFrame, scenario: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = window[["day_of_week", "weather_tag", "event_tag", "expected_attendance"]].copy()
    train["trend_index"] = np.arange(len(train), dtype=float)
    train["expected_attendance"] = train["expected_attendance"].astype(float) / 1000.0

    predict = pd.DataFrame([scenario])
    predict["trend_index"] = float(len(train))
    predict["expected_attendance"] = predict["expected_attendance"].astype(float) / 1000.0

    combined = pd.concat([train, predict], ignore_index=True)
    encoded = pd.get_dummies(
        combined,
        columns=["day_of_week", "weather_tag", "event_tag"],
        dtype=float,
    )
    return encoded.iloc[: len(train)].copy(), encoded.iloc[len(train) :].copy()


def _top_coefficients(model: LinearRegression, feature_names: List[str], limit: int = 5) -> List[Dict[str, float]]:
    values = [
        {"feature": feature, "coefficient": round(float(coef), 3)}
        for feature, coef in zip(feature_names, model.coef_)
        if abs(float(coef)) > 0.001
    ]
    return sorted(values, key=lambda row: abs(row["coefficient"]), reverse=True)[:limit]


def forecast_after_school_count(
    history: pd.DataFrame,
    *,
    day_of_week: str,
    weather_tag: str,
    event_tag: str,
    expected_attendance: int,
    training_days: int = TWO_WEEK_SCHOOL_DAYS,
) -> AfterschoolForecast:
    """Predict after-school students from the last two school weeks of aggregate counts."""
    daily = daily_after_school_counts(history)
    window = daily.tail(training_days).reset_index(drop=True)
    if len(window) < 2:
        fallback = int(
            np.clip(
                daily["actual_after_school_students"].mean(),
                MIN_AFTER_SCHOOL_COUNT,
                MAX_AFTER_SCHOOL_COUNT,
            )
        )
        return AfterschoolForecast(
            predicted_count=fallback,
            low=max(MIN_AFTER_SCHOOL_COUNT, fallback - 15),
            high=min(MAX_AFTER_SCHOOL_COUNT, fallback + 15),
            model_type="Fallback mean from available aggregate after-school counts",
            training_days=len(window),
            mean_absolute_error=0.0,
            r2=0.0,
            coefficients=[],
            history_window=window,
        )

    scenario = {
        "day_of_week": day_of_week,
        "weather_tag": weather_tag,
        "event_tag": event_tag,
        "expected_attendance": expected_attendance,
    }
    x_train, x_today = _encoded_features(window, scenario)
    y_train = window["actual_after_school_students"].astype(float).to_numpy()

    model = LinearRegression()
    model.fit(x_train, y_train)

    fitted = model.predict(x_train)
    prediction = float(model.predict(x_today)[0])
    residuals = y_train - fitted
    mae = float(np.mean(np.abs(residuals)))
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 2 else mae
    uncertainty = max(8.0, residual_std, mae)

    clipped = int(round(np.clip(prediction, MIN_AFTER_SCHOOL_COUNT, MAX_AFTER_SCHOOL_COUNT)))
    low = int(round(np.clip(clipped - 1.15 * uncertainty, MIN_AFTER_SCHOOL_COUNT, MAX_AFTER_SCHOOL_COUNT)))
    high = int(round(np.clip(clipped + 1.15 * uncertainty, MIN_AFTER_SCHOOL_COUNT, MAX_AFTER_SCHOOL_COUNT)))

    display_window = window.copy()
    display_window["date"] = display_window["date"].dt.date.astype(str)
    display_window = display_window.rename(
        columns={
            "day_of_week": "day",
            "weather_tag": "weather",
            "event_tag": "event",
            "expected_attendance": "expected_lunch_attendance",
        }
    )

    return AfterschoolForecast(
        predicted_count=clipped,
        low=low,
        high=max(high, low),
        model_type=f"LinearRegression over the last {len(window)} school days",
        training_days=len(window),
        mean_absolute_error=round(mae, 1),
        r2=round(float(model.score(x_train, y_train)), 3),
        coefficients=_top_coefficients(model, list(x_train.columns)),
        history_window=display_window,
    )
