"""Leakage-safe seasonal naive baseline for retail unit sales.

This stage evaluates observed historical targets, not live future predictions.
The forecast horizon is explicitly passed by the caller, never fixed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["date", "store_nbr", "family"]


def predict_seasonal_naive(
    data: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    """Score each observed target using an exact past calendar reference.

    For each target date, select the same day in the latest complete 7-day
    calendar cycle ending at the cutoff. Longer horizons reuse that anchor.
    Missing reference dates remain NaN, never silently become zero sales.
    Historical targets after the cutoff are never used to predict.
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    required = set(KEYS + ["sales"])
    if missing := required.difference(data.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = data.loc[:, KEYS + ["sales"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if frame[KEYS + ["sales"]].isna().any().any():
        raise ValueError("Input contains missing dates, series keys or sales")
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate date/store_nbr/family combinations")
    if (frame["sales"] < 0).any():
        raise ValueError("Sales targets must be nonnegative")

    cutoff = pd.Timestamp(cutoff).normalize()
    end = cutoff + pd.Timedelta(days=horizon)
    future = frame.loc[(frame["date"] > cutoff) & (frame["date"] <= end)].copy()
    if future.empty:
        raise ValueError("No observed targets inside the evaluation interval")

    offsets = (future["date"] - cutoff).dt.days
    cycles_back = ((offsets - 1) // 7) + 1
    future["reference_date"] = future["date"] - pd.to_timedelta(
        cycles_back * 7, unit="D"
    )

    # Restrict the reference table to dates known at the forecast cutoff.
    # Only the required past calendar dates are included in the join.
    reference_dates = future["reference_date"].unique()
    historical = frame.loc[
        (frame["date"] <= cutoff) & (frame["date"].isin(reference_dates)),
        KEYS + ["sales"],
    ].rename(columns={"date": "reference_date", "sales": "prediction"})

    result = future.merge(
        historical,
        on=["reference_date", "store_nbr", "family"],
        how="left",
        validate="many_to_one",
    )
    return result.sort_values(KEYS).reset_index(drop=True)


def calculate_metrics(predictions: pd.DataFrame) -> dict:
    """Calculate metrics only where both target and prediction exist."""
    observed = len(predictions)
    scored = predictions.dropna(subset=["prediction"])
    metrics = {
        "observed_targets": int(observed),
        "scored_targets": int(len(scored)),
        "coverage": float(len(scored) / observed) if observed else 0.0,
        "mae": None,
        "rmsle": None,
    }
    if len(scored):
        true = scored["sales"].to_numpy(dtype="float64")
        predicted = scored["prediction"].to_numpy(dtype="float64")
        metrics["mae"] = float(np.mean(np.abs(true - predicted)))
        metrics["rmsle"] = float(
            np.sqrt(np.mean((np.log1p(true) - np.log1p(predicted)) ** 2))
        )
    return metrics
