"""Rolling-origin evaluation of a simple seasonal naive forecasting baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.baseline import KEYS, calculate_metrics, predict_seasonal_naive


def run_backtest(
    data: pd.DataFrame,
    horizon: int,
    origins: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate disjoint forward windows using only information at each cutoff.

    Each more recent origin may use data observed before that origin, including
    earlier scored intervals, as it would in a realistic sequential simulation.
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(origins, int) or isinstance(origins, bool) or origins <= 0:
        raise ValueError("origins must be a positive integer")
    if data.empty:
        raise ValueError("No source records")

    final_date = pd.to_datetime(data["date"]).max().normalize()
    earliest = pd.to_datetime(data["date"]).min().normalize()
    cutoffs = [final_date - pd.Timedelta(days=horizon * i) for i in range(origins, 0, -1)]
    if cutoffs[0] - pd.Timedelta(days=7) < earliest:
        raise ValueError("Not enough history for the requested horizons and origins")

    chunks: list[pd.DataFrame] = []
    per_origin: list[dict] = []
    for cutoff in cutoffs:
        predictions = predict_seasonal_naive(data, cutoff=cutoff, horizon=horizon)
        predictions["cutoff"] = cutoff
        chunks.append(predictions)
        per_origin.append({
            "cutoff": str(cutoff.date()),
            "target_start": str((cutoff + pd.Timedelta(days=1)).date()),
            "target_end": str((cutoff + pd.Timedelta(days=horizon)).date()),
            **calculate_metrics(predictions),
        })

    scored_windows = pd.concat(chunks, ignore_index=True)
    report = {
        "method": "seasonal_naive_calendar_aligned",
        "forecast_horizon": horizon,
        "evaluation_origins": origins,
        "per_origin": per_origin,
        "overall": calculate_metrics(scored_windows),
        "note": (
            "Only observed target rows are evaluated; missing historical "
            "reference dates are not filled with zero. Each cutoff uses only "
            "data available by that date."
        ),
    }
    return scored_windows, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a seasonal naive forecasting baseline")
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--horizon", type=int, required=True, help="Forecast length in calendar days")
    parser.add_argument("--origins", type=int, default=3, help="Number of non-overlapping evaluation cutoffs")
    args = parser.parse_args()

    if not args.features.is_file():
        parser.error(f"Feature file missing: {args.features}. Run python -m src.data_pipeline first.")
    data = pd.read_parquet(args.features, columns=KEYS + ["sales"])
    predictions, report = run_backtest(data, args.horizon, args.origins)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.output_dir / "baseline_backtest.parquet", index=False)
    (args.output_dir / "baseline_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved predictions to: {args.output_dir / 'baseline_backtest.parquet'}")


if __name__ == "__main__":
    main()
