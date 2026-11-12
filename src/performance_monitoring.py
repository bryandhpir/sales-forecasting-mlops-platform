
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline import calculate_metrics


REQUIRED = {
    "cutoff", "date", "store_nbr",
    "family_code", "sales", "prediction",
}


def summarize_performance(data):
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if data.empty:
        raise ValueError("No backtest observations")

    frame = data.copy()

    frame["cutoff"] = pd.to_datetime(frame["cutoff"])
    frame["date"] = pd.to_datetime(frame["date"])

    keys = ["cutoff", "date", "store_nbr", "family_code"]

    if frame.duplicated(keys).any():
        raise ValueError("Duplicate forecast observations")

    if (
        frame["cutoff"].isna().any()
        or frame["date"].isna().any()
        or (frame["date"] <= frame["cutoff"]).any()
    ):
        raise ValueError("Invalid forecast dates")

    actual = frame["sales"].to_numpy(dtype=float)
    predicted = frame["prediction"].dropna().to_numpy(dtype=float)

    if not np.isfinite(actual).all() or (actual < 0).any():
        raise ValueError("Invalid actual sales")

    if not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError("Invalid predictions")

    overall = calculate_metrics(frame)

    per_cutoff = []
    for cutoff, group in frame.groupby("cutoff"):
        per_cutoff.append({
            "cutoff": str(cutoff.date()),
            **calculate_metrics(group),
        })

    per_family = []
    for family, group in frame.groupby("family_code"):
        per_family.append({
            "family_code": int(family),
            **calculate_metrics(group),
        })

    per_family.sort(
        key=lambda row: (
            row["mae"] is None,
            -(row["mae"] or 0),
        )
    )

    return {
        "overall": overall,
        "per_cutoff": per_cutoff,
        "per_family": per_family,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/gradient_boosting_backtest.parquet"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/monitoring/model_performance.json"
        ),
    )

    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Missing backtest file: {args.input}")

    data = pd.read_parquet(args.input)

    report = summarize_performance(data)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Overall:", report["overall"])

    print("\nPerformance by cutoff:")
    for row in report["per_cutoff"]:
        print(row)

    print("\nHighest MAE families:")
    for row in report["per_family"][:5]:
        print(row)

    print("\nReport saved:", args.output)


if __name__ == "__main__":
    main()
