"""Repeatable, calendar-correct sales feature preparation.

Run from the repository root:
    python -m src.data_pipeline

Historical lag features are suitable for exploration and training examples,
but must not use observations from the evaluation period when scoring a
fixed-origin, multi-step forecast. Use origin-safe backtesting for evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_TRAIN_COLUMNS = {"date", "store_nbr", "family", "sales", "onpromotion"}
KEY_COLUMNS = ["date", "store_nbr", "family"]


def load_raw_data(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_file = raw_dir / "train.csv"
    stores_file = raw_dir / "stores.csv"
    missing = [str(p) for p in (train_file, stores_file) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Kaggle dataset files: " + ", ".join(missing)
            + ". Download from https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data"
        )

    train = pd.read_csv(
        train_file,
        usecols=["date", "store_nbr", "family", "sales", "onpromotion"],
        parse_dates=["date"],
        dtype={
            "store_nbr": "int16",
            "family": "category",
            "sales": "float32",
            "onpromotion": "int32",
        },
    )
    stores = pd.read_csv(stores_file, dtype={"store_nbr": "int16"})
    return train, stores


def validate_data(train: pd.DataFrame, stores: pd.DataFrame) -> None:
    if missing := REQUIRED_TRAIN_COLUMNS.difference(train.columns):
        raise ValueError(f"Missing train columns: {sorted(missing)}")
    if train.empty:
        raise ValueError("train.csv is empty")
    if train[list(REQUIRED_TRAIN_COLUMNS)].isna().any().any():
        raise ValueError("Missing values in required train columns")
    if train.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate date/store_nbr/family combinations")
    if (train["sales"] < 0).any() or (train["onpromotion"] < 0).any():
        raise ValueError("Unexpected negative sales or onpromotion counts")
    if "store_nbr" not in stores.columns or stores["store_nbr"].duplicated().any():
        raise ValueError("stores.csv must have one row per store_nbr")
    if not set(train["store_nbr"]).issubset(set(stores["store_nbr"])):
        raise ValueError("Unknown store_nbr values in train.csv")


def _daily_sales_lookup(frame: pd.DataFrame):
    """Return a function looking up sales exactly N calendar days earlier.

    Unlike groupby.shift(N), this handles dates missing from the source dataset.
    """
    history_index = pd.MultiIndex.from_frame(frame[KEY_COLUMNS])
    history = pd.Series(frame["sales"].to_numpy(), index=history_index)
    store_values = frame["store_nbr"]
    family_values = frame["family"]
    dates = frame["date"]

    def lookup(days_back: int) -> np.ndarray:
        index = pd.MultiIndex.from_arrays(
            [dates - pd.Timedelta(days=days_back), store_values, family_values],
            names=KEY_COLUMNS,
        )
        return history.reindex(index).to_numpy(dtype="float32", na_value=np.nan)

    return lookup


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar features and true daily lag / trailing-7-day features."""
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(KEY_COLUMNS).reset_index(drop=True)

    frame["year"] = frame["date"].dt.year.astype("int16")
    frame["month"] = frame["date"].dt.month.astype("int8")
    frame["day_of_week"] = frame["date"].dt.day_of_week.astype("int8")
    frame["day_of_month"] = frame["date"].dt.day.astype("int8")
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype("int8")

    lookup = _daily_sales_lookup(frame)
    # Each lag uses the exact calendar date (not just the Nth prior row).
    total_7 = np.zeros(len(frame), dtype="float32")
    has_all_7 = np.ones(len(frame), dtype=bool)
    for lag in range(1, 8):
        past = lookup(lag)
        total_7 += np.nan_to_num(past, nan=0.0)
        has_all_7 &= ~np.isnan(past)
        if lag in (1, 7):
            frame[f"sales_lag_{lag}"] = past

    frame["sales_rolling_mean_7"] = np.where(
        has_all_7, total_7 / 7.0, np.nan
    ).astype("float32")

    for lag in (14, 28):
        frame[f"sales_lag_{lag}"] = lookup(lag)

    return frame


def build_data(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    train, stores = load_raw_data(raw_dir)
    validate_data(train, stores)
    merged = train.merge(stores, on="store_nbr", how="left", validate="many_to_one")
    features = add_features(merged)
    max_date = features["date"].max()
    report = {
        "row_count": int(len(features)),
        "min_date": str(features["date"].min().date()),
        "max_date": str(max_date.date()),
        "stores": int(features["store_nbr"].nunique()),
        "product_families": int(features["family"].nunique()),
        "unique_date_store_family": bool(not features.duplicated(KEY_COLUMNS).any()),
        "missing_target_sales": int(features["sales"].isna().sum()),
        "missing_lag_7": int(features["sales_lag_7"].isna().sum()),
        "missing_lag_28": int(features["sales_lag_28"].isna().sum()),
        "evaluation_note": (
            "Historical lag features require origin-safe reconstruction "
            "before evaluating multi-step predictions."
        ),
    }
    return features, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare forecasting data")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")
    args = parser.parse_args()

    features, report = build_data(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_format == "parquet":
        output = args.output_dir / "features.parquet"
        features.to_parquet(output, index=False)
    else:
        output = args.output_dir / "features.csv.gz"
        features.to_csv(output, index=False, compression="gzip")

    (args.output_dir / "data_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved feature table: {output}")


if __name__ == "__main__":
    main()
