"""Train and backtest an origin-safe direct retail sales forecaster.

Feature values use only information available at each forecast cutoff.
The target rows are observed outcomes used to train or evaluate the model,
never to construct the input features of that same prediction window.

Run from repository root:
    python -m src.train_model --horizon 7 --origins 3 --train-origins 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.baseline import calculate_metrics

SOURCE_COLUMNS = [
    "date", "store_nbr", "family", "sales", "sales_lag_7",
    "sales_lag_28", "sales_rolling_mean_7",
]
KEYS = ["store_nbr", "family_code"]
MODEL_FEATURES = [
    "store_nbr", "family_code", "year", "month", "day_of_month",
    "day_of_week", "is_weekend", "horizon_step", "baseline_reference",
    "cutoff_sales", "cutoff_lag_7", "cutoff_lag_28",
    "cutoff_rolling_mean_7",
]


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare compact typed data and one stable code per product family."""
    missing = set(SOURCE_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    df = data[SOURCE_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if df[["date", "store_nbr", "family", "sales"]].isna().any().any():
        raise ValueError("Missing dates, series keys, or targets")
    if df.duplicated(["date", "store_nbr", "family"]).any():
        raise ValueError("Duplicate date/store/family keys")
    if (df["sales"] < 0).any():
        raise ValueError("Negative unit sales")
    # Categories are derived from labels only: never from future sales values.
    df["family"] = df["family"].astype("category")
    df["family_code"] = df["family"].cat.codes.astype("int16")
    return df.sort_values("date").reset_index(drop=True)


def build_origin_examples(
    df: pd.DataFrame, cutoff: str | pd.Timestamp, horizon: int,
) -> pd.DataFrame:
    """Construct examples using exclusively <= cutoff feature observations.

    Ground-truth target 'sales' is included for supervised learning/measurement,
    but it is NEVER used as a future feature. Do not use lag columns of target rows.
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    cutoff = pd.Timestamp(cutoff).normalize()
    end = cutoff + pd.Timedelta(days=horizon)
    targets = df.loc[
        (df["date"] > cutoff) & (df["date"] <= end),
        ["date", "store_nbr", "family_code", "sales"],
    ].copy()
    if targets.empty:
        raise ValueError(f"No target observations for cutoff={cutoff.date()}")

    targets["horizon_step"] = (targets["date"] - cutoff).dt.days.astype("int16")
    steps = targets["horizon_step"].to_numpy()
    # For each future weekday, take its last same-weekday occurrence <= cutoff.
    cycles = ((steps - 1) // 7) + 1
    targets["reference_date"] = targets["date"] - pd.to_timedelta(cycles * 7, unit="D")

    # Fast calendar-specific lookups, from historical data only.
    reference_dates = targets["reference_date"].unique()
    past = df.loc[
        (df["date"] <= cutoff) & (df["date"].isin(reference_dates)),
        ["date", *KEYS, "sales"],
    ].rename(columns={"date": "reference_date", "sales": "baseline_reference"})
    targets = targets.merge(past, on=["reference_date", *KEYS], how="left", validate="many_to_one")

    # Snapshot at the origin. All lag columns in this snapshot refer to earlier dates.
    snapshot = df.loc[
        df["date"] == cutoff,
        [*KEYS, "sales", "sales_lag_7", "sales_lag_28", "sales_rolling_mean_7"],
    ].rename(columns={
        "sales": "cutoff_sales",
        "sales_lag_7": "cutoff_lag_7",
        "sales_lag_28": "cutoff_lag_28",
        "sales_rolling_mean_7": "cutoff_rolling_mean_7",
    })
    if snapshot.empty:
        raise ValueError(f"No historical snapshot at cutoff={cutoff.date()}")
    targets = targets.merge(snapshot, on=KEYS, how="left", validate="many_to_one")

    targets["year"] = targets["date"].dt.year.astype("int16")
    targets["month"] = targets["date"].dt.month.astype("int8")
    targets["day_of_month"] = targets["date"].dt.day.astype("int8")
    targets["day_of_week"] = targets["date"].dt.day_of_week.astype("int8")
    targets["is_weekend"] = (targets["day_of_week"] >= 5).astype("int8")
    targets["cutoff"] = cutoff
    return targets


def backtest_gradient_boosting(
    data: pd.DataFrame, horizon: int, origins: int = 3,
    train_origins: int = 12, max_iter: int = 100,
) -> tuple[pd.DataFrame, dict, HistGradientBoostingRegressor]:
    """Train on older, disjoint windows; evaluate on later disjoint windows."""
    for name, val in [("horizon", horizon), ("origins", origins),
                      ("train_origins", train_origins), ("max_iter", max_iter)]:
        if not isinstance(val, int) or isinstance(val, bool) or val < 1:
            raise ValueError(f"{name} must be a positive integer")
    df = prepare_data(data)
    end_date = df["date"].max()
    first_date = df["date"].min()
    evaluation_cutoffs = [
        end_date - pd.Timedelta(days=horizon * i)
        for i in range(origins, 0, -1)
    ]
    first_eval_cutoff = evaluation_cutoffs[0]
    training_cutoffs = [
        first_eval_cutoff - pd.Timedelta(days=horizon * i)
        for i in range(train_origins, 0, -1)
    ]
    if training_cutoffs[0] - pd.Timedelta(days=35) < first_date:
        raise ValueError("Not enough history for the selected training/evaluation cutoffs")

    train_parts = [build_origin_examples(df, c, horizon) for c in training_cutoffs]
    train = pd.concat(train_parts, ignore_index=True)
    if train["date"].max() > first_eval_cutoff:
        raise AssertionError("Training target overlaps evaluation interval")
    if len(train) < 100:
        raise ValueError("Insufficient training examples")

    # All input features are computed using known snapshots and calendar data.
    # HistGradientBoostingRegressor handles missing lag/reference values directly.
    model = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=max_iter, max_leaf_nodes=25,
        learning_rate=0.06, min_samples_leaf=40, l2_regularization=2.0,
        categorical_features=["store_nbr", "family_code"],
        random_state=42,
    )
    # An entirely missing feature carries no information and may cause
    # threshold-binning errors in some scikit-learn versions. Keep partially
    # missing features: the estimator handles missing observations natively.
    all_missing_features = [
        name for name in MODEL_FEATURES if train[name].isna().all()
    ]
    active_features = [
        name for name in MODEL_FEATURES if name not in all_missing_features
    ]
    if not active_features:
        raise ValueError("No usable training features")
    X_train = train[active_features]
    y_train = np.log1p(train["sales"].to_numpy(dtype="float64"))
    model.fit(X_train, y_train)

    predictions = []
    per_origin = []
    for cutoff in evaluation_cutoffs:
        chunk = build_origin_examples(df, cutoff, horizon)
        log_pred = model.predict(chunk[active_features])
        chunk["prediction"] = np.expm1(np.clip(log_pred, 0.0, 25.0))
        predictions.append(chunk[["date", "store_nbr", "family_code", "cutoff", "sales", "prediction"]])
        per_origin.append({
            "cutoff": str(cutoff.date()),
            "target_start": str((cutoff + pd.Timedelta(days=1)).date()),
            "target_end": str((cutoff + pd.Timedelta(days=horizon)).date()),
            **calculate_metrics(chunk),
        })

    result = pd.concat(predictions, ignore_index=True)
    metrics = {
        "method": "hist_gradient_boosting_direct_origin_safe",
        "forecast_horizon": horizon,
        "evaluation_origins": origins,
        "train_origins": train_origins,
        "train_rows": int(len(train)),
        "train_last_target_date": str(train["date"].max().date()),
        "first_eval_cutoff": str(first_eval_cutoff.date()),
        "feature_columns": active_features,
        "skipped_all_missing_features": all_missing_features,
        "per_origin": per_origin,
        "overall": calculate_metrics(result),
        "note": "Training labels do not overlap evaluation targets. Each forecast input uses only observations at or before its cutoff. Observed target rows only are scored.",
    }
    return result, metrics, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate an origin-safe gradient boosting sales forecaster")
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--origins", type=int, default=3)
    parser.add_argument("--train-origins", type=int, default=12)
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()
    if not args.features.exists():
        parser.error(f"Missing features: {args.features}. Run python -m src.data_pipeline")
    df = pd.read_parquet(args.features, columns=SOURCE_COLUMNS)
    predictions, metrics, _ = backtest_gradient_boosting(
        df, horizon=args.horizon, origins=args.origins,
        train_origins=args.train_origins, max_iter=args.max_iter,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.output_dir / "gradient_boosting_backtest.parquet", index=False)

    baseline_file = args.output_dir / "baseline_metrics.json"
    if baseline_file.is_file():
        baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
        if (baseline.get("forecast_horizon") == args.horizon
                and baseline.get("evaluation_origins") == args.origins
                and baseline.get("per_origin", [{}])[0].get("cutoff")
                == metrics["per_origin"][0]["cutoff"]):
            b = baseline["overall"]
            m = metrics["overall"]
            metrics["baseline_comparison"] = {
                "baseline_mae": b["mae"], "baseline_rmsle": b["rmsle"],
                "mae_delta_model_minus_baseline": m["mae"] - b["mae"],
                "rmsle_delta_model_minus_baseline": m["rmsle"] - b["rmsle"],
                "interpretation": "Negative delta means lower error for the ML model",
            }
        else:
            print("Baseline metrics do not match these evaluation cutoffs; no comparison generated.")
    (args.output_dir / "gradient_boosting_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved predictions to {args.output_dir / 'gradient_boosting_backtest.parquet'}")


if __name__ == "__main__":
    main()
