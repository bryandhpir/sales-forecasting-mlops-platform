"""Generate exploratory analysis figures from the processed feature dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_features(processed_dir: Path) -> pd.DataFrame:
    parquet_path = processed_dir / "features.parquet"
    csv_path = processed_dir / "features.csv.gz"
    if parquet_path.is_file():
        return pd.read_parquet(parquet_path)
    if csv_path.is_file():
        return pd.read_csv(csv_path, parse_dates=["date"])
    raise FileNotFoundError("Run `python -m src.data_pipeline` before EDA")


def create_charts(features: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])

    daily = df.groupby("date", observed=True)["sales"].sum().sort_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    daily.plot(ax=ax, linewidth=1)
    ax.set(title="Total daily sales", xlabel="Date", ylabel="Sales (dataset units)")
    fig.tight_layout()
    fig.savefig(figures_dir / "daily_sales.png", dpi=150)
    plt.close(fig)

    # Mean sales per record, not total daily sales, for weekday comparison.
    by_day = df.groupby("day_of_week", observed=True)["sales"].mean()
    by_day = by_day.reindex(range(7))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], by_day)
    ax.set(title="Average sales per record by weekday", ylabel="Average sales")
    fig.tight_layout()
    fig.savefig(figures_dir / "weekday_seasonality.png", dpi=150)
    plt.close(fig)

    top_families = (
        df.groupby("family", observed=True)["sales"].sum()
        .sort_values(ascending=False)
        .head(10).sort_values()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    top_families.plot.barh(ax=ax)
    ax.set(title="Top 10 product families by total sales", xlabel="Total sales")
    fig.tight_layout()
    fig.savefig(figures_dir / "top_families.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA charts")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    df = load_features(args.processed_dir)
    create_charts(df, args.figures_dir)
    print(f"Generated 3 charts in {args.figures_dir.resolve()}")


if __name__ == "__main__":
    main()
