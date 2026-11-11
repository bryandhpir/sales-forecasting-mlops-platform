
import numpy as np
import pandas as pd

from src.monitoring import compare_windows


def make_sample(current_shift=0):
    dates = pd.date_range(
        "2024-01-01",
        periods=200,
    )

    baseline = np.arange(1, 101, dtype=float)

    values = np.concatenate([
        baseline,
        baseline + current_shift,
    ])

    return pd.DataFrame({
        "date": dates,
        "sales_lag_7": values,
    })


def run_comparison(data):
    dates = data["date"]

    return compare_windows(
        data,
        reference_start=dates.iloc[0],
        reference_end=dates.iloc[99],
        current_start=dates.iloc[100],
        current_end=dates.iloc[199],
        features=["sales_lag_7"],
    )


def test_no_drift():
    report = run_comparison(make_sample())

    result = report["features"]["sales_lag_7"]

    assert result["ks_statistic"] == 0
    assert result["drift_detected"] is False


def test_detects_drift():
    report = run_comparison(
        make_sample(current_shift=500)
    )

    result = report["features"]["sales_lag_7"]

    assert result["ks_statistic"] == 1
    assert result["drift_detected"] is True
