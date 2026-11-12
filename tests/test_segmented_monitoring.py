
import numpy as np
import pandas as pd

from src.segmented_monitoring import compare_segments


def test_segmented_median_decrease():
    dates = pd.date_range(
        "2024-01-01",
        periods=120,
    )

    values = np.concatenate([
        np.arange(100, 160, dtype=float),
        np.arange(50, 110, dtype=float),
    ])

    data = pd.DataFrame({
        "date": dates,
        "store_nbr": 1,
        "family": "BEVERAGES",
        "sales_lag_7": values,
        "sales_rolling_mean_7": values,
    })

    result, skipped = compare_segments(
        data,
        reference_start="2024-01-01",
        reference_end="2024-02-29",
        current_start="2024-03-01",
        current_end="2024-04-29",
    )

    assert skipped == 0
    assert len(result) == 2

    row = result.loc[
        result["feature"] == "sales_lag_7"
    ].iloc[0]

    assert row["reference_median"] == 129.5
    assert row["current_median"] == 79.5
    assert row["median_delta"] == -50.0
    assert row["median_change_pct"] < 0
    assert bool(row["drift_detected"])
