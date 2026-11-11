
import pandas as pd
import pytest

from src.forecast_features import build_future_calendar


def test_future_calendar():

    result = build_future_calendar(
        cutoff="2017-08-15",
        store_nbr=1,
        family="BEVERAGES",
        horizon=7
    )

    # Có đúng số dòng không?
    assert len(result) == 7

    # Ngày đầu và ngày cuối
    assert result["date"].iloc[0] == pd.Timestamp("2017-08-16")
    assert result["date"].iloc[-1] == pd.Timestamp("2017-08-22")

    # Kiểm tra horizon_step
    assert result["horizon_step"].tolist() == list(range(1, 8))

    # Chủ nhật phải là weekend
    sunday = result.loc[
        result["date"] == pd.Timestamp("2017-08-20")
    ]

    assert sunday["day_of_week"].iloc[0] == 6
    assert sunday["is_weekend"].iloc[0] == 1


def test_invalid_horizon():

    with pytest.raises(ValueError):
        build_future_calendar(
            cutoff="2017-08-15",
            store_nbr=1,
            family="BEVERAGES",
            horizon=0
        )

import numpy as np
from src.forecast_features import attach_historical_features


def test_attach_historical_features():
    dates = pd.date_range("2024-01-01", periods=8)

    history = pd.DataFrame({
        "date": dates,
        "store_nbr": 1,
        "family_code": 3,
        "sales": [10, 20, 30, 40, 50, 60, 70, 80]
    })

    history["sales_lag_7"] = history["sales"].shift(7)
    history["sales_lag_28"] = history["sales"].shift(28)
    history["sales_rolling_mean_7"] = (
        history["sales"].shift(1).rolling(7).mean()
    )

    future = build_future_calendar(
        cutoff="2024-01-08",
        store_nbr=1,
        family="BEVERAGES",
        horizon=2
    )

    future["family_code"] = 3

    result = attach_historical_features(
        future,
        history,
        cutoff="2024-01-08"
    )

    assert len(result) == 2

    np.testing.assert_allclose(
        result["baseline_reference"],
        [20, 30]
    )

    np.testing.assert_allclose(
        result["cutoff_sales"],
        [80, 80]
    )

    np.testing.assert_allclose(
        result["cutoff_lag_7"],
        [10, 10]
    )

    np.testing.assert_allclose(
        result["cutoff_rolling_mean_7"],
        [40, 40]
    )

    assert result["cutoff_lag_28"].isna().all()
