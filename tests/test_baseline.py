import numpy as np
import pandas as pd
import pytest

from src.baseline import calculate_metrics, predict_seasonal_naive
from src.backtesting import run_backtest


def sample_data():
    rows = []
    for store in (1, 2):
        for i, day in enumerate(pd.date_range("2024-01-01", periods=42)):
            rows.append({
                "date": day,
                "store_nbr": store,
                "family": "FOOD",
                "sales": float(i + 100 * store),
            })
    return pd.DataFrame(rows)


def test_configurable_horizon_and_calendar_reference():
    result = predict_seasonal_naive(sample_data(), "2024-01-21", horizon=10)
    store_1 = result[result["store_nbr"] == 1]
    assert store_1["date"].min() == pd.Timestamp("2024-01-22")
    assert store_1["date"].max() == pd.Timestamp("2024-01-31")
    assert store_1.loc[store_1["date"] == "2024-01-22", "prediction"].iloc[0] == 114.0
    assert store_1.loc[store_1["date"] == "2024-01-29", "prediction"].iloc[0] == 114.0


def test_target_changes_after_cutoff_do_not_affect_prediction():
    data = sample_data()
    original = predict_seasonal_naive(data, "2024-01-21", 10)
    data.loc[data["date"] > "2024-01-21", "sales"] = 999999
    modified = predict_seasonal_naive(data, "2024-01-21", 10)
    assert original["prediction"].tolist() == modified["prediction"].tolist()


def test_missing_reference_is_not_treated_as_zero():
    data = sample_data()
    data = data.loc[~((data["store_nbr"] == 1) & (data["date"] == "2024-01-15"))]
    predicted = predict_seasonal_naive(data, "2024-01-21", 10)
    row = predicted.loc[(predicted["store_nbr"] == 1) & (predicted["date"] == "2024-01-22")].iloc[0]
    assert pd.isna(row["prediction"])
    assert calculate_metrics(predicted)["coverage"] < 1


def test_metrics_exact_values():
    predictions = pd.DataFrame({"sales": [0.0, 3.0], "prediction": [0.0, 1.0]})
    result = calculate_metrics(predictions)
    assert result["mae"] == pytest.approx(1.0)
    assert result["rmsle"] == pytest.approx(np.sqrt((np.log(4) - np.log(2)) ** 2 / 2))


def test_multiple_origins_are_chronological_and_disjoint():
    predictions, metrics = run_backtest(sample_data(), horizon=5, origins=3)
    assert metrics["evaluation_origins"] == 3
    assert metrics["forecast_horizon"] == 5
    cutoffs = [pd.Timestamp(x["cutoff"]) for x in metrics["per_origin"]]
    assert cutoffs == sorted(cutoffs)
    assert cutoffs[1] - cutoffs[0] == pd.Timedelta(days=5)
    assert cutoffs[2] - cutoffs[1] == pd.Timedelta(days=5)
    assert not predictions.duplicated(["date", "store_nbr", "family"]).any()
    assert metrics["overall"]["scored_targets"] == len(predictions)


def test_invalid_parameters_fail():
    with pytest.raises(ValueError, match="positive"):
        predict_seasonal_naive(sample_data(), "2024-01-21", 0)
    with pytest.raises(ValueError, match="positive"):
        run_backtest(sample_data(), horizon=5, origins=0)
