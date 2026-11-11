import numpy as np
import pandas as pd
import pytest

from src.train_model import (
    MODEL_FEATURES,
    backtest_gradient_boosting,
    build_origin_examples,
    prepare_data,
)


def sample_features():
    dates = pd.date_range("2024-01-01", periods=140, freq="D")
    items = []
    for store in (1, 2):
        for family in ("A", "B"):
            for i, date in enumerate(dates):
                if date == pd.Timestamp("2024-01-04"):
                    continue
                sales = float(4 + store * 10 + 5 * (family == "B") + (i % 7))
                items.append({"date": date, "store_nbr": store, "family": family, "sales": sales})
    frame = pd.DataFrame(items).sort_values(["store_nbr", "family", "date"])
    for lag in (7, 28):
        lookup = frame[["date", "store_nbr", "family", "sales"]].rename(
            columns={"sales": f"sales_lag_{lag}"}
        )
        lookup["date"] += pd.Timedelta(days=lag)
        frame = frame.merge(lookup, on=["date", "store_nbr", "family"], how="left")
    # The exact numeric values of this trailing statistic don't affect the leakage test.
    frame["sales_rolling_mean_7"] = np.nan
    return frame


def test_future_sales_never_become_features():
    original = sample_features()
    cutoff = pd.Timestamp("2024-04-20")
    base = build_origin_examples(prepare_data(original), cutoff, horizon=5)
    altered = original.copy()
    altered.loc[altered.date > cutoff, "sales"] += 99999
    # Simulate the features.parquet target rows having been computed using future values.
    altered.loc[altered.date > cutoff, "sales_lag_7"] = 99999
    altered.loc[altered.date > cutoff, "sales_rolling_mean_7"] = 99999
    new = build_origin_examples(prepare_data(altered), cutoff, horizon=5)
    pd.testing.assert_frame_equal(base[MODEL_FEATURES], new[MODEL_FEATURES])
    assert not base["sales"].equals(new["sales"])


def test_missing_reference_not_filled_as_zero():
    data = prepare_data(sample_features())
    # Jan 11 -> Jan 4 is missing for all series.
    chunk = build_origin_examples(data, pd.Timestamp("2024-01-10"), horizon=1)
    assert chunk["baseline_reference"].isna().all()


def test_train_and_eval_windows_do_not_overlap():
    data = sample_features()
    results, metrics, model = backtest_gradient_boosting(
        data, horizon=5, origins=2, train_origins=6, max_iter=10,
    )
    assert metrics["train_last_target_date"] <= metrics["first_eval_cutoff"]
    assert len(results) == metrics["overall"]["observed_targets"]
    assert np.isfinite(results["prediction"]).all()
    assert (results["prediction"] >= 0).all()
    assert metrics["forecast_horizon"] == 5
    # Test data intentionally makes the trailing statistic entirely missing.
    # The model must skip it instead of failing in histogram binning.
    assert "cutoff_rolling_mean_7" in metrics["skipped_all_missing_features"]
    assert "cutoff_rolling_mean_7" not in metrics["feature_columns"]
    assert list(model.feature_names_in_) == metrics["feature_columns"]


def test_invalid_horizon_rejected():
    with pytest.raises(ValueError, match="horizon"):
        build_origin_examples(prepare_data(sample_features()), "2024-04-20", 0)
