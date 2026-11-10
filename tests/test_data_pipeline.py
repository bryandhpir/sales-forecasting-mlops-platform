import pandas as pd
import pytest

from src.data_pipeline import add_features, validate_data


def sample_data():
    days = pd.date_range("2024-01-01", periods=36, freq="D")
    rows = []
    for store in (1, 2):
        for i, date in enumerate(days):
            if store == 1 and i == 3:  # intentionally missing Jan 4
                continue
            rows.append({
                "date": date, "store_nbr": store, "family": "GROCERY",
                "sales": float(i + 1000 * (store - 1)), "onpromotion": 0,
            })
    return pd.DataFrame(rows)


def test_lag_is_exact_calendar_day_and_does_not_cross_stores():
    features = add_features(sample_data())
    jan8 = pd.Timestamp("2024-01-08")
    store1 = features[(features.store_nbr == 1) & (features.date == jan8)].iloc[0]
    store2 = features[(features.store_nbr == 2) & (features.date == jan8)].iloc[0]
    assert store1.sales_lag_7 == 0.0
    assert store2.sales_lag_7 == 1000.0
    assert pd.isna(store1.sales_rolling_mean_7)  # Jan 4 really is missing
    assert store2.sales_rolling_mean_7 == pytest.approx(1003.0)


def test_lag_1_uses_date_not_previous_available_row():
    features = add_features(sample_data())
    jan5 = features[(features.store_nbr == 1) & (features.date == "2024-01-05")].iloc[0]
    assert pd.isna(jan5.sales_lag_1)
    jan29 = features[(features.store_nbr == 1) & (features.date == "2024-01-29")].iloc[0]
    assert jan29.sales_lag_28 == 0.0


def test_duplicates_are_rejected():
    df = sample_data()
    duplicates = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    stores = pd.DataFrame({"store_nbr": [1, 2]})
    with pytest.raises(ValueError, match="Duplicate"):
        validate_data(duplicates, stores)
