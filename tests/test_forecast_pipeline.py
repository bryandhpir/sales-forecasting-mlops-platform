
import numpy as np
import pandas as pd

from src.forecast_pipeline import forecast_sales


class DummyModel:
    feature_names_in_ = np.array([
        "baseline_reference",
        "cutoff_sales",
        "horizon_step",
    ])

    def predict(self, X):
        return np.log1p(
            X["baseline_reference"].to_numpy(dtype=float)
        )


def test_forecast_pipeline():
    dates = pd.date_range("2024-01-01", periods=15)
    sales = pd.Series(np.arange(1, 16, dtype=float))

    history = pd.DataFrame({
        "date": dates,
        "store_nbr": 1,
        "family_code": 3,
        "sales": sales,
    })

    history["sales_lag_7"] = sales.shift(7)
    history["sales_lag_28"] = sales.shift(28)
    history["sales_rolling_mean_7"] = (
        sales.shift(1).rolling(7).mean()
    )

    result = forecast_sales(
        model=DummyModel(),
        history=history,
        family_mapping={"BEVERAGES": 3},
        cutoff="2024-01-15",
        store_nbr=1,
        family="BEVERAGES",
        horizon=3,
    )

    assert len(result) == 3

    np.testing.assert_allclose(
        result["predicted_sales"].to_numpy(),
        [9, 10, 11],
    )
