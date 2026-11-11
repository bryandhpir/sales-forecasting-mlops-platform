
import numpy as np
import pandas as pd

from fastapi.testclient import TestClient
import src.api as api


class DummyModel:
    feature_names_in_ = np.array([
        "baseline_reference",
        "cutoff_sales",
        "horizon_step",
    ])

    def predict(self, X):
        # Kiểm tra snapshot lấy đúng tại cutoff
        assert (X["cutoff_sales"] == 15).all()

        return np.log1p(
            X["baseline_reference"].to_numpy(dtype=float)
        )


def test_api_forecast_pipeline(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=15)
    sales = pd.Series(np.arange(1, 16, dtype=float))

    history = pd.DataFrame({
        "date": dates,
        "store_nbr": 1,
        "family_code": 3,
        "sales": sales,
        "sales_lag_7": sales.shift(7),
        "sales_lag_28": sales.shift(28),
        "sales_rolling_mean_7": (
            sales.shift(1).rolling(7).mean()
        ),
    })

    # Chỉ giả lập tải tài nguyên
    monkeypatch.setattr(
        api,
        "get_resources",
        lambda: (
            DummyModel(),
            history,
            {"BEVERAGES": 3},
        ),
    )

    response = TestClient(api.app).post(
        "/predict",
        json={
            "cutoff": "2024-01-15",
            "store_nbr": 1,
            "family": "BEVERAGES",
            "horizon": 3,
        },
    )

    assert response.status_code == 200, response.text

    predictions = response.json()

    assert len(predictions) == 3

    assert [row["date"] for row in predictions] == [
        "2024-01-16",
        "2024-01-17",
        "2024-01-18",
    ]

    np.testing.assert_allclose(
        [row["predicted_sales"] for row in predictions],
        [9, 10, 11],
    )
