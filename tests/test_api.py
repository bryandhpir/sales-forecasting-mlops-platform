
import pandas as pd

from fastapi.testclient import TestClient
import src.api as api


client = TestClient(api.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_invalid_horizon():
    response = client.post(
        "/predict",
        json={
            "cutoff": "2017-08-15",
            "store_nbr": 1,
            "family": "BEVERAGES",
            "horizon": 0
        }
    )

    assert response.status_code == 422


def test_predict_endpoint(monkeypatch):
    # Không tải MLflow thật trong unit test
    monkeypatch.setattr(
        api,
        "get_resources",
        lambda: (None, None, {"BEVERAGES": 3})
    )

    # Giả lập kết quả Forecasting Pipeline
    def fake_forecast_sales(**kwargs):
        return pd.DataFrame({
            "date": [pd.Timestamp("2017-08-16")],
            "store_nbr": [1],
            "family": ["BEVERAGES"],
            "horizon_step": [1],
            "predicted_sales": [120.5]
        })

    monkeypatch.setattr(
        api,
        "forecast_sales",
        fake_forecast_sales
    )

    response = client.post(
        "/predict",
        json={
            "cutoff": "2017-08-15",
            "store_nbr": 1,
            "family": "BEVERAGES",
            "horizon": 1
        }
    )

    assert response.status_code == 200
    assert response.json()[0]["predicted_sales"] == 120.5

def test_unknown_family_returns_400(monkeypatch):

    # Giả lập tài nguyên đã tải thành công
    monkeypatch.setattr(
        api,
        "get_resources",
        lambda: (None, None, {"BEVERAGES": 3})
    )

    response = client.post(
        "/predict",
        json={
            "cutoff": "2017-08-15",
            "store_nbr": 1,
            "family": "UNKNOWN",
            "horizon": 7
        }
    )

    assert response.status_code == 400

    assert "Unknown family" in response.json()["detail"]
