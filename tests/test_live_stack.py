
import os
import math

import httpx
import pytest


# Chỉ chạy khi chủ động bật kiểm thử hệ thống
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_STACK") != "1",
    reason="Live Docker stack test is disabled"
)

API_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8002"
)


def test_health_endpoint():
    response = httpx.get(
        f"{API_URL}/health",
        timeout=10
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_forecast_endpoint():
    payload = {
        "cutoff": "2017-08-15",
        "store_nbr": 1,
        "family": "BEVERAGES",
        "horizon": 7
    }

    response = httpx.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=120
    )

    assert response.status_code == 200, response.text

    predictions = response.json()

    # Phải trả đủ số ngày
    assert len(predictions) == payload["horizon"]

    # Ngày dự báo phải đúng thứ tự
    assert [item["date"] for item in predictions] == [
        "2017-08-16",
        "2017-08-17",
        "2017-08-18",
        "2017-08-19",
        "2017-08-20",
        "2017-08-21",
        "2017-08-22"
    ]

    # Kiểm tra từng dự báo
    for i, item in enumerate(predictions, start=1):
        assert item["store_nbr"] == 1
        assert item["family"] == "BEVERAGES"
        assert item["horizon_step"] == i

        sales = item["predicted_sales"]

        assert math.isfinite(sales)
        assert sales >= 0
