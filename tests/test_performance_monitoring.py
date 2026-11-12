
import pandas as pd
import pytest

from src.performance_monitoring import summarize_performance


def test_performance_by_cutoff():
    data = pd.DataFrame({
        "cutoff": [
            "2024-01-01", "2024-01-01",
            "2024-01-10", "2024-01-10",
        ],
        "date": [
            "2024-01-02", "2024-01-03",
            "2024-01-11", "2024-01-12",
        ],
        "store_nbr": [1, 1, 1, 1],
        "family_code": [3, 3, 3, 3],
        "sales": [10, 20, 30, 40],
        "prediction": [9, 19, 25, 35],
    })

    report = summarize_performance(data)

    assert report["overall"]["mae"] == pytest.approx(3.0)
    assert report["overall"]["coverage"] == 1.0

    assert report["per_cutoff"][0]["mae"] == pytest.approx(1.0)
    assert report["per_cutoff"][1]["mae"] == pytest.approx(5.0)

    assert len(report["per_family"]) == 1
