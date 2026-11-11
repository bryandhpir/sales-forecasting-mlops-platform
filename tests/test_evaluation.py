
import numpy as np
import pandas as pd
import pytest

from src.evaluation import evaluate_model

class DummyModel:

    def predict(self, X):
        return np.full(
            len(X),
            np.log1p(100.0)
        )

def test_evaluate_model_mae():

    # Tạo 2 bản ghi giả định
    samples = pd.DataFrame({
        "cutoff": pd.to_datetime(
            ["2017-07-04", "2017-07-04"]
        ),
        "date": pd.to_datetime(
            ["2017-07-05", "2017-07-06"]
        ),
        "store_nbr": [1, 1],
        "family_code": [3, 3],
        "sales": [90.0, 110.0]
    })

    X = pd.DataFrame({
        "feature": [1, 2]
    })

    model = DummyModel()

    report, evaluated = evaluate_model(
        "Dummy Model",
        model,
        X,
        samples
    )

    assert report["mae"] == pytest.approx(10.0)
    assert report["coverage"] == 1.0
    assert len(evaluated) == 2

def test_evaluate_model_invalid_length():

    X = pd.DataFrame({
        "feature": [1, 2, 3]
    })

    samples = pd.DataFrame({
        "sales": [100, 200]
    })

    model = DummyModel()

    with pytest.raises(ValueError):
        evaluate_model(
            "Dummy Model",
            model,
            X,
            samples
        )

def test_evaluate_model_rmsle():

    samples = pd.DataFrame({
        "cutoff": pd.to_datetime([
            "2017-07-04", "2017-07-04"
        ]),
        "date": pd.to_datetime([
            "2017-07-05", "2017-07-06"
        ]),
        "store_nbr": [1, 1],
        "family_code": [3, 3],
        "sales": [90.0, 110.0]
    })

    X = pd.DataFrame({
        "feature": [1, 2]
    })

    model = DummyModel()

    report, evaluated = evaluate_model(
        "Dummy Model",
        model,
        X,
        samples
    )

    # DummyModel luôn dự báo sales = 100
    actual = np.array([90.0, 110.0])
    predicted = np.array([100.0, 100.0])

    expected_rmsle = np.sqrt(
        np.mean(
            (np.log1p(actual) - np.log1p(predicted)) ** 2
        )
    )

    assert report["rmsle"] == pytest.approx(expected_rmsle)

    np.testing.assert_allclose(
        evaluated["prediction"].to_numpy(),
        predicted
    )

def test_evaluate_model_reordered_rows():

    samples = pd.DataFrame({
        "cutoff": pd.to_datetime([
            "2017-07-04", "2017-07-04"
        ]),
        "date": pd.to_datetime([
            "2017-07-05", "2017-07-06"
        ]),
        "store_nbr": [1, 2],
        "family_code": [3, 3],
        "sales": [100.0, 500.0]
    })

    X = pd.DataFrame({
        "feature": [10, 20]
    })

    # Cố tình đảo thứ tự các dòng
    X_reordered = X.iloc[::-1]

    model = DummyModel()

    with pytest.raises(ValueError, match="index"):
        evaluate_model(
            "Dummy Model",
            model,
            X_reordered,
            samples
        )
