
import numpy as np
import pandas as pd
import pytest

from src.inference import predict_sales


class DummyLogModel:

    feature_names_in_ = np.array(["a", "b"])

    def predict(self, X):
        assert list(X.columns) == ["a", "b"]
        return np.log1p(X["a"] + X["b"])


def test_predict_sales():

    model = DummyLogModel()

    # Cố tình đảo thứ tự cột
    X = pd.DataFrame({
        "b": [2, 4],
        "a": [1, 2]
    })

    predictions = predict_sales(model, X)

    np.testing.assert_allclose(
        predictions.to_numpy(),
        [3, 6]
    )


def test_missing_feature():

    model = DummyLogModel()

    X = pd.DataFrame({
        "a": [1, 2]
    })

    with pytest.raises(ValueError, match="Missing features"):
        predict_sales(model, X)
