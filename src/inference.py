
import numpy as np
import pandas as pd


def predict_sales(model, features):

    # Kiểm tra dữ liệu đầu vào
    if not isinstance(features, pd.DataFrame):
        raise TypeError("Features must be a DataFrame")

    if features.empty:
        raise ValueError("Features cannot be empty")

    if not features.columns.is_unique:
        raise ValueError("Feature names must be unique")

    # Lấy danh sách features từ mô hình đã huấn luyện
    required_features = list(model.feature_names_in_)

    # Kiểm tra các cột còn thiếu
    missing = [
        col for col in required_features
        if col not in features.columns
    ]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Sắp xếp lại các cột đúng thứ tự huấn luyện
    X = features.loc[:, required_features]

    # Dự báo trên thang log
    log_predictions = np.asarray(
        model.predict(X),
        dtype=float
    )

    if (
        log_predictions.ndim != 1
        or len(log_predictions) != len(features)
        or not np.isfinite(log_predictions).all()
    ):
        raise ValueError("Invalid model predictions")

    # Chuyển log-sales thành sales
    predictions = np.expm1(
        np.clip(log_predictions, 0, 25)
    )

    return pd.Series(
        predictions,
        index=features.index,
        name="predicted_sales"
    )
