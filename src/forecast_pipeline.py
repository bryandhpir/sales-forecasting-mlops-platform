
import pandas as pd

from src.forecast_features import (
    build_future_calendar,
    attach_historical_features,
)
from src.inference import predict_sales


def forecast_sales(
    model,
    history,
    family_mapping,
    cutoff,
    store_nbr,
    family,
    horizon,
    max_horizon=7,
):
    # Chỉ cho phép khoảng dự báo đã được đánh giá
    if horizon > max_horizon:
        raise ValueError(
            f"horizon exceeds validated limit: {max_horizon}"
        )

    # Kiểm tra family
    if family not in family_mapping:
        raise ValueError(f"Unknown family: {family}")

    # 1. Tạo lịch dự báo
    future = build_future_calendar(
        cutoff=cutoff,
        store_nbr=store_nbr,
        family=family,
        horizon=horizon,
    )

    # 2. Sử dụng đúng family_code từ Training
    future["family_code"] = family_mapping[family]

    # 3. Bổ sung historical features
    features = attach_historical_features(
        future=future,
        history=history,
        cutoff=cutoff,
    )

    # 4. Dự báo bằng mô hình đã huấn luyện
    predictions = predict_sales(
        model=model,
        features=features,
    )

    # 5. Trả về bảng kết quả
    result = features[
        ["date", "store_nbr", "family", "horizon_step"]
    ].copy()

    result["predicted_sales"] = predictions

    return result
