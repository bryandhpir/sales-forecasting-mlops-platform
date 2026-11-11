
import numpy as np
import pandas as pd

from src.baseline import calculate_metrics


def evaluate_model(name, model, X, samples):

    if len(X) != len(samples):
        raise ValueError(
            "Features and samples must have the same length"
        )
    
    if not X.index.is_unique or not samples.index.is_unique:
        raise ValueError(
            "X and samples must have unique indexes"
        )

    if not X.index.equals(samples.index):
        raise ValueError(
            "X and samples index order must match"
    )


    # Dự báo trên thang log
    log_pred = model.predict(X)

    # Chuyển dự báo về đơn vị sales
    predictions = np.expm1(
        np.clip(log_pred, 0, 25)
    )

    # Tạo bảng đánh giá
    evaluated = samples[
        ["cutoff", "date", "store_nbr",
         "family_code", "sales"]
    ].copy()

    evaluated["prediction"] = predictions

    # Tính metrics
    metrics = calculate_metrics(evaluated)

    report = {
        "model": name,
        "mae": metrics["mae"],
        "rmsle": metrics["rmsle"],
        "coverage": metrics["coverage"]
    }

    return report, evaluated
