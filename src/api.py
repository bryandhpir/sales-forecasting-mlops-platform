import os
from datetime import date
import logging
from time import perf_counter
logger = logging.getLogger(__name__)
from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="Sales Forecasting API",
    version="0.1.0"
)


class ForecastRequest(BaseModel):
    cutoff: date
    store_nbr: int = Field(gt=0)
    family: str = Field(min_length=1)
    horizon: int = Field(ge=1, le=7)


class ForecastResponse(BaseModel):
    date: date
    store_nbr: int
    family: str
    horizon_step: int
    predicted_sales: float


@app.get("/health")
def health():
    return {"status": "ok"}

import json
import mlflow
import mlflow.sklearn
import pandas as pd

from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException

from src.train_model import prepare_data
from src.forecast_pipeline import forecast_sales


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def get_resources():
    start = perf_counter()
    print("[1] Starting resource loading...", flush=True)
    # Kết nối MLflow
    mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )
)


    # Tải mô hình đã đăng ký
    model = mlflow.sklearn.load_model(
        "models:/sales_forecasting_gradient_boosting@candidate"
    )
    print(
        "[2] MLflow model loaded:",
        round(perf_counter() - start, 2),
        "seconds",
        flush=True
    )

    # Đọc và chuẩn bị dữ liệu lịch sử
    raw_history = pd.read_parquet(
        ROOT / "data/processed/features.parquet"
    )
    print(
        "[3] Parquet loaded:",
        round(perf_counter() - start, 2),
        "seconds",
        flush=True
    )

    history = prepare_data(raw_history)

    print(
        "[4] Data prepared:",
        round(perf_counter() - start, 2),
        "seconds",
        flush=True
    )

    # Đọc mapping đã lưu
    mapping_path = (
        ROOT / "data/processed/family_mapping.json"
    )

    family_mapping = json.loads(
        mapping_path.read_text(encoding="utf-8")
    )

    # Kiểm tra mapping khớp với preprocessing
    pairs = history[
        ["family", "family_code"]
    ].drop_duplicates()

    actual_mapping = {
        str(family): int(code)
        for family, code in zip(
            pairs["family"],
            pairs["family_code"]
        )
    }

    if family_mapping != actual_mapping:
        raise ValueError(
            "Family mapping does not match training preprocessing"
        )

    print(
        "[5] All resources ready:",
        round(perf_counter() - start, 2),
        "seconds",
        flush=True
    )
    return model, history, family_mapping

@app.post(
    "/predict",
    response_model=list[ForecastResponse]
)
def predict(request: ForecastRequest):
    request_start = perf_counter()
    print("[API] Request received", flush=True)
    # Tải tài nguyên đã cache
    
    try:
        model, history, family_mapping = get_resources()

    except Exception as exc:
        logger.exception("Failed to load forecast resources")

        raise HTTPException(
            status_code=503,
            detail="Forecast model or data unavailable"
        ) from exc



    # Chạy Forecasting Pipeline
    try:
        result = forecast_sales(
            model=model,
            history=history,
            family_mapping=family_mapping,
            cutoff=request.cutoff,
            store_nbr=request.store_nbr,
            family=request.family,
            horizon=request.horizon,
        )
        print(
            "[API] Forecast completed:",
            round(perf_counter() - request_start, 2),
            "seconds",
            flush=True
        )        

    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    # Chuyển DataFrame thành JSON-compatible response
    return [
        ForecastResponse(
            date=pd.Timestamp(row["date"]).date(),
            store_nbr=int(row["store_nbr"]),
            family=str(row["family"]),
            horizon_step=int(row["horizon_step"]),
            predicted_sales=float(row["predicted_sales"])
        )
        for row in result.to_dict(orient="records")
    ]
