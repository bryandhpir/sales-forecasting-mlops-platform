
import pandas as pd


def build_future_calendar(
    cutoff,
    store_nbr,
    family,
    horizon
):
    # Kiểm tra horizon
    if (
        not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or horizon <= 0
    ):
        raise ValueError("horizon must be a positive integer")

    # Chuyển cutoff sang kiểu Timestamp
    cutoff = pd.Timestamp(cutoff).normalize()

    if pd.isna(cutoff):
        raise ValueError("cutoff must be a valid date")

    # Tạo các ngày cần dự báo
    future_dates = pd.date_range(
        start=cutoff + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    # Tạo DataFrame
    future = pd.DataFrame({
        "date": future_dates,
        "store_nbr": store_nbr,
        "family": family,
        "horizon_step": range(1, horizon + 1)
    })

    # Tạo calendar features
    future["year"] = future["date"].dt.year
    future["month"] = future["date"].dt.month
    future["day_of_month"] = future["date"].dt.day
    future["day_of_week"] = future["date"].dt.day_of_week

    future["is_weekend"] = (
        future["day_of_week"] >= 5
    ).astype(int)

    # Giữ công thức reference_date giống Training
    steps = future["horizon_step"]

    cycles = ((steps - 1) // 7) + 1

    future["reference_date"] = (
        future["date"]
        - pd.to_timedelta(cycles * 7, unit="D")
    )

    # Không được tham chiếu tới ngày sau cutoff
    assert (future["reference_date"] <= cutoff).all()

    return future

def attach_historical_features(future, history, cutoff):
    cutoff = pd.Timestamp(cutoff).normalize()

    keys = ["store_nbr", "family_code"]

    lag_columns = [
        "sales_lag_7",
        "sales_lag_28",
        "sales_rolling_mean_7"
    ]

    # Chỉ sử dụng lịch sử đến cutoff
    past_data = history.loc[
        pd.to_datetime(history["date"]) <= cutoff
    ].copy()

    past_data["date"] = pd.to_datetime(
        past_data["date"]
    ).dt.normalize()

    if past_data.duplicated(["date", *keys]).any():
        raise ValueError("Duplicate historical keys")

    result = future.copy()

    # Đảm bảo không tham chiếu tương lai
    if (
        (result["date"] <= cutoff).any()
        or (result["reference_date"] > cutoff).any()
        or result[["date", "reference_date"]].isna().any().any()
    ):
        raise ValueError("Invalid forecast or reference dates")

    # 1. Lấy doanh số tại reference_date
    references = past_data[
        ["date", *keys, "sales"]
    ].rename(columns={
        "date": "reference_date",
        "sales": "baseline_reference"
    })

    result = result.merge(
        references,
        on=["reference_date", *keys],
        how="left",
        validate="many_to_one"
    )

    # 2. Lấy snapshot tại cutoff
    snapshot = past_data.loc[
        past_data["date"] == cutoff,
        [*keys, "sales", *lag_columns]
    ].rename(columns={
        "sales": "cutoff_sales",
        "sales_lag_7": "cutoff_lag_7",
        "sales_lag_28": "cutoff_lag_28",
        "sales_rolling_mean_7": "cutoff_rolling_mean_7"
    })

    result = result.merge(
        snapshot,
        on=keys,
        how="left",
        validate="many_to_one",
        indicator=True
    )

    # Mỗi chuỗi phải có snapshot tại cutoff
    if (result["_merge"] != "both").any():
        raise ValueError("Missing cutoff snapshot")

    return result.drop(columns="_merge")
