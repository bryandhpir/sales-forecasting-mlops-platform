# Hướng dẫn chạy dự án — Data Preparation & EDA

## Mục tiêu

Kiểm tra chất lượng dữ liệu, xây dựng feature table, phân tích phân phối doanh số và điểm bất thường trước khi xây dựng mô hình dự báo có thể cấu hình phạm vi dự báo.

## Thiết lập môi trường

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

Tải tập dữ liệu từ https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data và đặt `train.csv`, `stores.csv` vào `data/raw/`. Không đẩy file dữ liệu gốc lên GitHub.

## Chạy pipeline

```powershell
.\.venv\Scripts\python.exe -m src.data_pipeline
```

Kiểm tra `data/processed/features.parquet` và `data/processed/data_quality_report.json`.

## EDA và kiểm tra anomalies

```powershell
.\.venv\Scripts\python.exe -m src.eda
```

Mở `notebooks/01_eda.ipynb`, chọn kernel `.venv` (Python 3.13) và chạy các cell phân tích. Nếu không đọc được Parquet, kiểm tra thư mục làm việc bằng `Path.cwd()`; khi notebook chạy từ thư mục `notebooks/`, dùng `../data/processed/features.parquet`.

Phân biệt rõ các ngày có doanh số thực tế bằng 0 với ngày hoàn toàn không có bản ghi. Đặc biệt lưu ý ngày 25/12 và 01/01 qua các năm.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Kiểm tra trước khi commit

```powershell
git status --short
```

Không commit `data/raw/*.csv`, `data/processed/*.parquet`, `.venv/`, `.env` hoặc token truy cập. Có thể commit source, notebook và hình EDA.

## Các câu hỏi kỹ thuật cần trả lời khi phát triển mô hình

1. Vì sao chuỗi thời gian cần chia train/validation theo thứ tự thời gian?
2. Tại sao độ dài lịch sử (lag) khác với phạm vi dự báo?
3. Vì sao lag dựa trên doanh số quan sát trong tập đánh giá gây data leakage?
4. Làm sao tạo đặc trưng chỉ từ thông tin có sẵn ở thời điểm dự báo?
