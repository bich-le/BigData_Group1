# BigData_Group1-
supervised learning pipeline spark subject

Đồ án xây dựng quy trình học có giám sát đầu-cuối bằng PySpark MLlib cho dữ liệu chuyến bay.

Hai bài toán của đồ án:

- Hồi quy: dự đoán số phút chuyến bay đến trễ.
- Phân loại nhị phân: dự đoán chuyến bay bị trễ nghiêm trọng.

## Phần Suongmai đã hoàn thành

- Pipeline PySpark MLlib: Imputer → StringIndexer → OneHotEncoder →
  VectorAssembler → StandardScaler → PCA (`k=10`).
- Adapter tích hợp feature của Kiên trước khi chạy MLlib pipeline.
- Temporal train/test split khi dùng feature lookback của Kiên.
- Biểu đồ cumulative explained variance tại `docs/assets/pca_variance.png`.
- MLflow log parameters, metrics, PCA/config artifacts **và fitted
  preprocessing PipelineModel**.
- Smoke test end-to-end có đo thời gian và RSS memory.

`features_pca` là đầu ra bàn giao cho phần model selection/tuning. Repository
chưa tuyên bố hoàn thành benchmark cuối, streaming inference hay registry vì đó
thuộc phần việc các thành viên khác.

## Cấu trúc repository

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── custom_transformers.py
├── mllib_pipeline.py
├── inference.py
├── benchmark_results.py
├── submit_pipeline.sh
├── data/
│   └── README.md
├── notebooks/
│   └── README.md
├── docs/
│   └── REPORT.md
└── models/
    └── README.md
```

## Thiết lập môi trường local

### Yêu cầu

- Eclipse Temurin JDK 17.
- Apache Spark 3.5.9.
- Conda và Python 3.11.

Đặt `JAVA_HOME` trỏ tới thư mục JDK 17 trên máy của bạn và thêm thư mục `bin` của JDK vào `PATH`. Không sao chép đường dẫn cài đặt từ máy khác.

Ví dụ cấu hình cho phiên PowerShell hiện tại:

```powershell
$env:JAVA_HOME = '<đường-dẫn-tới-jdk-17>'
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
```

Tạo và kích hoạt môi trường Conda:

```powershell
conda create -n bigdata-spark python=3.11 pip -y
conda activate bigdata-spark
```

Cài các dependency đã khóa phiên bản:

```powershell
python -m pip install -r requirements.txt
```

Kiểm tra môi trường:

```powershell
java -version
javac -version
python --version
python -c "import pyspark; print('pyspark', pyspark.__version__)"
python -c "import pyarrow; print('pyarrow', pyarrow.__version__)"
python -c "import pandas; print('pandas', pandas.__version__)"
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import mlflow; print('mlflow', mlflow.__version__)"
```

## Chuẩn bị dữ liệu local

Mỗi thành viên tự đặt raw CSV trong `data/raw` và tạo typed Parquet tại `data/typed-parquet`. Xem lệnh dành riêng cho WSL/Bash và PowerShell trong [hướng dẫn chuẩn bị dữ liệu](data/README.md); WSL/Linux được khuyến nghị cho bước ghi Parquet.

## Chạy phần pipeline của Suongmai

Trên Windows, script tự cấu hình `PYSPARK_PYTHON` về đúng Python/Conda đang
chạy script để Spark worker không gọi nhầm lệnh Unix `python3`; khi log model,
MLflow dùng thư mục Temp của Windows thay vì đường dẫn Unix `/tmp`. Bản nộp
cũng kèm `vendor/hadoop-bare-naked-local-fs-0.1.0.jar` để Spark local có thể
lưu model mà không cần cài `winutils.exe`.

Chạy smoke test tích hợp toàn bộ feature của Kiên, PCA và MLflow:

```powershell
python smoke_test.py `
  --rows 160 `
  --with-kien-features `
  --mlflow-experiment "SuongMai-E22-Verification" `
  --output-dir .
```

Kết quả cần kiểm tra:

- `artifacts/smoke_test_metrics.json` có `"smoke_test": "PASS"`.
- `artifacts/preprocessing_summary.json` có
  `"preprocessing_model_logged": true` trong mục `mlflow`.
- MLflow run có PCA plot, config JSON, manifest và artifact
  `preprocessing_pipeline/`.
- Minh chứng đã kiểm tra được ghi tại `artifacts/E22_MLFLOW_EVIDENCE.md` và
  `artifacts/E23_SMOKE_TEST_EVIDENCE.md`.

Chạy trên typed Parquet thật:

```powershell
python mllib_pipeline.py `
  --input data/typed-parquet/flights `
  --airports data/typed-parquet/airports `
  --with-kien-features `
  --mlflow-experiment "BigData-Group1" `
  --output-dir .
```
