# BigData_Group1-
supervised learning pipeline spark subject

Đồ án xây dựng quy trình học có giám sát đầu-cuối bằng PySpark MLlib cho dữ liệu chuyến bay.

Hai bài toán của đồ án:

- Hồi quy: dự đoán số phút chuyến bay đến trễ.
- Phân loại nhị phân: dự đoán chuyến bay bị trễ nghiêm trọng.

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

Các file Python hiện tại mới là bộ khung của đồ án; các chức năng xử lý dữ liệu, pipeline, mô hình và streaming chưa được triển khai.
