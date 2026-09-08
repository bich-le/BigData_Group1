# Dữ liệu

Thư mục `data/` chứa dữ liệu dùng cho project. Raw CSV và full Parquet chỉ lưu ở máy local, **không commit lên Git**.

## 1. Cấu trúc dữ liệu

Mỗi thành viên tự chuẩn bị đúng ba file raw:

```text
data/
├── README.md
├── raw/
│   ├── flights.csv
│   ├── airports.csv
│   └── airlines.csv
└── typed-parquet/
    ├── flights/
    │   ├── MONTH=1/
    │   ├── ...
    │   └── MONTH=12/
    ├── airports/
    └── airlines/
```

## 2. Tạo typed Parquet

Khuyến nghị chạy bằng **WSL/Linux** với environment đã xác minh:

- Java 17
- Python 3.11.16
- PySpark 3.5.9

Từ repository root:

```bash
python scripts/prepare_data.py \
  --input-dir "data/raw" \
  --output-dir "data/typed-parquet" \
  --master "local[2]" \
  --shuffle-partitions 12
```

`prepare_data.py` thực hiện:

1. đọc CSV raw dưới dạng string;
2. kiểm tra header, thứ tự cột và row count;
3. cast theo explicit schema;
4. kiểm tra non-missing value không bị mất khi cast;
5. kiểm tra một số domain/invariant;
6. ghi staging Parquet bằng Snappy;
7. đọc lại để validate schema, row count, lookup và partition;
8. chỉ promote staging thành final output khi toàn bộ validation PASS.

Script **không**:

- impute/drop missing;
- xử lý outlier;
- mapping DOT → IATA;
- tạo feature hoặc label;
- split train/test;
- train model.

Nếu `data/typed-parquet` đã tồn tại, script sẽ từ chối overwrite.

## 3. Kết quả chuyển đổi hiện tại

| Hạng mục | Kết quả |
|---|---:|
| Raw CSV size | 592,430,817 bytes (~565 MB) |
| Typed Parquet size | 142,878,769 bytes (~137 MB) |
| Size reduction | ~75.9% |
| CSV/Parquet ratio | ~4.15x |
| Compression | Snappy |
| Parquet part files | 14 |

`flights` được `partitionBy("MONTH")`, tạo đủ `MONTH=1` đến `MONTH=12`.
Khi query có filter theo tháng, Spark có thể đọc đúng partition cần thiết thay vì scan toàn bộ flights.

## 4. Đọc dữ liệu bằng PySpark

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
root = "data/typed-parquet"

flights = spark.read.parquet(f"{root}/flights")
airports = spark.read.parquet(f"{root}/airports")
airlines = spark.read.parquet(f"{root}/airlines")
```

Typed Parquet vẫn là **typed raw data**, chưa phải cleaned/model-ready dataset.

## 5. Data quality notebook

Notebook profiling chính:

```text
notebooks/01_data_quality_profile.ipynb
```

Notebook chỉ kiểm tra và ghi nhận:

- null / blank;
- validity;
- uniqueness;
- cardinality;
- referential integrity;
- target / eligibility;
- label distribution;
- numeric profile;
- leakage;
- data-quality scorecard.

Notebook **không sửa raw CSV hoặc typed Parquet**.

## 6. Git

Không commit:

```text
data/raw/
data/typed-parquet/
*.csv
*.parquet
```

Tài liệu liên quan:

- `docs/data_dictionary.md`
