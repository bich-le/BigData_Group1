"""
build_modeling_dataset.py  [LOCAL ONLY — KHÔNG push git, chỉ để test ra
output trực tiếp trên máy cá nhân]

Đây là file điều phối (orchestration), KHÔNG chứa logic feature engineering
-- toàn bộ logic thật nằm trong custom_transformers.py (3 file chính thức:
custom_transformers.py, tests/test_custom_transformers.py,
docs/feature_dictionary.md). File này chỉ GỌI LẠI các hàm đã có, theo pattern
tiết kiệm memory (không cần server mạnh):

  1. Build "base" (grain = 1 dòng/1 chuyến bay, composite key
     [AIRLINE, FLIGHT_NUMBER, NGAY_DL, SCHEDULED_DEPARTURE]), checkpoint ra
     Parquet.
  2. Với TỪNG nhóm feature (Haversine, Outlier+flag, TargetEncoder, giờ/
     congestion, Airline-level, Airport-level, Airplane-level, Flight-level,
     Temporal-level):
       a. Đọc lại "base" (+ raw columns cần thiết) TỪ ĐĨA -- không giữ
          lineage/DataFrame cũ trong memory.
       b. Gọi đúng hàm Spark trong custom_transformers.py để tính feature.
       c. Merge kết quả vào "base" theo composite key, ghi đè "base" ra
          Parquet.
       d. Xóa hẳn tham chiếu DataFrame trung gian (del + unpersist) trước
          khi sang nhóm feature tiếp theo.
  3. Lặp lại tới khi base có đủ toàn bộ feature -> đây là dataset DUY NHẤT
     dùng để split train/validation/test (đảm bảo cùng schema, cùng grain).

LƯU Ý VỀ NGUỒN DỮ LIỆU: đọc từ data/parquet (pipeline chính thức của nhóm,
đã qua validate), KHÔNG tự tải lại CSV qua kagglehub -- tránh 2 nguồn dữ
liệu song song không đồng bộ.
"""

import gc
import os
import sys

python_executable = sys.executable
os.environ["PYSPARK_PYTHON"] = python_executable
os.environ["PYSPARK_DRIVER_PYTHON"] = python_executable

os.environ["SPARK_LOCAL_DIRS"] = r"F:\SparkTemp"
from pyspark.sql import SparkSession, functions as F

from custom_transformers import (
    OutlierIQRTruncator,
    TargetEncoder,
    add_flight_timestamp_column,
    add_haversine_distance_feature,
    build_airport_code_mapping,
    add_route_congestion_feature,
    add_scheduled_hour_feature,
    build_airline_level_features,
    build_airplane_level_features,
    build_airport_level_features,
    build_flight_level_features,
    build_temporal_level_features,
)

# -----------------------------------------------------------------------------
# Cấu hình
# -----------------------------------------------------------------------------
DATA_ROOT = "data/parquet"         
BASE_PATH = "data/model-ready/base"  
MODELS_DIR = "models"

# Airport mapping lookup tables used by build_airport_code_mapping().
# These are lookup inputs only; mapping logic remains in custom_transformers.py.
AIRPORT_REF_PATH = "data/raw/L_AIRPORT.csv"
AIRPORT_ID_PATH = "data/raw/L_AIRPORT_ID.csv"

LOOKBACK_WINDOWS_DAYS = (1, 3, 7)         # Airline/Airplane-level
AIRPORT_LOOKBACK_HOURS = (3, 6, 12)        # Airport-level
AIRLINE_AVG_WINDOW_FOR_FLIGHT = 7
TRAIN_TEST_SPLIT = (0.8, 0.2)
SPLIT_SEED = 42
IQR_MULTIPLIER = 1.5

# Composite key chuyến bay -- grain của base, dùng để merge mọi feature vào
FLIGHT_KEY_COLS = ["AIRLINE", "FLIGHT_NUMBER", "YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE"]

LEAKAGE_COLUMNS = [
    "DEPARTURE_TIME", "TAXI_OUT", "WHEELS_OFF", "ELAPSED_TIME", "AIR_TIME",
    "WHEELS_ON", "TAXI_IN", "ARRIVAL_TIME", "CANCELLATION_REASON",
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY", "DIVERTED", "CANCELLED",
]


def get_spark():
    return (
        SparkSession.builder.appName("build_modeling_dataset")
        .master("local[2]")
        .config("spark.pyspark.python", python_executable)
        .config("spark.pyspark.driver.python", python_executable)
        .config("spark.driver.memory", "8g")  # điều chỉnh theo RAM máy bạn
        .config("spark.local.dir", r"F:\SparkTemp")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "12")
        .getOrCreate()
    )


def read_lookup_table(spark, path):
    """Read a lookup table used by the airport-code mapping.

    Supports CSV (the notebook/source files) and Parquet without moving any
    mapping logic into this orchestration script.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Airport mapping lookup file not found: {path}\n"
            "Make sure L_AIRPORT.csv and L_AIRPORT_ID.csv are available at the configured paths."
        )

    if path.lower().endswith(".csv"):
        return spark.read.option("header", True).option("inferSchema", True).csv(path)
    return spark.read.parquet(path)


def checkpoint_and_reload(spark, df, path):
    """Ghi DataFrame ra Parquet rồi đọc lại -- cắt lineage, giải phóng memory
    của toàn bộ query plan cũ trước khi tiếp tục. Đây là bước cốt lõi của
    pattern tiết kiệm memory: không bao giờ để 1 DAG dài tích lũy qua nhiều
    bước feature engineering."""
    df.write.mode("overwrite").parquet(path)
    return spark.read.parquet(path)


def merge_feature_into_base(spark, feature_df, feature_cols, base_path=BASE_PATH):
    """
    Đọc base hiện tại từ đĩa, merge (join theo FLIGHT_KEY_COLS) với
    feature_df (chỉ lấy đúng key + feature_cols mới), ghi đè base ra Parquet,
    rồi XÓA hẳn mọi tham chiếu DataFrame trung gian trước khi trả về base mới.
    """
    base = spark.read.parquet(base_path)
    to_merge = feature_df.select(*FLIGHT_KEY_COLS, *feature_cols)
    merged = base.join(to_merge, on=FLIGHT_KEY_COLS, how="left")
    merged.write.mode("overwrite").parquet(base_path + "_tmp")

    # Xóa base cũ, thay bằng bản mới -- tránh giữ 2 bản base cùng lúc trên đĩa
    import shutil

    shutil.rmtree(base_path, ignore_errors=True)
    shutil.move(base_path + "_tmp", base_path)

    del base, to_merge, merged, feature_df
    gc.collect()

    return spark.read.parquet(base_path)


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    os.makedirs("data/model-ready", exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # -------------------------------------------------------------------
    # BƯỚC 0 — Build base: grain = 1 dòng/1 chuyến bay, composite key,
    # eligibility filter, checkpoint ra Parquet ngay lập tức
    # -------------------------------------------------------------------
    flights = spark.read.parquet(f"{DATA_ROOT}/flights")
    airports = spark.read.parquet(f"{DATA_ROOT}/airports")
    airport_ref = read_lookup_table(spark, AIRPORT_REF_PATH)
    airport_id = read_lookup_table(spark, AIRPORT_ID_PATH)

    base = flights.filter(
        (F.col("ARRIVAL_DELAY").isNotNull())
        & (F.col("CANCELLED") == 0)
        & (F.col("DIVERTED") == 0)
    ).withColumn(
        "label_severe_delay", F.when(F.col("ARRIVAL_DELAY") > 30, 1).otherwise(0)
    )
    base = add_flight_timestamp_column(base)

    n_base = base.count()
    n_distinct_key = base.select(*FLIGHT_KEY_COLS).distinct().count()
    print(f"[BASE] {n_base:,} dòng, {n_distinct_key:,} composite key khác nhau")
    if n_distinct_key != n_base:
        print(
            f"⚠️  CẢNH BÁO: {n_base - n_distinct_key:,} dòng trùng composite key -- "
            "cần xem xét thêm cột vào FLIGHT_KEY_COLS trước khi tiếp tục."
        )

    base.write.mode("overwrite").parquet(BASE_PATH)
    del flights
    gc.collect()
    print(f"[BASE] Đã checkpoint ra {BASE_PATH}\n")

    # -------------------------------------------------------------------
    # BƯỚC 1 — Section 1: Haversine distance
    # -------------------------------------------------------------------
    base = spark.read.parquet(BASE_PATH)

    # Build DOT/IATA mapping once from the same eligible flight universe, then
    # pass the resulting mapping table into the custom transformer. The actual
    # matching logic (direct/level 1/2/3/manual) stays in custom_transformers.py.
    airport_mapping = build_airport_code_mapping(
        base,
        airports,
        airport_ref_df=airport_ref,
        airport_id_df=airport_id,
    )
    print("[1] Airport code mapping built.")

    haversine_result = add_haversine_distance_feature(
        base,
        airports,
        airport_mapping_df=airport_mapping,
    )
    base = merge_feature_into_base(
        spark, haversine_result,
        ["haversine_distance_km", "route_distance_km", "has_haversine_distance", "distance_feature_km"],
    )
    del haversine_result, airport_mapping, airport_ref, airport_id
    gc.collect()
    print(f"[1] Section 1 (Haversine) — {base.count():,} dòng, {len(base.columns)} cột\n")

    # -------------------------------------------------------------------
    # BƯỚC 2 — Section 4: scheduled_hour + route_congestion_count
    # (cần đọc thêm SCHEDULED_DEPARTURE/ORIGIN_AIRPORT/MONTH/DAY -- đã có sẵn
    # trong base từ đầu, không cần join thêm từ flights gốc)
    # -------------------------------------------------------------------
    base = spark.read.parquet(BASE_PATH)
    hour_result = add_scheduled_hour_feature(base)
    congestion_result = add_route_congestion_feature(hour_result)
    base = merge_feature_into_base(
        spark, congestion_result, ["scheduled_hour", "route_congestion_count"]
    )
    del hour_result, congestion_result
    gc.collect()
    print(f"[2] Section 4 (giờ + congestion) — {base.count():,} dòng, {len(base.columns)} cột\n")

    # -------------------------------------------------------------------
    # BƯỚC 3 — Section 5.1: Airline-level
    # -------------------------------------------------------------------
    base = spark.read.parquet(BASE_PATH)
    airline_result = build_airline_level_features(base, lookback_windows=LOOKBACK_WINDOWS_DAYS)
    airline_new_cols = [c for c in airline_result.columns if c not in base.columns]
    base = merge_feature_into_base(spark, airline_result, airline_new_cols)
    del airline_result
    gc.collect()
    print(f"[3] Airline-level — {base.count():,} dòng, {len(base.columns)} cột\n")

    # -------------------------------------------------------------------
    # BƯỚC 4 — Section 5.2: Airport-level (kỳ phái sinh theo GIỜ)
    # -------------------------------------------------------------------
    base = spark.read.parquet(BASE_PATH)
    airport_result = build_airport_level_features(base, lookback_windows_hours=AIRPORT_LOOKBACK_HOURS)
    airport_new_cols = [c for c in airport_result.columns if c not in base.columns]
    base = merge_feature_into_base(spark, airport_result, airport_new_cols)
    del airport_result
    gc.collect()
    print(f"[4] Airport-level — {base.count():,} dòng, {len(base.columns)} cột\n")

    # -------------------------------------------------------------------
    # BƯỚC 5 — Section 5.3: Airplane-level
    # -------------------------------------------------------------------
    base = spark.read.parquet(BASE_PATH)
    airplane_result = build_airplane_level_features(base, lookback_windows=LOOKBACK_WINDOWS_DAYS)
    airplane_new_cols = [c for c in airplane_result.columns if c not in base.columns]
    base = merge_feature_into_base(spark, airplane_result, airplane_new_cols)
    del airplane_result
    gc.collect()
    print(f"[5] Airplane-level — {base.count():,} dòng, {len(base.columns)} cột\n")

    # -------------------------------------------------------------------
    # BƯỚC 6 — Section 5.4: Flight-level (cần airline_*_lag7d đã có ở bước 3)
    # -------------------------------------------------------------------
    base = spark.read.parquet(BASE_PATH)
    flight_result = build_flight_level_features(base, airline_avg_window=AIRLINE_AVG_WINDOW_FOR_FLIGHT)
    flight_new_cols = [c for c in flight_result.columns if c not in base.columns]
    base = merge_feature_into_base(spark, flight_result, flight_new_cols)
    del flight_result
    gc.collect()
    print(f"[6] Flight-level — {base.count():,} dòng, {len(base.columns)} cột\n")

    # -------------------------------------------------------------------
    # BƯỚC 7 — Section 5.5: Temporal-level
    # -------------------------------------------------------------------
    base = spark.read.parquet(BASE_PATH)
    temporal_result = build_temporal_level_features(base)
    temporal_new_cols = [c for c in temporal_result.columns if c not in base.columns]
    base = merge_feature_into_base(spark, temporal_result, temporal_new_cols)
    del temporal_result
    gc.collect()
    print(f"[7] Temporal-level — {base.count():,} dòng, {len(base.columns)} cột\n")

    # -------------------------------------------------------------------
    # BƯỚC 8 — Time-based split
    # 80% thời gian đầu = train
    # 20% thời gian cuối = test
    # -------------------------------------------------------------------

    base = spark.read.parquet(BASE_PATH)

    split_date = base.select(
        F.expr(
            "percentile_approx(to_date(FLIGHT_TIMESTAMP), 0.8)"
        ).alias("split_date")
    ).collect()[0]["split_date"]

    train_df = base.filter(
        F.to_date(F.col("FLIGHT_TIMESTAMP")) < F.lit(split_date)
    )

    test_df = base.filter(
        F.to_date(F.col("FLIGHT_TIMESTAMP")) >= F.lit(split_date)
    )

    n_train = train_df.count()
    n_test = test_df.count()

    print(f"[8] Time split tại: {split_date}")
    print(f"[8] Train: {n_train:,} dòng | Test: {n_test:,} dòng")
    print(f"[8] Train ratio: {n_train / (n_train + n_test):.2%}")
    print(f"[8] Test ratio:  {n_test / (n_train + n_test):.2%}\n")

    # Persist the split before releasing DataFrames so the following
    # leakage-safe fit step can reload clean Parquet datasets.
    train_path = "data/model-ready/train_raw"
    test_path = "data/model-ready/test_raw"

    train_df.write.mode("overwrite").parquet(train_path)
    test_df.write.mode("overwrite").parquet(test_path)

    del base, train_df, test_df
    gc.collect()

    train_df = spark.read.parquet(train_path)
    test_df = spark.read.parquet(test_path)

    n_train = train_df.count()
    n_test = test_df.count()

    print(f"[8] Time split tại: {split_date}")
    print(f"[8] Train: {n_train:,} dòng | Test: {n_test:,} dòng")
    print(f"[8] Train ratio: {n_train / (n_train + n_test):.2%}")
    print(f"[8] Test ratio:  {n_test / (n_train + n_test):.2%}\n")

    # -------------------------------------------------------------------
    # BƯỚC 9 — Section 2 (Outlier + flag) + Section 3 (Target Encoding)
    # PHẢI fit CHỈ trên train, transform cả train/test
    # -------------------------------------------------------------------
    outlier_truncator = OutlierIQRTruncator(
        inputCols=["distance_feature_km", "SCHEDULED_TIME"],
        outputCols=["distance_feature_km_clipped", "scheduled_time_clipped"],
        outlierFlagCols=["distance_feature_km_is_outlier", "scheduled_time_is_outlier"],
        iqrMultiplier=IQR_MULTIPLIER,
    )
    outlier_model = outlier_truncator.fit(train_df)
    train_df = outlier_model.transform(train_df)
    test_df = outlier_model.transform(test_df)

    target_encoder = TargetEncoder(
        inputCols=["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "TAIL_NUMBER"],
        outputCols=["airline_encoded", "origin_airport_encoded", "destination_airport_encoded", "tail_number_encoded"],
        targetCol="ARRIVAL_DELAY",
        smoothing=10.0,
    )
    encoder_model = target_encoder.fit(train_df)
    train_df = encoder_model.transform(train_df)
    test_df = encoder_model.transform(test_df)

    print("[9] Đã fit Outlier + TargetEncoder trên train, transform cả train/test")
    print(outlier_model.bounds_summary().to_string(index=False))

    # -------------------------------------------------------------------
    # BƯỚC 10 — Loại cột leakage, sắp xếp cột, lưu output cuối
    # -------------------------------------------------------------------

    existing_leakage_cols = [
        c for c in LEAKAGE_COLUMNS
        if c in train_df.columns
    ]

    train_final = train_df.drop(*existing_leakage_cols)
    test_final = test_df.drop(*existing_leakage_cols)

    key_cols_first = (
        FLIGHT_KEY_COLS
        + [
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "label_severe_delay",
        ]
    )

    key_cols_present = [
        c for c in key_cols_first
        if c in train_final.columns
    ]

    remaining_cols = [
        c for c in train_final.columns
        if c not in key_cols_present
    ]

    ordered_cols = key_cols_present + remaining_cols

    train_final = train_final.select(*ordered_cols)
    test_final = test_final.select(*ordered_cols)

    # -------------------------------------------------------------------
    # Ghi output cuối
    # -------------------------------------------------------------------

    train_path = "data/model-ready/train"
    test_path = "data/model-ready/test"

    train_final.write.mode("overwrite").parquet(train_path)
    test_final.write.mode("overwrite").parquet(test_path)

    # -------------------------------------------------------------------
    # Lưu fitted models
    # -------------------------------------------------------------------

    outlier_model.write().overwrite().save(
        f"{MODELS_DIR}/outlier_truncator"
    )

    encoder_model.write().overwrite().save(
        f"{MODELS_DIR}/target_encoder"
    )

    print("\n[10] Đã lưu:")
    print(
        f"  - {train_path} "
        f"({n_train:,} dòng, {len(train_final.columns)} cột)"
    )
    print(
        f"  - {test_path} "
        f"({n_test:,} dòng, {len(test_final.columns)} cột)"
    )
    print(
        f"  - {MODELS_DIR}/outlier_truncator, "
        f"{MODELS_DIR}/target_encoder"
    )
    print(f"\nCột đầu bảng (khóa): {key_cols_present}")

    # ---------------------------------------------------------------
    # Lưu fitted models
    # ---------------------------------------------------------------
    outlier_model.write().overwrite().save(
        f"{MODELS_DIR}/outlier_truncator"
    )

    encoder_model.write().overwrite().save(
        f"{MODELS_DIR}/target_encoder"
    )

    print("\n[10] Đã lưu:")
    print(
        f"  - {train_path} "
        f"({n_train:,} dòng, {len(train_final.columns)} cột)"
    )
    print(
        f"  - {test_path} "
        f"({n_test:,} dòng, {len(test_final.columns)} cột)"
    )
    print(
        f"  - {MODELS_DIR}/outlier_truncator, "
        f"{MODELS_DIR}/target_encoder"
    )
    print(f"\nCột đầu bảng (khóa): {key_cols_present}")


if __name__ == "__main__":
    main()