"""
tests/test_custom_transformers.py

Unit test chính thức cho custom_transformers.py, chạy bằng pytest.
Khác với sanity check trong __main__ (chạy trên data thật, chỉ để mắt
kiểm tra nhanh), test ở đây dùng dữ liệu giả lập nhỏ, có expected value
rõ ràng, để CI/kiểm thử tự động phát hiện lỗi khi code thay đổi.

Chạy: pytest tests/test_custom_transformers.py -v
"""

import math
import sys

import pytest
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

sys.path.insert(0, ".")  # cho phép import custom_transformers từ project root
from custom_transformers import (
    OutlierIQRTruncator,
    TargetEncoder,
    TargetEncoderModel,
    add_haversine_distance_feature,
    haversine_km,
    patch_missing_airport_coordinates,
)


@pytest.fixture(scope="module")
def spark():
    import os

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("test_custom_transformers")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )
    yield spark
    spark.stop()


# =============================================================================
# SECTION 1 — Haversine UDF & fallback
# =============================================================================


def test_haversine_same_point_is_zero(spark):
    df = spark.createDataFrame([(10.0, 20.0, 10.0, 20.0)], ["lat1", "lon1", "lat2", "lon2"])
    result = df.withColumn(
        "d", haversine_km(F.col("lat1"), F.col("lon1"), F.col("lat2"), F.col("lon2"))
    ).collect()[0]["d"]
    assert result == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance_hanoi_hcmc(spark):
    # Hà Nội (21.0285, 105.8542) - TP.HCM (10.7626, 106.6602)
    # Khoảng cách great-circle thực tế ~ 1140-1160 km, dùng để kiểm chứng công thức
    df = spark.createDataFrame(
        [(21.0285, 105.8542, 10.7626, 106.6602)], ["lat1", "lon1", "lat2", "lon2"]
    )
    result = df.withColumn(
        "d", haversine_km(F.col("lat1"), F.col("lon1"), F.col("lat2"), F.col("lon2"))
    ).collect()[0]["d"]
    assert 1100 < result < 1200


def test_haversine_null_coordinate_returns_null(spark):
    schema = StructType(
        [
            StructField("lat1", DoubleType(), True),
            StructField("lon1", DoubleType(), True),
            StructField("lat2", DoubleType(), True),
            StructField("lon2", DoubleType(), True),
        ]
    )
    df = spark.createDataFrame([(None, 20.0, 10.0, 20.0)], schema)
    result = df.withColumn(
        "d", haversine_km(F.col("lat1"), F.col("lon1"), F.col("lat2"), F.col("lon2"))
    ).collect()[0]["d"]
    assert result is None


def test_haversine_invalid_domain_returns_null(spark):
    # lat=200 ngoài domain hợp lệ [-90, 90] -> phải trả NULL, không tính bừa
    df = spark.createDataFrame([(200.0, 20.0, 10.0, 20.0)], ["lat1", "lon1", "lat2", "lon2"])
    result = df.withColumn(
        "d", haversine_km(F.col("lat1"), F.col("lon1"), F.col("lat2"), F.col("lon2"))
    ).collect()[0]["d"]
    assert result is None


def test_patch_missing_airport_coordinates_fills_only_target_codes(spark):
    airports = spark.createDataFrame(
        [("ECP", None, None), ("JFK", 40.6413, -73.7781)],
        ["IATA_CODE", "LATITUDE", "LONGITUDE"],
    )
    patched = patch_missing_airport_coordinates(airports)
    rows = {r["IATA_CODE"]: (r["LATITUDE"], r["LONGITUDE"]) for r in patched.collect()}

    assert rows["ECP"] == pytest.approx((30.3417, -85.7973))
    # JFK không nằm trong danh sách patch -> giữ nguyên, không bị ghi đè
    assert rows["JFK"] == pytest.approx((40.6413, -73.7781))


def test_add_haversine_distance_feature_fallback_when_airport_unknown(spark):
    flights = spark.createDataFrame(
        [("ZZZ", "YYY", 500)],  # mã sân bay giả, không có trong bảng airports
        ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DISTANCE"],
    )
    airports = spark.createDataFrame(
        [], StructType(
            [
                StructField("IATA_CODE", StringType(), True),
                StructField("LATITUDE", DoubleType(), True),
                StructField("LONGITUDE", DoubleType(), True),
            ]
        )
    )
    result = add_haversine_distance_feature(flights, airports).collect()[0]

    assert result["has_haversine_distance"] is False
    assert result["distance_feature_km"] == pytest.approx(500 * 1.60934)


# =============================================================================
# SECTION 2 — OutlierIQRTruncator
# =============================================================================


def test_outlier_truncator_clips_extreme_value(spark):
    data = [(v,) for v in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1000]]
    df = spark.createDataFrame(data, ["value"])

    truncator = OutlierIQRTruncator(
        inputCols=["value"], outputCols=["value_clipped"], iqrMultiplier=1.5
    )
    model = truncator.fit(df)
    result = {r["value"]: r["value_clipped"] for r in model.transform(df).collect()}

    assert result[1000] < 1000  # outlier phải bị clip
    assert result[15] == 15  # giá trị bình thường không đổi


def test_outlier_truncator_bounds_computed_only_from_fit_data(spark):
    """
    Kiểm chứng KHÔNG leak: bounds tính từ train, áp dụng lên test có outlier
    cực đoan (1000, 2000, -500) mà KHÔNG được tính lại theo test.
    """
    train = spark.createDataFrame([(v,) for v in range(1, 51)], ["value"])  # 1..50, không outlier
    test = spark.createDataFrame([(v,) for v in [1000, 2000, -500]], ["value"])

    truncator = OutlierIQRTruncator(inputCols=["value"], outputCols=["value_clipped"])
    model = truncator.fit(train)

    upper_bound_before = model.bounds_summary().iloc[0]["upper_bound"]

    test_result = model.transform(test).collect()
    for row in test_result:
        # mọi giá trị test đều phải bị clip về đúng bound đã tính từ train
        assert row["value_clipped"] <= upper_bound_before + 1e-6

    upper_bound_after = model.bounds_summary().iloc[0]["upper_bound"]
    # bounds không được thay đổi chỉ vì transform() được gọi trên data khác
    assert upper_bound_before == upper_bound_after


def test_outlier_truncator_mismatched_cols_raises(spark):
    df = spark.createDataFrame([(1,), (2,)], ["value"])
    truncator = OutlierIQRTruncator(inputCols=["value"], outputCols=["a", "b"])
    with pytest.raises(ValueError):
        truncator.fit(df)


def test_outlier_truncator_multiple_columns_independent_bounds(spark):
    data = [(v, v * 100) for v in [10, 11, 12, 13, 14, 1000]]
    df = spark.createDataFrame(data, ["small_scale", "large_scale"])

    truncator = OutlierIQRTruncator(
        inputCols=["small_scale", "large_scale"],
        outputCols=["small_scale_clipped", "large_scale_clipped"],
    )
    model = truncator.fit(df)
    bounds = model.bounds_summary().set_index("output_col")

    # 2 cột phải có bounds khác nhau (không dùng chung 1 giá trị nhầm cột)
    assert bounds.loc["small_scale_clipped", "upper_bound"] != bounds.loc[
        "large_scale_clipped", "upper_bound"
    ]


def test_outlier_truncator_save_load_roundtrip(spark, tmp_path):
    """
    Kiểm chứng bản vá persistence (task 4.4): bounds giờ lưu qua Param JSON
    thay vì thuộc tính Python thường -- save()/load() phải hoạt động đúng.
    """
    from custom_transformers import OutlierIQRTruncatorModel

    df = spark.createDataFrame([(v,) for v in [10, 11, 12, 13, 14, 1000]], ["value"])
    truncator = OutlierIQRTruncator(inputCols=["value"], outputCols=["value_clipped"])
    model = truncator.fit(df)

    save_path = str(tmp_path / "outlier_truncator_model")
    model.write().overwrite().save(save_path)

    loaded_model = OutlierIQRTruncatorModel.load(save_path)

    original_result = {r["value"]: r["value_clipped"] for r in model.transform(df).collect()}
    loaded_result = {r["value"]: r["value_clipped"] for r in loaded_model.transform(df).collect()}

    assert original_result == pytest.approx(loaded_result)


# =============================================================================
# SECTION 3 — TargetEncoder
# =============================================================================


def test_target_encoder_computes_correct_smoothed_mean(spark):
    # A: target=[10,20,30] -> mean=20, count=3
    # B: target=[100,200]  -> mean=150, count=2
    # global_mean = (10+20+30+100+200)/5 = 72
    # smoothing=10 (default)
    # encoded_A = (3*20 + 10*72) / (3+10) = 780/13 = 60.0
    # encoded_B = (2*150 + 10*72) / (2+10) = 1020/12 = 85.0
    df = spark.createDataFrame(
        [("A", 10.0), ("A", 20.0), ("A", 30.0), ("B", 100.0), ("B", 200.0)],
        ["category", "target"],
    )
    encoder = TargetEncoder(
        inputCols=["category"], outputCols=["category_encoded"], targetCol="target", smoothing=10.0
    )
    model = encoder.fit(df)

    result = {r["category"]: r["category_encoded"] for r in model.transform(df).collect()}
    # Nhiều dòng cùng category -> lấy 1 giá trị đại diện để so sánh
    assert result["A"] == pytest.approx(60.0, rel=1e-3)
    assert result["B"] == pytest.approx(85.0, rel=1e-3)


def test_target_encoder_unseen_category_falls_back_to_global_mean(spark):
    train = spark.createDataFrame(
        [("A", 10.0), ("A", 20.0), ("B", 100.0), ("B", 200.0)], ["category", "target"]
    )
    # global_mean(train) = (10+20+100+200)/4 = 82.5
    encoder = TargetEncoder(
        inputCols=["category"], outputCols=["category_encoded"], targetCol="target"
    )
    model = encoder.fit(train)

    test_schema = StructType(
        [
            StructField("category", StringType(), True),
            StructField("target", DoubleType(), True),
        ]
    )
    test = spark.createDataFrame([("C", None)], test_schema)  # "C" chưa từng thấy
    result = model.transform(test).collect()[0]["category_encoded"]

    assert result == pytest.approx(82.5, rel=1e-3)


def test_target_encoder_no_leak_bounds_fixed_after_fit(spark):
    """
    Kiểm chứng KHÔNG leak: encoding tính từ train, áp dụng lên test có
    phân phối khác hẳn mà KHÔNG được tính lại theo test.
    """
    train = spark.createDataFrame(
        [("X", 5.0), ("X", 5.0), ("X", 5.0)], ["category", "target"]
    )  # mean=5.0
    encoder = TargetEncoder(
        inputCols=["category"], outputCols=["category_encoded"], targetCol="target", smoothing=0.0
    )
    model = encoder.fit(train)  # smoothing=0 -> encoded("X") = 5.0 chính xác

    # test có target cực đoan khác hẳn — NHƯNG target ở test không được dùng
    # để tính lại encoding, vì transform() chỉ tra cứu map đã fit sẵn
    test = spark.createDataFrame([("X", 9999.0), ("X", -9999.0)], ["category", "target"])
    result = model.transform(test).collect()

    for row in result:
        assert row["category_encoded"] == pytest.approx(5.0, rel=1e-3)


def test_target_encoder_save_load_roundtrip(spark, tmp_path):
    """
    Kiểm chứng đúng vấn đề persistence đã phát hiện khi thiết kế: encoding
    map lưu qua Param JSON phải load lại đúng, không mất dữ liệu.
    """
    df = spark.createDataFrame(
        [("A", 10.0), ("A", 20.0), ("B", 100.0)], ["category", "target"]
    )
    encoder = TargetEncoder(
        inputCols=["category"], outputCols=["category_encoded"], targetCol="target"
    )
    model = encoder.fit(df)

    save_path = str(tmp_path / "target_encoder_model")
    model.write().overwrite().save(save_path)

    loaded_model = TargetEncoderModel.load(save_path)

    original_result = {r["category"]: r["category_encoded"] for r in model.transform(df).collect()}
    loaded_result = {r["category"]: r["category_encoded"] for r in loaded_model.transform(df).collect()}

    assert original_result == pytest.approx(loaded_result)


# =============================================================================
# SECTION 4 — Feature cơ bản khác (giờ trong ngày, route congestion)
# =============================================================================


def test_scheduled_hour_normal_case(spark):
    from custom_transformers import add_scheduled_hour_feature

    df = spark.createDataFrame([("0930",), ("1745",)], ["SCHEDULED_DEPARTURE"])
    result = {r["SCHEDULED_DEPARTURE"]: r["scheduled_hour"] for r in add_scheduled_hour_feature(df).collect()}

    assert result["0930"] == 9
    assert result["1745"] == 17


def test_scheduled_hour_handles_2400_convention(spark):
    from custom_transformers import add_scheduled_hour_feature

    # "2400" là convention hợp lệ theo data_dictionary.md -> phải quy về giờ 0
    df = spark.createDataFrame([("2400",)], ["SCHEDULED_DEPARTURE"])
    result = add_scheduled_hour_feature(df).collect()[0]["scheduled_hour"]

    assert result == 0


def test_scheduled_hour_handles_unpadded_time(spark):
    from custom_transformers import add_scheduled_hour_feature

    # "5" (giờ 00:05, một số dòng dữ liệu gốc không pad đủ 4 ký tự)
    df = spark.createDataFrame([("5",)], ["SCHEDULED_DEPARTURE"])
    result = add_scheduled_hour_feature(df).collect()[0]["scheduled_hour"]

    assert result == 0


def test_route_congestion_counts_correctly(spark):
    from custom_transformers import add_route_congestion_feature

    # 3 chuyến cùng sân bay/ngày/giờ, 1 chuyến khác giờ -> congestion phải phân biệt đúng
    df = spark.createDataFrame(
        [
            ("JFK", 1, 15, 9),
            ("JFK", 1, 15, 9),
            ("JFK", 1, 15, 9),
            ("JFK", 1, 15, 10),
        ],
        ["ORIGIN_AIRPORT", "MONTH", "DAY", "scheduled_hour"],
    )
    result = add_route_congestion_feature(df).collect()

    congestion_by_hour = {}
    for row in result:
        congestion_by_hour[row["scheduled_hour"]] = row["route_congestion_count"]

    assert congestion_by_hour[9] == 3
    assert congestion_by_hour[10] == 1