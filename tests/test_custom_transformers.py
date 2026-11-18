"""
tests/test_custom_transformers.py

Unit test chính thức cho custom_transformers.py, chạy bằng pytest.
Khác với sanity check trong __main__ (chạy trên data thật, chỉ để mắt
kiểm tra nhanh), test ở đây dùng dữ liệu giả lập nhỏ, có expected value
rõ ràng, để CI/kiểm thử tự động phát hiện lỗi khi code thay đổi.

Chạy: pytest tests/test_custom_transformers.py -v
"""

<<<<<<< HEAD
import math
=======
import os
>>>>>>> Kien
import sys

import pytest
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

sys.path.insert(0, ".")  # cho phép import custom_transformers từ project root
<<<<<<< HEAD
from mllib_pipeline import configure_local_filesystem, configure_pyspark_python
=======
>>>>>>> Kien
from custom_transformers import (
    OutlierIQRTruncator,
    TargetEncoder,
    TargetEncoderModel,
    add_haversine_distance_feature,
<<<<<<< HEAD
=======
    build_airport_code_mapping,
>>>>>>> Kien
    haversine_km,
    patch_missing_airport_coordinates,
)


<<<<<<< HEAD
@pytest.fixture(scope="module")
def spark():
    configure_pyspark_python()
    spark = configure_local_filesystem(
=======
# -----------------------------------------------------------------------------
# MỘT SparkSession duy nhất cho cả file (scope="session").
# Trước đây fixture `spark` bị định nghĩa HAI LẦN trong file này -> định nghĩa
# sau ghi đè định nghĩa trước, và mỗi lần chạy lại tạo/stop session, rất dễ
# làm chết SparkContext giữa chừng (triệu chứng: test đầu pass, toàn bộ test
# sau fail hàng loạt). Kèm theo đó là cấu hình mặc định
# spark.sql.shuffle.partitions=200 -- với dữ liệu test vài dòng thì đây là
# nguyên nhân chính khiến suite chạy tới vài phút.
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    session = (
>>>>>>> Kien
        SparkSession.builder.master("local[2]")
        .appName("test_custom_transformers")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
<<<<<<< HEAD
    ).getOrCreate()
    yield spark
    spark.stop()
=======
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


# =============================================================================
# SECTION 0 — DOT -> IATA airport mapping
# =============================================================================


def make_airports(spark):
    return spark.createDataFrame([
        ("ATL", "Hartsfield Airport", "Atlanta", "GA", 33.6407, -84.4277),
        ("XYZ", "Test Airport", "Testville", "TX", 30.0, -97.0),
        ("AUS", "Austin Airport", "Austin", "TX", 30.1975, -97.6664),
        ("SMF", "Sacramento International", "Sacramento", "CA", 38.6951, -121.5908),
        ("CVG", "Cincinnati/Northern Kentucky International", "Cincinnati", "KY", 39.0488, -84.6678),
    ], ["IATA_CODE", "AIRPORT", "CITY", "STATE", "LATITUDE", "LONGITUDE"])


def make_ref(spark):
    return spark.createDataFrame([
        ("10001", "Atlanta, GA: Hartsfield Airport"),   # Level 1
        ("10002", "Austin, TX: Something Else"),        # Level 2
        ("10003", "Mystery, CA: Sacramento International"), # Level 3
    ], ["Code", "Description"])


def make_id(spark):
    return spark.createDataFrame([
        ("10001", "Atlanta, GA: Hartsfield Airport"),
        ("10002", "Austin, TX: Something Else"),
        ("10003", "Mystery, CA: Sacramento International"),
        ("11193", "Cincinnati, OH: Manual fallback"),
    ], ["Code", "Description"])


def test_mapping_direct_iata_is_preserved(spark):
    flights = spark.createDataFrame([("ATL", "XYZ")], ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])
    m = build_airport_code_mapping(flights, make_airports(spark), make_ref(spark), make_id(spark), manual_mapping={})
    rows = {r["AIRPORT_CODE"]: r["IATA_CODE"] for r in m.collect()}
    assert rows["ATL"] == "ATL"
    assert rows["XYZ"] == "XYZ"


def test_mapping_level_1_uses_state_city_and_airport(spark):
    flights = spark.createDataFrame([("10001", "ATL")], ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])
    m = build_airport_code_mapping(flights, make_airports(spark), make_ref(spark), make_id(spark), manual_mapping={})
    rows = {r["AIRPORT_CODE"]: r["IATA_CODE"] for r in m.collect()}
    assert rows["10001"] == "ATL"


def test_mapping_level_2_uses_state_and_city(spark):
    flights = spark.createDataFrame([("10002", "ATL")], ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])
    m = build_airport_code_mapping(flights, make_airports(spark), make_ref(spark), make_id(spark), manual_mapping={})
    rows = {r["AIRPORT_CODE"]: r["IATA_CODE"] for r in m.collect()}
    assert rows["10002"] == "AUS"


def test_mapping_level_3_uses_state_and_airport_name(spark):
    flights = spark.createDataFrame([("10003", "ATL")], ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])
    m = build_airport_code_mapping(flights, make_airports(spark), make_ref(spark), make_id(spark), manual_mapping={})
    rows = {r["AIRPORT_CODE"]: r["IATA_CODE"] for r in m.collect()}
    assert rows["10003"] == "SMF"


def test_mapping_manual_fallback_is_used(spark):
    flights = spark.createDataFrame([("11193", "ATL")], ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])
    m = build_airport_code_mapping(
        flights, make_airports(spark), make_ref(spark), make_id(spark),
        manual_mapping={"11193": "CVG"},
    )
    rows = {r["AIRPORT_CODE"]: r["IATA_CODE"] for r in m.collect()}
    assert rows["11193"] == "CVG"


def test_mapping_has_one_row_per_airport_code(spark):
    flights = spark.createDataFrame(
        [("ATL", "10001"), ("10002", "11193"), ("10003", "ATL")],
        ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"],
    )
    m = build_airport_code_mapping(
        flights, make_airports(spark), make_ref(spark), make_id(spark),
        manual_mapping={"11193": "CVG"},
    )
    assert m.groupBy("AIRPORT_CODE").count().filter(F.col("count") > 1).count() == 0


def test_haversine_uses_mapping_for_numeric_dot_code(spark):
    flights = spark.createDataFrame(
        [("10001", "ATL", 100.0)],
        ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DISTANCE"],
    )
    airports = make_airports(spark)
    mapping = build_airport_code_mapping(
        flights, airports, make_ref(spark), make_id(spark), manual_mapping={}
    )
    result = add_haversine_distance_feature(flights, airports, mapping).collect()[0]
    assert result["has_haversine_distance"] is True
    assert result["haversine_distance_km"] is not None
    assert result["distance_feature_km"] == pytest.approx(result["haversine_distance_km"])


def test_haversine_still_falls_back_for_unmapped_code(spark):
    flights = spark.createDataFrame(
        [("99999", "ATL", 500.0)],
        ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DISTANCE"],
    )
    airports = make_airports(spark)
    mapping = build_airport_code_mapping(
        flights, airports, make_ref(spark), make_id(spark), manual_mapping={}
    )
    result = add_haversine_distance_feature(flights, airports, mapping).collect()[0]
    assert result["has_haversine_distance"] is False
    assert result["distance_feature_km"] == pytest.approx(500.0 * 1.60934)
>>>>>>> Kien


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


def test_outlier_truncator_flag_marks_values_outside_q1_q3(spark):
    """
    Kiểm chứng binary flag (quyết định mới thay cho việc dùng giá trị clip
    trong training): flag = 1 nếu nằm ngoài [Q1, Q3] -- KHÔNG dùng bound đã
    nhân iqrMultiplier, khác với cột đã clip.
    """
    data = [(v,) for v in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1000]]
    df = spark.createDataFrame(data, ["value"])

    truncator = OutlierIQRTruncator(
        inputCols=["value"],
        outputCols=["value_clipped"],
        outlierFlagCols=["value_is_outlier"],
    )
    model = truncator.fit(df)
    result = {r["value"]: r["value_is_outlier"] for r in model.transform(df).collect()}

    assert result[1000] == 1  # ngoài Q3 -> flag = 1
    assert result[10] in (0, 1)  # 10 có thể nằm ngoài Q1 tùy phân vị -- không assert cứng giá trị biên
    assert result[15] == 0  # giá trị giữa phân phối -> chắc chắn không phải outlier


def test_outlier_truncator_flag_uses_q1_q3_not_multiplied_bounds(spark):
    """
    Flag phải dùng đúng Q1/Q3 thô, KHÔNG phải lower/upper đã nhân
    iqrMultiplier -- 2 ngưỡng này khác nhau, dễ nhầm lẫn nếu code sai.
    """
    df = spark.createDataFrame([(v,) for v in range(1, 101)], ["value"])  # 1..100

    truncator = OutlierIQRTruncator(
        inputCols=["value"],
        outputCols=["value_clipped"],
        outlierFlagCols=["value_is_outlier"],
        iqrMultiplier=1.5,
    )
    model = truncator.fit(df)
    bounds = model.bounds_summary().iloc[0]

    # Với 1.5x multiplier trên phân phối 1..100, lower/upper phải RỘNG hơn Q1/Q3
    assert bounds["lower_bound"] < bounds["q1"]
    assert bounds["upper_bound"] > bounds["q3"]

    result = model.transform(df).collect()
    # Giá trị nằm giữa Q1 và Q3 phải luôn có flag = 0
    for row in result:
        if bounds["q1"] <= row["value"] <= bounds["q3"]:
            assert row["value_is_outlier"] == 0


def test_outlier_truncator_no_flag_when_outlierFlagCols_not_set(spark):
    """Không đặt outlierFlagCols -> không tạo cột flag nào, giữ hành vi cũ
    (backward-compatible với code đã viết trước khi có quyết định thêm flag)."""
    df = spark.createDataFrame([(v,) for v in [10, 20, 30, 1000]], ["value"])
    truncator = OutlierIQRTruncator(inputCols=["value"], outputCols=["value_clipped"])
    model = truncator.fit(df)
    result_cols = model.transform(df).columns

    assert "value_clipped" in result_cols
    assert not any(c.endswith("_is_outlier") for c in result_cols)


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


def test_outlier_truncator_flag_save_load_roundtrip(spark, tmp_path):
    """Round-trip cho trường hợp CÓ outlierFlagCols -- đảm bảo flag_col
    (embed trong boundsMapJson) không bị mất khi save()/load()."""
    from custom_transformers import OutlierIQRTruncatorModel

    df = spark.createDataFrame([(v,) for v in [10, 20, 30, 1000]], ["value"])
    truncator = OutlierIQRTruncator(
        inputCols=["value"], outputCols=["value_clipped"], outlierFlagCols=["value_is_outlier"]
    )
    model = truncator.fit(df)

    save_path = str(tmp_path / "outlier_truncator_flag_model")
    model.write().overwrite().save(save_path)
    loaded_model = OutlierIQRTruncatorModel.load(save_path)

    original_result = {r["value"]: r["value_is_outlier"] for r in model.transform(df).collect()}
    loaded_result = {r["value"]: r["value_is_outlier"] for r in loaded_model.transform(df).collect()}

    assert original_result == loaded_result


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

<<<<<<< HEAD
    # "2400" là convention hợp lệ theo docs/dictionaries/data_dictionary.md -> phải quy về giờ 0
=======
    # "2400" là convention hợp lệ theo data_dictionary.md -> phải quy về giờ 0
>>>>>>> Kien
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

<<<<<<< HEAD
    # 3 chuyến cùng sân bay/ngày/giờ, 1 chuyến khác giờ -> congestion phải phân biệt đúng
    df = spark.createDataFrame(
        [
            ("JFK", 1, 15, 9),
            ("JFK", 1, 15, 9),
            ("JFK", 1, 15, 9),
            ("JFK", 1, 15, 10),
        ],
        ["ORIGIN_AIRPORT", "MONTH", "DAY", "scheduled_hour"],
=======
    # 3 chuyến cùng sân bay/năm/ngày/giờ, 1 chuyến khác giờ -> congestion phải phân biệt đúng
    df = spark.createDataFrame(
        [
            ("JFK", 2015, 1, 15, 9),
            ("JFK", 2015, 1, 15, 9),
            ("JFK", 2015, 1, 15, 9),
            ("JFK", 2015, 1, 15, 10),
        ],
        ["ORIGIN_AIRPORT", "YEAR", "MONTH", "DAY", "scheduled_hour"],
>>>>>>> Kien
    )
    result = add_route_congestion_feature(df).collect()

    congestion_by_hour = {}
    for row in result:
        congestion_by_hour[row["scheduled_hour"]] = row["route_congestion_count"]

    assert congestion_by_hour[9] == 3
    assert congestion_by_hour[10] == 1


<<<<<<< HEAD
=======
def test_route_congestion_does_not_collide_across_years(spark):
    """Kiểm chứng bản vá (feature_dictionary.md, vấn đề còn treo #1): thiếu
    YEAR trong group key sẽ gộp nhầm cùng MONTH/DAY/scheduled_hour của các
    năm khác nhau vào 1 nhóm. Sau khi thêm YEAR, 2 chuyến cùng
    sân bay/tháng/ngày/giờ nhưng KHÁC năm phải được đếm ĐỘC LẬP."""
    from custom_transformers import add_route_congestion_feature

    df = spark.createDataFrame(
        [
            ("JFK", 2015, 1, 15, 9),
            ("JFK", 2015, 1, 15, 9),
            ("JFK", 2016, 1, 15, 9),  # cùng MONTH/DAY/hour, KHÁC năm
        ],
        ["ORIGIN_AIRPORT", "YEAR", "MONTH", "DAY", "scheduled_hour"],
    )
    result = {
        row["YEAR"]: row["route_congestion_count"]
        for row in add_route_congestion_feature(df).collect()
    }

    # Nếu group key thiếu YEAR, cả 3 dòng sẽ gộp thành 1 nhóm -> count = 3 cho cả 2 năm.
    assert result[2015] == 2
    assert result[2016] == 1


def test_route_congestion_year_col_none_restores_old_behavior(spark):
    """year_col=None tắt hẳn việc dùng YEAR -- dùng khi chắc chắn dataset chỉ
    có đúng 1 năm và muốn giữ hành vi cũ (không khuyến khích cho dữ liệu
    nhiều năm)."""
    from custom_transformers import add_route_congestion_feature

    df = spark.createDataFrame(
        [
            ("JFK", 2015, 1, 15, 9),
            ("JFK", 2016, 1, 15, 9),
        ],
        ["ORIGIN_AIRPORT", "YEAR", "MONTH", "DAY", "scheduled_hour"],
    )
    result = add_route_congestion_feature(df, year_col=None).collect()

    # year_col=None -> 2 dòng khác năm nhưng cùng MONTH/DAY/hour bị gộp chung
    assert all(r["route_congestion_count"] == 2 for r in result)


>>>>>>> Kien
# =============================================================================
# SECTION 5 — 5 cấp độ (Airline/Airport/Airplane/Flight/Temporal-level)
# =============================================================================


def test_flight_timestamp_orders_correctly_including_2400(spark):
    from custom_transformers import add_flight_timestamp_column

    df = spark.createDataFrame(
        [(2015, 1, 1, "0800"), (2015, 1, 1, "2400"), (2015, 1, 2, "0100")],
        ["YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE"],
    )
    result = add_flight_timestamp_column(df).orderBy("_flight_timestamp_epoch").collect()

    # "2400" ngày 1/1 phải tương đương 00:00 ngày 2/1 -> đứng TRƯỚC "0100" ngày 2/1
    ordered_departures = [r["SCHEDULED_DEPARTURE"] for r in result]
    assert ordered_departures == ["0800", "2400", "0100"]


def test_lookback_agg_excludes_future_and_same_timestamp(spark):
    """
    Kiểm chứng cốt lõi của leak-safety mới: 2 chuyến CÙNG NGÀY nhưng khác giờ
    -- chuyến buổi sáng KHÔNG được thấy dữ liệu của chuyến buổi tối cùng ngày.
    """
    from custom_transformers import add_flight_timestamp_column, derive_lookback_agg

    df = spark.createDataFrame(
        [
            ("AA", 2015, 1, 1, "0800", 5.0),   # chuyến sáng, delay=5
            ("AA", 2015, 1, 1, "2000", 100.0),  # chuyến tối CÙNG NGÀY, delay=100
            ("AA", 2015, 1, 2, "0800", 0.0),   # chuyến hôm sau
        ],
        ["AIRLINE", "YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE", "ARRIVAL_DELAY"],
    )
    df = add_flight_timestamp_column(df)
    df = derive_lookback_agg(
        df, ["AIRLINE"], "avg_delay_lag", lookback_days=7, agg_fn=F.avg, target_col="ARRIVAL_DELAY"
    )
    result = {r["SCHEDULED_DEPARTURE"] + "_" + str(r["DAY"]): r["avg_delay_lag"] for r in df.collect()}

    # Chuyến sáng ngày 1: không có gì trước đó -> NULL
    assert result["0800_1"] is None
    # Chuyến tối ngày 1: chỉ thấy chuyến sáng CÙNG NGÀY (5.0), KHÔNG thấy chính nó
    assert result["2000_1"] == pytest.approx(5.0)
    # Chuyến ngày 2: thấy CẢ 2 chuyến ngày 1 -> avg(5, 100) = 52.5
    assert result["0800_2"] == pytest.approx(52.5)


def test_schedule_count_includes_full_day_regardless_of_order(spark):
    """
    derive_schedule_count KHÔNG cắt theo timestamp -- an toàn dùng cả ngày vì
    là lịch bay (Airport-level), khác hẳn derive_lookback_agg.
    """
    from custom_transformers import derive_schedule_count

    df = spark.createDataFrame(
        [
            ("JFK", 2015, 1, 1, "0800"),
            ("JFK", 2015, 1, 1, "2000"),
            ("JFK", 2015, 1, 2, "0800"),
        ],
        ["ORIGIN_AIRPORT", "YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE"],
    )
    result = derive_schedule_count(
        df, ["ORIGIN_AIRPORT", "YEAR", "MONTH", "DAY"], "airport_scheduled_departures_same_day"
    ).collect()

    counts_by_day = {r["DAY"]: r["airport_scheduled_departures_same_day"] for r in result}
    # Chuyến sáng ngày 1 (0800) VẪN thấy đủ 2 chuyến cùng ngày (kể cả chuyến tối 2000
    # xảy ra "sau" nó) -- vì đây là lịch bay, không phải outcome
    assert counts_by_day[1] == 2
    assert counts_by_day[2] == 1


def test_safe_divide_special_rules(spark):
    from custom_transformers import safe_divide

    df = spark.createDataFrame(
        [(10.0, 2.0), (0.0, 0.0), (5.0, 0.0), (-5.0, 0.0), (None, 3.0)],
        StructType(
            [
                StructField("num", DoubleType(), True),
                StructField("den", DoubleType(), True),
            ]
        ),
    )
    result = df.withColumn("r", safe_divide(F.col("num"), F.col("den"))).collect()
    values = [row["r"] for row in result]

    assert values[0] == pytest.approx(5.0)       # 10/2 = 5
    assert values[1] == pytest.approx(0.0)       # 0/0 = 0
    assert values[2] == pytest.approx(99999.9999)   # 5/0, tử>0
    assert values[3] == pytest.approx(-99999.9999)  # -5/0, tử<0
    assert values[4] is None                      # tử NULL -> NULL


def test_airline_level_features_run_end_to_end(spark):
    from custom_transformers import add_flight_timestamp_column, build_airline_level_features

    df = spark.createDataFrame(
        [
            ("AA", "N100", 2015, 1, 1, "0800", 5.0),
            ("AA", "N101", 2015, 1, 1, "2000", 40.0),
            ("AA", "N100", 2015, 1, 2, "0800", 0.0),
        ],
        ["AIRLINE", "TAIL_NUMBER", "YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE", "ARRIVAL_DELAY"],
    )
    df = add_flight_timestamp_column(df)
    result = (
        build_airline_level_features(df, lookback_windows=(7,))
        .orderBy("_flight_timestamp_epoch")
        .collect()
    )

    # Dòng thứ 3 (ngày 2) phải thấy đủ 2 chuyến ngày 1 -> flights_count_lag7d = 2
    assert result[2]["airline_flights_count_lag7d"] == 2
    # Ngưỡng "late" đã chốt = ARRIVAL_DELAY > 30 -> chỉ chuyến delay=40 tính là "late"
    assert result[2]["airline_late_flights_count_lag7d"] == 1


def test_airline_level_features_multi_window_no_overwrite(spark):
    """Kiểm chứng bản vá multi-window: gọi 1 lần với nhiều cửa sổ, mỗi cửa sổ
    phải có cột riêng biệt (suffix khác nhau), không ghi đè lẫn nhau."""
    from custom_transformers import add_flight_timestamp_column, build_airline_level_features

    df = spark.createDataFrame(
        [
            ("AA", "N100", 2015, 1, 1, "0800", 40.0),
            ("AA", "N100", 2015, 1, 5, "0800", 40.0),
            ("AA", "N100", 2015, 1, 9, "0800", 0.0),  # dòng cần kiểm tra
        ],
        ["AIRLINE", "TAIL_NUMBER", "YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE", "ARRIVAL_DELAY"],
    )
    df = add_flight_timestamp_column(df)
    result = (
        build_airline_level_features(df, lookback_windows=(1, 3, 7))
        .orderBy("_flight_timestamp_epoch")
        .collect()
    )
    last_row = result[2]  # ngày 9/1

    # Cửa sổ 1 ngày (chỉ ngày 8/1): không có chuyến nào -> 0
    assert last_row["airline_flights_count_lag1d"] == 0
    # Cửa sổ 7 ngày (2/1 - 8/1): thấy đúng chuyến ngày 5/1 -> 1 chuyến
    assert last_row["airline_flights_count_lag7d"] == 1
    # 3 cột phải TỒN TẠI ĐỘC LẬP, không ghi đè nhau
    assert "airline_flights_count_lag1d" in result[0].asDict()
    assert "airline_flights_count_lag3d" in result[0].asDict()
    assert "airline_flights_count_lag7d" in result[0].asDict()


def test_airplane_level_schedule_count_same_day(spark):
    from custom_transformers import build_airplane_level_features, add_flight_timestamp_column

    df = spark.createDataFrame(
        [
<<<<<<< HEAD
            ("N100", 2015, 1, 1, "0800", 5.0, 100.0),
            ("N100", 2015, 1, 1, "1400", 0.0, 200.0),
        ],
        ["TAIL_NUMBER", "YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE", "ARRIVAL_DELAY", "distance_feature_km"],
=======
            ("N100", 2015, 1, 1, "0800", "1000", 5.0, 100.0),
            ("N100", 2015, 1, 1, "1400", "1600", 0.0, 200.0),
        ],
        ["TAIL_NUMBER", "YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL",
         "ARRIVAL_DELAY", "distance_feature_km"],
>>>>>>> Kien
    )
    df = add_flight_timestamp_column(df)
    result = build_airplane_level_features(df, lookback_windows=(7,)).collect()

    assert all(r["airplane_flights_scheduled_same_day"] == 2 for r in result)


def test_temporal_level_weekend_and_time_bucket(spark):
    from custom_transformers import build_temporal_level_features

    df = spark.createDataFrame(
        [(6, 9), (7, 14), (1, 19), (2, 2)], ["DAY_OF_WEEK", "scheduled_hour"]
    )
    result = build_temporal_level_features(df).collect()
    by_hour = {r["scheduled_hour"]: (r["is_weekend"], r["time_of_day_bucket"]) for r in result}

    assert by_hour[9] == (1, "morning")
    assert by_hour[14] == (1, "afternoon")
    assert by_hour[19] == (0, "evening")
    assert by_hour[2] == (0, "night")
<<<<<<< HEAD
=======


# =============================================================================
# SECTION 5 (BỔ SUNG) — prev-flight (Airplane), hour-window (Airport),
# default flight info (Flight-level, không kỳ phái sinh)
# =============================================================================


def test_airplane_prev_flight_arrival_delay_uses_only_immediate_previous(spark):
    from custom_transformers import add_flight_timestamp_column, build_airplane_level_features

    df = spark.createDataFrame(
        [
            ("N100", 2015, 1, 1, "0800", "1000", 5.0, 100.0),
            ("N100", 2015, 1, 1, "1400", "1600", 40.0, 100.0),
            ("N100", 2015, 1, 2, "0800", "1000", 0.0, 100.0),
        ],
        ["TAIL_NUMBER", "YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL",
         "ARRIVAL_DELAY", "distance_feature_km"],
    )
    df = add_flight_timestamp_column(df)
    result = (
        build_airplane_level_features(df, lookback_windows=(7,))
        .orderBy("_flight_timestamp_epoch")
        .collect()
    )

    # Chuyến đầu tiên: không có chuyến nào trước -> NULL
    assert result[0]["airplane_prev_flight_arrival_delay"] is None
    # Chuyến thứ 2 (14h ngày 1): chuyến liền trước là 8h ngày 1 (delay=5.0)
    assert result[1]["airplane_prev_flight_arrival_delay"] == pytest.approx(5.0)
    # Chuyến thứ 3 (ngày 2): chuyến liền trước là 14h ngày 1 (delay=40.0),
    # KHÔNG phải trung bình/tổng nhiều chuyến -- chỉ đúng 1 chuyến ngay trước
    assert result[2]["airplane_prev_flight_arrival_delay"] == pytest.approx(40.0)


def test_airplane_prev_flight_delay_vs_current_schedule_computation(spark):
    from custom_transformers import add_flight_timestamp_column, build_airplane_level_features

    df = spark.createDataFrame(
        [
            ("N100", 2015, 1, 1, "0800", "1000", 30.0, 100.0),   # chuyến 1: delay=30
            ("N100", 2015, 1, 1, "1400", "1630", 0.0, 100.0),    # chuyến 2: SCHEDULED_ARRIVAL=16:30 -> 990 phut
        ],
        ["TAIL_NUMBER", "YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL",
         "ARRIVAL_DELAY", "distance_feature_km"],
    )
    df = add_flight_timestamp_column(df)
    result = (
        build_airplane_level_features(df, lookback_windows=(7,))
        .orderBy("_flight_timestamp_epoch")
        .collect()
    )
    # Chuyến 2: prev_flight_arrival_delay=30.0, SCHEDULED_ARRIVAL="1630" -> 16*60+30=990 phut
    # 30.0 - 990 = -960.0
    assert result[1]["airplane_prev_flight_delay_vs_current_schedule"] == pytest.approx(30.0 - 990)


def test_airport_level_hour_based_window_naming_and_causality(spark):
    from custom_transformers import add_flight_timestamp_column, build_airport_level_features

    df = spark.createDataFrame(
        [
            ("JFK", "LAX", 2015, 1, 1, "0800", 50.0),  # 8h sang, delay lon
            ("JFK", "LAX", 2015, 1, 1, "1100", 0.0),   # 11h sang, cach chuyen truoc 3 gio
            ("JFK", "LAX", 2015, 1, 1, "2000", 0.0),   # 20h toi, cach xa hon 6h
        ],
        ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE", "ARRIVAL_DELAY"],
    )
    df = add_flight_timestamp_column(df)
    result = (
        build_airport_level_features(df, lookback_windows_hours=(3, 6))
        .orderBy("_flight_timestamp_epoch")
        .collect()
    )

    # Cột phải có suffix "_lag3h"/"_lag6h", KHÔNG phải "_lagXd"
    assert "airport_avg_delay_lag3h" in result[0].asDict()
    assert "airport_avg_delay_lag6h" in result[0].asDict()

    # Chuyến 11h: cách chuyến 8h đúng 3 giờ -> cửa sổ 3h vẫn thấy được (biên bao gồm)
    row_11h = result[1]
    assert row_11h["airport_avg_delay_lag3h"] == pytest.approx(50.0)

    # Chuyến 20h: cách chuyến 8h tới 12 giờ -> cửa sổ 6h KHÔNG thấy được chuyến 8h nữa
    row_20h = result[2]
    assert row_20h["airport_avg_delay_lag6h"] is None


def test_flight_level_default_info_no_period_applied(spark):
    from custom_transformers import build_flight_level_features

    df = spark.createDataFrame(
        [("0800", "1030", 750.0)],  # 08:00 -> 10:30 = 150 phut, distance=750km
        ["SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL", "distance_feature_km"],
    )
    # Không cần add_flight_timestamp_column vì đây là feature KHÔNG áp dụng kỳ phái sinh
    df = df.withColumn("DEPARTURE_DELAY", F.lit(None).cast("double"))
    result = build_flight_level_features(df).collect()[0]

    assert result["flight_computed_scheduled_duration_minutes"] == 150
    assert result["flight_scheduled_minutes_per_km"] == pytest.approx(150 / 750)


def test_flight_level_default_info_handles_overnight(spark):
    from custom_transformers import build_flight_level_features

    df = spark.createDataFrame(
        [("2330", "0130", 500.0)],  # 23:30 -> 01:30 (qua đêm) = 120 phút
        ["SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL", "distance_feature_km"],
    )
    df = df.withColumn("DEPARTURE_DELAY", F.lit(None).cast("double"))
    result = build_flight_level_features(df).collect()[0]

    assert result["flight_computed_scheduled_duration_minutes"] == 120
>>>>>>> Kien
