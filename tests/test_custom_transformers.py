"""Unit tests for flight feature transformers and model persistence."""

import math
import sys

import pytest
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

sys.path.insert(0, ".")
from mllib_pipeline import configure_local_filesystem, configure_pyspark_python
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
    configure_pyspark_python()
    spark = configure_local_filesystem(
        SparkSession.builder.master("local[2]")
        .appName("test_custom_transformers")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
    ).getOrCreate()
    yield spark
    spark.stop()


def test_haversine_same_point_is_zero(spark):
    df = spark.createDataFrame([(10.0, 20.0, 10.0, 20.0)], ["lat1", "lon1", "lat2", "lon2"])
    result = df.withColumn(
        "d", haversine_km(F.col("lat1"), F.col("lon1"), F.col("lat2"), F.col("lon2"))
    ).collect()[0]["d"]
    assert result == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance_hanoi_hcmc(spark):


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

    assert rows["JFK"] == pytest.approx((40.6413, -73.7781))


def test_add_haversine_distance_feature_fallback_when_airport_unknown(spark):
    flights = spark.createDataFrame(
        [("ZZZ", "YYY", 500)],
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


def test_outlier_truncator_clips_extreme_value(spark):
    data = [(v,) for v in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1000]]
    df = spark.createDataFrame(data, ["value"])

    truncator = OutlierIQRTruncator(
        inputCols=["value"], outputCols=["value_clipped"], iqrMultiplier=1.5
    )
    model = truncator.fit(df)
    result = {r["value"]: r["value_clipped"] for r in model.transform(df).collect()}

    assert result[1000] < 1000
    assert result[15] == 15


def test_outlier_truncator_bounds_computed_only_from_fit_data(spark):
    """Outlier truncator bounds computed only from fit data."""
    train = spark.createDataFrame([(v,) for v in range(1, 51)], ["value"])
    test = spark.createDataFrame([(v,) for v in [1000, 2000, -500]], ["value"])

    truncator = OutlierIQRTruncator(inputCols=["value"], outputCols=["value_clipped"])
    model = truncator.fit(train)

    upper_bound_before = model.bounds_summary().iloc[0]["upper_bound"]

    test_result = model.transform(test).collect()
    for row in test_result:

        assert row["value_clipped"] <= upper_bound_before + 1e-6

    upper_bound_after = model.bounds_summary().iloc[0]["upper_bound"]

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


    assert bounds.loc["small_scale_clipped", "upper_bound"] != bounds.loc[
        "large_scale_clipped", "upper_bound"
    ]


def test_outlier_truncator_flag_marks_values_outside_q1_q3(spark):
    """Outlier truncator flag marks values outside q1 q3."""
    data = [(v,) for v in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1000]]
    df = spark.createDataFrame(data, ["value"])

    truncator = OutlierIQRTruncator(
        inputCols=["value"],
        outputCols=["value_clipped"],
        outlierFlagCols=["value_is_outlier"],
    )
    model = truncator.fit(df)
    result = {r["value"]: r["value_is_outlier"] for r in model.transform(df).collect()}

    assert result[1000] == 1
    assert result[10] in (0, 1)
    assert result[15] == 0


def test_outlier_truncator_flag_uses_q1_q3_not_multiplied_bounds(spark):
    """Outlier truncator flag uses q1 q3 not multiplied bounds."""
    df = spark.createDataFrame([(v,) for v in range(1, 101)], ["value"])

    truncator = OutlierIQRTruncator(
        inputCols=["value"],
        outputCols=["value_clipped"],
        outlierFlagCols=["value_is_outlier"],
        iqrMultiplier=1.5,
    )
    model = truncator.fit(df)
    bounds = model.bounds_summary().iloc[0]


    assert bounds["lower_bound"] < bounds["q1"]
    assert bounds["upper_bound"] > bounds["q3"]

    result = model.transform(df).collect()

    for row in result:
        if bounds["q1"] <= row["value"] <= bounds["q3"]:
            assert row["value_is_outlier"] == 0


def test_outlier_truncator_no_flag_when_outlierFlagCols_not_set(spark):
    """Outlier truncator no flag when outlierflagcols not set."""
    df = spark.createDataFrame([(v,) for v in [10, 20, 30, 1000]], ["value"])
    truncator = OutlierIQRTruncator(inputCols=["value"], outputCols=["value_clipped"])
    model = truncator.fit(df)
    result_cols = model.transform(df).columns

    assert "value_clipped" in result_cols
    assert not any(c.endswith("_is_outlier") for c in result_cols)


def test_outlier_truncator_save_load_roundtrip(spark, tmp_path):
    """Outlier truncator save load roundtrip."""
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
    """Outlier truncator flag save load roundtrip."""
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


def test_target_encoder_computes_correct_smoothed_mean(spark):


    df = spark.createDataFrame(
        [("A", 10.0), ("A", 20.0), ("A", 30.0), ("B", 100.0), ("B", 200.0)],
        ["category", "target"],
    )
    encoder = TargetEncoder(
        inputCols=["category"], outputCols=["category_encoded"], targetCol="target", smoothing=10.0
    )
    model = encoder.fit(df)

    result = {r["category"]: r["category_encoded"] for r in model.transform(df).collect()}

    assert result["A"] == pytest.approx(60.0, rel=1e-3)
    assert result["B"] == pytest.approx(85.0, rel=1e-3)


def test_target_encoder_unseen_category_falls_back_to_global_mean(spark):
    train = spark.createDataFrame(
        [("A", 10.0), ("A", 20.0), ("B", 100.0), ("B", 200.0)], ["category", "target"]
    )

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
    test = spark.createDataFrame([("C", None)], test_schema)
    result = model.transform(test).collect()[0]["category_encoded"]

    assert result == pytest.approx(82.5, rel=1e-3)


def test_target_encoder_no_leak_bounds_fixed_after_fit(spark):
    """Target encoder no leak bounds fixed after fit."""
    train = spark.createDataFrame(
        [("X", 5.0), ("X", 5.0), ("X", 5.0)], ["category", "target"]
    )
    encoder = TargetEncoder(
        inputCols=["category"], outputCols=["category_encoded"], targetCol="target", smoothing=0.0
    )
    model = encoder.fit(train)


    test = spark.createDataFrame([("X", 9999.0), ("X", -9999.0)], ["category", "target"])
    result = model.transform(test).collect()

    for row in result:
        assert row["category_encoded"] == pytest.approx(5.0, rel=1e-3)


def test_target_encoder_save_load_roundtrip(spark, tmp_path):
    """Target encoder save load roundtrip."""
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


def test_scheduled_hour_normal_case(spark):
    from custom_transformers import add_scheduled_hour_feature

    df = spark.createDataFrame([("0930",), ("1745",)], ["SCHEDULED_DEPARTURE"])
    result = {r["SCHEDULED_DEPARTURE"]: r["scheduled_hour"] for r in add_scheduled_hour_feature(df).collect()}

    assert result["0930"] == 9
    assert result["1745"] == 17


def test_scheduled_hour_handles_2400_convention(spark):
    from custom_transformers import add_scheduled_hour_feature


    df = spark.createDataFrame([("2400",)], ["SCHEDULED_DEPARTURE"])
    result = add_scheduled_hour_feature(df).collect()[0]["scheduled_hour"]

    assert result == 0


def test_scheduled_hour_handles_unpadded_time(spark):
    from custom_transformers import add_scheduled_hour_feature


    df = spark.createDataFrame([("5",)], ["SCHEDULED_DEPARTURE"])
    result = add_scheduled_hour_feature(df).collect()[0]["scheduled_hour"]

    assert result == 0


def test_route_congestion_counts_correctly(spark):
    from custom_transformers import add_route_congestion_feature


    df = spark.createDataFrame(
        [
            ("JFK", 1, 15, 9),
            ("JFK", 1, 15, 9),
            ("JFK", 1, 15, 9),
            ("JFK", 1, 15, 10),
        ],
        ["ORIGIN_AIRPORT", "MONTH", "DAY", "scheduled_hour"],
    )
    df = df.withColumn("YEAR", F.lit(2015))
    result = add_route_congestion_feature(df).collect()

    congestion_by_hour = {}
    for row in result:
        congestion_by_hour[row["scheduled_hour"]] = row["route_congestion_count"]

    assert congestion_by_hour[9] == 3
    assert congestion_by_hour[10] == 1


def test_flight_timestamp_orders_correctly_including_2400(spark):
    from custom_transformers import add_flight_timestamp_column

    df = spark.createDataFrame(
        [(2015, 1, 1, "0800"), (2015, 1, 1, "2400"), (2015, 1, 2, "0100")],
        ["YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE"],
    )
    result = add_flight_timestamp_column(df).orderBy("_flight_timestamp_epoch").collect()


    ordered_departures = [r["SCHEDULED_DEPARTURE"] for r in result]
    assert ordered_departures == ["0800", "2400", "0100"]


def test_lookback_agg_excludes_future_and_same_timestamp(spark):
    """Lookback agg excludes future and same timestamp."""
    from custom_transformers import add_flight_timestamp_column, derive_lookback_agg

    df = spark.createDataFrame(
        [
            ("AA", 2015, 1, 1, "0800", 5.0),
            ("AA", 2015, 1, 1, "2000", 100.0),
            ("AA", 2015, 1, 2, "0800", 0.0),
        ],
        ["AIRLINE", "YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE", "ARRIVAL_DELAY"],
    )
    df = add_flight_timestamp_column(df)
    df = derive_lookback_agg(
        df, ["AIRLINE"], "avg_delay_lag", lookback_days=7, agg_fn=F.avg, target_col="ARRIVAL_DELAY"
    )
    result = {r["SCHEDULED_DEPARTURE"] + "_" + str(r["DAY"]): r["avg_delay_lag"] for r in df.collect()}


    assert result["0800_1"] is None

    assert result["2000_1"] == pytest.approx(5.0)

    assert result["0800_2"] == pytest.approx(52.5)


def test_schedule_count_includes_full_day_regardless_of_order(spark):
    """Schedule count includes full day regardless of order."""
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

    assert values[0] == pytest.approx(5.0)
    assert values[1] == pytest.approx(0.0)
    assert values[2] == pytest.approx(99999.9999)
    assert values[3] == pytest.approx(-99999.9999)
    assert values[4] is None


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


    assert result[2]["airline_flights_count_lag7d"] == 2

    assert result[2]["airline_late_flights_count_lag7d"] == 1


def test_airline_level_features_multi_window_no_overwrite(spark):
    """Airline level features multi window no overwrite."""
    from custom_transformers import add_flight_timestamp_column, build_airline_level_features

    df = spark.createDataFrame(
        [
            ("AA", "N100", 2015, 1, 1, "0800", 40.0),
            ("AA", "N100", 2015, 1, 5, "0800", 40.0),
            ("AA", "N100", 2015, 1, 9, "0800", 0.0),
        ],
        ["AIRLINE", "TAIL_NUMBER", "YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE", "ARRIVAL_DELAY"],
    )
    df = add_flight_timestamp_column(df)
    result = (
        build_airline_level_features(df, lookback_windows=(1, 3, 7))
        .orderBy("_flight_timestamp_epoch")
        .collect()
    )
    last_row = result[2]


    assert last_row["airline_flights_count_lag1d"] == 0

    assert last_row["airline_flights_count_lag7d"] == 1

    assert "airline_flights_count_lag1d" in result[0].asDict()
    assert "airline_flights_count_lag3d" in result[0].asDict()
    assert "airline_flights_count_lag7d" in result[0].asDict()


def test_airplane_level_schedule_count_same_day(spark):
    from custom_transformers import build_airplane_level_features, add_flight_timestamp_column

    df = spark.createDataFrame(
        [
            ("N100", 2015, 1, 1, "0800", 5.0, 100.0),
            ("N100", 2015, 1, 1, "1400", 0.0, 200.0),
        ],
        ["TAIL_NUMBER", "YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE", "ARRIVAL_DELAY", "distance_feature_km"],
    )
    df = add_flight_timestamp_column(df)
    df = df.withColumn("SCHEDULED_ARRIVAL", F.lit("1600"))
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


def test_congestion_does_not_mix_years(spark):
    from custom_transformers import add_route_congestion_feature
    data = spark.createDataFrame([
        ("JFK", 2015, 1, 1, 9), ("JFK", 2015, 1, 1, 9), ("JFK", 2016, 1, 1, 9),
    ], ["ORIGIN_AIRPORT", "YEAR", "MONTH", "DAY", "scheduled_hour"])
    rows = add_route_congestion_feature(data).select("YEAR", "route_congestion_count").collect()
    assert {(row.YEAR, row.route_congestion_count) for row in rows} == {(2015, 2), (2016, 1)}


def test_airport_windows_use_hours_and_exclude_current_flight(spark):
    from custom_transformers import add_flight_timestamp_column, build_airport_level_features
    data = spark.createDataFrame([
        ("JFK", "LAX", 2015, 1, 1, "0800", 40.0),
        ("JFK", "LAX", 2015, 1, 1, "1200", 0.0),
    ], ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE", "ARRIVAL_DELAY"])
    rows = build_airport_level_features(add_flight_timestamp_column(data), lookback_windows_hours=(3, 6)).orderBy("SCHEDULED_DEPARTURE").collect()
    assert rows[1]["airport_avg_delay_lag3h"] is None
    assert rows[1]["airport_avg_delay_lag6h"] == 40.0


def test_haversine_mapping_preserves_original_airport_columns(spark):
    airports = spark.createDataFrame([("JFK", 40.64, -73.78), ("LAX", 33.94, -118.41)], ["IATA_CODE", "LATITUDE", "LONGITUDE"])
    flights = spark.createDataFrame([("12478", "LAX", 2475.0)], ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DISTANCE"])
    mapping = spark.createDataFrame([("12478", "JFK")], ["AIRPORT_CODE", "IATA_CODE"])
    row = add_haversine_distance_feature(flights, airports, airport_mapping_df=mapping).first()
    assert row.ORIGIN_AIRPORT == "12478"
    assert row.DESTINATION_AIRPORT == "LAX"
    assert row.has_haversine_distance
    assert 3900 < row.distance_feature_km < 4100
