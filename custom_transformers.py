"""Custom PySpark transformers and point-in-time flight feature engineering."""

import numpy as np
import pandas as pd
import json
from pyspark import keyword_only
from pyspark.ml import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import DoubleType


# IATA coordinate overrides for airports with missing lookup coordinates.
MANUAL_AIRPORT_COORDS = {

    "ECP": (30.3417, -85.7973),
    "PBG": (44.65083, -73.46806),
    "UST": (29.95925, -81.33972),
}


def patch_missing_airport_coordinates(airports_df):
    """Fill missing airport coordinates using the fixed IATA coordinate lookup."""
    result = airports_df
    for iata_code, (lat, lon) in MANUAL_AIRPORT_COORDS.items():
        result = result.withColumn(
            "LATITUDE",
            F.when(
                (F.col("IATA_CODE") == iata_code) & F.col("LATITUDE").isNull(),
                F.lit(lat),
            ).otherwise(F.col("LATITUDE")),
        ).withColumn(
            "LONGITUDE",
            F.when(
                (F.col("IATA_CODE") == iata_code) & F.col("LONGITUDE").isNull(),
                F.lit(lon),
            ).otherwise(F.col("LONGITUDE")),
        )
    return result


@F.pandas_udf(DoubleType())
def haversine_km(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """Compute vectorized great-circle distances in kilometres.

    Null or out-of-range coordinates produce NaN; latitude must be in
    [-90, 90] and longitude in [-180, 180]."""
    R = 6371.0088

    lat1_arr = lat1.to_numpy(dtype="float64", na_value=np.nan)
    lon1_arr = lon1.to_numpy(dtype="float64", na_value=np.nan)
    lat2_arr = lat2.to_numpy(dtype="float64", na_value=np.nan)
    lon2_arr = lon2.to_numpy(dtype="float64", na_value=np.nan)


    def invalid_lat(a):
        return (a < -90) | (a > 90)

    def invalid_lon(a):
        return (a < -180) | (a > 180)

    invalid_mask = (
        invalid_lat(lat1_arr)
        | invalid_lat(lat2_arr)
        | invalid_lon(lon1_arr)
        | invalid_lon(lon2_arr)
    )

    lat1_rad = np.radians(lat1_arr)
    lat2_rad = np.radians(lat2_arr)
    dlat = np.radians(lat2_arr - lat1_arr)
    dlon = np.radians(lon2_arr - lon1_arr)

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(
        dlon / 2.0
    ) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    distance = R * c

    distance = np.where(invalid_mask, np.nan, distance)
    return pd.Series(distance)


def add_haversine_distance_feature(flights_df, airports_df):
    """Join airport coordinates and derive distance features in kilometres.

    Use Haversine distance when coordinates are available, otherwise convert
    the original DISTANCE from miles. Retain a flag indicating the source."""
    airports_df = patch_missing_airport_coordinates(airports_df)

    origin_coords = airports_df.select(
        F.col("IATA_CODE").alias("ORIGIN_AIRPORT"),
        F.col("LATITUDE").alias("origin_lat"),
        F.col("LONGITUDE").alias("origin_lon"),
    )
    dest_coords = airports_df.select(
        F.col("IATA_CODE").alias("DESTINATION_AIRPORT"),
        F.col("LATITUDE").alias("dest_lat"),
        F.col("LONGITUDE").alias("dest_lon"),
    )

    joined = (
        flights_df.join(origin_coords, on="ORIGIN_AIRPORT", how="left")
        .join(dest_coords, on="DESTINATION_AIRPORT", how="left")
    )

    joined = joined.withColumn(
        "haversine_distance_km",
        haversine_km(
            F.col("origin_lat"), F.col("origin_lon"), F.col("dest_lat"), F.col("dest_lon")
        ),
    )

    MILES_TO_KM = 1.60934
    joined = joined.withColumn(
        "route_distance_km", F.col("DISTANCE") * F.lit(MILES_TO_KM)
    )

    joined = joined.withColumn(
        "has_haversine_distance", F.col("haversine_distance_km").isNotNull()
    )


    joined = joined.withColumn(
        "distance_feature_km",
        F.coalesce(F.col("haversine_distance_km"), F.col("route_distance_km")),
    )

    return joined.drop("origin_lat", "origin_lon", "dest_lat", "dest_lon")


if __name__ == "__main__":
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("custom_transformers_sanity_check").getOrCreate()

    flights = spark.read.parquet("data/typed-parquet/flights")
    airports = spark.read.parquet("data/typed-parquet/airports")

    eligible = flights.filter(
        (F.col("ARRIVAL_DELAY").isNotNull())
        & (F.col("CANCELLED") == 0)
        & (F.col("DIVERTED") == 0)
    )

    result = add_haversine_distance_feature(eligible, airports)

    print("Phân bố has_haversine_distance theo tháng:")
    result.groupBy("MONTH", "has_haversine_distance").count().orderBy(
        "MONTH", "has_haversine_distance"
    ).show(24)

    print("So sánh haversine_distance_km vs DISTANCE gốc (mẫu 10 dòng có Haversine):")
    result.filter(F.col("has_haversine_distance")).select(
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "DISTANCE",
        "route_distance_km",
        "haversine_distance_km",
    ).show(10, truncate=False)


class HasIQRMultiplier(Params):
    iqrMultiplier = Param(
        Params._dummy(),
        "iqrMultiplier",
        "Hệ số nhân IQR để xác định outlier bounds (mặc định 1.5)",
        typeConverter=TypeConverters.toFloat,
    )

    def __init__(self):
        super(HasIQRMultiplier, self).__init__()
        self._setDefault(iqrMultiplier=1.5)

    def getIqrMultiplier(self):
        return self.getOrDefault(self.iqrMultiplier)


class HasOutlierFlagCols(Params):
    """Configure optional binary flags for values outside the raw [Q1, Q3] range."""

    outlierFlagCols = Param(
        Params._dummy(),
        "outlierFlagCols",
        "Danh sách tên cột flag binary (song song với outputCols); dùng '' "
        "cho vị trí không cần flag",
        typeConverter=TypeConverters.toListString,
    )

    def getOutlierFlagCols(self):
        return self.getOrDefault(self.outlierFlagCols)


class HasBoundsMap(Params):
    """Persist fitted IQR bounds as a JSON Spark Param for model save/load."""

    boundsMapJson = Param(
        Params._dummy(),
        "boundsMapJson",
        "JSON string chứa toàn bộ IQR bounds đã fit",
        typeConverter=TypeConverters.toString,
    )

    def getBoundsMapJson(self):
        return self.getOrDefault(self.boundsMapJson)


class OutlierIQRTruncator(
    Estimator,
    HasInputCols,
    HasOutputCols,
    HasIQRMultiplier,
    HasOutlierFlagCols,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """Estimate fixed clipping bounds using distributed approximate quantiles.

    Fit only on training data. Place this estimator inside the pipeline so
    cross-validation learns bounds independently for each training fold."""

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, iqrMultiplier=1.5, outlierFlagCols=None):
        super(OutlierIQRTruncator, self).__init__()
        self._setDefault(iqrMultiplier=1.5, outlierFlagCols=[])
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None, iqrMultiplier=1.5, outlierFlagCols=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _fit(self, dataset):
        input_cols = self.getInputCols()
        output_cols = self.getOutputCols()
        multiplier = self.getIqrMultiplier()
        flag_cols = self.getOutlierFlagCols() or []

        if len(input_cols) != len(output_cols):
            raise ValueError("inputCols và outputCols phải cùng độ dài")
        if flag_cols and len(flag_cols) != len(input_cols):
            raise ValueError("outlierFlagCols (nếu dùng) phải cùng độ dài với inputCols")

        bounds = {}
        for idx, (in_col, out_col) in enumerate(zip(input_cols, output_cols)):
            q1, q3 = dataset.approxQuantile(in_col, [0.25, 0.75], relativeError=0.01)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            flag_col = flag_cols[idx] if flag_cols else ""
            bounds[out_col] = {
                "input_col": in_col,
                "q1": q1,
                "q3": q3,
                "lower": lower,
                "upper": upper,
                "flag_col": flag_col,
            }

        bounds_json = json.dumps(bounds)
        return OutlierIQRTruncatorModel(
            inputCols=input_cols, outputCols=output_cols, boundsMapJson=bounds_json
        )


class OutlierIQRTruncatorModel(
    Model, HasInputCols, HasOutputCols, HasBoundsMap, DefaultParamsReadable, DefaultParamsWritable
):
    """Apply fitted clipping bounds without recomputing statistics.

    Optional binary flags use raw [Q1, Q3], rather than the multiplied IQR
    clipping bounds. Bounds and flag names persist through a JSON Spark Param."""

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, boundsMapJson=None):
        super(OutlierIQRTruncatorModel, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, dataset):
        bounds = json.loads(self.getBoundsMapJson())
        result = dataset
        for out_col, info in bounds.items():
            in_col = info["input_col"]
            lower = info["lower"]
            upper = info["upper"]
            result = result.withColumn(
                out_col,
                F.when(F.col(in_col) < F.lit(lower), F.lit(lower))
                .when(F.col(in_col) > F.lit(upper), F.lit(upper))
                .otherwise(F.col(in_col)),
            )
            flag_col = info.get("flag_col", "")
            if flag_col:
                q1, q3 = info["q1"], info["q3"]
                result = result.withColumn(
                    flag_col,
                    F.when(
                        (F.col(in_col) < F.lit(q1)) | (F.col(in_col) > F.lit(q3)), F.lit(1)
                    ).otherwise(F.lit(0)),
                )
        return result

    def bounds_summary(self) -> pd.DataFrame:
        """Return fitted clipping bounds as a compact table."""
        bounds = json.loads(self.getBoundsMapJson())
        rows = [
            {
                "output_col": out_col,
                "input_col": info["input_col"],
                "q1": info["q1"],
                "q3": info["q3"],
                "lower_bound": info["lower"],
                "upper_bound": info["upper"],
                "flag_col": info.get("flag_col", ""),
            }
            for out_col, info in bounds.items()
        ]
        return pd.DataFrame(rows)


def _section2_sanity_check(spark):
    flights = spark.read.parquet("data/typed-parquet/flights")
    airports = spark.read.parquet("data/typed-parquet/airports")

    eligible = flights.filter(
        (F.col("ARRIVAL_DELAY").isNotNull())
        & (F.col("CANCELLED") == 0)
        & (F.col("DIVERTED") == 0)
    )
    with_distance = add_haversine_distance_feature(eligible, airports)


    train_df, test_df = with_distance.randomSplit([0.8, 0.2], seed=42)

    truncator = OutlierIQRTruncator(
        inputCols=["distance_feature_km", "SCHEDULED_TIME"],
        outputCols=["distance_feature_km_clipped", "scheduled_time_clipped"],
        iqrMultiplier=1.5,
    )
    model = truncator.fit(train_df)

    print("Bounds tính được (chỉ từ train):")
    print(model.bounds_summary().to_string(index=False))

    train_clipped = model.transform(train_df)
    test_clipped = model.transform(test_df)

    for out_col in ["distance_feature_km_clipped", "scheduled_time_clipped"]:
        n_clipped_train = train_clipped.filter(
            F.col(out_col) != F.col(
                dict(
                    distance_feature_km_clipped="distance_feature_km",
                    scheduled_time_clipped="SCHEDULED_TIME",
                )[out_col]
            )
        ).count()
        print(f"Số dòng bị clip trong train ở {out_col}: {n_clipped_train:,}")

    return train_clipped, test_clipped


if __name__ == "__main__" and False:


    pass


class HasTargetCol(Params):
    targetCol = Param(
        Params._dummy(),
        "targetCol",
        "Tên cột target dùng để tính mean encoding",
        typeConverter=TypeConverters.toString,
    )

    def getTargetCol(self):
        return self.getOrDefault(self.targetCol)


class HasSmoothing(Params):
    smoothing = Param(
        Params._dummy(),
        "smoothing",
        "Hệ số smoothing (m-estimate) — category càng ít dữ liệu càng bị "
        "kéo về gần global mean",
        typeConverter=TypeConverters.toFloat,
    )

    def __init__(self):
        super(HasSmoothing, self).__init__()
        self._setDefault(smoothing=10.0)

    def getSmoothing(self):
        return self.getOrDefault(self.smoothing)


class HasEncodingMap(Params):
    """Persist category encodings as a JSON Spark Param for model save/load."""

    encodingMapJson = Param(
        Params._dummy(),
        "encodingMapJson",
        "JSON string chứa toàn bộ encoding map đã fit",
        typeConverter=TypeConverters.toString,
    )

    def getEncodingMapJson(self):
        return self.getOrDefault(self.encodingMapJson)


class TargetEncoder(
    Estimator,
    HasInputCols,
    HasOutputCols,
    HasTargetCol,
    HasSmoothing,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """Fit category target means with m-estimate smoothing on training data.

    Place this estimator inside the pipeline to fit each training fold
    independently. Encoding is not out-of-fold within the training fold."""

    @keyword_only
    def __init__(
        self, inputCols=None, outputCols=None, targetCol=None, smoothing=10.0
    ):
        super(TargetEncoder, self).__init__()
        self._setDefault(smoothing=10.0)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @keyword_only
    def setParams(
        self, inputCols=None, outputCols=None, targetCol=None, smoothing=10.0
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _fit(self, dataset):
        input_cols = self.getInputCols()
        output_cols = self.getOutputCols()
        target_col = self.getTargetCol()
        smoothing = self.getSmoothing()

        if len(input_cols) != len(output_cols):
            raise ValueError("inputCols và outputCols phải cùng độ dài")

        columns_map = {}
        for in_col, out_col in zip(input_cols, output_cols):
            global_mean_row = dataset.agg(F.mean(F.col(target_col))).collect()[0]
            global_mean = float(global_mean_row[0])

            stats = (
                dataset.groupBy(in_col)
                .agg(
                    F.mean(F.col(target_col)).alias("cat_mean"),
                    F.count(F.col(target_col)).alias("cat_count"),
                )
                .withColumn(
                    "smoothed",
                    (
                        F.col("cat_count") * F.col("cat_mean")
                        + F.lit(smoothing) * F.lit(global_mean)
                    )
                    / (F.col("cat_count") + F.lit(smoothing)),
                )
            )


            category_map = {
                str(row[in_col]): float(row["smoothed"])
                for row in stats.select(in_col, "smoothed").collect()
                if row[in_col] is not None
            }

            columns_map[out_col] = {
                "input_col": in_col,
                "global_mean": global_mean,
                "map": category_map,
            }

        encoding_json = json.dumps(columns_map)
        return TargetEncoderModel(
            inputCols=input_cols, outputCols=output_cols, encodingMapJson=encoding_json
        )


class TargetEncoderModel(
    Model, HasInputCols, HasOutputCols, HasEncodingMap, DefaultParamsReadable, DefaultParamsWritable
):
    """Apply fitted category encodings; unseen categories use the global mean."""

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, encodingMapJson=None):
        super(TargetEncoderModel, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, dataset):
        spark = dataset.sparkSession
        columns_map = json.loads(self.getEncodingMapJson())

        result = dataset
        for out_col, info in columns_map.items():
            in_col = info["input_col"]
            global_mean = info["global_mean"]
            category_map = info["map"]

            if len(category_map) == 0:
                result = result.withColumn(out_col, F.lit(global_mean))
                continue

            mapping_rows = [(cat, val) for cat, val in category_map.items()]
            mapping_df = spark.createDataFrame(mapping_rows, [in_col, out_col])


            result = result.withColumn(f"__{in_col}_str", F.col(in_col).cast("string"))
            mapping_df = mapping_df.withColumnRenamed(in_col, f"__{in_col}_str")

            result = result.join(F.broadcast(mapping_df), on=f"__{in_col}_str", how="left")
            result = result.withColumn(
                out_col, F.coalesce(F.col(out_col), F.lit(global_mean))
            ).drop(f"__{in_col}_str")

        return result

    def encoding_summary(self, output_col: str, top_n: int = 5) -> pd.DataFrame:
        """Return a sample of fitted category encodings."""
        columns_map = json.loads(self.getEncodingMapJson())
        info = columns_map[output_col]
        rows = sorted(info["map"].items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        df = pd.DataFrame(rows, columns=["category", "encoded_value"])
        df["global_mean"] = info["global_mean"]
        return df


def add_scheduled_hour_feature(
    flights_df, time_col: str = "SCHEDULED_DEPARTURE", output_col: str = "scheduled_hour"
):
    """Extract hour from HHMM values, mapping 2400 to hour zero."""
    padded = F.lpad(F.col(time_col), 4, "0")
    hour_int = F.substring(padded, 1, 2).cast("int")
    normalized_hour = F.when(hour_int == 24, F.lit(0)).otherwise(hour_int)
    return flights_df.withColumn(output_col, normalized_hour)


def add_route_congestion_feature(
    flights_df,
    airport_col: str = "ORIGIN_AIRPORT",
    hour_col: str = "scheduled_hour",
    output_col: str = "route_congestion_count",
):
    """Count scheduled flights per origin, date and hour.

    Requires hour_col. Counts use the published schedule rather than outcomes."""
    window_cols = [airport_col, "MONTH", "DAY", hour_col]
    congestion = flights_df.groupBy(*window_cols).agg(
        F.count(F.lit(1)).alias(output_col)
    )
    return flights_df.join(congestion, on=window_cols, how="left")


def safe_divide(numerator_col, denominator_col):
    """Divide columns with explicit null and zero-denominator handling.

    Null operands yield null; 0/0 yields zero. A zero denominator with a
    nonzero numerator yields a signed sentinel (99999.9999 by default)."""
    return (
        F.when(numerator_col.isNull() | denominator_col.isNull(), F.lit(None))
        .when((numerator_col == 0) & (denominator_col == 0), F.lit(0.0))
        .when((denominator_col == 0) & (numerator_col > 0), F.lit(99999.9999))
        .when((denominator_col == 0) & (numerator_col < 0), F.lit(-99999.9999))
        .otherwise(numerator_col / denominator_col)
    )


def add_flight_timestamp_column(
    flights_df,
    date_cols=("YEAR", "MONTH", "DAY"),
    time_col: str = "SCHEDULED_DEPARTURE",
    output_col: str = "flight_timestamp",
):
    """Build timestamps and epoch seconds from date and scheduled departure.

    The HHMM value 2400 maps to midnight on the following day, preserving
    chronological order for lookback windows."""
    year_col, month_col, day_col = date_cols
    padded_time = F.lpad(F.col(time_col), 4, "0")
    hour = F.substring(padded_time, 1, 2).cast("int")
    minute = F.substring(padded_time, 3, 2).cast("int")

    base_date = F.to_date(F.concat_ws("-", F.col(year_col), F.col(month_col), F.col(day_col)))
    is_2400 = hour == 24

    date_part = F.when(is_2400, F.date_add(base_date, 1)).otherwise(base_date)
    hour_part = F.when(is_2400, F.lit(0)).otherwise(hour)

    timestamp_col = F.to_timestamp(
        F.concat_ws(
            " ",
            F.date_format(date_part, "yyyy-MM-dd"),
            F.format_string("%02d:%02d:00", hour_part, minute),
        )
    )
    df = flights_df.withColumn(output_col, timestamp_col)
    return df.withColumn(f"_{output_col}_epoch", F.col(output_col).cast("long"))


def derive_lookback_agg(
    df,
    partition_cols: list,
    output_col: str,
    lookback_days: float,
    agg_fn,
    target_col: str,
    timestamp_epoch_col: str = "_flight_timestamp_epoch",
):
    """Aggregate entity history strictly before each flight timestamp.

    Requires timestamp_epoch_col. The lookback window is measured in seconds
    and supports fractional days; records with the same timestamp are excluded."""
    lookback_seconds = int(lookback_days * 86400)
    window = (
        Window.partitionBy(*partition_cols)
        .orderBy(timestamp_epoch_col)
        .rangeBetween(-lookback_seconds, -1)
    )
    return df.withColumn(output_col, agg_fn(F.col(target_col)).over(window))


def derive_schedule_count(df, group_cols: list, output_col: str):
    """Count flights by schedule keys across the complete published schedule."""
    counts = df.groupBy(*group_cols).agg(F.count(F.lit(1)).alias(output_col))
    return df.join(counts, on=group_cols, how="left")


def _window_suffix(lookback_days) -> str:
    """Format a lookback duration as a column suffix, such as _lag1d or _lag0.5d."""
    n = int(lookback_days) if float(lookback_days).is_integer() else lookback_days
    return f"_lag{n}d"


LATE_ARRIVAL_THRESHOLD_MINUTES = 30


def build_airline_level_features(df, lookback_windows=(1, 3, 7), late_threshold=LATE_ARRIVAL_THRESHOLD_MINUTES):
    """Build airline history for each lookback window with distinct column suffixes.

    Late arrivals exceed late_threshold minutes, which defaults to 30."""
    for w in lookback_windows:
        suf = _window_suffix(w)
        df = derive_lookback_agg(
            df, ["AIRLINE"], f"airline_late_flights_count{suf}", w,
            agg_fn=lambda c: F.sum(F.when(c > late_threshold, 1).otherwise(0)), target_col="ARRIVAL_DELAY",
        )
        df = derive_lookback_agg(
            df, ["AIRLINE"], f"airline_total_delay_minutes{suf}", w,
            agg_fn=F.sum, target_col="ARRIVAL_DELAY",
        )
        df = derive_lookback_agg(
            df, ["AIRLINE"], f"airline_flights_count{suf}", w,
            agg_fn=lambda c: F.count(c), target_col="ARRIVAL_DELAY",
        )
        df = derive_lookback_agg(
            df, ["AIRLINE"], f"airline_distinct_tails_late{suf}", w,
            agg_fn=lambda c: F.approx_count_distinct(F.when(c > late_threshold, F.col("TAIL_NUMBER"))),
            target_col="ARRIVAL_DELAY",
        )
        df = df.withColumn(
            f"airline_late_flight_ratio{suf}",
            safe_divide(F.col(f"airline_late_flights_count{suf}"), F.col(f"airline_flights_count{suf}")),
        )
        df = df.withColumn(
            f"airline_delay_per_tail{suf}",
            safe_divide(F.col(f"airline_total_delay_minutes{suf}"), F.col(f"airline_distinct_tails_late{suf}")),
        )
    return df


def build_airport_level_features(df, lookback_windows=(1, 3, 7), late_threshold=LATE_ARRIVAL_THRESHOLD_MINUTES):

    df = derive_schedule_count(df, ["ORIGIN_AIRPORT", "YEAR", "MONTH", "DAY"], "airport_scheduled_departures_same_day")
    df = derive_schedule_count(df, ["DESTINATION_AIRPORT", "YEAR", "MONTH", "DAY"], "airport_scheduled_arrivals_same_day")


    for w in lookback_windows:
        suf = _window_suffix(w)
        df = derive_lookback_agg(
            df, ["ORIGIN_AIRPORT"], f"airport_late_flights_count{suf}", w,
            agg_fn=lambda c: F.sum(F.when(c > late_threshold, 1).otherwise(0)), target_col="ARRIVAL_DELAY",
        )
        df = derive_lookback_agg(
            df, ["ORIGIN_AIRPORT"], f"airport_avg_delay{suf}", w,
            agg_fn=F.avg, target_col="ARRIVAL_DELAY",
        )
    return df


def build_airplane_level_features(df, lookback_windows=(1, 3, 7)):


    df = derive_schedule_count(df, ["TAIL_NUMBER", "YEAR", "MONTH", "DAY"], "airplane_flights_scheduled_same_day")


    for w in lookback_windows:
        suf = _window_suffix(w)
        df = derive_lookback_agg(
            df, ["TAIL_NUMBER"], f"airplane_total_delay_minutes{suf}", w,
            agg_fn=F.sum, target_col="ARRIVAL_DELAY",
        )
        df = df.withColumn(
            f"airplane_delay_per_km{suf}",
            safe_divide(F.col(f"airplane_total_delay_minutes{suf}"), F.col("distance_feature_km")),
        )
    return df


def build_flight_level_features(df, airline_avg_window=7):
    df = df.withColumn(
        "flight_departure_delay_per_km",
        safe_divide(F.col("DEPARTURE_DELAY"), F.col("distance_feature_km")),
    )
    suf = _window_suffix(airline_avg_window)
    total_col, count_col = f"airline_total_delay_minutes{suf}", f"airline_flights_count{suf}"
    if total_col in df.columns and count_col in df.columns:
        avg_airline_delay = safe_divide(F.col(total_col), F.col(count_col))
        df = df.withColumn(
            f"flight_departure_delay_vs_airline_avg_rt{suf}",
            safe_divide(F.col("DEPARTURE_DELAY"), avg_airline_delay),
        )
    return df


def build_temporal_level_features(df, weekend_days=(6, 7)):
    """Derive calendar features using ISO weekdays (Monday=1, Sunday=7).

    The default weekend is Saturday and Sunday (6, 7)."""
    df = df.withColumn("is_weekend", F.col("DAY_OF_WEEK").isin(list(weekend_days)).cast("int"))
    df = df.withColumn(
        "time_of_day_bucket",
        F.when(F.col("scheduled_hour").between(5, 11), F.lit("morning"))
        .when(F.col("scheduled_hour").between(12, 16), F.lit("afternoon"))
        .when(F.col("scheduled_hour").between(17, 20), F.lit("evening"))
        .otherwise(F.lit("night")),
    )
    return df
