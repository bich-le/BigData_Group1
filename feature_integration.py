"""Integration helpers for the feature-engineering module."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from pyspark.ml import Estimator
from pyspark.sql import DataFrame, functions as F


LOOKBACK_WINDOWS = (1, 3, 7)
AIRPORT_LOOKBACK_HOURS = (3, 6, 12)
IQR_INPUT_COLS = ("distance_feature_km", "SCHEDULED_TIME")
IQR_OUTPUT_COLS = ("distance_feature_km_clipped", "scheduled_time_clipped")
IQR_FLAG_COLS = ("distance_feature_km_is_outlier", "scheduled_time_is_outlier")


def _feature_module():
    """Load custom_transformers.py from the project root."""

    try:
        module = importlib.import_module("custom_transformers")
    except ImportError as exc:
        raise ImportError(
            "custom_transformers.py is required for --with-flight-features. "
            "Ensure the module and its dependencies are installed."
        ) from exc

    required = (
        "OutlierIQRTruncator",
        "add_haversine_distance_feature",
        "add_scheduled_hour_feature",
        "add_flight_timestamp_column",
        "build_airline_level_features",
        "build_airport_level_features",
        "build_airplane_level_features",
        "build_flight_level_features",
        "build_temporal_level_features",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            "custom_transformers.py is missing required functions/classes: "
            + ", ".join(missing)
        )
    return module


@dataclass(frozen=True)
class FlightFeatureContract:
    """Feature columns used by the preprocessing pipeline."""

    lookback_windows: tuple[int, ...] = LOOKBACK_WINDOWS
    include_departure_delay_features: bool = True

    @property
    def categorical_cols(self) -> tuple[str, ...]:
        # FLIGHT_NUMBER is excluded from the categorical feature contract.
        return (
            "AIRLINE",
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "TAIL_NUMBER",
            "time_of_day_bucket",
        )

    @property
    def numeric_cols(self) -> tuple[str, ...]:
        cols = [
            "YEAR",
            "MONTH",
            "DAY",
            "DAY_OF_WEEK",
            "SCHEDULED_TIME",
            "scheduled_hour",
            "distance_feature_km",
            "has_haversine_distance_num",
            "route_congestion_count",
            "is_weekend",
            *IQR_FLAG_COLS,
        ]

        for window in self.lookback_windows:
            suffix = f"_lag{window}d"
            cols.extend(
                [
                    f"airline_flights_count{suffix}",
                    f"airline_late_flights_count{suffix}",
                    f"airline_total_delay_minutes{suffix}",
                    f"airline_distinct_tails_late{suffix}",
                    f"airline_late_flight_ratio{suffix}",
                    f"airline_delay_per_tail{suffix}",
                    f"airplane_total_delay_minutes{suffix}",
                    f"airplane_delay_per_km{suffix}",
                ]
            )

        for hours in AIRPORT_LOOKBACK_HOURS:
            cols.extend([f"airport_late_flights_count_lag{hours}h", f"airport_avg_delay_lag{hours}h"])

        cols.extend(
            [
                "airport_scheduled_departures_same_day",
                "airport_scheduled_arrivals_same_day",
                "airplane_flights_scheduled_same_day",
                "airplane_prev_flight_arrival_delay",
            ]
        )

        if self.include_departure_delay_features:
            cols.extend(
                [
                    "DEPARTURE_DELAY",
                    "flight_departure_delay_per_km",
                    "flight_departure_delay_vs_airline_avg_rt_lag7d",
                    "flight_computed_scheduled_duration_minutes",
                    "flight_scheduled_minutes_per_km",
                ]
            )
        return tuple(cols)


def _require_columns(df: DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns for feature preparation: {missing}")


def add_route_congestion_feature_with_year(df: DataFrame) -> DataFrame:
    """Count scheduled flights by airport, date, and hour."""

    keys = ["ORIGIN_AIRPORT", "YEAR", "MONTH", "DAY", "scheduled_hour"]
    counts = df.groupBy(*keys).agg(F.count(F.lit(1)).alias("route_congestion_count"))
    return df.join(counts, on=keys, how="left")


def build_flight_feature_frame(
    flights: DataFrame,
    airports: DataFrame,
    contract: FlightFeatureContract = FlightFeatureContract(),
) -> DataFrame:
    """Build the feature-engineered DataFrame before train/test splitting."""

    _require_columns(
        flights,
        (
            "YEAR",
            "MONTH",
            "DAY",
            "DAY_OF_WEEK",
            "AIRLINE",
            "TAIL_NUMBER",
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "SCHEDULED_DEPARTURE",
            "SCHEDULED_TIME",
            "SCHEDULED_ARRIVAL",
            "DISTANCE",
            "ARRIVAL_DELAY",
        ),
    )
    if contract.include_departure_delay_features:
        _require_columns(flights, ("DEPARTURE_DELAY",))
    _require_columns(airports, ("IATA_CODE", "LATITUDE", "LONGITUDE"))

    feature_module = _feature_module()
    df = feature_module.add_haversine_distance_feature(flights, airports)
    df = feature_module.add_scheduled_hour_feature(df)
    # Include YEAR in the congestion key so dates from different years are separate.
    df = add_route_congestion_feature_with_year(df)
    df = feature_module.add_flight_timestamp_column(df)
    df = feature_module.build_airline_level_features(
        df, lookback_windows=contract.lookback_windows
    )
    df = feature_module.build_airport_level_features(
        df, lookback_windows_hours=AIRPORT_LOOKBACK_HOURS
    )
    df = feature_module.build_airplane_level_features(
        df, lookback_windows=contract.lookback_windows
    )
    if contract.include_departure_delay_features:
        df = feature_module.build_flight_level_features(df, airline_avg_window=7)
    df = feature_module.build_temporal_level_features(df)

    return df.withColumn(
        "has_haversine_distance_num",
        F.col("has_haversine_distance").cast("double"),
    )


def build_flight_iqr_stage() -> Estimator:
    """Create the IQR estimator used inside the Spark ML pipeline."""

    feature_module = _feature_module()
    return feature_module.OutlierIQRTruncator(
        inputCols=list(IQR_INPUT_COLS),
        outputCols=list(IQR_OUTPUT_COLS),
        outlierFlagCols=list(IQR_FLAG_COLS),
        iqrMultiplier=1.5,
    )
