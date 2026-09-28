"""Smoke test for the preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import psutil
from pyspark.sql import SparkSession

from mllib_pipeline import (
    PipelineConfig,
    configure_local_filesystem,
    configure_pyspark_python,
    fit_preprocessing,
    kien_pipeline_config,
    prepare_eligible_flights,
)


def synthetic_flights(rows: int) -> list[tuple[object, ...]]:
    airlines = ["AA", "DL", "UA", "WN"]
    origins = ["ATL", "LAX", "ORD", "DFW"]
    destinations = ["JFK", "SFO", "SEA", "BOS"]
    result = []
    for i in range(rows):
        result.append(
            (
                2015,
                1,
                1 + ((i // 4) % 28),
                (i % 7) + 1,
                airlines[i % len(airlines)],
                f"{100 + i % 50}",
                f"N{1000 + i % 12}",
                origins[i % len(origins)],
                destinations[i % len(destinations)],
                f"{(6 + i % 16):02d}{(i * 7) % 60:02d}",
                90 + (i % 180),
                250 + (i % 1200),
                float((i % 100) - 20),
                float((i % 45) - 8),
                0,
                0,
            )
        )
    return result


def synthetic_airports() -> list[tuple[str, float, float]]:
    return [
        ("ATL", 33.6407, -84.4277),
        ("LAX", 33.9416, -118.4085),
        ("ORD", 41.9742, -87.9073),
        ("DFW", 32.8998, -97.0403),
        ("JFK", 40.6413, -73.7781),
        ("SFO", 37.6213, -122.3790),
        ("SEA", 47.4502, -122.3088),
        ("BOS", 42.3656, -71.0096),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--mlflow-experiment")
    parser.add_argument("--with-kien-features", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    process = psutil.Process(os.getpid())
    started = time.perf_counter()

    configure_pyspark_python()
    spark = configure_local_filesystem(
        SparkSession.builder.master("local[2]").appName("preprocessing-smoke-test")
    ).getOrCreate()
    try:
        columns = [
            "YEAR",
            "MONTH",
            "DAY",
            "DAY_OF_WEEK",
            "AIRLINE",
            "FLIGHT_NUMBER",
            "TAIL_NUMBER",
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "SCHEDULED_DEPARTURE",
            "SCHEDULED_TIME",
            "DISTANCE",
            "ARRIVAL_DELAY",
            "DEPARTURE_DELAY",
            "CANCELLED",
            "DIVERTED",
        ]
        flights = spark.createDataFrame(synthetic_flights(args.rows), columns)
        stages = []
        config = PipelineConfig()

        if args.with_kien_features:
            from feature_integration import (
                build_kien_feature_frame,
                build_kien_iqr_stage,
            )

            airports = spark.createDataFrame(
                synthetic_airports(),
                ["IATA_CODE", "LATITUDE", "LONGITUDE"],
            )
            config = kien_pipeline_config()
            flights = build_kien_feature_frame(
                prepare_eligible_flights(flights, config),
                airports,
            )
            stages = [build_kien_iqr_stage()]

        _, _, transformed, summary = fit_preprocessing(
            flights,
            output,
            config,
            stages,
            mlflow_experiment=args.mlflow_experiment,
        )
        assert transformed.select("features_pca").limit(1).count() == 1

        summary.update(
            {
                "smoke_test": "PASS",
                "synthetic_rows": args.rows,
                "kien_features": args.with_kien_features,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "process_rss_mb": round(
                    process.memory_info().rss / 1024 / 1024,
                    2,
                ),
                "spark_master": spark.sparkContext.master,
                "spark_version": spark.version,
            }
        )

        artifact = output / "artifacts" / "smoke_test_metrics.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
