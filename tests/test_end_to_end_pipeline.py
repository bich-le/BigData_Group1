"""Small raw-input save/load smoke test for the production pipeline."""
from pathlib import Path
import sys

import pytest
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from mllib_pipeline import (
    PipelineConfig, configure_local_filesystem, configure_pyspark_python,
    fit_and_save_end_to_end_pipeline,
)


@pytest.fixture(scope="module")
def spark():
    configure_pyspark_python()
    session = configure_local_filesystem(
        SparkSession.builder.master("local[2]").appName("test_end_to_end_pipeline")
    ).getOrCreate()
    yield session
    session.stop()


def test_full_pipeline_scores_raw_serving_rows_after_reload(spark, tmp_path: Path):
    rows = [
        ("AA", "JFK", "LAX", 1, 1, 1, 900, 900, 2475.0, 0, 0, 5.0),
        ("AA", "JFK", "LAX", 2, 2, 2, 1000, 1000, 2475.0, 0, 0, 45.0),
        ("DL", "ATL", "SFO", 3, 3, 3, 1100, 1100, 2139.0, 0, 0, 10.0),
        ("DL", "ATL", "SFO", 4, 4, 4, 1200, 1200, 2139.0, 0, 0, 60.0),
    ]
    columns = [
        "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "MONTH", "DAY",
        "DAY_OF_WEEK", "SCHEDULED_TIME", "SCHEDULED_DEPARTURE", "DISTANCE",
        "CANCELLED", "DIVERTED", "ARRIVAL_DELAY",
    ]
    path = tmp_path / "full_pipeline"
    model = fit_and_save_end_to_end_pipeline(
        spark.createDataFrame(rows, columns), path, "classification", PipelineConfig(pca_k=2)
    )
    assert type(model.stages[0]).__name__ == "FlightEligibilityAndLabelTransformer"
    reloaded = PipelineModel.load(str(path))
    serving = spark.createDataFrame(
        [("AA", "JFK", "LAX", 5, 5, 5, 900, 900, 2475.0, 0, 0)],
        columns[:-1],
    )
    assert reloaded.transform(serving).select("prediction").count() == 1
