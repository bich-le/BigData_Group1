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


def test_gamma_pipeline_restores_delay_units_and_streams_after_reload(spark, tmp_path):
    import json
    from benchmark_results import evaluate_predictions, write_diagnostics
    from inference import input_schema, load_verified_model
    from scripts.train_tournament import make_tournament

    columns = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "MONTH", "DAY", "DAY_OF_WEEK", "SCHEDULED_TIME", "SCHEDULED_DEPARTURE", "DISTANCE", "CANCELLED", "DIVERTED", "ARRIVAL_DELAY"]
    rows = [("AA" if i % 2 else "DL", "JFK", "LAX", i % 12 + 1, i % 28 + 1, i % 7 + 1, float(80 + i), float(600 + i % 12 * 100), float(500 + i * 7), 0.0, 0.0, float(-10 + i % 50)) for i in range(40)]
    data = spark.createDataFrame(rows, columns)
    cv, _ = make_tournament("regression", PipelineConfig(pca_k=2), "glm", folds=2, small_grid=True)
    model = cv.getEstimator().fit(data, cv.getEstimatorParamMaps()[0])
    path = tmp_path / "gamma"
    model.write().save(str(path))
    loaded = load_verified_model(path, "regression")
    predictions = loaded.transform(data).cache()
    try:
        assert set(evaluate_predictions(predictions, "regression", "ARRIVAL_DELAY")) == {"rmse", "mae", "r2"}
        assert predictions.filter("prediction < 0").count() > 0
        diagnostics = write_diagnostics(loaded, tmp_path / "diagnostics")
        assert len(diagnostics["pca_explained_variance"]) == 2
        assert "pca_feature_importances" not in diagnostics
    finally:
        predictions.unpersist()
    schema = input_schema(loaded)
    assert "ARRIVAL_DELAY" not in schema.fieldNames()
    assert len(schema.fieldNames()) == len(set(schema.fieldNames()))
    source = tmp_path / "input"
    source.mkdir()
    event = dict(zip(columns[:-1], rows[0][:-1]))
    event["event_id"] = "stream-reload"
    (source / "events.json").write_text(json.dumps(event) + "\n")
    incoming = spark.readStream.schema(schema).json(str(source))
    query = (loaded.transform(incoming).select("event_id", "prediction").writeStream
             .format("json").option("path", str(tmp_path / "scored"))
             .option("checkpointLocation", str(tmp_path / "checkpoint"))
             .trigger(availableNow=True).start())
    query.awaitTermination()
    scored = spark.read.json(str(tmp_path / "scored")).collect()
    assert len(scored) == 1
    expected = loaded.transform(spark.read.schema(schema).json(str(source))).first()["prediction"]
    assert scored[0]["prediction"] == pytest.approx(expected)


@pytest.mark.parametrize("task,algorithms", [
    ("classification", ("linear_svc", "random_forest", "gbt")),
    ("regression", ("linear_regression", "glm", "random_forest", "gbt")),
])
def test_all_tournament_families_have_fold_local_preprocessing(task, algorithms):
    from scripts.train_tournament import make_tournament
    from custom_transformers import OutlierIQRTruncator
    for algorithm in algorithms:
        cv, _ = make_tournament(task, PipelineConfig(), algorithm)
        assert cv.getNumFolds() == 5
        assert cv.getParallelism() == 4
        assert any(isinstance(stage, OutlierIQRTruncator) for stage in cv.getEstimator().getStages())
        assert cv.getEstimatorParamMaps()
