"""Preprocessing, PCA, and MLflow tracking for the flight-delay project."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from pyspark.ml import Estimator, Pipeline, PipelineModel, Transformer
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    PCA,
    PCAModel,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame, SparkSession, functions as F


PipelineStage = Estimator | Transformer


@dataclass(frozen=True)
class PipelineConfig:
    categorical_cols: tuple[str, ...] = (
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
    )
    numeric_cols: tuple[str, ...] = (
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "SCHEDULED_TIME",
        "DISTANCE",
    )
    scheduled_departure_col: str = "SCHEDULED_DEPARTURE"
    regression_label: str = "ARRIVAL_DELAY"
    classification_label: str = "severe_delay"
    pca_k: int = 10
    split_train_ratio: float = 0.80
    split_seed: int = 42
    split_strategy: str = "random"
    temporal_split_col: str | None = None
    add_scheduled_hour_stage: bool = True
    pipeline_generated_numeric_cols: tuple[str, ...] = ()


class ScheduledHourTransformer(
    Transformer, DefaultParamsReadable, DefaultParamsWritable
):
    """Create scheduled_hour from an HHMM scheduled-departure column."""

    inputCol = Param(
        Params._dummy(),
        "inputCol",
        "Raw scheduled departure HHMM column",
        TypeConverters.toString,
    )
    outputCol = Param(
        Params._dummy(),
        "outputCol",
        "Derived scheduled hour column",
        TypeConverters.toString,
    )

    def __init__(
        self,
        inputCol: str = "SCHEDULED_DEPARTURE",
        outputCol: str = "scheduled_hour",
    ):
        super().__init__()
        self._setDefault(inputCol="SCHEDULED_DEPARTURE", outputCol="scheduled_hour")
        self._set(inputCol=inputCol, outputCol=outputCol)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        raw = F.trim(F.col(self.getOrDefault(self.inputCol)).cast("string"))
        hhmm = raw.cast("int")
        hour = F.when(hhmm == 2400, F.lit(0)).when(
            (hhmm >= 0) & (hhmm <= 2359),
            F.floor(hhmm / F.lit(100)),
        )
        return dataset.withColumn(
            self.getOrDefault(self.outputCol), hour.cast("double")
        )


def _require_columns(df: DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")


def prepare_eligible_flights(
    df: DataFrame,
    config: PipelineConfig = PipelineConfig(),
) -> DataFrame:
    """Filter eligible flights and create the binary classification label."""

    _require_columns(df, [config.regression_label, "CANCELLED", "DIVERTED"])
    return (
        df.filter(
            F.col(config.regression_label).isNotNull()
            & (F.col("CANCELLED") == 0)
            & (F.col("DIVERTED") == 0)
        )
        .withColumn(
            config.classification_label,
            (F.col(config.regression_label) > F.lit(30)).cast("double"),
        )
    )


def kien_pipeline_config() -> PipelineConfig:
    """Return the pipeline configuration for Kiên's feature set."""

    from feature_integration import IQR_FLAG_COLS, KienFeatureContract

    contract = KienFeatureContract()
    return PipelineConfig(
        categorical_cols=contract.categorical_cols,
        numeric_cols=contract.numeric_cols,
        split_strategy="temporal",
        temporal_split_col="_flight_timestamp_epoch",
        add_scheduled_hour_stage=False,
        pipeline_generated_numeric_cols=IQR_FLAG_COLS,
    )


def build_preprocessing_pipeline(
    config: PipelineConfig = PipelineConfig(),
    feature_stages: Sequence[PipelineStage] = (),
) -> Pipeline:
    """Build imputation, encoding, scaling, and PCA stages."""

    numeric_input = list(config.numeric_cols)
    imputed_cols = [f"{col}__imputed" for col in numeric_input]
    indexed_cols = [f"{col}__idx" for col in config.categorical_cols]
    encoded_cols = [f"{col}__ohe" for col in config.categorical_cols]

    stages: list[PipelineStage] = [
        *(
            [ScheduledHourTransformer(config.scheduled_departure_col, "scheduled_hour")]
            if config.add_scheduled_hour_stage
            else []
        ),
        *feature_stages,
        Imputer(
            strategy="median",
            inputCols=numeric_input,
            outputCols=imputed_cols,
        ),
        *[
            StringIndexer(
                inputCol=col,
                outputCol=indexed,
                handleInvalid="keep",
            )
            for col, indexed in zip(config.categorical_cols, indexed_cols)
        ],
        OneHotEncoder(
            inputCols=indexed_cols,
            outputCols=encoded_cols,
            handleInvalid="keep",
            dropLast=False,
        ),
        VectorAssembler(
            inputCols=[*imputed_cols, *encoded_cols],
            outputCol="features_raw",
            handleInvalid="keep",
        ),
        StandardScaler(
            inputCol="features_raw",
            outputCol="features_scaled",
            withStd=True,
            withMean=False,
        ),
        PCA(
            k=config.pca_k,
            inputCol="features_scaled",
            outputCol="features_pca",
        ),
    ]
    return Pipeline(stages=stages)


def split_train_test(
    df: DataFrame,
    config: PipelineConfig = PipelineConfig(),
) -> tuple[DataFrame, DataFrame]:
    """Split the input DataFrame using the configured strategy."""

    if config.split_strategy == "temporal":
        if not config.temporal_split_col:
            raise ValueError(
                "temporal_split_col is required when split_strategy='temporal'"
            )
        _require_columns(df, [config.temporal_split_col])
        cutoff = df.approxQuantile(
            config.temporal_split_col,
            [config.split_train_ratio],
            0.001,
        )[0]
        train = df.filter(F.col(config.temporal_split_col) <= F.lit(cutoff))
        test = df.filter(F.col(config.temporal_split_col) > F.lit(cutoff))
        if train.limit(1).count() == 0 or test.limit(1).count() == 0:
            raise ValueError("Temporal split produced an empty train or test set")
        return train, test

    if config.split_strategy != "random":
        raise ValueError(f"Unsupported split strategy: {config.split_strategy}")

    train, test = df.randomSplit(
        [config.split_train_ratio, 1 - config.split_train_ratio],
        seed=config.split_seed,
    )
    return train, test


def explained_variance(model: PipelineModel) -> list[float]:
    """Return the explained-variance ratio for each PCA component."""

    pca_model = next(
        (stage for stage in model.stages if isinstance(stage, PCAModel)),
        None,
    )
    if pca_model is None:
        raise ValueError("No fitted PCAModel found in the pipeline")
    return [float(value) for value in pca_model.explainedVariance.toArray().tolist()]


def save_pca_variance_plot(
    variances: Sequence[float],
    output_path: str | Path,
) -> Path:
    """Save per-component and cumulative PCA explained variance."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cumulative: list[float] = []
    running = 0.0
    for value in variances:
        running += value
        cumulative.append(running)

    components = list(range(1, len(variances) + 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        components,
        cumulative,
        marker="o",
        label="Cumulative explained variance",
    )
    ax.bar(
        components,
        variances,
        alpha=0.35,
        label="Per-component variance",
    )
    ax.set(
        xlabel="Principal component",
        ylabel="Explained variance",
        ylim=(0, 1.05),
    )
    ax.set_xticks(components)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def log_preprocessing_run(
    config: PipelineConfig,
    train_rows: int,
    test_rows: int,
    variances: Sequence[float],
    plot_path: Path,
    experiment_name: str,
) -> dict[str, str]:
    """Log preprocessing parameters, metrics, and artifacts to MLflow."""

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow is not installed") from exc

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="preprocessing-pca") as run:
        mlflow.log_params(
            {
                "pca_k": config.pca_k,
                "split_strategy": config.split_strategy,
                "split_train_ratio": config.split_train_ratio,
                "split_seed": config.split_seed,
                "categorical_cols": ",".join(config.categorical_cols),
                "numeric_cols": ",".join(config.numeric_cols),
            }
        )
        metrics = {
            "train_rows": float(train_rows),
            "test_rows": float(test_rows),
            "pca_variance_cumulative": float(sum(variances)),
        }
        metrics.update(
            {f"pca_variance_pc{i}": float(value) for i, value in enumerate(variances, 1)}
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(plot_path), artifact_path="plots")
        mlflow.log_dict(asdict(config), "config/pipeline_config.json")
        return {"run_id": run.info.run_id}


def fit_preprocessing(
    flights: DataFrame,
    output_dir: str | Path,
    config: PipelineConfig = PipelineConfig(),
    feature_stages: Sequence[PipelineStage] = (),
    mlflow_experiment: str | None = None,
) -> tuple[PipelineModel, DataFrame, DataFrame, dict[str, object]]:
    """Fit preprocessing on train data and transform the test set."""

    _require_columns(
        flights,
        [
            *config.categorical_cols,
            *[
                col
                for col in config.numeric_cols
                if col not in config.pipeline_generated_numeric_cols
            ],
            *(
                [config.scheduled_departure_col]
                if config.add_scheduled_hour_stage
                else []
            ),
        ],
    )

    eligible = prepare_eligible_flights(flights, config)
    train, test = split_train_test(eligible, config)

    pipeline = build_preprocessing_pipeline(config, feature_stages)
    model = pipeline.fit(train)
    transformed_test = model.transform(test)

    output = Path(output_dir)
    variances = explained_variance(model)
    plot = save_pca_variance_plot(
        variances,
        output / "docs" / "assets" / "pca_variance.png",
    )

    train_rows = train.count()
    test_rows = transformed_test.count()
    summary: dict[str, object] = {
        "config": asdict(config),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "explained_variance": variances,
        "pca_plot": str(plot),
    }

    if mlflow_experiment:
        summary["mlflow"] = log_preprocessing_run(
            config,
            train_rows,
            test_rows,
            variances,
            plot,
            mlflow_experiment,
        )

    artifact_dir = output / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return model, train, transformed_test, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the preprocessing and PCA pipeline on typed flight Parquet data."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to data/typed-parquet/flights",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for artifacts and the PCA plot",
    )
    parser.add_argument(
        "--mlflow-experiment",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--with-kien-features",
        action="store_true",
        help="Use feature engineering from custom_transformers.py",
    )
    parser.add_argument(
        "--airports",
        help="Path to typed airports Parquet; required with --with-kien-features",
    )
    args = parser.parse_args()

    spark = SparkSession.builder.appName("flight-preprocessing-pca").getOrCreate()
    try:
        flights = spark.read.parquet(args.input)
        stages: list[PipelineStage] = []

        if args.with_kien_features:
            if not args.airports:
                parser.error("--airports is required with --with-kien-features")

            from feature_integration import (
                build_kien_feature_frame,
                build_kien_iqr_stage,
            )

            config = kien_pipeline_config()
            airports = spark.read.parquet(args.airports)
            flights = build_kien_feature_frame(
                prepare_eligible_flights(flights, config),
                airports,
            )
            stages = [build_kien_iqr_stage()]
        else:
            config = PipelineConfig()

        _, _, _, summary = fit_preprocessing(
            flights,
            args.output_dir,
            config,
            stages,
            args.mlflow_experiment,
        )
        print(json.dumps(summary, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
