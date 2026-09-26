"""PySpark version of `02_sfdws_model copy.ipynb` and `function_utils (1).py`.

The processing order, feature list, filters, train/validation/OOT split, grid
search and final Random Forest workflow are retained.  This file deliberately
uses Spark DataFrames and Spark ML only; it does not collect the source data
to pandas.
"""

import argparse
import gc
import json
import math
from itertools import product
from pathlib import Path

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, Window, functions as F
from pyspark.sql.types import BooleanType, DoubleType, IntegerType, NumericType, StringType, StructField, StructType

import os
import sys

# Force Spark Python workers to use the current conda environment
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession, Window, functions as F

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT.parent / "huy/model-ready/base"
DEFAULT_MODEL_PATH = ROOT / "models/random_forest_regression_final_spark"

# Honour spark-submit runtime settings such as --master and --driver-memory.
spark = (
    SparkSession.builder
    .appName("FlightDelayRegressionTraining")
    .config("spark.pyspark.python", sys.executable)
    .config("spark.pyspark.driver.python", sys.executable)
    .getOrCreate()
)
# ---------------------------------------------------------------------------
# Library and configuration (the first code cell in the supplied notebook)
# ---------------------------------------------------------------------------

CONFIGURE = {
    "oot_split_threshold": "2015-11-01",
    "missing_threshold": 0.7,
    "concentration_threshold": 0.9,
    "psi_threshold": 0.25,
    "corr_threshold": 0.7,
    "linear_svc": {
        "C": 1, "penalty": "l2", "loss": "squared_hinge", "class_weight": "balanced",
        "dual": False, "tol": 1e-3, "max_iter": 1000, "random_state": 42,
    },
    "linear_svc_grid": {
        "C": [0.001, 0.01, 0.1, 1, 10, 30, 100], "penalty": ["l1", "l2"],
        "loss": ["hinge", "squared_hinge"], "class_weight": [None, "balanced"],
        "dual": [True, False],
    },
    "random_forest": {
        "n_estimators": 300, "criterion": "variance", "max_depth": 15,
        "min_samples_leaf": 5, "max_features": "sqrt",
        "bootstrap": True, "random_state": 42,
    },
    "random_forest_grid": {
        "n_estimators": [300, 500], "max_depth": [12, 15, 18],
        "min_samples_leaf": [5, 10],
        "max_features": ["sqrt"], "criterion": ["variance"], "bootstrap": [True],
    },
    "gradient_boosting": {
        "n_estimators": 200, "learning_rate": 0.05, "max_depth": 3,
        "min_samples_split": 20, "min_samples_leaf": 10, "subsample": 0.8,
        "max_features": None, "random_state": 42,
    },
    "gradient_boosting_grid": {
        "n_estimators": [150, 250], "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5], "min_samples_leaf": [10], "subsample": [0.8],
        "max_features": [None],
    },
}


def _is_numeric(df, column):
    """Return whether a Spark column is numeric, excluding Boolean."""
    data_type = df.schema[column].dataType
    return isinstance(data_type, NumericType) and not isinstance(data_type, BooleanType)


def _safe_div(numerator, denominator):
    return F.when(denominator != 0, numerator / denominator)


def _parameter_grid(param_grid):
    """Spark-only equivalent of sklearn.model_selection.ParameterGrid."""
    keys = list(param_grid)
    for values in product(*(param_grid[key] for key in keys)):
        yield dict(zip(keys, values))



def _rf_estimator(params, label_col="ARRIVAL_DELAY", features_col="features"):
    """Build a Spark RandomForestRegressor from the notebook-style parameters."""
    estimator = RandomForestRegressor(
        labelCol=label_col,
        featuresCol=features_col,
        predictionCol="prediction",
        numTrees=int(params.get("n_estimators", params.get("numTrees", 100))),
        maxDepth=int(params.get("max_depth", params.get("maxDepth", 10))),
        minInstancesPerNode=int(
            params.get("min_samples_leaf", params.get("minInstancesPerNode", 1))
        ),
        featureSubsetStrategy={
            "sqrt": "sqrt",
            "log2": "log2",
            None: "auto",
        }.get(params.get("max_features"), params.get("max_features", "auto")),
        impurity=params.get("criterion", params.get("impurity", "variance")),
        bootstrap=bool(params.get("bootstrap", True)),
        seed=int(params.get("random_state", params.get("seed", 42))),
    )
    return estimator


def fit_random_forest(df, feature_cols, params, label_col="ARRIVAL_DELAY"):
    """Fit Spark RandomForestRegressor."""
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="keep",
    )
    regressor = _rf_estimator(params, label_col, "features")
    return Pipeline(stages=[assembler, regressor]).fit(df)


def score_with_model(model, df):
    return model.transform(df)


# ---------------------------------------------------------------------------
# Feature-filter functions converted from function_utils (1).py
# ---------------------------------------------------------------------------
def missing_filter_fct(df, missing_threshold):
    missing_rates = df.select(*[
        F.avg(F.when(F.col(column).isNull(), F.lit(1.0)).otherwise(F.lit(0.0))).alias(column)
        for column in df.columns
    ]).first().asDict()
    return [column for column, rate in missing_rates.items() if rate is not None and rate > missing_threshold]


def concentration_filter_fct(df, concentration_threshold):
    drop_cols = []
    for column in df.columns:
        non_null = df.where(F.col(column).isNotNull())
        n_rows = non_null.count()
        if n_rows:
            maximum = non_null.groupBy(column).count().agg(F.max("count").alias("maximum")).first()["maximum"]
            if maximum / n_rows > concentration_threshold:
                drop_cols.append(column)
    return drop_cols


def psi_filter_fct(X_train, X_test, psi_threshold, num_bins=10):
    """Population Stability Index, retaining the supplied pandas algorithm's rules."""
    result = []
    eps = 1e-4
    total_base = X_train.count()
    total_comp = X_test.count()
    if not total_base or not total_comp:
        raise ValueError("PSI requires non-empty train and comparison DataFrames.")

    for column in X_train.columns:
        base_missing = X_train.where(F.col(column).isNull()).count()
        comp_missing = X_test.where(F.col(column).isNull()).count()
        probabilities = []
        if base_missing or comp_missing:
            probabilities.append((base_missing / total_base, comp_missing / total_comp))

        base_clean = X_train.where(F.col(column).isNotNull())
        comp_clean = X_test.where(F.col(column).isNotNull())
        if base_clean.limit(1).count():
            if _is_numeric(X_train, column):
                quantiles = base_clean.approxQuantile(column, [i / num_bins for i in range(num_bins + 1)], 0.0)
                bounds = sorted(set(quantiles))
                if len(bounds) > 1:
                    base_min_max = base_clean.agg(F.min(column).alias("min"), F.max(column).alias("max")).first()
                    comp_min_max = comp_clean.agg(F.min(column).alias("min"), F.max(column).alias("max")).first()
                    low = min(base_min_max["min"], comp_min_max["min"] if comp_min_max["min"] is not None else base_min_max["min"])
                    high = max(base_min_max["max"], comp_min_max["max"] if comp_min_max["max"] is not None else base_min_max["max"])
                    bounds[0], bounds[-1] = low - 1e-5, high + 1e-5
                    for index in range(len(bounds) - 1):
                        lower, upper = bounds[index], bounds[index + 1]
                        predicate = (F.col(column) >= lower) & (F.col(column) < upper)
                        if index == len(bounds) - 2:
                            predicate = (F.col(column) >= lower) & (F.col(column) <= upper)
                        probabilities.append((base_clean.where(predicate).count() / total_base, comp_clean.where(predicate).count() / total_comp))
            else:
                base_counts = base_clean.groupBy(column).count().withColumnRenamed("count", "base_count")
                comp_counts = comp_clean.groupBy(column).count().withColumnRenamed("count", "comp_count")
                for row in base_counts.join(comp_counts, column, "full").fillna(0).collect():
                    probabilities.append((row["base_count"] / total_base, row["comp_count"] / total_comp))

        psi = sum((max(p, eps) - max(q, eps)) * math.log(max(p, eps) / max(q, eps)) for p, q in probabilities)
        result.append((column, float(psi)))

    schema = StructType([StructField("variable_name", StringType(), False), StructField("psi", DoubleType(), False)])
    df_psi = spark.createDataFrame(result, schema=schema)
    return df_psi, [row["variable_name"] for row in df_psi.where(F.col("psi") > psi_threshold).select("variable_name").collect()]


_CLASSING_SCHEMA = StructType([
    StructField("Bins", StringType(), True), StructField("Bin_Lower", DoubleType(), True), StructField("Bin_Upper", DoubleType(), True),
    StructField("Good", DoubleType(), True), StructField("Bad", DoubleType(), True), StructField("%Good", DoubleType(), True),
    StructField("%Bad", DoubleType(), True), StructField("%Odds", DoubleType(), True), StructField("SFD_Rate", DoubleType(), True),
    StructField("Observation_Number", DoubleType(), True), StructField("%Observation_Number", DoubleType(), True), StructField("WOE", DoubleType(), True),
])


def fine_classing(X, y, variable, target_name="Bad"):
    """Fine classing using Spark quantiles/grouping; returns a Spark DataFrame."""
    label_df = y if hasattr(y, "columns") else None
    if label_df is None:
        raise TypeError("y must be a Spark DataFrame containing the target column.")
    joined = X.select(variable, "__row_id").join(label_df.select("__row_id", F.col(target_name).alias(target_name)), "__row_id")
    is_numeric = _is_numeric(X, variable)
    if is_numeric:
        quantiles = joined.where(F.col(variable).isNotNull()).approxQuantile(variable, [i / 10 for i in range(11)], 0.0)
        bounds = sorted(set(quantiles))
        if len(bounds) <= 1:
            return spark.createDataFrame([], _CLASSING_SCHEMA)
        bucket_expr = None
        for i in range(len(bounds) - 1):
            condition = (F.col(variable) >= bounds[i]) & (F.col(variable) < bounds[i + 1])
            if i == len(bounds) - 2:
                condition = (F.col(variable) >= bounds[i]) & (F.col(variable) <= bounds[i + 1])
            value = F.lit(f"({bounds[i]},{bounds[i + 1]}]")
            bucket_expr = F.when(condition, value) if bucket_expr is None else bucket_expr.when(condition, value)
        grouped = joined.where(F.col(variable).isNotNull()).withColumn("Bins", bucket_expr).groupBy("Bins").agg(
            F.min(variable).cast("double").alias("Bin_Lower"), F.max(variable).cast("double").alias("Bin_Upper"),
            F.count(variable).cast("double").alias("Observation_Number"), F.sum(target_name).cast("double").alias("Bad"),
        )
    else:
        grouped = joined.withColumn("Bins", F.coalesce(F.col(variable).cast("string"), F.lit("NULL"))).groupBy("Bins").agg(
            F.first(F.lit(None).cast("double")).alias("Bin_Lower"), F.first(F.lit(None).cast("double")).alias("Bin_Upper"),
            F.count(F.lit(1)).cast("double").alias("Observation_Number"), F.sum(target_name).cast("double").alias("Bad"),
        )
    grouped = grouped.withColumn("Good", F.col("Observation_Number") - F.col("Bad"))
    totals = grouped.agg(F.sum("Good").alias("good"), F.sum("Bad").alias("bad"), F.sum("Observation_Number").alias("obs")).first()
    detailed = grouped.select(
        "Bins", "Bin_Lower", "Bin_Upper", "Good", "Bad",
        (_safe_div(F.col("Good"), F.lit(totals["good"])) * 100).alias("%Good"),
        (_safe_div(F.col("Bad"), F.lit(totals["bad"])) * 100).alias("%Bad"),
        (_safe_div(F.col("Bad"), F.col("Good")) * 100).alias("%Odds"),
        (_safe_div(F.col("Bad"), F.col("Observation_Number")) * 100).alias("SFD_Rate"), "Observation_Number",
        (_safe_div(F.col("Observation_Number"), F.lit(totals["obs"])) * 100).alias("%Observation_Number"),
    )
    detailed = detailed.withColumn("WOE", F.when((F.col("%Good") > 0) & (F.col("%Bad") > 0), F.log(F.col("%Good") / F.col("%Bad"))))
    total_row = spark.createDataFrame([("Total", None, None, totals["good"], totals["bad"], 100.0, 100.0, None, (totals["bad"] / totals["obs"] * 100) if totals["obs"] else None, totals["obs"], 100.0, None)], _CLASSING_SCHEMA)
    return detailed.select(_CLASSING_SCHEMA.fieldNames()).unionByName(total_row)


def coarse_classing(df_fine, min_obs_pct=5, min_bad_pct=5, min_bad_count=50,
                    max_concentration_numeric=35, max_concentration_categorical=90,
                    min_groups=2, max_groups=8):
    """Coarse classing. Spark performs aggregation; small bin metadata is merged on the driver."""
    rows = [row.asDict() for row in df_fine.where(F.col("Bins") != "Total").collect()]
    if not rows:
        return spark.createDataFrame([], _CLASSING_SCHEMA)
    numeric = any(row["Bin_Lower"] is not None for row in rows)
    rows.sort(key=lambda row: (row["Bin_Lower"] is None, row["Bin_Lower"] or 0, row["Bins"]))
    total_good, total_bad = sum(row["Good"] for row in rows), sum(row["Bad"] for row in rows)
    total_obs = total_good + total_bad

    def refresh(values):
        for row in values:
            row["Observation_Number"] = row["Good"] + row["Bad"]
            row["%Observation_Number"] = row["Observation_Number"] / total_obs * 100 if total_obs else None
            row["%Good"] = row["Good"] / total_good * 100 if total_good else None
            row["%Bad"] = row["Bad"] / total_bad * 100 if total_bad else None
            row["SFD_Rate"] = row["Bad"] / row["Observation_Number"] * 100 if row["Observation_Number"] else None
            row["%Odds"] = row["Bad"] / row["Good"] * 100 if row["Good"] else None
            row["WOE"] = math.log(row["%Good"] / row["%Bad"]) if row["%Good"] and row["%Bad"] else None
        return values

    def merge(values, i, j):
        left, right = values[i], values[j]
        if numeric:
            left["Bin_Lower"] = min(left["Bin_Lower"], right["Bin_Lower"])
            left["Bin_Upper"] = max(left["Bin_Upper"], right["Bin_Upper"])
            left["Bins"] = f"({left['Bin_Lower']},{left['Bin_Upper']}]"
        else:
            left["Bins"] = f"{left['Bins']},{right['Bins']}"
        left["Good"], left["Bad"] = left["Good"] + right["Good"], left["Bad"] + right["Bad"]
        values.pop(j)
        return refresh(values)

    rows = refresh(rows)
    limit = max_concentration_numeric if numeric else max_concentration_categorical
    while len(rows) > max_groups:
        violations = [i for i, row in enumerate(rows) if row["%Observation_Number"] < min_obs_pct or (row["%Bad"] < min_bad_pct and row["Bad"] < min_bad_count) or row["%Observation_Number"] > limit]
        if not violations:
            break
        i = violations[0]
        neighbours = ([v for v in (i - 1, i + 1) if 0 <= v < len(rows)] if numeric else [v for v in range(len(rows)) if v != i])
        if not neighbours:
            break
        j = min(neighbours, key=lambda v: abs(rows[v]["SFD_Rate"] - rows[i]["SFD_Rate"]))
        rows = merge(rows, min(i, j), max(i, j))
    while len(rows) > min_groups:
        too_concentrated = next((i for i, row in enumerate(rows) if row["%Observation_Number"] > limit), None)
        if too_concentrated is None:
            break
        i = too_concentrated
        neighbours = ([v for v in (i - 1, i + 1) if 0 <= v < len(rows)] if numeric else [v for v in range(len(rows)) if v != i])
        if not neighbours:
            break
        j = min(neighbours, key=lambda v: abs(rows[v]["SFD_Rate"] - rows[i]["SFD_Rate"]))
        rows = merge(rows, min(i, j), max(i, j))
    total = {"Bins": "Total", "Bin_Lower": None, "Bin_Upper": None, "Good": total_good, "Bad": total_bad, "%Good": 100.0, "%Bad": 100.0, "%Odds": total_bad / total_good * 100 if total_good else None, "SFD_Rate": total_bad / total_obs * 100 if total_obs else None, "Observation_Number": total_obs, "%Observation_Number": 100.0, "WOE": None}
    return spark.createDataFrame(rows + [total], _CLASSING_SCHEMA)


def feature_label_correlation_fct(df, label_col="ARRIVAL_DELAY"):
    """Measure Pearson correlation between every numeric feature and ARRIVAL_DELAY."""
    rows = []
    for column in df.columns:
        if column == label_col or column == "__row_id":
            continue
        if _is_numeric(df, column) and _is_numeric(df, label_col):
            corr = df.stat.corr(column, label_col)
            rows.append((column, float(corr) if corr is not None else 0.0))

    schema = StructType([
        StructField("Variable", StringType(), False),
        StructField("Corr_With_Label", DoubleType(), False),
        StructField("Abs_Corr_With_Label", DoubleType(), False),
    ])
    return spark.createDataFrame(
        [(v, c, abs(c)) for v, c in rows],
        schema=schema,
    ).orderBy(F.desc("Abs_Corr_With_Label"))


def multicollinearity_filter_fct(
    df,
    df_feature_corr,
    corr_threshold=0.7,
    exclude_cols=None,
):
    """
    Remove one feature from each highly-correlated feature pair.

    Selection priority is based on |corr(feature, ARRIVAL_DELAY)|:
    when |corr(A,B)| > threshold, keep the feature having stronger
    absolute correlation with ARRIVAL_DELAY and drop the other one.
    """
    exclude_cols = exclude_cols or []
    cols = [c for c in df.columns if c not in exclude_cols and c != "__row_id"]
    numeric_cols = [c for c in cols if _is_numeric(df, c)]

    label_corr = {
        row["Variable"]: row["Abs_Corr_With_Label"]
        for row in df_feature_corr.collect()
    }

    associations = []
    for i, left in enumerate(numeric_cols):
        for right in numeric_cols[i + 1:]:
            value = df.stat.corr(left, right)
            associations.append(
                (left, right, float(value) if value is not None else 0.0, "Pearson")
            )

    schema = StructType([
        StructField("Variable A", StringType(), False),
        StructField("Variable B", StringType(), False),
        StructField("Correlation", DoubleType(), False),
        StructField("Method", StringType(), False),
    ])
    df_corr = spark.createDataFrame(associations, schema=schema)

    dropped = set()
    # Process strongest feature-label correlations first.
    sorted_cols = sorted(
        numeric_cols,
        key=lambda c: label_corr.get(c, 0.0),
        reverse=True,
    )

    for i, column in enumerate(sorted_cols):
        if column in dropped:
            continue

        for other in sorted_cols[i + 1:]:
            if other in dropped:
                continue

            pair = df_corr.where(
                ((F.col("Variable A") == column) & (F.col("Variable B") == other))
                | ((F.col("Variable A") == other) & (F.col("Variable B") == column))
            ).select("Correlation").first()

            pair_corr = abs(pair["Correlation"]) if pair else 0.0
            if pair_corr > corr_threshold:
                # column is earlier because it has stronger label correlation.
                dropped.add(other)

    return df_corr, sorted(dropped)


def recursive_feature_elimination_cv(
    X_train,
    y_train,
    model_params,
    cv=3,
    n_jobs=1,
    save_path="rfecv_profile",
    label_col="ARRIVAL_DELAY",
):
    """Spark RFECV-style feature elimination for regression."""
    features = [column for column in X_train.columns if column != "__row_id"]
    data = X_train.join(y_train, "__row_id")
    folds = data.withColumn(
        "__fold",
        F.pmod(
            F.row_number().over(Window.orderBy(F.rand(42))) - 1,
            F.lit(cv),
        ),
    ).repartition(160).persist(StorageLevel.DISK_ONLY)

    evaluator = RegressionEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="rmse",
    )
    profile = []
    current = features[:]

    while current:
        scores = []
        for fold in range(cv):
            train_fold = folds.where(F.col("__fold") != fold)
            valid_fold = folds.where(F.col("__fold") == fold)
            model = fit_random_forest(train_fold, current, model_params, label_col)
            scores.append(evaluator.evaluate(score_with_model(model, valid_fold)))

        profile.append((len(current), *scores, sum(scores) / len(scores), json.dumps(current)))

        full_model = fit_random_forest(data, current, model_params, label_col)
        importances = full_model.stages[-1].featureImportances.toArray().tolist()
        current.pop(importances.index(min(importances)))

    schema = StructType(
        [StructField("n_features", IntegerType(), False)]
        + [StructField(f"rmse_cv{i}", DoubleType(), True) for i in range(1, cv + 1)]
        + [StructField("rmse_mean", DoubleType(), True), StructField("features", StringType(), False)]
    )
    profile_df = spark.createDataFrame(profile, schema=schema).orderBy("n_features")
    best = profile_df.orderBy("rmse_mean").first()
    folds.unpersist(blocking=True)

    return json.loads(best["features"]), profile_df


# ---------------------------------------------------------------------------
# Modelling and metrics converted from function_utils (1).py
# ---------------------------------------------------------------------------
def grid_search_cv_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
    model,
    param_grid,
    n_jobs=-1,
    label_col="ARRIVAL_DELAY",
    checkpoint_path=None,
):
    """Three-fold Spark CV grid search for RandomForestRegressor.

    Completed combinations are atomically checkpointed on the driver so an
    interrupted multi-day run can resume without changing the grid or metrics.
    """
    feature_cols = [column for column in X_train.columns if column != "__row_id"]
    train = X_train.join(y_train, "__row_id")
    valid = X_valid.join(y_valid, "__row_id")

    folds = train.withColumn(
        "__fold",
        F.pmod(
            F.row_number().over(Window.orderBy(F.rand(42))) - 1,
            F.lit(3),
        ),
    ).repartition(160).persist(StorageLevel.DISK_ONLY)

    evaluators = {
        "rmse": RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse"),
        "mae": RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae"),
        "r2": RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2"),
    }

    def evaluate_all(fitted, data):
        scored = fitted.transform(data).select(label_col, "prediction").persist(StorageLevel.DISK_ONLY)
        try:
            return {name: evaluator.evaluate(scored) for name, evaluator in evaluators.items()}
        finally:
            scored.unpersist(blocking=True)

    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    checkpoint_rows = []
    if checkpoint and checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved.get("feature_cols") != feature_cols:
            raise ValueError("Grid checkpoint feature contract does not match this run")
        checkpoint_rows = saved.get("rows", [])
    completed = {row["params"]: row for row in checkpoint_rows}

    result_fields = [
        "rmse_cv", "mae_cv", "r2_cv", "rmse_train", "rmse_valid",
        "mae_train", "mae_valid", "r2_train", "r2_valid", "params",
    ]
    results = []
    grid = list(_parameter_grid(param_grid))
    print(
        f"[3] Starting full RF grid: {len(grid)} combinations, 3 folds each; "
        f"resuming {len(completed)} completed combinations",
        flush=True,
    )
    for grid_index, params in enumerate(grid, start=1):
        merged = dict(model)
        merged.update(params)
        params_json = json.dumps(merged, sort_keys=True)
        if params_json in completed:
            row = completed[params_json]
            results.append(tuple(row[field] for field in result_fields))
            print(f"[3] Grid {grid_index}/{len(grid)} restored from checkpoint", flush=True)
            continue

        print(f"[3] Grid {grid_index}/{len(grid)} starting: {merged}", flush=True)
        cv_metrics = {name: [] for name in evaluators}
        for fold in range(3):
            fitted = fit_random_forest(
                folds.where(F.col("__fold") != fold),
                feature_cols,
                merged,
                label_col,
            )
            fold_metrics = evaluate_all(fitted, folds.where(F.col("__fold") == fold))
            for name, value in fold_metrics.items():
                cv_metrics[name].append(value)
            del fitted
            gc.collect()
            spark._jvm.System.gc()

        full_model = fit_random_forest(train, feature_cols, merged, label_col)
        train_metrics = evaluate_all(full_model, train)
        valid_metrics = evaluate_all(full_model, valid)
        del full_model
        gc.collect()
        spark._jvm.System.gc()

        result = (
            sum(cv_metrics["rmse"]) / 3,
            sum(cv_metrics["mae"]) / 3,
            sum(cv_metrics["r2"]) / 3,
            train_metrics["rmse"],
            valid_metrics["rmse"],
            train_metrics["mae"],
            valid_metrics["mae"],
            train_metrics["r2"],
            valid_metrics["r2"],
            params_json,
        )
        results.append(result)
        saved_row = dict(zip(result_fields, result))
        checkpoint_rows.append(saved_row)
        if checkpoint:
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            temporary.write_text(json.dumps({
                "feature_cols": feature_cols,
                "split": {"train_fraction": 0.8, "seed": 42, "oot_start": "2015-11-01"},
                "rows": checkpoint_rows,
            }, indent=2))
            temporary.replace(checkpoint)
        print(
            f"[3] Grid {grid_index}/{len(grid)} complete: "
            f"cv_rmse={result[0]:.6f}, valid_rmse={result[4]:.6f}, "
            f"valid_mae={result[6]:.6f}, valid_r2={result[8]:.6f}",
            flush=True,
        )

    folds.unpersist(blocking=True)

    schema = StructType([
        StructField("rmse_cv", DoubleType(), False),
        StructField("mae_cv", DoubleType(), False),
        StructField("r2_cv", DoubleType(), False),
        StructField("rmse_train", DoubleType(), False),
        StructField("rmse_valid", DoubleType(), False),
        StructField("mae_train", DoubleType(), False),
        StructField("mae_valid", DoubleType(), False),
        StructField("r2_train", DoubleType(), False),
        StructField("r2_valid", DoubleType(), False),
        StructField("params", StringType(), False),
    ])
    return spark.createDataFrame(results, schema=schema).orderBy("rmse_valid")

def run_single_cv_fold(
    X_train,
    y_train,
    model,
    param_grid,
    fold_number,
    label_col="ARRIVAL_DELAY",
    checkpoint_path=None,
):
    """Fit and checkpoint exactly one of the established three CV folds."""
    feature_cols = [column for column in X_train.columns if column != "__row_id"]
    grid = list(_parameter_grid(param_grid))
    if len(grid) != 1:
        raise ValueError("Single-fold mode requires exactly one grid combination")
    if fold_number not in (1, 2, 3):
        raise ValueError("fold_number must be 1, 2, or 3")

    train = X_train.join(y_train, "__row_id")
    folds = train.withColumn(
        "__fold",
        F.pmod(
            F.row_number().over(Window.orderBy(F.rand(42))) - 1,
            F.lit(3),
        ),
    ).repartition(160).persist(StorageLevel.DISK_ONLY)
    params = dict(model)
    params.update(grid[0])
    params_json = json.dumps(params, sort_keys=True)
    fold_index = fold_number - 1
    print(f"[3] Single-fold run starting: fold={fold_number}/3 params={params}", flush=True)

    fitted = fit_random_forest(
        folds.where(F.col("__fold") != fold_index),
        feature_cols,
        params,
        label_col,
    )
    scored = fitted.transform(
        folds.where(F.col("__fold") == fold_index)
    ).select(label_col, "prediction").persist(StorageLevel.DISK_ONLY)
    try:
        metrics = {
            name: RegressionEvaluator(
                labelCol=label_col,
                predictionCol="prediction",
                metricName=name,
            ).evaluate(scored)
            for name in ("rmse", "mae", "r2")
        }
    finally:
        scored.unpersist(blocking=True)
        folds.unpersist(blocking=True)

    row = {
        "fold": fold_number,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "params": params_json,
    }
    if checkpoint_path:
        checkpoint = Path(checkpoint_path)
        saved = {
            "feature_cols": feature_cols,
            "split": {"train_fraction": 0.8, "seed": 42, "oot_start": "2015-11-01"},
            "cv_folds": 3,
            "rows": [],
        }
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved.get("feature_cols") != feature_cols:
                raise ValueError("Fold checkpoint feature contract does not match this run")
        saved["rows"] = [
            existing for existing in saved.get("rows", [])
            if not (existing["fold"] == fold_number and existing["params"] == params_json)
        ] + [row]
        temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        temporary.write_text(json.dumps(saved, indent=2))
        temporary.replace(checkpoint)
    print(
        f"[3] Single-fold run complete: fold={fold_number}/3 "
        f"rmse={metrics['rmse']:.6f}, mae={metrics['mae']:.6f}, r2={metrics['r2']:.6f}",
        flush=True,
    )
    del fitted
    gc.collect()
    spark._jvm.System.gc()
    return row


def model_result(
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_oot,
    y_oot,
    model,
    label_col="ARRIVAL_DELAY",
):
    """Return RMSE/MAE/R2 on train, validation and OOT."""
    evaluator_metrics = {
        "RMSE": RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse"),
        "MAE": RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae"),
        "R2": RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2"),
    }

    datasets = {
        "Train": score_with_model(model, X_train.join(y_train, "__row_id")),
        "Valid": score_with_model(model, X_valid.join(y_valid, "__row_id")),
        "OOT": score_with_model(model, X_oot.join(y_oot, "__row_id")),
    }

    rows = []
    for metric, evaluator in evaluator_metrics.items():
        rows.append((
            metric,
            *[float(evaluator.evaluate(datasets[name])) for name in ("Train", "Valid", "OOT")]
        ))

    indicators = spark.createDataFrame(rows, ["Metric", "Train", "Valid", "OOT"])

    output = (
        datasets["Train"].select("prediction", F.col(label_col).alias("y_actual"))
        .withColumn("dataset", F.lit("Train"))
        .unionByName(
            datasets["Valid"].select("prediction", F.col(label_col).alias("y_actual"))
            .withColumn("dataset", F.lit("Valid"))
        )
        .unionByName(
            datasets["OOT"].select("prediction", F.col(label_col).alias("y_actual"))
            .withColumn("dataset", F.lit("OOT"))
        )
    )
    return indicators, output


def run_pipeline(
    data_path,
    model_output_path,
    label_col="ARRIVAL_DELAY",
    param_grid=None,
    grid_only=False,
    single_cv_fold=None,
):
    """
    Random Forest Regression pipeline.

    Flow:
      1. Filter all eligible records.
      2. Split into train/valid/OOT.
      3. Missing + concentration filters.
      4. Measure corr(feature, ARRIVAL_DELAY).
      5. Multicollinearity: for highly correlated feature pairs, keep the
         feature with stronger |corr(feature, ARRIVAL_DELAY)|.
      6. RF regression CV.
      7. Final RF regression + RMSE/MAE/R2.
    """
    base = spark.read.parquet(data_path)

    # Use all eligible records; OOT remains excluded from model selection.
    base = (
        base.where(
            (F.col("ARRIVAL_DELAY").isNotNull())
            & (F.col("DIVERTED") == 0)
            & (F.col("CANCELLED") == 0)
        )
        .where(
            F.make_date(
                F.col("YEAR").cast("int"),
                F.col("MONTH").cast("int"),
                F.col("DAY").cast("int"),
            ) > F.lit("2015-01-07").cast("date")
        )
        .orderBy(F.rand(42))
    )

    base = base.withColumn(
        "NGAY_DL",
        F.make_date(
            F.col("YEAR").cast("int"),
            F.col("MONTH").cast("int"),
            F.col("DAY").cast("int"),
        ),
    )

    columns_to_drop = [
        "YEAR", "MONTH", "DAY", "DAY_OF_WEEK",
        "DEPARTURE_TIME", "DEPARTURE_DELAY", "TAXI_OUT", "WHEELS_OFF",
        "SCHEDULED_TIME", "ELAPSED_TIME", "DIVERTED", "CANCELLED",
        "AIR_TIME", "WHEELS_ON", "TAXI_IN", "ARRIVAL_TIME",
        "CANCELLATION_REASON", "AIR_SYSTEM_DELAY", "SECURITY_DELAY",
        "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
        "label_severe_delay", "flight_timestamp", "TAIL_NUMBER",
        "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
        "flight_departure_delay_per_km",
        "flight_departure_delay_vs_airline_avg_rt_lag7d",
    ]
    # IMPORTANT: ARRIVAL_DELAY is the regression target, so it is NOT dropped.
    base = base.drop(*[c for c in columns_to_drop if c in base.columns])

    if "time_of_day_bucket" in base.columns:
        base = (
            base.withColumn(
                "time_of_day_bucket_evening",
                (F.col("time_of_day_bucket") == "evening").cast("double"),
            )
            .drop("time_of_day_bucket")
        )

    base = base.withColumn("__row_id", F.monotonically_increasing_id()).persist(StorageLevel.DISK_ONLY)

    features = [
        "DISTANCE", "_flight_timestamp_epoch", "scheduled_hour",
        "airline_distinct_tails_late_lag1d", "airline_late_flight_ratio_lag1d",
        "airline_delay_per_tail_lag1d", "airline_total_delay_minutes_lag3d",
        "airline_delay_per_tail_lag3d", "airline_flights_count_lag7d",
        "airline_late_flight_ratio_lag7d", "airport_scheduled_departures_same_day",
        "airport_scheduled_arrivals_same_day", "airport_avg_delay_lag3h",
        "airport_late_flights_count_lag6h", "airplane_flights_scheduled_same_day",
        "airplane_total_delay_minutes_lag1d", "airplane_total_delay_minutes_lag3d",
        "airplane_delay_per_km_lag7d", "airplane_prev_flight_arrival_delay",
        "flight_computed_scheduled_duration_minutes",
        "flight_scheduled_minutes_per_km", "is_weekend",
        "time_of_day_bucket_evening",
    ]

    missing_features = sorted(set(features) - set(base.columns))
    if missing_features:
        raise ValueError(f"Expected features are absent from input data: {missing_features}")

    # OOT by date; train/valid are a random 80/20 split inside the pre-OOT period.
    train_valid = base.where(
        F.col("NGAY_DL") < F.lit(CONFIGURE["oot_split_threshold"]).cast("date")
    )
    oot = base.where(
        F.col("NGAY_DL") >= F.lit(CONFIGURE["oot_split_threshold"]).cast("date")
    )

    train_df, valid_df = train_valid.randomSplit([0.8, 0.2], seed=42)

    X_train = train_df.select("__row_id", *features)
    y_train = train_df.select("__row_id", label_col)
    X_valid = valid_df.select("__row_id", *features)
    y_valid = valid_df.select("__row_id", label_col)
    X_oot = oot.select("__row_id", *features)
    y_oot = oot.select("__row_id", label_col)

    print(f"[0] Eligible records: {base.count():,}", flush=True)
    print(f"[0] Train: {train_df.count():,} | Valid: {valid_df.count():,} | OOT: {oot.count():,}", flush=True)

    # Variable filtering.
    missing_drop_cols = missing_filter_fct(
        X_train.drop("__row_id"),
        CONFIGURE["missing_threshold"],
    )
    concentration_drop_cols = concentration_filter_fct(
        X_train.drop("__row_id"),
        CONFIGURE["concentration_threshold"],
    )

    selected = [
        c for c in features
        if c not in set(missing_drop_cols + concentration_drop_cols)
    ]
    print(f"[1] Dropped by missingness: {missing_drop_cols}", flush=True)
    print(f"[1] Dropped by concentration: {concentration_drop_cols}", flush=True)

    X_train = X_train.select("__row_id", *selected)

    # NEW STEP: correlation(feature, ARRIVAL_DELAY), replacing IV.
    corr_label_df = (
        X_train.join(y_train, "__row_id")
        .select(*selected, label_col)
    )
    df_feature_corr = feature_label_correlation_fct(
        corr_label_df,
        label_col=label_col,
    )

    print("\n[1] Correlation with ARRIVAL_DELAY:")
    df_feature_corr.show(truncate=False)

    # Multicollinearity: use feature-label correlation as the priority.
    df_corr, corr_drop_cols = multicollinearity_filter_fct(
        X_train.drop("__row_id"),
        df_feature_corr,
        CONFIGURE["corr_threshold"],
    )

    selected = [c for c in selected if c not in corr_drop_cols]

    print(f"[2] Dropped by multicollinearity: {corr_drop_cols}")
    print(f"[2] Remaining features ({len(selected)}): {selected}", flush=True)

    X_train = X_train.select("__row_id", *selected)
    X_valid = X_valid.select("__row_id", *selected)
    X_oot = X_oot.select("__row_id", *selected)

    # Spark RF does not accept null feature values.
    X_train = X_train.fillna(0, subset=selected)
    X_valid = X_valid.fillna(0, subset=selected)
    X_oot = X_oot.fillna(0, subset=selected)

    grid_config = param_grid or CONFIGURE["random_forest_grid"]
    if single_cv_fold is not None:
        run_single_cv_fold(
            X_train,
            y_train,
            CONFIGURE["random_forest"],
            grid_config,
            single_cv_fold,
            label_col=label_col,
            checkpoint_path=model_output_path.parent / "regression_fold_checkpoint.json",
        )
        return None, None, df_feature_corr, df_corr, None

    # Huy-confirmed full grid: 12 combinations, each evaluated by 3-fold CV.
    df_hyperparams_opts = grid_search_cv_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
        CONFIGURE["random_forest"],
        grid_config,
        n_jobs=1,
        label_col=label_col,
        checkpoint_path=model_output_path.parent / "regression_grid_checkpoint.json",
    )

    print("\n[3] RF Regression CV results:")
    df_hyperparams_opts.show(truncate=False)

    if grid_only:
        print("[3] Grid-only shard complete; final fitting is intentionally deferred.", flush=True)
        return None, None, df_feature_corr, df_corr, df_hyperparams_opts

    chosen = df_hyperparams_opts.limit(1).collect()
    final_hyperparams = (
        json.loads(chosen[0]["params"])
        if chosen
        else CONFIGURE["random_forest"]
    )

    print(f"[3] Winning hyperparameters: {final_hyperparams}", flush=True)
    training_df = X_train.join(y_train, "__row_id")
    model = fit_random_forest(
        training_df,
        selected,
        final_hyperparams,
        label_col=label_col,
    )

    model.write().save(str(model_output_path))
    print(f"[4] Saved fitted PipelineModel: {model_output_path}", flush=True)

    df_result_indicators, df_output_model = model_result(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_oot,
        y_oot,
        model,
        label_col=label_col,
    )

    print("\n[4] Final regression metrics:")
    df_result_indicators.show(truncate=False)

    return (
        df_result_indicators,
        df_output_model,
        df_feature_corr,
        df_corr,
        df_hyperparams_opts,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--grid-num-trees", type=int, nargs="+")
    parser.add_argument("--grid-max-depth", type=int, nargs="+")
    parser.add_argument("--grid-min-leaf", type=int, nargs="+")
    parser.add_argument(
        "--grid-only",
        action="store_true",
        help="Run/checkpoint the requested grid shard without fitting a final model.",
    )
    parser.add_argument(
        "--single-cv-fold",
        type=int,
        choices=(1, 2, 3),
        help="Fit and checkpoint only this one-based CV fold; skip full-model fitting.",
    )
    args = parser.parse_args()
    if not args.data_path.is_dir():
        parser.error(f"Training data directory is missing: {args.data_path}")
    if args.model_output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.model_output}")
    if not args.model_output.parent.is_dir():
        parser.error(f"Model output parent is missing: {args.model_output.parent}")

    grid = {key: list(values) for key, values in CONFIGURE["random_forest_grid"].items()}
    requested = {
        "n_estimators": args.grid_num_trees,
        "max_depth": args.grid_max_depth,
        "min_samples_leaf": args.grid_min_leaf,
    }
    for key, values in requested.items():
        if values is None:
            continue
        unsupported = sorted(set(values) - set(grid[key]))
        if unsupported:
            parser.error(f"Unsupported {key} values: {unsupported}; allowed: {grid[key]}")
        grid[key] = values

    try:
        spark.sparkContext.setLogLevel("WARN")
        df_result_indicators, df_output_model, *_ = run_pipeline(
            data_path=str(args.data_path),
            model_output_path=args.model_output,
            param_grid=grid,
            grid_only=args.grid_only,
            single_cv_fold=args.single_cv_fold,
        )
        if df_result_indicators is not None:
            df_result_indicators.show(truncate=False)
            df_output_model.show(20, truncate=False)
    finally:
        spark.stop()
