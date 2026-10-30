import time
import os
import sys
import json
import logging
import gc
import shutil
import resource
import platform
import argparse
import subprocess
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats as scipy_stats

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    roc_auc_score, average_precision_score
)
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier as SkGradientBoostingClassifier
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA as SkPCA
from sklearn.preprocessing import StandardScaler as SkStandardScaler

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, FloatType, IntegerType
from pyspark.storagelevel import StorageLevel

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, PCA, PCAModel, UnivariateFeatureSelector
)
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.linalg import DenseVector, SparseVector

try:
    from py4j.protocol import Py4JJavaError, Py4JError
except ImportError:
    Py4JJavaError = Exception
    Py4JError = Exception

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

SPARK_MASTER = "spark://127.0.0.1:7077"
SPARK_APP_NAME = "NSCLC"
SPARK_EXECUTOR_MEMORY = "20g"
SPARK_DRIVER_MEMORY = "32g"
SPARK_DRIVER_OVERHEAD = "10g"
SPARK_EXECUTOR_OVERHEAD = "8g"
SPARK_SHUFFLE_PARTITIONS = 8
SPARK_MEMORY_FRACTION = 0.4
SPARK_MEMORY_STORAGE_FRACTION = 0.2

# Data Paths
DATA_PATH = ("./datset-path")
RESULTS_DIR = "./results"
CHECKPOINT_DIR = "/tmp/Lung_checkpoints"

SPLIT_CACHE_DIR = os.path.join(RESULTS_DIR, "_shared_split_cache")
FEATURE_CACHE_DIR = os.path.join(RESULTS_DIR, "_feature_cache")

# Pipeline Parameters
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
NUM_SELECTED_GENES = 300
PCA_COMPONENTS = 50
N_FOLDS = 5
RANDOM_SEEDS = [42, 123, 456, 789, 999]
DEFAULT_THRESHOLD = 0.3
FOLD_PARTITIONS = 8


TUNING_TRAIN_RATIO = 0.8

# Hybrid feature selection parameters
ANOVA_FILTER_FACTOR = 2  

# Learning curve parameters
LEARNING_CURVE_TRAIN_SIZES = np.linspace(0.1, 1.0, 10) 
LEARNING_CURVE_CV = 3  
LEARNING_CURVE_N_JOBS = 1

FEATURE_METHODS = ["rf", "anova", "pca", "hybrid"]
FEATURE_METHOD_LABELS = {
    "rf": "Random Forest Importance",
    "anova": "ANOVA F-test",
    "pca": "PCA",
    "hybrid": "ANOVA + RF Hybrid"
}

# Base Models 
BASE_MODELS = ["logistic_regression", "random_forest", "gradient_boosting"]

MODEL_LABELS = {
    'logistic_regression': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting Trees',
    'stacking_ensemble': 'Stacking Ensemble'
}

COLORS = {
    'logistic_regression': '#2E86AB',
    'random_forest': '#A23B72',
    'gradient_boosting': '#F18F01',
    'stacking_ensemble': '#C73E1D'
}


def split_cache_dir_for_seed(seed: int) -> str:
    return os.path.join(SPLIT_CACHE_DIR, f"seed_{seed}")


def feature_cache_dir_for(method: str, seed: int) -> str:
    return os.path.join(FEATURE_CACHE_DIR, method, f"seed_{seed}")


# =============================================================================
# SPARK SESSION MANAGEMENT
# =============================================================================

def create_spark_session(label) -> SparkSession:
    spark = (SparkSession.builder
             .master(SPARK_MASTER)
             .appName(f"{SPARK_APP_NAME}_{label}")
             .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
             .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
             .config("spark.driver.memoryOverhead", SPARK_DRIVER_OVERHEAD)
             .config("spark.executor.memoryOverhead", SPARK_EXECUTOR_OVERHEAD)
             .config("spark.driver.maxResultSize", "8g")
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.sql.shuffle.partitions", str(SPARK_SHUFFLE_PARTITIONS))
             .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
             .config("spark.memory.fraction", str(SPARK_MEMORY_FRACTION))
             .config("spark.memory.storageFraction", str(SPARK_MEMORY_STORAGE_FRACTION))
             .config("spark.rdd.compress", "true")
             .config("spark.sql.autoBroadcastJoinThreshold", "10m")
             .config("spark.sql.broadcastTimeout", "300")
             .config("spark.cleaner.referenceTracking", "true")
             .config("spark.cleaner.referenceTracking.blocking", "false")
             .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
             .config("spark.cleaner.periodicGC.interval", "1min")
             .config("spark.driver.extraJavaOptions",
                     "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35")
             .config("spark.executor.extraJavaOptions",
                     "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35")
             .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def force_jvm_gc(spark: SparkSession):
    try:
        spark.sparkContext._jvm.System.gc()
    except Exception:
        pass
    gc.collect()


def is_spark_alive(spark: Optional[SparkSession]) -> bool:
    if spark is None:
        return False
    try:
        return not spark.sparkContext._jsc.sc().isStopped()
    except Exception:
        return False


def safe_stop_spark(spark: Optional[SparkSession]):
    if spark is None:
        return
    try:
        if is_spark_alive(spark):
            spark.catalog.clearCache()
    except Exception:
        pass
    try:
        spark.stop()
    except Exception:
        pass


# =============================================================================
# RESOURCE TRACKING
# =============================================================================

def get_dir_size_bytes(path: str) -> int:
    if not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def get_peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == "Darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_gene_columns(df) -> List[str]:
    exclude_cols = {
        'SampleID', 'label', 'class_weight', 'fold', 'split',
        'pca_features', 'pca_raw_features',
        'anova_raw_features', 'anova_selected_features'
    }
    return [c for c in df.columns if c not in exclude_cols]


def probability_udf():
    def _get_probability(vector):
        if isinstance(vector, (DenseVector, SparseVector)):
            return float(vector[1]) if len(vector) > 1 else 0.0
        return 0.0
    return F.udf(_get_probability, DoubleType())


def checkpoint_df(spark, df, label: str, partitions: int = FOLD_PARTITIONS):
    df = df.repartition(partitions)
    df = df.persist(StorageLevel.MEMORY_AND_DISK)
    df = df.checkpoint()
    count = df.count()
    logger.info(f"Checkpointed [{label}] - rows: {count:,}, partitions: {partitions}")
    return df


def cleanup_cache(spark):
    if not is_spark_alive(spark):
        return
    try:
        spark.catalog.clearCache()
    except Exception:
        pass
    force_jvm_gc(spark)


def safe_get_param(obj, param_name):
    try:
        attr = getattr(obj, param_name)
        if callable(attr):
            return attr()
        return attr
    except Exception:
        return None


def hash_fraction(sample_id: str, salt: str = "") -> float:
    h = hashlib.md5(f"{sample_id}{salt}".encode("utf-8")).hexdigest()
    return (int(h[:8], 16) % 100) / 100.0


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(spark, path: str):
    logger.info(f"Loading data from {path}")
    df = spark.read.parquet(path)
    if "SampleID" not in df.columns:
        df = df.withColumn("SampleID", F.monotonically_increasing_id().cast(StringType()))
    total_samples = df.count()
    total_genes = len(get_gene_columns(df))
    logger.info(f"Loaded {total_samples:,} samples with {total_genes:,} genes")
    return df, total_genes


# =============================================================================
# DATA SPLITTING
# =============================================================================

def deterministic_stratified_split(df, train_ratio: float = 0.80, val_ratio: float = 0.10,
                                    seed: Optional[int] = None):
    salt = "" if seed is None else f"::seed{seed}"
    logger.info(f"Performing split (train={train_ratio:.0%}, val={val_ratio:.0%}, seed={seed})")

    df = (df
          .withColumn("sample_hash", F.md5(F.concat(F.col("SampleID").cast(StringType()), F.lit(salt))))
          .withColumn("hash_value",
                     (F.conv(F.substring(F.col("sample_hash"), 1, 8), 16, 10) % 100)
                     .cast(FloatType()) / 100))

    window = Window.partitionBy("label").orderBy("hash_value")
    df = (df
          .withColumn("rank", F.row_number().over(window))
          .withColumn("total", F.count("*").over(Window.partitionBy("label")))
          .withColumn("split",
                     F.when(F.col("rank") / F.col("total") <= train_ratio, "train")
                      .when(F.col("rank") / F.col("total") <= train_ratio + val_ratio, "val")
                      .otherwise("test"))
          .drop("rank", "total", "sample_hash", "hash_value"))

    spark = df.sparkSession
    train_df = df.filter(F.col("split") == "train").drop("split")
    val_df = df.filter(F.col("split") == "val").drop("split")
    test_df = df.filter(F.col("split") == "test").drop("split")

    train_df = checkpoint_df(spark, train_df, "train")
    val_df = checkpoint_df(spark, val_df, "val")
    test_df = checkpoint_df(spark, test_df, "test")

    logger.info(f"Split sizes - train: {train_df.count():,}, val: {val_df.count():,}, test: {test_df.count():,}")
    return train_df, val_df, test_df


def compute_class_weights(df):
    logger.info("Computing class weights...")
    class_stats = df.groupBy("label").count().collect()
    total_samples = sum(row["count"] for row in class_stats)
    weights = {}
    for row in class_stats:
        label = float(row["label"])
        count = row["count"]
        weights[label] = total_samples / (2.0 * count)
        logger.info(f"  Class {int(label)}: count={count:,}, weight={weights[label]:.4f}")
    return weights


def add_weight_column(df, weights: Dict):
    @F.udf(FloatType())
    def weight_udf(label):
        return weights.get(float(label), 1.0)
    return df.withColumn("class_weight", weight_udf(F.col("label")))


def precompute_split_for_seed(data_path: str, seed: int) -> Dict[str, Any]:
    cache_dir = split_cache_dir_for_seed(seed)
    meta_path = os.path.join(cache_dir, "metadata.json")
    train_path = os.path.join(cache_dir, "train.parquet")
    val_path = os.path.join(cache_dir, "val.parquet")
    test_path = os.path.join(cache_dir, "test.parquet")

    if all(os.path.exists(p) for p in [meta_path, train_path, val_path, test_path]):
        with open(meta_path) as f:
            meta = json.load(f)
        logger.info(f"✓ Reusing cached split for seed={seed}")
        return meta

    logger.info(f"No cached split found for seed={seed} - computing...")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    spark = create_spark_session(f"prep_split_seed{seed}")
    try:
        spark.sparkContext.setCheckpointDir(os.path.join(CHECKPOINT_DIR, f"prep_split_seed{seed}"))
        df, total_genes = load_data(spark, data_path)
        train_df, val_df, test_df = deterministic_stratified_split(df, TRAIN_RATIO, VAL_RATIO, seed=seed)

        weights = compute_class_weights(train_df)
        train_df = add_weight_column(train_df, weights)

        train_df.write.mode("overwrite").parquet(train_path)
        val_df.write.mode("overwrite").parquet(val_path)
        test_df.write.mode("overwrite").parquet(test_path)

        meta = {
            "seed": seed,
            "total_genes": total_genes,
            "train_rows": train_df.count(),
            "val_rows": val_df.count(),
            "test_rows": test_df.count(),
            "class_weights": weights,
            "source_data_path": data_path,
            "source_dataset_size_bytes": get_dir_size_bytes(data_path),
            "computed_at": datetime.now().isoformat()
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info(f"✓ Cached split for seed={seed}")
        return meta
    finally:
        safe_stop_spark(spark)
        gc.collect()
        time.sleep(2)


# =============================================================================
# FEATURE SELECTION - WITH HYBRID METHOD
# =============================================================================

def select_top_features_rf(df, gene_cols: List[str], num_select: int, seed: int):
    logger.info(f"[rf] Selecting top {num_select} features...")
    assembler = VectorAssembler(inputCols=gene_cols, outputCol="features", handleInvalid="skip")
    rf = RandomForestClassifier(
        featuresCol="features", labelCol="label", weightCol="class_weight",
        numTrees=50, seed=seed
    )
    pipeline = Pipeline(stages=[assembler, rf])
    model = pipeline.fit(df)
    importance = model.stages[-1].featureImportances.toArray()
    sorted_indices = np.argsort(importance)[::-1]
    selected_genes = [gene_cols[i] for i in sorted_indices[:num_select]]
    logger.info(f"[rf] Selected {len(selected_genes)} genes")
    return selected_genes


def select_top_features_anova(df, gene_cols: List[str], num_select: int):
    logger.info(f"[anova] Selecting top {num_select} features...")
    assembler = VectorAssembler(inputCols=gene_cols, outputCol="anova_raw_features", handleInvalid="skip")
    assembled = assembler.transform(df)

    selector = UnivariateFeatureSelector(
        featuresCol="anova_raw_features",
        outputCol="anova_selected_features",
        labelCol="label",
    )
    selector.setFeatureType("continuous")
    selector.setLabelType("categorical")
    selector.setSelectionMode("numTopFeatures")
    selector.setSelectionThreshold(num_select)

    model = selector.fit(assembled)
    selected_indices = list(model.selectedFeatures)
    selected_genes = [gene_cols[i] for i in selected_indices]
    logger.info(f"[anova] Selected {len(selected_genes)} genes")
    return selected_genes


def select_top_features_hybrid(df, gene_cols: List[str], num_select: int, seed: int):
    """
    Hybrid feature selection: First use ANOVA for filtering, then RF for ranking.
    """
    logger.info(f"[hybrid] Starting hybrid feature selection...")
    logger.info(f"[hybrid] Total genes: {len(gene_cols):,}")

    anova_keep_count = int(num_select * ANOVA_FILTER_FACTOR)
    logger.info(f"[hybrid] Step 1: ANOVA filtering to top {anova_keep_count} genes...")

    assembler = VectorAssembler(inputCols=gene_cols, outputCol="anova_raw_features", handleInvalid="skip")
    assembled = assembler.transform(df)

    selector = UnivariateFeatureSelector(
        featuresCol="anova_raw_features",
        outputCol="anova_selected_features",
        labelCol="label",
    )
    selector.setFeatureType("continuous")
    selector.setLabelType("categorical")
    selector.setSelectionMode("numTopFeatures")
    selector.setSelectionThreshold(anova_keep_count)

    selector_model = selector.fit(assembled)
    anova_selected_indices = list(selector_model.selectedFeatures)
    anova_selected_genes = [gene_cols[i] for i in anova_selected_indices]

    logger.info(f"[hybrid] ANOVA filtered to {len(anova_selected_genes)} genes")
    logger.info(f"[hybrid] Step 2: RF importance ranking on filtered genes...")

    selected_cols = ["SampleID", "label", "class_weight"] + anova_selected_genes
    df_filtered = df.select(*selected_cols)

    assembler_rf = VectorAssembler(inputCols=anova_selected_genes, outputCol="features", handleInvalid="skip")
    rf = RandomForestClassifier(
        featuresCol="features", labelCol="label", weightCol="class_weight",
        numTrees=50, seed=seed
    )
    pipeline = Pipeline(stages=[assembler_rf, rf])
    model = pipeline.fit(df_filtered)

    importance = model.stages[-1].featureImportances.toArray()
    sorted_indices = np.argsort(importance)[::-1]

    selected_genes = [anova_selected_genes[i] for i in sorted_indices[:num_select]]

    logger.info(f"[hybrid] Final selection: {len(selected_genes)} genes")
    logger.info(f"[hybrid] Top 10 genes: {selected_genes[:10]}")

    return selected_genes


def fit_pca_model(df, gene_cols: List[str], k: int, max_features_for_pca: int = 2000):
    logger.info(f"[pca] Starting with {len(gene_cols):,} genes")
    logger.info(f"[pca] Variance filtering to top {max_features_for_pca} genes...")

    var_exprs = [F.variance(F.col(c)).alias(c) for c in gene_cols]
    var_df = df.select(var_exprs).collect()[0]

    gene_var = {}
    for c in gene_cols:
        val = var_df[c]
        if val is not None:
            gene_var[c] = val

    sorted_genes = sorted(gene_var.items(), key=lambda x: x[1], reverse=True)
    top_genes = [g for g, _ in sorted_genes[:max_features_for_pca]]

    logger.info(f"[pca] Kept {len(top_genes):,} genes for PCA")

    assembler = VectorAssembler(inputCols=top_genes, outputCol="pca_raw_features", handleInvalid="skip")
    assembled = assembler.transform(df)

    pca = PCA(k=k, inputCol="pca_raw_features", outputCol="pca_features")
    pca_model = pca.fit(assembled)

    explained_variance = pca_model.explainedVariance.toArray()
    logger.info(f"[pca] Fitted. Explained variance (top 5): {[round(v, 4) for v in explained_variance[:5]]}")

    return pca_model


def precompute_feature_selection(method: str, seed: int) -> Dict[str, Any]:
    if method == "rf":
        return {"method": "rf"}

    method_cache_dir = feature_cache_dir_for(method, seed)
    split_cache_dir = split_cache_dir_for_seed(seed)

    if method == "anova":
        genes_path = os.path.join(method_cache_dir, "selected_genes.json")
        if os.path.exists(genes_path):
            with open(genes_path) as f:
                selected_genes = json.load(f)
            logger.info(f"✓ Reusing cached ANOVA selection for seed={seed}")
            return {"method": "anova", "selected_genes": selected_genes}

        os.makedirs(method_cache_dir, exist_ok=True)
        spark = create_spark_session(f"prep_anova_seed{seed}")
        try:
            spark.sparkContext.setCheckpointDir(os.path.join(CHECKPOINT_DIR, f"prep_anova_seed{seed}"))
            train_df = spark.read.parquet(os.path.join(split_cache_dir, "train.parquet"))
            gene_cols = get_gene_columns(train_df)
            selected_genes = select_top_features_anova(train_df, gene_cols, NUM_SELECTED_GENES)
            with open(genes_path, "w") as f:
                json.dump(selected_genes, f, indent=2)
            return {"method": "anova", "selected_genes": selected_genes}
        finally:
            safe_stop_spark(spark)
            gc.collect()
            time.sleep(2)

    elif method == "hybrid":
        genes_path = os.path.join(method_cache_dir, "selected_genes.json")
        anova_filtered_path = os.path.join(method_cache_dir, "anova_filtered_genes.json")

        if os.path.exists(genes_path):
            with open(genes_path) as f:
                selected_genes = json.load(f)
            logger.info(f"✓ Reusing cached hybrid selection for seed={seed}")
            return {"method": "hybrid", "selected_genes": selected_genes}

        os.makedirs(method_cache_dir, exist_ok=True)
        spark = create_spark_session(f"prep_hybrid_seed{seed}")
        try:
            spark.sparkContext.setCheckpointDir(os.path.join(CHECKPOINT_DIR, f"prep_hybrid_seed{seed}"))
            train_df = spark.read.parquet(os.path.join(split_cache_dir, "train.parquet"))
            gene_cols = get_gene_columns(train_df)

            anova_keep_count = int(NUM_SELECTED_GENES * ANOVA_FILTER_FACTOR)
            anova_filtered_genes = select_top_features_anova(train_df, gene_cols, anova_keep_count)

            with open(anova_filtered_path, "w") as f:
                json.dump(anova_filtered_genes, f, indent=2)

            selected_genes = select_top_features_hybrid(train_df, gene_cols, NUM_SELECTED_GENES, seed)

            with open(genes_path, "w") as f:
                json.dump(selected_genes, f, indent=2)

            return {
                "method": "hybrid",
                "selected_genes": selected_genes,
                "anova_filtered_genes": anova_filtered_genes
            }
        finally:
            safe_stop_spark(spark)
            gc.collect()
            time.sleep(2)

    elif method == "pca":
        model_path = os.path.join(method_cache_dir, "pca_model")
        gene_order_path = os.path.join(method_cache_dir, "gene_order.json")
        variance_filtered_path = os.path.join(method_cache_dir, "variance_filtered_genes.json")

        if os.path.exists(model_path) and os.path.exists(gene_order_path):
            with open(gene_order_path) as f:
                gene_order = json.load(f)
            filtered_genes = None
            if os.path.exists(variance_filtered_path):
                with open(variance_filtered_path) as f:
                    filtered_genes = json.load(f)
            logger.info(f"✓ Reusing cached PCA model for seed={seed}")
            return {
                "method": "pca",
                "gene_order": gene_order,
                "pca_model_path": model_path,
                "variance_filtered_genes": filtered_genes
            }

        os.makedirs(method_cache_dir, exist_ok=True)
        spark = create_spark_session(f"prep_pca_seed{seed}")
        try:
            spark.sparkContext.setCheckpointDir(os.path.join(CHECKPOINT_DIR, f"prep_pca_seed{seed}"))
            train_df = spark.read.parquet(os.path.join(split_cache_dir, "train.parquet"))
            gene_cols = get_gene_columns(train_df)

            max_features_for_pca = 2000
            pca_model = fit_pca_model(train_df, gene_cols, PCA_COMPONENTS, max_features_for_pca)

            pca_model.write().overwrite().save(model_path)

            with open(gene_order_path, "w") as f:
                json.dump(gene_cols, f, indent=2)

            var_exprs = [F.variance(F.col(c)).alias(c) for c in gene_cols]
            var_df = train_df.select(var_exprs).collect()[0]
            gene_var = {c: var_df[c] for c in gene_cols if var_df[c] is not None}
            sorted_genes = sorted(gene_var.items(), key=lambda x: x[1], reverse=True)
            filtered_genes = [g for g, _ in sorted_genes[:max_features_for_pca]]

            with open(variance_filtered_path, "w") as f:
                json.dump(filtered_genes, f, indent=2)

            logger.info(f"✓ PCA model cached to {model_path}")

            return {
                "method": "pca",
                "gene_order": gene_cols,
                "pca_model_path": model_path,
                "variance_filtered_genes": filtered_genes
            }
        finally:
            safe_stop_spark(spark)
            gc.collect()
            time.sleep(2)
    else:
        raise ValueError(f"Unknown feature method: {method}")


# =============================================================================
# BASE MODEL BUILDERS 
# =============================================================================

def build_model_pipeline(model_type: str, genes: List[str], params: Dict = None,
                          precomputed_col: Optional[str] = None):
    if params is None:
        params = {}

    if precomputed_col is not None:
        if model_type == "logistic_regression":
            scaler = StandardScaler(inputCol=precomputed_col, outputCol="features", withMean=True, withStd=True)
            classifier = LogisticRegression(
                featuresCol="features", labelCol="label", weightCol="class_weight",
                family="binomial", maxIter=100,
                regParam=float(params.get("regParam", 0.01)),
                elasticNetParam=float(params.get("elasticNetParam", 0.0))
            )
            return Pipeline(stages=[scaler, classifier]), classifier

        elif model_type == "random_forest":
            classifier = RandomForestClassifier(
                featuresCol=precomputed_col, labelCol="label", weightCol="class_weight",
                numTrees=int(params.get("numTrees", 50)),
                maxDepth=int(params.get("maxDepth", 8)),
                minInstancesPerNode=5,
                seed=params.get("seed", 42)
            )
            return Pipeline(stages=[classifier]), classifier

        elif model_type == "gradient_boosting":
            classifier = GBTClassifier(
                featuresCol=precomputed_col, labelCol="label", weightCol="class_weight",
                maxIter=int(params.get("maxIter", 30)),
                maxDepth=int(params.get("maxDepth", 3)),
                stepSize=float(params.get("stepSize", 0.1)),
                seed=params.get("seed", 42)
            )
            return Pipeline(stages=[classifier]), classifier

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    assembler = VectorAssembler(
        inputCols=genes,
        outputCol="raw_features" if model_type == "logistic_regression" else "features",
        handleInvalid="skip"
    )

    if model_type == "logistic_regression":
        scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
        classifier = LogisticRegression(
            featuresCol="features", labelCol="label", weightCol="class_weight",
            family="binomial", maxIter=100,
            regParam=float(params.get("regParam", 0.01)),
            elasticNetParam=float(params.get("elasticNetParam", 0.0))
        )
        return Pipeline(stages=[assembler, scaler, classifier]), classifier

    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            featuresCol="features", labelCol="label", weightCol="class_weight",
            numTrees=int(params.get("numTrees", 50)),
            maxDepth=int(params.get("maxDepth", 8)),
            minInstancesPerNode=5,
            seed=params.get("seed", 42)
        )
        return Pipeline(stages=[assembler, classifier]), classifier

    elif model_type == "gradient_boosting":
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
        classifier = GBTClassifier(
            featuresCol="scaled_features", labelCol="label", weightCol="class_weight",
            maxIter=int(params.get("maxIter", 30)),
            maxDepth=int(params.get("maxDepth", 3)),
            stepSize=float(params.get("stepSize", 0.1)),
            seed=params.get("seed", 42)
        )
        return Pipeline(stages=[assembler, scaler, classifier]), classifier

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_param_grid(model_type: str, classifier):
    if model_type == "logistic_regression":
        return (ParamGridBuilder()
                .addGrid(classifier.regParam, [0.0001, 0.001, 0.01, 0.1])
                .addGrid(classifier.elasticNetParam, [0.0, 0.5, 1.0])
                .build())
    elif model_type == "random_forest":
        return (ParamGridBuilder()
                .addGrid(classifier.numTrees, [30, 50])
                .addGrid(classifier.maxDepth, [5, 8])
                .build())
    elif model_type == "gradient_boosting":
        return (ParamGridBuilder()
                .addGrid(classifier.maxIter, [20, 30, 40])
                .addGrid(classifier.maxDepth, [3, 5, 7])
                .addGrid(classifier.stepSize, [0.01, 0.1])
                .build())
    return []


# =============================================================================
# SKLEARN PARAMETER GRID (FOR COMPARISON)
# =============================================================================

def sk_param_grid(model_type: str) -> List[Dict[str, float]]:
    """Mirrors get_param_grid() above exactly, so the sklearn engine's
    grid search covers the identical hyperparameter space as Spark's
    TrainValidationSplit."""
    if model_type == "logistic_regression":
        grid = []
        for reg_param in [0.0001, 0.001, 0.01, 0.1]:
            for elastic_net in [0.0, 0.5, 1.0]:
                grid.append({"regParam": reg_param, "elasticNetParam": elastic_net})
        return grid
    elif model_type == "random_forest":
        grid = []
        for num_trees in [30, 50]:
            for max_depth in [5, 8]:
                grid.append({"numTrees": num_trees, "maxDepth": max_depth})
        return grid
    elif model_type == "gradient_boosting":
        grid = []
        for max_iter in [20, 30, 40]:
            for max_depth in [3, 5, 7]:
                for step_size in [0.01, 0.1]:
                    grid.append({"maxIter": max_iter, "maxDepth": max_depth, "stepSize": step_size})
        return grid
    return [{}]


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_metrics(y_true, y_prob, y_pred, threshold: float = 0.5) -> Dict:
    tp = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 0) and (yp == 0) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1) and (yp == 0) for yt, yp in zip(y_true, y_pred))

    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-10)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)

    return {
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
        "PR-AUC": float(average_precision_score(y_true, y_prob)),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "F1-score": float(f1),
        "Accuracy": float(accuracy),
        "Balanced Accuracy": float((sensitivity + specificity) / 2),
        "Threshold": float(threshold),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    }


# =============================================================================
# MAIN PIPELINE 
# =============================================================================

class GenomicStackingPipeline:
    def __init__(self, spark, feature_method: str, num_genes: int, seed: int = 42):
        self.spark = spark
        self.feature_method = feature_method
        self.num_genes = num_genes
        self.seed = seed

        self.train_df = None
        self.val_df = None
        self.test_df = None

        self.selected_genes = None
        self.precomputed_col = None
        self.class_weights = None
        self.base_models = {}
        self.best_params = {}
        self.meta_model = None
        self.optimal_threshold = DEFAULT_THRESHOLD

        self.metrics = {}
        self.execution_times = {}
        self.predictions = {}
        self.cached_dfs = []
        self.resource_metrics = {}

        self.learning_curves = {}

        self.seed_checkpoint_dir = os.path.join(CHECKPOINT_DIR, feature_method, f"seed_{seed}")

        logger.info(f"Initialized Spark pipeline (method={feature_method}, seed={seed})")

    def _persist_df(self, df, label: str):
        df = checkpoint_df(self.spark, df, label)
        self.cached_dfs.append(df)
        return df

    def _cleanup_cached_dfs(self):
        for df in self.cached_dfs:
            try:
                df.unpersist()
            except Exception:
                pass
        self.cached_dfs = []
        cleanup_cache(self.spark)

    def _ensure_checkpoint_dir(self):
        os.makedirs(self.seed_checkpoint_dir, exist_ok=True)
        self.spark.sparkContext.setCheckpointDir(self.seed_checkpoint_dir)

    def load_cached_split(self, split_meta: Dict):
        start = time.time()
        self._ensure_checkpoint_dir()

        self.total_genes = split_meta["total_genes"]
        self.class_weights = {float(k): v for k, v in split_meta["class_weights"].items()}
        seed_split_dir = split_cache_dir_for_seed(self.seed)

        train_df = self.spark.read.parquet(os.path.join(seed_split_dir, "train.parquet"))
        val_df = self.spark.read.parquet(os.path.join(seed_split_dir, "val.parquet"))
        test_df = self.spark.read.parquet(os.path.join(seed_split_dir, "test.parquet"))

        self.train_df = self._persist_df(train_df, "train_cached")
        self.val_df = self._persist_df(val_df, "val_cached")
        self.test_df = self._persist_df(test_df, "test_cached")

        self.execution_times["load_cached_split"] = time.time() - start
        return self

    def select_features(self, feature_meta: Dict):
        start = time.time()
        method = self.feature_method
        logger.info(f"Running feature selection: method={method}")

        if method == "rf":
            gene_cols = get_gene_columns(self.train_df)
            self.selected_genes = select_top_features_rf(
                self.train_df, gene_cols, self.num_genes, self.seed
            )
            self.precomputed_col = None
            self._trim_to_selected_genes()

        elif method == "anova":
            self.selected_genes = feature_meta["selected_genes"]
            self.precomputed_col = None
            self._trim_to_selected_genes()

        elif method == "hybrid":
            self.selected_genes = feature_meta["selected_genes"]
            self.precomputed_col = None
            self._trim_to_selected_genes()
            if "anova_filtered_genes" in feature_meta:
                logger.info(f"[hybrid] ANOVA filtered {len(feature_meta['anova_filtered_genes'])} genes "
                           f"before RF selection of {len(self.selected_genes)} genes")

        elif method == "pca":
            pca_model = PCAModel.load(feature_meta["pca_model_path"])

            if "variance_filtered_genes" in feature_meta and feature_meta["variance_filtered_genes"]:
                filtered_genes = feature_meta["variance_filtered_genes"]
                logger.info(f"[pca] Using variance-filtered genes: {len(filtered_genes)} features")
            else:
                filtered_genes = feature_meta["gene_order"]
                logger.warning(f"[pca] No variance-filtered genes found, using all {len(filtered_genes)} genes")

            assembler = VectorAssembler(
                inputCols=filtered_genes,
                outputCol="pca_raw_features",
                handleInvalid="skip"
            )

            def add_pca(df, has_weight: bool):
                out = pca_model.transform(assembler.transform(df))
                cols = ["SampleID", "label", "pca_features"]
                if has_weight:
                    cols.insert(2, "class_weight")
                return out.select(*cols)

            self.train_df = add_pca(self.train_df, has_weight=True)
            self.val_df = add_pca(self.val_df, has_weight=False)
            self.test_df = add_pca(self.test_df, has_weight=False)

            self.train_df = self._persist_df(self.train_df, "train_pca")
            self.val_df = self._persist_df(self.val_df, "val_pca")
            self.test_df = self._persist_df(self.test_df, "test_pca")

            self.selected_genes = [f"PC{i+1}" for i in range(PCA_COMPONENTS)]
            self.precomputed_col = "pca_features"

        else:
            raise ValueError(f"Unknown feature method: {method}")

        self.execution_times["feature_selection"] = time.time() - start
        return self

    def _trim_to_selected_genes(self):
        train_cols = ["SampleID", "label", "class_weight"] + self.selected_genes
        other_cols = ["SampleID", "label"] + self.selected_genes
        self.train_df = self.train_df.select(*train_cols)
        self.val_df = self.val_df.select(*other_cols)
        self.test_df = self.test_df.select(*other_cols)
        self.train_df = self._persist_df(self.train_df, "train_trimmed")
        self.val_df = self._persist_df(self.val_df, "val_trimmed")
        self.test_df = self._persist_df(self.test_df, "test_trimmed")

    def tune_and_train_base_models(self):
        start = time.time()
        logger.info("=" * 60)
        logger.info("TUNING AND TRAINING BASE MODELS (LR, RF, GBT)")
        logger.info("=" * 60)

        self._ensure_checkpoint_dir()
        evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

        for model_type in BASE_MODELS:
            logger.info(f"\n▶ {model_type.upper()}")

            pipeline, classifier = build_model_pipeline(
                model_type, self.selected_genes, {"seed": self.seed},
                precomputed_col=self.precomputed_col
            )
            param_grid = get_param_grid(model_type, classifier)

            tvs = TrainValidationSplit(
                estimator=pipeline, estimatorParamMaps=param_grid, evaluator=evaluator,
                trainRatio=TUNING_TRAIN_RATIO, seed=self.seed, parallelism=1
            )

            logger.info("  Running TrainValidationSplit tuning...")
            tvs_model = tvs.fit(self.train_df)

            best_model = tvs_model.bestModel
            best_classifier = best_model.stages[-1]

            if model_type == "logistic_regression":
                self.best_params[model_type] = {
                    "regParam": float(best_classifier.getRegParam()),
                    "elasticNetParam": float(best_classifier.getElasticNetParam())
                }
            elif model_type == "random_forest":
                num_trees = safe_get_param(best_classifier, "getNumTrees")
                max_depth = safe_get_param(best_classifier, "getMaxDepth")
                self.best_params[model_type] = {
                    "numTrees": int(num_trees) if num_trees is not None else 50,
                    "maxDepth": int(max_depth) if max_depth is not None else 8
                }
            elif model_type == "gradient_boosting":
                self.best_params[model_type] = {
                    "maxIter": int(best_classifier.getMaxIter()),
                    "maxDepth": int(best_classifier.getMaxDepth()),
                    "stepSize": float(best_classifier.getStepSize())
                }

            self.base_models[model_type] = best_model
            logger.info(f"  ✓ Best parameters: {self.best_params[model_type]}")

            del tvs_model, tvs, pipeline
            self._cleanup_cached_dfs()
            force_jvm_gc(self.spark)

        self.execution_times["base_models"] = time.time() - start
        logger.info(f"\n✓ Base models done in {self.execution_times['base_models']:.1f}s")
        return self

    def generate_oof_predictions(self, n_folds: int = 5):
        start = time.time()
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING OOF PREDICTIONS")
        logger.info("=" * 60)
        logger.info(f"Using {n_folds}-fold cross-validation")

        self._ensure_checkpoint_dir()
        self._cleanup_cached_dfs()

        df_with_folds = self._assign_folds(self.train_df, n_folds)
        oof_parts = []
        prob_udf = probability_udf()

        model_alias = {
            "logistic_regression": "pred_lr",
            "random_forest": "pred_rf",
            "gradient_boosting": "pred_gbt"
        }

        for fold_idx in range(n_folds):
            logger.info(f"\n▶ Fold {fold_idx + 1}/{n_folds}")

            train_fold = df_with_folds.filter(F.col("fold") != fold_idx).drop("fold")
            val_fold = df_with_folds.filter(F.col("fold") == fold_idx).drop("fold")

            train_fold = self._persist_df(train_fold, f"fold_{fold_idx}_train")
            val_fold = self._persist_df(val_fold, f"fold_{fold_idx}_val")

            fold_result = val_fold.select("SampleID", "label", "class_weight")

            for model_type in BASE_MODELS:
                pipeline, _ = build_model_pipeline(
                    model_type, self.selected_genes, self.best_params.get(model_type, {}),
                    precomputed_col=self.precomputed_col
                )
                model = pipeline.fit(train_fold)

                predictions = model.transform(val_fold)
                prob_col = predictions.select(
                    "SampleID", prob_udf(F.col("probability")).alias(model_alias[model_type])
                )
                fold_result = fold_result.join(prob_col, on="SampleID", how="left")
                del model, pipeline

            fold_result = self._persist_df(fold_result, f"oof_fold_{fold_idx}")
            oof_parts.append(fold_result)

            train_fold.unpersist()
            val_fold.unpersist()

            self._cleanup_cached_dfs()
            force_jvm_gc(self.spark)

        logger.info("\n▶ Combining OOF predictions...")
        oof_df = oof_parts[0]
        for part in oof_parts[1:]:
            oof_df = oof_df.unionByName(part)

        oof_df = self._persist_df(oof_df, "oof_all")
        for part in oof_parts:
            if part != oof_df:
                part.unpersist()

        self.execution_times["oof_generation"] = time.time() - start
        logger.info(f"\n✓ OOF generation complete in {self.execution_times['oof_generation']:.1f}s")
        return oof_df

    def _assign_folds(self, df, n_folds: int):
        df = (df
              .withColumn("sample_hash", F.md5(F.col("SampleID").cast(StringType())))
              .withColumn("hash_value",
                         (F.conv(F.substring(F.col("sample_hash"), 1, 8), 16, 10) % 100)
                         .cast(FloatType()) / 100))

        window = Window.partitionBy("label").orderBy("hash_value")
        df = (df
              .withColumn("rank", F.row_number().over(window))
              .withColumn("total", F.count("*").over(Window.partitionBy("label")))
              .withColumn("fold",
                         ((F.col("rank") - 1) * n_folds / F.col("total")).cast(IntegerType()))
              .drop("rank", "total", "sample_hash", "hash_value"))
        return self._persist_df(df, "folded_data")

    def train_meta_learner(self, oof_df):
        start = time.time()
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING META LEARNER")
        logger.info("=" * 60)

        self._ensure_checkpoint_dir()

        meta_features = ["pred_lr", "pred_rf", "pred_gbt"]
        logger.info(f"Meta-features: {meta_features}")

        assembler = VectorAssembler(inputCols=meta_features, outputCol="meta_features", handleInvalid="skip")
        scaler = StandardScaler(inputCol="meta_features", outputCol="meta_scaled", withMean=True, withStd=True)
        meta_lr = LogisticRegression(
            featuresCol="meta_scaled", labelCol="label", weightCol="class_weight",
            family="binomial", maxIter=100
        )
        meta_pipeline = Pipeline(stages=[assembler, scaler, meta_lr])

        logger.info("  Tuning meta-learner...")
        param_grid = (ParamGridBuilder()
                      .addGrid(meta_lr.regParam, [0.001, 0.01, 0.1])
                      .addGrid(meta_lr.elasticNetParam, [0.0, 0.5])
                      .build())

        evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")

        tvs = TrainValidationSplit(
            estimator=meta_pipeline, estimatorParamMaps=param_grid, evaluator=evaluator,
            trainRatio=TUNING_TRAIN_RATIO, seed=self.seed, parallelism=1
        )

        tvs_model = tvs.fit(oof_df)
        self.meta_model = tvs_model.bestModel

        best_meta_lr = self.meta_model.stages[-1]
        coefficients = best_meta_lr.coefficients.toArray()

        logger.info("\n▶ Meta-learner coefficients:")
        for name, coef in zip(meta_features, coefficients):
            logger.info(f"    {name:>12}: {coef:>8.4f}")

        logger.info(f"\n  Best hyperparameters:")
        logger.info(f"    regParam: {best_meta_lr.getRegParam():.4f}")
        logger.info(f"    elasticNetParam: {best_meta_lr.getElasticNetParam():.4f}")

        self.execution_times["meta_training"] = time.time() - start

        del tvs_model, tvs, meta_pipeline
        oof_df.unpersist()
        self._cleanup_cached_dfs()
        force_jvm_gc(self.spark)

        logger.info(f"\n✓ Meta-learner done in {self.execution_times['meta_training']:.1f}s")
        return self.meta_model

    def optimize_threshold(self, n_thresholds: int = 21):
        start = time.time()
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZING THRESHOLD")
        logger.info("=" * 60)

        stacked_preds = self.predict_stacked(self.val_df)
        stacked_preds = self._persist_df(stacked_preds, "stacked_val_preds")

        prob_df = stacked_preds.select(
            F.col("label").cast(DoubleType()), F.col("prob_stack").alias("prob")
        ).persist(StorageLevel.MEMORY_AND_DISK)

        try:
            stats = prob_df.agg(F.min("prob"), F.max("prob")).collect()[0]
            min_prob = float(stats[0] or 0.01)
            max_prob = float(stats[1] or 0.99)

            thresholds = np.linspace(max(min_prob, 0.01), min(max_prob, 0.99), n_thresholds).tolist()
            logger.info(f"  Threshold range: {thresholds[0]:.4f} to {thresholds[-1]:.4f}")

            threshold_expr = F.explode(F.array([F.lit(float(t)) for t in thresholds]))

            threshold_stats = (prob_df
                .withColumn("t", threshold_expr)
                .withColumn("pred", F.when(F.col("prob") >= F.col("t"), 1).otherwise(0))
                .withColumn("tp", F.when((F.col("label") == 1) & (F.col("pred") == 1), 1).otherwise(0))
                .withColumn("tn", F.when((F.col("label") == 0) & (F.col("pred") == 0), 1).otherwise(0))
                .withColumn("fp", F.when((F.col("label") == 0) & (F.col("pred") == 1), 1).otherwise(0))
                .withColumn("fn", F.when((F.col("label") == 1) & (F.col("pred") == 0), 1).otherwise(0))
                .groupBy("t")
                .agg(F.sum("tp").alias("tp"), F.sum("tn").alias("tn"),
                     F.sum("fp").alias("fp"), F.sum("fn").alias("fn"))
                .withColumn("sens", F.col("tp") / (F.col("tp") + F.col("fn") + 1e-10))
                .withColumn("spec", F.col("tn") / (F.col("tn") + F.col("fp") + 1e-10))
                .withColumn("bacc", (F.col("sens") + F.col("spec")) / 2)
                .orderBy(F.desc("bacc"))
                .first()
            )

            if threshold_stats:
                self.optimal_threshold = float(threshold_stats["t"])
                logger.info(f"\n▶ Optimal threshold: {self.optimal_threshold:.4f}")
                logger.info(f"  Best balanced accuracy: {float(threshold_stats['bacc']):.4f}")
            else:
                self.optimal_threshold = 0.5
                logger.warning("No valid threshold found, using default 0.5")
        finally:
            prob_df.unpersist()
            stacked_preds.unpersist()

        self.execution_times["threshold_optimization"] = time.time() - start
        return self.optimal_threshold

    def _make_meta_features(self, df):
        prob_udf = probability_udf()
        model_alias = {
            "logistic_regression": "pred_lr",
            "random_forest": "pred_rf",
            "gradient_boosting": "pred_gbt"
        }
        result = df.select("SampleID", "label")
        for model_type, base_model in self.base_models.items():
            predictions = base_model.transform(df)
            prob_col = predictions.select(
                "SampleID", prob_udf(F.col("probability")).alias(model_alias[model_type])
            )
            result = result.join(prob_col, on="SampleID", how="left")
        return result

    def predict_stacked(self, df):
        meta_df = self._make_meta_features(df)
        meta_predictions = self.meta_model.transform(meta_df)
        prob_udf = probability_udf()
        return meta_predictions.withColumn("prob_stack", prob_udf(F.col("probability")))

    def evaluate(self):
        logger.info("\n" + "=" * 60)
        logger.info("FINAL EVALUATION ON TEST SET (LR, RF, GBT, Stacking)")
        logger.info("=" * 60)

        threshold = self.optimal_threshold
        prob_udf = probability_udf()

        logger.info("\n▶ Evaluating Base Models...")
        for model_type in BASE_MODELS:
            self._evaluate_base_model(model_type, threshold, prob_udf)

        logger.info("\n▶ Evaluating Stacking Ensemble...")
        self._evaluate_ensemble(threshold, prob_udf)
        return self.metrics

    def _evaluate_base_model(self, model_type: str, threshold: float, prob_udf):
        model = self.base_models[model_type]
        predictions = model.transform(self.test_df)

        preds_with_probs = (predictions
            .withColumn("prob_pos", prob_udf(F.col("probability")))
            .withColumn("pred_opt", F.when(F.col("prob_pos") >= threshold, 1.0).otherwise(0.0))
        ).select("label", "prob_pos", "pred_opt").persist(StorageLevel.MEMORY_AND_DISK)

        rows = preds_with_probs.collect()
        y_true = [row["label"] for row in rows]
        y_prob = [row["prob_pos"] for row in rows]
        y_pred = [row["pred_opt"] for row in rows]

        metrics = calculate_metrics(y_true, y_prob, y_pred, threshold)
        self.metrics[model_type] = metrics
        preds_with_probs = preds_with_probs.withColumnRenamed("pred_opt", "pred")
        self.predictions[model_type] = preds_with_probs

        logger.info(f"  {MODEL_LABELS[model_type]}: ROC-AUC={metrics['ROC-AUC']:.4f}")

    def _evaluate_ensemble(self, threshold: float, prob_udf):
        meta_df = self._make_meta_features(self.test_df)
        meta_predictions = self.meta_model.transform(meta_df)

        ensemble_preds = (meta_predictions
            .withColumn("prob_pos", prob_udf(F.col("probability")))
            .withColumn("pred", F.when(F.col("prob_pos") >= threshold, 1.0).otherwise(0.0))
        ).select("label", "prob_pos", "pred").persist(StorageLevel.MEMORY_AND_DISK)

        rows = ensemble_preds.collect()
        y_true = [row["label"] for row in rows]
        y_prob = [row["prob_pos"] for row in rows]
        y_pred = [row["pred"] for row in rows]

        metrics = calculate_metrics(y_true, y_prob, y_pred, threshold)
        self.metrics["stacking_ensemble"] = metrics
        self.predictions["stacking_ensemble"] = ensemble_preds

        logger.info(f"  ROC-AUC: {metrics['ROC-AUC']:.4f}")
        logger.info(f"  PR-AUC: {metrics['PR-AUC']:.4f}")
        logger.info(f"  F1-score: {metrics['F1-score']:.4f}")
        logger.info(f"  Accuracy: {metrics['Accuracy']:.4f}")

    def collect_resource_metrics(self, split_meta: Dict):
        total_time = sum(v for v in self.execution_times.values() if isinstance(v, (int, float)))
        self.resource_metrics = {
            "engine": "spark",
            "seed": self.seed,
            "feature_method": self.feature_method,
            "total_training_time_sec": total_time,
            "peak_driver_memory_mb": get_peak_memory_mb(),
            "cpu_cores_available": os.cpu_count(),
            "spark_default_parallelism": self.spark.sparkContext.defaultParallelism,
            "dataset_train_rows": split_meta.get("train_rows"),
            "dataset_val_rows": split_meta.get("val_rows"),
            "dataset_test_rows": split_meta.get("test_rows"),
            "dataset_total_genes": split_meta.get("total_genes"),
            "dataset_selected_genes": self.num_genes,
            "dataset_size_bytes": split_meta.get("source_dataset_size_bytes"),
            "step_times_sec": self.execution_times,
        }
        return self.resource_metrics

    # =========================================================================
    # LEARNING CURVES 
    # =========================================================================

    def compute_learning_curves_spark(self, train_sizes: List[float] = None, n_folds=None):
        if n_folds is not None:
            logger.info("compute_learning_curves_spark: n_folds is no longer used "
                        "(val_df is used directly as the held-out set) - ignoring.")

        if train_sizes is None:
            train_sizes = LEARNING_CURVE_TRAIN_SIZES

        start = time.time()
        logger.info("\n" + "=" * 60)
        logger.info("COMPUTING LEARNING CURVES (Spark) - corrected, leakage-free")
        logger.info("=" * 60)

        self._ensure_checkpoint_dir()

        learning_curves = {}

        for model_type in BASE_MODELS + ["stacking_ensemble"]:
            logger.info(f"\n▶ Learning curve for {MODEL_LABELS.get(model_type, model_type)}")

            train_scores, val_scores, n_samples_list, sizes_used = [], [], [], []

            for frac in train_sizes:
                frac = float(frac)
                sample_df = self._stratified_sample_deterministic(
                    self.train_df, frac, seed_salt=f"lc_{model_type}_{frac:.3f}"
                )
                sample_df = self._persist_df(sample_df, f"lc_sample_{model_type}_{frac:.2f}")
                actual_n = sample_df.count()

                if actual_n < 10:
                    logger.warning(f"  size={frac:.0%} -> only {actual_n} rows, skipping (too small)")
                    sample_df.unpersist()
                    continue

                try:
                    if model_type == "stacking_ensemble":
                        train_auc, val_auc = self._learning_curve_point_ensemble(sample_df)
                    else:
                        train_auc, val_auc = self._learning_curve_point_base(model_type, sample_df)
                except Exception as e:
                    logger.warning(f"  size={frac:.0%} ({actual_n} rows) - point failed, skipping: {e}")
                    sample_df.unpersist()
                    self._cleanup_cached_dfs()
                    continue

                sample_df.unpersist()
                self._cleanup_cached_dfs()
                force_jvm_gc(self.spark)

                train_scores.append(train_auc)
                val_scores.append(val_auc)
                n_samples_list.append(actual_n)
                sizes_used.append(frac)

                logger.info(f"  size={frac:.0%} (n={actual_n})  "
                           f"train_AUC={train_auc:.4f}  val_AUC={val_auc:.4f}")

            learning_curves[model_type] = {
                "train_size_fractions": sizes_used,
                "n_samples": n_samples_list,
                "train_scores": train_scores,
                "val_scores": val_scores,
            }

        self.learning_curves = learning_curves
        self.execution_times["learning_curves"] = time.time() - start
        logger.info(f"\n✓ Learning curves computed in {self.execution_times['learning_curves']:.1f}s")
        return learning_curves

    def _stratified_sample_deterministic(self, df, frac: float, seed_salt: str):
        salted = (df
            .withColumn("lc_hash", F.md5(F.concat(F.col("SampleID").cast(StringType()), F.lit(seed_salt))))
            .withColumn("lc_hash_value",
                       (F.conv(F.substring(F.col("lc_hash"), 1, 8), 16, 10) % 100).cast(FloatType()) / 100))
        window = Window.partitionBy("label").orderBy("lc_hash_value")
        ranked = (salted
            .withColumn("lc_rank", F.row_number().over(window))
            .withColumn("lc_total", F.count("*").over(Window.partitionBy("label"))))
        sampled = (ranked
            .filter(F.col("lc_rank") / F.col("lc_total") <= frac)
            .drop("lc_hash", "lc_hash_value", "lc_rank", "lc_total"))
        return sampled

    def _learning_curve_point_base(self, model_type: str, sample_df):
        pipeline, _ = build_model_pipeline(
            model_type, self.selected_genes, self.best_params.get(model_type, {}),
            precomputed_col=self.precomputed_col
        )
        model = pipeline.fit(sample_df)

        evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

        train_auc = evaluator.evaluate(model.transform(sample_df))
        val_auc = evaluator.evaluate(model.transform(self.val_df))

        del model, pipeline
        return float(train_auc), float(val_auc)

    def _learning_curve_point_ensemble(self, sample_df):
        prob_udf = probability_udf()
        model_alias = {
            "logistic_regression": "pred_lr",
            "random_forest": "pred_rf",
            "gradient_boosting": "pred_gbt"
        }

        base_train = self._stratified_sample_deterministic(sample_df, 0.7, seed_salt="lc_inner_split")
        meta_train = sample_df.join(base_train.select("SampleID"), on="SampleID", how="left_anti")

        base_train = self._persist_df(base_train, "lc_base_train")
        meta_train = self._persist_df(meta_train, "lc_meta_train")

        base_models_cv = {}
        for model_type in BASE_MODELS:
            pipeline, _ = build_model_pipeline(
                model_type, self.selected_genes, self.best_params.get(model_type, {}),
                precomputed_col=self.precomputed_col
            )
            base_models_cv[model_type] = pipeline.fit(base_train)

        def make_meta_features(df):
            result = df.select("SampleID", "label")
            for model_type, model in base_models_cv.items():
                preds = model.transform(df)
                prob_col = preds.select(
                    "SampleID", prob_udf(F.col("probability")).alias(model_alias[model_type])
                )
                result = result.join(prob_col, on="SampleID", how="left")
            return result

        meta_features_train = make_meta_features(meta_train).join(
            meta_train.select("SampleID", "class_weight"), on="SampleID", how="left"
        )
        meta_features_train = self._persist_df(meta_features_train, "lc_meta_features_train")

        meta_feature_cols = ["pred_lr", "pred_rf", "pred_gbt"]
        assembler = VectorAssembler(inputCols=meta_feature_cols, outputCol="meta_features", handleInvalid="skip")
        scaler = StandardScaler(inputCol="meta_features", outputCol="meta_scaled", withMean=True, withStd=True)
        meta_lr = LogisticRegression(
            featuresCol="meta_scaled", labelCol="label", weightCol="class_weight",
            family="binomial", maxIter=100, regParam=0.01, elasticNetParam=0.0
        )
        meta_pipeline = Pipeline(stages=[assembler, scaler, meta_lr])
        meta_model = meta_pipeline.fit(meta_features_train)

        evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

        train_auc = evaluator.evaluate(meta_model.transform(meta_features_train))

        meta_features_val = make_meta_features(self.val_df)
        val_auc = evaluator.evaluate(meta_model.transform(meta_features_val))

        base_train.unpersist()
        meta_train.unpersist()
        meta_features_train.unpersist()

        del base_models_cv, meta_model, meta_pipeline
        return float(train_auc), float(val_auc)


# =============================================================================
# SKLEARN ENGINE
# =============================================================================

def _sk_stratified_split(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int):
    salt = f"::seed{seed}"
    hv = df["SampleID"].astype(str).apply(lambda s: hash_fraction(s, salt))
    df = df.assign(hash_value=hv.values)
    df["rank"] = df.groupby("label")["hash_value"].rank(method="first")
    df["total"] = df.groupby("label")["label"].transform("count")
    frac = df["rank"] / df["total"]
    split = np.where(frac <= train_ratio, "train",
             np.where(frac <= train_ratio + val_ratio, "val", "test"))
    df = df.drop(columns=["hash_value", "rank", "total"])
    train_df = df[split == "train"].reset_index(drop=True)
    val_df = df[split == "val"].reset_index(drop=True)
    test_df = df[split == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


def _sk_assign_folds(df: pd.DataFrame, n_folds: int):
    hv = df["SampleID"].astype(str).apply(lambda s: hash_fraction(s, ""))
    df = df.assign(hash_value=hv.values)
    df["rank"] = df.groupby("label")["hash_value"].rank(method="first")
    df["total"] = df.groupby("label")["label"].transform("count")
    fold = (((df["rank"] - 1) * n_folds) / df["total"]).astype(int)
    df = df.assign(fold=fold.values).drop(columns=["hash_value", "rank", "total"])
    return df


def _sk_class_weights(train_df: pd.DataFrame) -> Dict[float, float]:
    counts = train_df["label"].value_counts()
    total = counts.sum()
    return {float(lbl): total / (2.0 * cnt) for lbl, cnt in counts.items()}


def _sk_build_base_model(model_type: str, seed: int, params: Dict = None):
    """
    Builds a base model matching the given hyperparameters. For logistic
    regression this now properly honors `elasticNetParam` (previously it
    was silently ignored and the model always used plain L2 regardless of
    what was passed in) - mapped the same way Spark's LogisticRegression
    interprets it: 0.0 -> pure L2 (ridge), 1.0 -> pure L1 (lasso), anything
    in between -> elastic net with that mixing ratio (sklearn's l1_ratio).
    """
    params = params or {}
    if model_type == "logistic_regression":
        reg_param = float(params.get("regParam", 0.01))
        elastic_net = float(params.get("elasticNetParam", 0.0))
        C = 1.0 / max(reg_param, 1e-6)
        if elastic_net <= 0.0:
            return SkLogisticRegression(max_iter=500, C=C, penalty="l2", solver="lbfgs")
        elif elastic_net >= 1.0:
            return SkLogisticRegression(max_iter=500, C=C, penalty="l1", solver="saga")
        else:
            return SkLogisticRegression(max_iter=500, C=C, penalty="elasticnet",
                                        solver="saga", l1_ratio=elastic_net)
    elif model_type == "random_forest":
        return SkRandomForestClassifier(
            n_estimators=int(params.get("numTrees", 50)),
            max_depth=int(params.get("maxDepth", 8)),
            min_samples_leaf=5, random_state=seed, n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        return SkGradientBoostingClassifier(
            n_estimators=int(params.get("maxIter", 30)),
            max_depth=int(params.get("maxDepth", 3)),
            learning_rate=float(params.get("stepSize", 0.1)),
            random_state=seed
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def _sk_select_features_hybrid(train_df: pd.DataFrame, gene_cols: List[str],
                                num_select: int, seed: int) -> List[str]:
    """Hybrid feature selection for sklearn: ANOVA filtering then RF ranking."""
    X_train = train_df[gene_cols].to_numpy(dtype=np.float64)
    y_train = train_df["label"].to_numpy()
    w_train = train_df["class_weight"].to_numpy()

    anova_keep_count = int(num_select * ANOVA_FILTER_FACTOR)
    f_vals, _ = f_classif(X_train, y_train)
    f_vals = np.nan_to_num(f_vals, nan=-np.inf)
    anova_order = np.argsort(f_vals)[::-1]
    anova_filtered_genes = [gene_cols[i] for i in anova_order[:anova_keep_count]]

    X_filtered = train_df[anova_filtered_genes].to_numpy(dtype=np.float64)
    rf = SkRandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
    rf.fit(X_filtered, y_train, sample_weight=w_train)

    importance_order = np.argsort(rf.feature_importances_)[::-1]
    selected_genes = [anova_filtered_genes[i] for i in importance_order[:num_select]]

    return selected_genes


class SklearnStackingPipeline:
    def __init__(self, feature_method: str, num_genes: int, seed: int, resource_log_path: str):
        self.feature_method = feature_method
        self.num_genes = num_genes
        self.seed = seed
        self.resource_log_path = resource_log_path

        self.selected_genes = None
        self.scaler_for_lr = None
        self.pca_model = None
        self.variance_filtered_genes = None
        self.anova_filtered_genes = None
        self.base_models = {}
        self.best_params = {}
        self.tuning_val_scores = {}
        self.meta_model = None
        self.meta_scaler = None
        self.optimal_threshold = DEFAULT_THRESHOLD
        self.metrics = {}
        self.predictions = {}
        self.execution_times = {}
        self.learning_curves = {}
        self._t0 = time.time()

    def _log_step(self, step_name: str):
        self.execution_times[step_name] = time.time() - self._t0
        self._t0 = time.time()
        partial = {
            "engine": "sklearn",
            "seed": self.seed,
            "feature_method": self.feature_method,
            "last_completed_step": step_name,
            "peak_driver_memory_mb": get_peak_memory_mb(),
            "step_times_sec": dict(self.execution_times),
        }
        os.makedirs(os.path.dirname(self.resource_log_path), exist_ok=True)
        with open(self.resource_log_path, "w") as f:
            json.dump(partial, f, indent=2, default=str)

    def load_and_split(self, data_path: str):
        df = pd.read_parquet(data_path)
        if "SampleID" not in df.columns:
            df = df.reset_index().rename(columns={"index": "SampleID"})
            df["SampleID"] = df["SampleID"].astype(str)
        self.gene_cols = get_gene_columns(df)
        self.total_genes = len(self.gene_cols)
        train_df, val_df, test_df = _sk_stratified_split(df, TRAIN_RATIO, VAL_RATIO, self.seed)
        self.class_weights = _sk_class_weights(train_df)
        train_df = train_df.assign(
            class_weight=train_df["label"].map(self.class_weights).astype(float)
        )
        self.train_df, self.val_df, self.test_df = train_df, val_df, test_df
        self._log_step("load_and_split")
        return self

    def select_features(self):
        gene_cols = self.gene_cols
        X_train = self.train_df[gene_cols].to_numpy(dtype=np.float64)
        y_train = self.train_df["label"].to_numpy()

        if self.feature_method == "rf":
            rf = SkRandomForestClassifier(n_estimators=50, random_state=self.seed, n_jobs=-1)
            rf.fit(X_train, y_train, sample_weight=self.train_df["class_weight"].to_numpy())
            order = np.argsort(rf.feature_importances_)[::-1]
            self.selected_genes = [gene_cols[i] for i in order[:self.num_genes]]

        elif self.feature_method == "anova":
            f_vals, _ = f_classif(X_train, y_train)
            f_vals = np.nan_to_num(f_vals, nan=-np.inf)
            order = np.argsort(f_vals)[::-1]
            self.selected_genes = [gene_cols[i] for i in order[:self.num_genes]]

        elif self.feature_method == "hybrid":
            self.selected_genes = _sk_select_features_hybrid(
                self.train_df, gene_cols, self.num_genes, self.seed
            )
            anova_keep_count = int(self.num_genes * ANOVA_FILTER_FACTOR)
            f_vals, _ = f_classif(X_train, y_train)
            f_vals = np.nan_to_num(f_vals, nan=-np.inf)
            anova_order = np.argsort(f_vals)[::-1]
            self.anova_filtered_genes = [gene_cols[i] for i in anova_order[:anova_keep_count]]

        elif self.feature_method == "pca":
            max_features_for_pca = 2000
            variances = np.nanvar(X_train, axis=0)
            order = np.argsort(variances)[::-1]
            self.variance_filtered_genes = [gene_cols[i] for i in order[:max_features_for_pca]]
            X_filtered = self.train_df[self.variance_filtered_genes].to_numpy(dtype=np.float64)
            self.pca_model = SkPCA(n_components=PCA_COMPONENTS, random_state=self.seed)
            self.pca_model.fit(X_filtered)
            self.selected_genes = [f"PC{i+1}" for i in range(PCA_COMPONENTS)]
        else:
            raise ValueError(f"Unknown feature method: {self.feature_method}")

        self._log_step("feature_selection")
        return self

    def _featurize(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature_method == "pca":
            X = df[self.variance_filtered_genes].to_numpy(dtype=np.float64)
            return self.pca_model.transform(X)
        return df[self.selected_genes].to_numpy(dtype=np.float64)


    # =========================================================================

    def _sk_inner_train_val_split(self, df: pd.DataFrame, train_ratio: float = TUNING_TRAIN_RATIO):
        hv = df["SampleID"].astype(str).apply(lambda s: hash_fraction(s, "::tune"))
        tmp = df.assign(_tune_hash=hv.values)
        tmp["_tune_rank"] = tmp.groupby("label")["_tune_hash"].rank(method="first")
        tmp["_tune_total"] = tmp.groupby("label")["label"].transform("count")
        frac = tmp["_tune_rank"] / tmp["_tune_total"]
        inner_train = (tmp[frac <= train_ratio]
                       .drop(columns=["_tune_hash", "_tune_rank", "_tune_total"])
                       .reset_index(drop=True))
        inner_val = (tmp[frac > train_ratio]
                     .drop(columns=["_tune_hash", "_tune_rank", "_tune_total"])
                     .reset_index(drop=True))
        return inner_train, inner_val

    def train_base_models(self):
        """
        Grid-search tuning + final refit, mirroring Spark's
        TrainValidationSplit exactly:
          1. Split self.train_df once into inner_train/inner_val
             (TUNING_TRAIN_RATIO, same ratio Spark uses).
          2. For every model type, fit every combination in sk_param_grid()
             on inner_train, score by ROC-AUC on inner_val, keep the best.
          3. Refit the winning hyperparameters on the FULL self.train_df
             (this matches TrainValidationSplitModel's behavior, which
             refits bestModel on the whole input dataset after selecting
             the best ParamMap).
        """
        start = time.time()
        logger.info("=" * 60)
        logger.info("TUNING AND TRAINING BASE MODELS (LR, RF, GBT) - sklearn grid search")
        logger.info("=" * 60)

        inner_train, inner_val = self._sk_inner_train_val_split(self.train_df)
        logger.info(f"  Tuning split: inner_train={len(inner_train)}, inner_val={len(inner_val)}")

        X_it = self._featurize(inner_train)
        y_it = inner_train["label"].to_numpy()
        w_it = inner_train["class_weight"].to_numpy()
        X_iv = self._featurize(inner_val)
        y_iv = inner_val["label"].to_numpy()

        for model_type in BASE_MODELS:
            logger.info(f"\n▶ {model_type.upper()}")
            grid = sk_param_grid(model_type)
            logger.info(f"  Grid search over {len(grid)} combinations...")

            best_score = -1.0
            best_params_for_model = grid[0] if grid else {}

            for params in grid:
                try:
                    if model_type == "logistic_regression":
                        scaler = SkStandardScaler().fit(X_it)
                        model = _sk_build_base_model(model_type, self.seed, params)
                        model.fit(scaler.transform(X_it), y_it, sample_weight=w_it)
                        val_prob = model.predict_proba(scaler.transform(X_iv))[:, 1]
                    else:
                        model = _sk_build_base_model(model_type, self.seed, params)
                        model.fit(X_it, y_it, sample_weight=w_it)
                        val_prob = model.predict_proba(X_iv)[:, 1]
                    score = roc_auc_score(y_iv, val_prob)
                except Exception as e:
                    logger.warning(f"    params={params} failed, skipping: {e}")
                    continue

                if score > best_score:
                    best_score = score
                    best_params_for_model = params

            self.best_params[model_type] = best_params_for_model
            self.tuning_val_scores[model_type] = best_score
            logger.info(f"  ✓ Best parameters: {best_params_for_model} (val ROC-AUC={best_score:.4f})")

        logger.info("\n  Refitting best models on full training set...")
        X_train = self._featurize(self.train_df)
        y_train = self.train_df["label"].to_numpy()
        w_train = self.train_df["class_weight"].to_numpy()

        for model_type in BASE_MODELS:
            params = self.best_params.get(model_type, {})
            if model_type == "logistic_regression":
                self.scaler_for_lr = SkStandardScaler().fit(X_train)
                Xs = self.scaler_for_lr.transform(X_train)
                model = _sk_build_base_model(model_type, self.seed, params)
                model.fit(Xs, y_train, sample_weight=w_train)
            else:
                model = _sk_build_base_model(model_type, self.seed, params)
                model.fit(X_train, y_train, sample_weight=w_train)
            self.base_models[model_type] = model

        self.execution_times["base_models"] = time.time() - start
        logger.info(f"\n✓ Base models tuned and trained in {self.execution_times['base_models']:.1f}s")
        self._log_step("base_models")
        return self

    def _base_model_proba(self, model_type: str, X: np.ndarray) -> np.ndarray:
        model = self.base_models[model_type]
        if model_type == "logistic_regression":
            X = self.scaler_for_lr.transform(X)
        return model.predict_proba(X)[:, 1]

    def generate_oof_predictions(self, n_folds: int = 5):
        """OOF folds now use the TUNED best_params (previously these were
        built with untuned defaults even though train_base_models had
        already found better hyperparameters) - matches the Spark engine's
        generate_oof_predictions, which always uses self.best_params."""
        folded = _sk_assign_folds(self.train_df, n_folds)
        oof_rows = []
        for fold_idx in range(n_folds):
            train_fold = folded[folded["fold"] != fold_idx]
            val_fold = folded[folded["fold"] == fold_idx]

            X_tr = self._featurize(train_fold)
            y_tr = train_fold["label"].to_numpy()
            w_tr = train_fold["class_weight"].to_numpy()
            X_val = self._featurize(val_fold)

            fold_preds = {"SampleID": val_fold["SampleID"].values, "label": val_fold["label"].values}
            for model_type in BASE_MODELS:
                params = self.best_params.get(model_type, {})
                if model_type == "logistic_regression":
                    scaler = SkStandardScaler().fit(X_tr)
                    model = _sk_build_base_model(model_type, self.seed, params)
                    model.fit(scaler.transform(X_tr), y_tr, sample_weight=w_tr)
                    proba = model.predict_proba(scaler.transform(X_val))[:, 1]
                else:
                    model = _sk_build_base_model(model_type, self.seed, params)
                    model.fit(X_tr, y_tr, sample_weight=w_tr)
                    proba = model.predict_proba(X_val)[:, 1]
                key = {"logistic_regression": "pred_lr", "random_forest": "pred_rf",
                       "gradient_boosting": "pred_gbt"}[model_type]
                fold_preds[key] = proba
            oof_rows.append(pd.DataFrame(fold_preds))

        oof_df = pd.concat(oof_rows, ignore_index=True)
        self._log_step("oof_generation")
        return oof_df

    def train_meta_learner(self, oof_df: pd.DataFrame):
        meta_features = ["pred_lr", "pred_rf", "pred_gbt"]
        X_meta = oof_df[meta_features].to_numpy(dtype=np.float64)
        y_meta = oof_df["label"].to_numpy()

        self.meta_scaler = SkStandardScaler().fit(X_meta)
        self.meta_model = SkLogisticRegression(max_iter=200)
        self.meta_model.fit(self.meta_scaler.transform(X_meta), y_meta)

        self._log_step("meta_training")
        return self.meta_model

    def _predict_stacked(self, df: pd.DataFrame) -> np.ndarray:
        X = self._featurize(df)
        preds = np.column_stack([
            self._base_model_proba("logistic_regression", X),
            self._base_model_proba("random_forest", X),
            self._base_model_proba("gradient_boosting", X),
        ])
        Xs = self.meta_scaler.transform(preds)
        return self.meta_model.predict_proba(Xs)[:, 1]

    def optimize_threshold(self, n_thresholds: int = 21):
        prob = self._predict_stacked(self.val_df)
        y_true = self.val_df["label"].to_numpy()
        lo, hi = max(float(prob.min()), 0.01), min(float(prob.max()), 0.99)
        best_t, best_bacc = 0.5, -1.0
        for t in np.linspace(lo, hi, n_thresholds):
            pred = (prob >= t).astype(int)
            tp = np.sum((y_true == 1) & (pred == 1)); fn = np.sum((y_true == 1) & (pred == 0))
            tn = np.sum((y_true == 0) & (pred == 0)); fp = np.sum((y_true == 0) & (pred == 1))
            sens = tp / (tp + fn + 1e-10); spec = tn / (tn + fp + 1e-10)
            bacc = (sens + spec) / 2
            if bacc > best_bacc:
                best_bacc, best_t = bacc, float(t)
        self.optimal_threshold = best_t
        self._log_step("threshold_optimization")
        return best_t

    def evaluate(self):
        threshold = self.optimal_threshold
        for model_type in BASE_MODELS:
            X_test = self._featurize(self.test_df)
            y_true = self.test_df["label"].to_numpy()
            prob = self._base_model_proba(model_type, X_test)
            pred_opt = (prob >= threshold).astype(float)
            metrics = calculate_metrics(y_true, prob, pred_opt, threshold)
            self.metrics[model_type] = metrics
            self.predictions[model_type] = pd.DataFrame(
                {"label": y_true, "prob_pos": prob, "pred": pred_opt})

        prob_stack = self._predict_stacked(self.test_df)
        y_true = self.test_df["label"].to_numpy()
        pred_stack = (prob_stack >= threshold).astype(float)
        metrics = calculate_metrics(y_true, prob_stack, pred_stack, threshold)
        self.metrics["stacking_ensemble"] = metrics
        self.predictions["stacking_ensemble"] = pd.DataFrame(
            {"label": y_true, "prob_pos": prob_stack, "pred": pred_stack})

        self._log_step("evaluation")
        return self.metrics

    # =========================================================================


    def compute_learning_curves_sklearn(self, train_sizes: List[float] = None, n_folds=None):
        if n_folds is not None:
            logger.info("compute_learning_curves_sklearn: n_folds is no longer used - ignoring.")

        if train_sizes is None:
            train_sizes = LEARNING_CURVE_TRAIN_SIZES

        start = time.time()
        logger.info("\n" + "=" * 60)
        logger.info("COMPUTING LEARNING CURVES (sklearn) - corrected, leakage-free")
        logger.info("=" * 60)

        learning_curves = {}

        for model_type in BASE_MODELS + ["stacking_ensemble"]:
            logger.info(f"\n▶ Learning curve for {MODEL_LABELS.get(model_type, model_type)}")

            train_scores, val_scores, n_samples_list, sizes_used = [], [], [], []

            for frac in train_sizes:
                frac = float(frac)
                sample_df = self._stratified_sample_pandas(
                    self.train_df, frac, salt=f"lc_{model_type}_{frac:.3f}"
                )
                actual_n = len(sample_df)
                if actual_n < 10:
                    logger.warning(f"  size={frac:.0%} -> only {actual_n} rows, skipping")
                    continue

                try:
                    if model_type == "stacking_ensemble":
                        train_auc, val_auc = self._learning_curve_point_ensemble_sklearn(sample_df)
                    else:
                        train_auc, val_auc = self._learning_curve_point_base_sklearn(model_type, sample_df)
                except Exception as e:
                    logger.warning(f"  size={frac:.0%} ({actual_n} rows) - point failed, skipping: {e}")
                    continue

                train_scores.append(train_auc)
                val_scores.append(val_auc)
                n_samples_list.append(actual_n)
                sizes_used.append(frac)

                logger.info(f"  size={frac:.0%} (n={actual_n})  "
                           f"train_AUC={train_auc:.4f}  val_AUC={val_auc:.4f}")

            learning_curves[model_type] = {
                "train_size_fractions": sizes_used,
                "n_samples": n_samples_list,
                "train_scores": train_scores,
                "val_scores": val_scores,
            }

        self.learning_curves = learning_curves
        self.execution_times["learning_curves"] = time.time() - start
        logger.info(f"\n✓ Learning curves computed in {self.execution_times['learning_curves']:.1f}s")
        return learning_curves

    def _stratified_sample_pandas(self, df: pd.DataFrame, frac: float, salt: str) -> pd.DataFrame:
        hv = df["SampleID"].astype(str).apply(lambda s: hash_fraction(s, salt))
        tmp = df.assign(_lc_hash=hv.values)
        tmp["_lc_rank"] = tmp.groupby("label")["_lc_hash"].rank(method="first")
        tmp["_lc_total"] = tmp.groupby("label")["label"].transform("count")
        keep = tmp["_lc_rank"] / tmp["_lc_total"] <= frac
        return tmp[keep].drop(columns=["_lc_hash", "_lc_rank", "_lc_total"]).reset_index(drop=True)

    def _learning_curve_point_base_sklearn(self, model_type: str, sample_df: pd.DataFrame):
        X_train = self._featurize(sample_df)
        y_train = sample_df["label"].to_numpy()
        w_train = sample_df["class_weight"].to_numpy()

        X_val = self._featurize(self.val_df)
        y_val = self.val_df["label"].to_numpy()

        if model_type == "logistic_regression":
            scaler = SkStandardScaler().fit(X_train)
            model = _sk_build_base_model(model_type, self.seed, self.best_params.get(model_type, {}))
            model.fit(scaler.transform(X_train), y_train, sample_weight=w_train)
            train_prob = model.predict_proba(scaler.transform(X_train))[:, 1]
            val_prob = model.predict_proba(scaler.transform(X_val))[:, 1]
        else:
            model = _sk_build_base_model(model_type, self.seed, self.best_params.get(model_type, {}))
            model.fit(X_train, y_train, sample_weight=w_train)
            train_prob = model.predict_proba(X_train)[:, 1]
            val_prob = model.predict_proba(X_val)[:, 1]

        train_auc = roc_auc_score(y_train, train_prob)
        val_auc = roc_auc_score(y_val, val_prob)
        return float(train_auc), float(val_auc)

    def _learning_curve_point_ensemble_sklearn(self, sample_df: pd.DataFrame):
        base_train = self._stratified_sample_pandas(sample_df, 0.7, salt="lc_inner_split")
        meta_train = sample_df[~sample_df["SampleID"].isin(base_train["SampleID"])].reset_index(drop=True)

        X_base_train = self._featurize(base_train)
        y_base_train = base_train["label"].to_numpy()
        w_base_train = base_train["class_weight"].to_numpy()

        base_models_cv = {}
        scalers_cv = {}
        for model_type in BASE_MODELS:
            params = self.best_params.get(model_type, {})
            if model_type == "logistic_regression":
                scaler = SkStandardScaler().fit(X_base_train)
                model = _sk_build_base_model(model_type, self.seed, params)
                model.fit(scaler.transform(X_base_train), y_base_train, sample_weight=w_base_train)
                scalers_cv[model_type] = scaler
            else:
                model = _sk_build_base_model(model_type, self.seed, params)
                model.fit(X_base_train, y_base_train, sample_weight=w_base_train)
            base_models_cv[model_type] = model

        def base_proba(model_type, X):
            model = base_models_cv[model_type]
            if model_type == "logistic_regression":
                X = scalers_cv[model_type].transform(X)
            return model.predict_proba(X)[:, 1]

        X_meta_train = self._featurize(meta_train)
        meta_feat_train = np.column_stack([
            base_proba("logistic_regression", X_meta_train),
            base_proba("random_forest", X_meta_train),
            base_proba("gradient_boosting", X_meta_train),
        ])
        y_meta_train = meta_train["label"].to_numpy()

        meta_scaler = SkStandardScaler().fit(meta_feat_train)
        meta_model = SkLogisticRegression(max_iter=200)
        meta_model.fit(meta_scaler.transform(meta_feat_train), y_meta_train)

        train_prob = meta_model.predict_proba(meta_scaler.transform(meta_feat_train))[:, 1]
        train_auc = roc_auc_score(y_meta_train, train_prob)

        X_val = self._featurize(self.val_df)
        meta_feat_val = np.column_stack([
            base_proba("logistic_regression", X_val),
            base_proba("random_forest", X_val),
            base_proba("gradient_boosting", X_val),
        ])
        y_val = self.val_df["label"].to_numpy()
        val_prob = meta_model.predict_proba(meta_scaler.transform(meta_feat_val))[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)

        return float(train_auc), float(val_auc)

    def collect_resource_metrics(self):
        total_time = sum(v for v in self.execution_times.values() if isinstance(v, (int, float)))
        return {
            "engine": "sklearn",
            "seed": self.seed,
            "feature_method": self.feature_method,
            "total_training_time_sec": total_time,
            "peak_driver_memory_mb": get_peak_memory_mb(),
            "cpu_cores_available": os.cpu_count(),
            "dataset_train_rows": len(self.train_df),
            "dataset_val_rows": len(self.val_df),
            "dataset_test_rows": len(self.test_df),
            "dataset_total_genes": self.total_genes,
            "dataset_selected_genes": self.num_genes,
            "dataset_size_bytes": get_dir_size_bytes(DATA_PATH),
            "step_times_sec": self.execution_times,
        }


def run_sklearn_seed_worker(method: str, seed: int, data_path: str, output_dir: str) -> bool:
    seed_output_dir = os.path.join(output_dir, method, f"seed_{seed}", "engine_sklearn")
    os.makedirs(seed_output_dir, exist_ok=True)
    resource_log_path = os.path.join(seed_output_dir, "resource_metrics_partial.json")
    num_genes = PCA_COMPONENTS if method == "pca" else NUM_SELECTED_GENES

    success = False
    try:
        pipeline = SklearnStackingPipeline(method, num_genes, seed, resource_log_path)
        pipeline.load_and_split(data_path)
        pipeline.select_features()
        pipeline.train_base_models()
        oof_df = pipeline.generate_oof_predictions(n_folds=N_FOLDS)
        pipeline.train_meta_learner(oof_df)
        pipeline.optimize_threshold(n_thresholds=21)
        pipeline.evaluate()

        pipeline.compute_learning_curves_sklearn(train_sizes=LEARNING_CURVE_TRAIN_SIZES)

        resource_metrics = pipeline.collect_resource_metrics()

        with open(os.path.join(seed_output_dir, "metrics.json"), "w") as f:
            json.dump(pipeline.metrics, f, indent=2, default=str)
        with open(os.path.join(seed_output_dir, "resource_metrics.json"), "w") as f:
            json.dump(resource_metrics, f, indent=2, default=str)
        with open(os.path.join(seed_output_dir, "learning_curves.json"), "w") as f:
            json.dump(pipeline.learning_curves, f, indent=2, default=str)
        with open(os.path.join(seed_output_dir, "config.json"), "w") as f:
            json.dump({
                "feature_method": pipeline.feature_method,
                "optimal_threshold": pipeline.optimal_threshold,
                "num_selected_genes": pipeline.num_genes,
                "seed": pipeline.seed,
                "execution_times": pipeline.execution_times,
                "best_params": pipeline.best_params,
                "tuning_val_scores": pipeline.tuning_val_scores,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2, default=str)

        plot_learning_curves_sklearn(pipeline.learning_curves, seed_output_dir, method)

        success = True
        failed_marker = os.path.join(seed_output_dir, "FAILED")
        if os.path.exists(failed_marker):
            os.remove(failed_marker)
    except MemoryError as e:
        logger.error(f"✗ sklearn OOM for method={method} seed={seed}: {e}")
        with open(os.path.join(seed_output_dir, "FAILED"), "w") as f:
            f.write(f"MemoryError: {e}")
        success = False
    except Exception as e:
        logger.error(f"✗ sklearn failed for method={method} seed={seed}: {e}", exc_info=True)
        with open(os.path.join(seed_output_dir, "FAILED"), "w") as f:
            f.write(str(e))
        success = False
    finally:
        gc.collect()

    return success


# =============================================================================


def run_single_seed_worker(method: str, seed: int, data_path: str, output_dir: str) -> bool:
    method_output_dir = os.path.join(output_dir, method)
    seed_checkpoint = os.path.join(CHECKPOINT_DIR, method, f"seed_{seed}")
    os.makedirs(seed_checkpoint, exist_ok=True)
    seed_output_dir = os.path.join(method_output_dir, f"seed_{seed}")
    os.makedirs(seed_output_dir, exist_ok=True)

    split_meta = precompute_split_for_seed(data_path, seed)
    feature_meta = precompute_feature_selection(method, seed)
    num_genes = PCA_COMPONENTS if method == "pca" else NUM_SELECTED_GENES

    spark = None
    success = False
    try:
        spark = create_spark_session(f"{method}_seed{seed}")
        spark.sparkContext.setCheckpointDir(seed_checkpoint)

        pipeline = GenomicStackingPipeline(spark, feature_method=method, num_genes=num_genes, seed=seed)
        pipeline.load_cached_split(split_meta)
        pipeline.select_features(feature_meta)
        pipeline.tune_and_train_base_models()
        oof_df = pipeline.generate_oof_predictions(n_folds=N_FOLDS)
        pipeline.train_meta_learner(oof_df)
        pipeline.optimize_threshold(n_thresholds=21)
        pipeline.evaluate()

        pipeline.compute_learning_curves_spark(train_sizes=LEARNING_CURVE_TRAIN_SIZES)

        pipeline.collect_resource_metrics(split_meta)

        save_results(pipeline, seed_output_dir)
        with open(os.path.join(seed_output_dir, "resource_metrics.json"), "w") as f:
            json.dump(pipeline.resource_metrics, f, indent=2, default=str)

        with open(os.path.join(seed_output_dir, "learning_curves.json"), "w") as f:
            json.dump(pipeline.learning_curves, f, indent=2, default=str)

        create_visualizations(pipeline, seed_output_dir)
        plot_learning_curves_spark(pipeline.learning_curves, seed_output_dir, method)
        print_summary(pipeline)

        pipeline._cleanup_cached_dfs()
        success = True

        failed_marker = os.path.join(seed_output_dir, "FAILED")
        if os.path.exists(failed_marker):
            os.remove(failed_marker)

    except Exception as e:
        logger.error(f"✗ Pipeline failed for method={method} seed={seed}: {e}", exc_info=True)
        with open(os.path.join(seed_output_dir, "FAILED"), "w") as f:
            f.write(str(e))
        success = False

    finally:
        safe_stop_spark(spark)
        gc.collect()

    return success



# =============================================================================

def plot_learning_curves_spark(learning_curves: Dict, output_dir: str, method_name: str):
    _plot_learning_curves_generic(learning_curves, output_dir, method_name, engine="Spark",
                                  filename="learning_curves_spark.png")


def plot_learning_curves_sklearn(learning_curves: Dict, output_dir: str, method_name: str):
    _plot_learning_curves_generic(learning_curves, output_dir, method_name, engine="sklearn",
                                  filename="learning_curves_sklearn.png")


def _plot_learning_curves_generic(learning_curves: Dict, output_dir: str, method_name: str,
                                  engine: str, filename: str):
    if not learning_curves:
        logger.warning(f"No learning curve data available for {engine}")
        return

    model_types = [m for m, d in learning_curves.items() if d.get("n_samples")]
    if not model_types:
        logger.warning("No valid learning curve data found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, model_type in enumerate(model_types[:4]):
        ax = axes[idx]
        data = learning_curves[model_type]
        x = data["n_samples"]
        train_scores = data["train_scores"]
        val_scores = data["val_scores"]

        ax.plot(x, train_scores, 'o-', color='#2E86AB', label='Training score', linewidth=2)
        ax.plot(x, val_scores, 'o-', color='#F18F01', label='Validation score (held-out)', linewidth=2)

        label = MODEL_LABELS.get(model_type, model_type).upper()
        ax.set_title(f'{label} - Learning Curve', fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Samples', fontsize=10)
        ax.set_ylabel('ROC-AUC Score', fontsize=10)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f'{int(v):,}'))

    for idx in range(len(model_types), 4):
        axes[idx].set_visible(False)

    plt.suptitle(f'Learning Curves - {FEATURE_METHOD_LABELS.get(method_name, method_name)} ({engine})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"✓ {engine} learning curves saved to {os.path.join(output_dir, filename)}")


def plot_aggregated_learning_curves(all_lc_data: Dict, output_dir: str, method_name: str, engine: str):
    if not all_lc_data:
        return

    model_types = [m for m in all_lc_data.keys() if all_lc_data[m].get("train_scores")]
    if not model_types:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, model_type in enumerate(model_types[:4]):
        ax = axes[idx]
        data = all_lc_data[model_type]

        train_lists = [s for s in data["train_scores"] if s]
        val_lists = [s for s in data["val_scores"] if s]

        if not train_lists or not val_lists:
            continue

        min_len = min(len(s) for s in train_lists)
        train_arr = np.array([s[:min_len] for s in train_lists])
        val_arr = np.array([s[:min_len] for s in val_lists])

        x = data["n_samples"][:min_len] if data.get("n_samples") else list(range(min_len))

        train_mean, train_std = train_arr.mean(axis=0), train_arr.std(axis=0)
        val_mean, val_std = val_arr.mean(axis=0), val_arr.std(axis=0)

        ax.plot(x, train_mean, 'o-', color='#2E86AB', label='Training score', linewidth=2)
        ax.fill_between(x, train_mean - train_std, train_mean + train_std, alpha=0.2, color='#2E86AB')

        ax.plot(x, val_mean, 'o-', color='#F18F01', label='Validation score', linewidth=2)
        ax.fill_between(x, val_mean - val_std, val_mean + val_std, alpha=0.2, color='#F18F01')

        label = MODEL_LABELS.get(model_type, model_type).upper()
        ax.set_title(f'{label} (Mean ± SD across {len(train_lists)} seeds)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Training Samples', fontsize=10)
        ax.set_ylabel('ROC-AUC Score', fontsize=10)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f'{int(v):,}'))

    for idx in range(len(model_types), 4):
        axes[idx].set_visible(False)

    plt.suptitle(f'Learning Curves - {FEATURE_METHOD_LABELS.get(method_name, method_name)} ({engine})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves_aggregated.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"✓ Aggregated learning curves saved to "
               f"{os.path.join(output_dir, 'learning_curves_aggregated.png')}")


# =============================================================================


def run_all_seeds_for_method(method: str, seeds: List[int], data_path: str, output_dir: str,
                              engines: List[str] = ("spark",)):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for seed in seeds:
        precompute_split_for_seed(data_path, seed)
        precompute_feature_selection(method, seed)

    method_output_dir = os.path.join(output_dir, method)
    os.makedirs(method_output_dir, exist_ok=True)

    seed_wall_times = {"spark": {}, "sklearn": {}}
    for seed in seeds:
        for engine in engines:
            logger.info("\n" + "=" * 80)
            logger.info(f"LAUNCHING SUBPROCESS: engine={engine} method={method} seed={seed}")
            logger.info("=" * 80)

            t0 = time.time()
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--feature-method", method,
                   "--seed", str(seed),
                   "--engine", engine,
                   "--data-path", data_path,
                   "--output-dir", output_dir]
            result = subprocess.run(cmd)
            wall_time = time.time() - t0
            seed_wall_times[engine][seed] = wall_time

            if result.returncode != 0:
                logger.error(f"✗ engine={engine} method={method} seed={seed} exited {result.returncode} "
                             f"(wall {wall_time:.1f}s) - continuing")
            else:
                logger.info(f"✓ engine={engine} method={method} seed={seed} completed in {wall_time:.1f}s")

    if "spark" in engines:
        compile_final_report(seeds, method_output_dir, seed_wall_times["spark"])

    if "sklearn" in engines:
        sklearn_metrics = {}
        sklearn_resource = {}
        sklearn_lc = {}
        failed_seeds = []

        for seed in seeds:
            sklearn_dir = os.path.join(method_output_dir, f"seed_{seed}", "engine_sklearn")
            metrics_path = os.path.join(sklearn_dir, "metrics.json")
            resource_path = os.path.join(sklearn_dir, "resource_metrics.json")
            lc_path = os.path.join(sklearn_dir, "learning_curves.json")

            if os.path.exists(os.path.join(sklearn_dir, "FAILED")) or not os.path.exists(metrics_path):
                failed_seeds.append(seed)
                continue

            with open(metrics_path) as f:
                metrics = json.load(f)
            resource_data = {}
            if os.path.exists(resource_path):
                with open(resource_path) as f:
                    resource_data = json.load(f)
            lc_data = {}
            if os.path.exists(lc_path):
                with open(lc_path) as f:
                    lc_data = json.load(f)

            sklearn_metrics[seed] = metrics
            sklearn_resource[seed] = resource_data
            sklearn_lc[seed] = lc_data

        if sklearn_metrics:
            compile_sklearn_final_report(seeds, method_output_dir, sklearn_metrics, sklearn_resource,
                                         sklearn_lc, failed_seeds, seed_wall_times["sklearn"])

    if "spark" in engines and "sklearn" in engines:
        compile_engine_comparison(method_output_dir, seeds, seed_wall_times)
    else:
        logger.info(f"Skipping engine comparison (only {engines} ran - need both spark and sklearn)")


def compile_sklearn_final_report(seeds: List[int], method_output_dir: str,
                                  sklearn_metrics: Dict, sklearn_resource: Dict,
                                  sklearn_lc: Dict, failed_seeds: List[int],
                                  seed_wall_times: Dict[int, float]):
    all_metrics = defaultdict(list)
    all_resource = defaultdict(list)
    per_seed_results = {}

    all_lc_data = {}

    for seed in seeds:
        if seed in failed_seeds:
            continue

        metrics = sklearn_metrics.get(seed, {})
        resource_data = sklearn_resource.get(seed, {})
        lc_data = sklearn_lc.get(seed, {})

        per_seed_results[f"seed_{seed}"] = {
            "metrics": metrics,
            "resource": resource_data,
            "learning_curves": lc_data,
            "subprocess_wall_time_sec": seed_wall_times.get(seed)
        }

        for model_name, m in metrics.items():
            for metric_name, value in m.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    all_metrics[f"{model_name}_{metric_name}"].append(value)

        for k in ["total_training_time_sec", "peak_driver_memory_mb", "cpu_cores_available",
                  "dataset_train_rows", "dataset_val_rows", "dataset_test_rows",
                  "dataset_total_genes", "dataset_selected_genes", "dataset_size_bytes"]:
            if k in resource_data and isinstance(resource_data[k], (int, float)):
                all_resource[k].append(resource_data[k])

        if seed in seed_wall_times:
            all_resource["subprocess_wall_time_sec"].append(seed_wall_times[seed])

        for model_name, lc in lc_data.items():
            if model_name not in all_lc_data:
                all_lc_data[model_name] = {
                    "n_samples": lc.get("n_samples", []),
                    "train_scores": [],
                    "val_scores": [],
                }
            if lc.get("train_scores"):
                all_lc_data[model_name]["train_scores"].append(lc["train_scores"])
            if lc.get("val_scores"):
                all_lc_data[model_name]["val_scores"].append(lc["val_scores"])

    if not per_seed_results:
        logger.error("No successful sklearn seeds to compile a report from.")
        return

    metric_rows = [{"Metric": k, "Mean": np.mean(v), "Std": np.std(v),
                    "Min": min(v), "Max": max(v), "N": len(v)} for k, v in all_metrics.items()]
    metric_df = pd.DataFrame(metric_rows).sort_values("Metric")

    resource_rows = [{"Resource": k, "Mean": np.mean(v), "Std": np.std(v),
                      "Min": min(v), "Max": max(v), "N": len(v)} for k, v in all_resource.items()]
    resource_df = pd.DataFrame(resource_rows).sort_values("Resource")

    agg_dir = os.path.join(method_output_dir, "aggregated_sklearn")
    os.makedirs(agg_dir, exist_ok=True)

    metric_df.to_csv(os.path.join(agg_dir, "aggregated_results_sklearn.csv"), index=False)
    resource_df.to_csv(os.path.join(agg_dir, "resource_summary_sklearn.csv"), index=False)

    method_name = os.path.basename(method_output_dir.rstrip("/"))
    final_report = {
        "engine": "sklearn",
        "feature_method": method_name,
        "n_seeds_requested": len(seeds),
        "n_seeds_succeeded": len(per_seed_results),
        "failed_seeds": failed_seeds,
        "per_seed": per_seed_results,
        "metric_summary": metric_df.to_dict(orient="records"),
        "resource_summary": resource_df.to_dict(orient="records"),
        "learning_curves_aggregated": all_lc_data,
        "generated_at": datetime.now().isoformat()
    }
    with open(os.path.join(agg_dir, "final_report_sklearn.json"), "w") as f:
        json.dump(final_report, f, indent=2, default=str)

    plot_aggregated_learning_curves(all_lc_data, agg_dir, method_name, "sklearn")

    print("\n" + "=" * 100)
    print(" " * 25 + f"SKLEARN ENGINE FINAL REPORT — {FEATURE_METHOD_LABELS.get(method_name, method_name).upper()} (with GBT, tuned)")
    print("=" * 100)
    print(f"Seeds requested: {len(seeds)} | Succeeded: {len(per_seed_results)} | "
          f"Failed: {failed_seeds if failed_seeds else 'none'}")

    print("\n▶ SKLEARN MODEL PERFORMANCE (Mean ± SD)")
    print("-" * 100)
    for _, row in metric_df.iterrows():
        if "stacking_ensemble" in row["Metric"]:
            print(f"  {row['Metric']:<48} {row['Mean']:>10.4f} ± {row['Std']:>8.4f} "
                  f"[{row['Min']:.4f}, {row['Max']:.4f}]  n={int(row['N'])}")

    print("\n▶ SKLEARN RESOURCE USAGE (Mean ± SD)")
    print("-" * 100)
    for _, row in resource_df.iterrows():
        print(f"  {row['Resource']:<35} {row['Mean']:>12.2f} ± {row['Std']:>8.2f} "
              f"[{row['Min']:.2f}, {row['Max']:.2f}]")

    print("=" * 100)
    logger.info(f"\n✓ Sklearn report saved to {os.path.join(agg_dir, 'final_report_sklearn.json')}")

    if len(metric_df) > 0:
        plot_aggregated_results(metric_df, agg_dir, method_name + " (sklearn)")


def compile_final_report(seeds: List[int], method_output_dir: str, seed_wall_times: Dict[int, float]):
    all_metrics = defaultdict(list)
    all_resource = defaultdict(list)
    all_lc_data = {}
    per_seed_results = {}
    failed_seeds = []

    for seed in seeds:
        seed_dir = os.path.join(method_output_dir, f"seed_{seed}")
        metrics_path = os.path.join(seed_dir, "metrics.json")
        resource_path = os.path.join(seed_dir, "resource_metrics.json")
        config_path = os.path.join(seed_dir, "config.json")
        lc_path = os.path.join(seed_dir, "learning_curves.json")

        if os.path.exists(os.path.join(seed_dir, "FAILED")) or not os.path.exists(metrics_path):
            failed_seeds.append(seed)
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)
        config = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
        resource_data = {}
        if os.path.exists(resource_path):
            with open(resource_path) as f:
                resource_data = json.load(f)
        lc_data = {}
        if os.path.exists(lc_path):
            with open(lc_path) as f:
                lc_data = json.load(f)

        per_seed_results[f"seed_{seed}"] = {
            "metrics": metrics, "config": config, "resource": resource_data,
            "learning_curves": lc_data,
            "subprocess_wall_time_sec": seed_wall_times.get(seed)
        }

        for model_name, m in metrics.items():
            for metric_name, value in m.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    all_metrics[f"{model_name}_{metric_name}"].append(value)

        for k in ["total_training_time_sec", "peak_driver_memory_mb", "cpu_cores_available",
                  "spark_default_parallelism", "dataset_train_rows", "dataset_val_rows",
                  "dataset_test_rows", "dataset_total_genes", "dataset_selected_genes",
                  "dataset_size_bytes"]:
            if k in resource_data and isinstance(resource_data[k], (int, float)):
                all_resource[k].append(resource_data[k])

        if seed in seed_wall_times:
            all_resource["subprocess_wall_time_sec"].append(seed_wall_times[seed])

        for model_name, lc in lc_data.items():
            if model_name not in all_lc_data:
                all_lc_data[model_name] = {
                    "n_samples": lc.get("n_samples", []),
                    "train_scores": [],
                    "val_scores": [],
                }
            if lc.get("train_scores"):
                all_lc_data[model_name]["train_scores"].append(lc["train_scores"])
            if lc.get("val_scores"):
                all_lc_data[model_name]["val_scores"].append(lc["val_scores"])

    if not per_seed_results:
        logger.error(f"No successful seeds to compile a report from.")
        return

    metric_rows = [{"Metric": k, "Mean": np.mean(v), "Std": np.std(v),
                    "Min": min(v), "Max": max(v), "N": len(v)} for k, v in all_metrics.items()]
    metric_df = pd.DataFrame(metric_rows).sort_values("Metric")

    resource_rows = [{"Resource": k, "Mean": np.mean(v), "Std": np.std(v),
                      "Min": min(v), "Max": max(v), "N": len(v)} for k, v in all_resource.items()]
    resource_df = pd.DataFrame(resource_rows).sort_values("Resource")

    agg_dir = os.path.join(method_output_dir, "aggregated")
    os.makedirs(agg_dir, exist_ok=True)

    metric_df.to_csv(os.path.join(agg_dir, "aggregated_results.csv"), index=False)
    resource_df.to_csv(os.path.join(agg_dir, "resource_summary.csv"), index=False)

    method_name = os.path.basename(method_output_dir.rstrip("/"))
    final_report = {
        "feature_method": method_name,
        "n_seeds_requested": len(seeds),
        "n_seeds_succeeded": len(per_seed_results),
        "failed_seeds": failed_seeds,
        "per_seed": per_seed_results,
        "metric_summary": metric_df.to_dict(orient="records"),
        "resource_summary": resource_df.to_dict(orient="records"),
        "learning_curves_aggregated": all_lc_data,
        "generated_at": datetime.now().isoformat()
    }
    with open(os.path.join(agg_dir, "final_report.json"), "w") as f:
        json.dump(final_report, f, indent=2, default=str)

    plot_aggregated_learning_curves(all_lc_data, agg_dir, method_name, "spark")

    print("\n" + "=" * 100)
    print(" " * 25 + f"FINAL COMPILED REPORT — {FEATURE_METHOD_LABELS.get(method_name, method_name).upper()} (with GBT)")
    print("=" * 100)
    print(f"Seeds requested: {len(seeds)} | Succeeded: {len(per_seed_results)} | "
          f"Failed: {failed_seeds if failed_seeds else 'none'}")

    print("\n▶ MODEL PERFORMANCE (Mean ± SD)")
    print("-" * 100)
    for _, row in metric_df.iterrows():
        if "stacking_ensemble" in row["Metric"]:
            print(f"  {row['Metric']:<48} {row['Mean']:>10.4f} ± {row['Std']:>8.4f} "
                  f"[{row['Min']:.4f}, {row['Max']:.4f}]  n={int(row['N'])}")

    print("\n▶ RESOURCE USAGE (Mean ± SD)")
    print("-" * 100)
    for _, row in resource_df.iterrows():
        print(f"  {row['Resource']:<35} {row['Mean']:>12.2f} ± {row['Std']:>8.2f} "
              f"[{row['Min']:.2f}, {row['Max']:.2f}]")

    print("=" * 100)
    logger.info(f"\n✓ Report saved to {os.path.join(agg_dir, 'final_report.json')}")

    if len(metric_df) > 0:
        plot_aggregated_results(metric_df, agg_dir, method_name)


def plot_aggregated_results(summary_df, output_dir: str, method_name: str = ""):
    key_metrics = ['stacking_ensemble_ROC-AUC', 'stacking_ensemble_PR-AUC',
                   'stacking_ensemble_F1-score', 'stacking_ensemble_Accuracy']
    plot_df = summary_df[summary_df['Metric'].isin(key_metrics)]
    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(plot_df))
    bars = ax.bar(x, plot_df['Mean'], 0.6, yerr=plot_df['Std'], capsize=5,
                  color='#2E86AB', edgecolor='black')
    for bar, mean_val, std_val in zip(bars, plot_df['Mean'], plot_df['Std']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
               f'{mean_val:.3f}±{std_val:.3f}', ha='center', va='bottom', fontsize=9)

    title = "Stacking Ensemble Performance Across Seeds (Mean ± SD)"
    if method_name:
        title += f" — {FEATURE_METHOD_LABELS.get(method_name, method_name)} (with GBT)"
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('stacking_ensemble_', '') for m in plot_df['Metric']])
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aggregated_performance.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================


def _load_json_if_exists(path: str) -> Optional[Dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def compile_engine_comparison(method_output_dir: str, seeds: List[int],
                               seed_wall_times: Dict[str, Dict[int, float]]):
    rows = []
    for seed in seeds:
        seed_dir = os.path.join(method_output_dir, f"seed_{seed}")
        spark_res = _load_json_if_exists(os.path.join(seed_dir, "resource_metrics.json"))

        sklearn_dir = os.path.join(seed_dir, "engine_sklearn")
        sklearn_res = _load_json_if_exists(os.path.join(sklearn_dir, "resource_metrics.json"))
        sklearn_status = "completed"
        if sklearn_res is None:
            sklearn_res = _load_json_if_exists(os.path.join(sklearn_dir, "resource_metrics_partial.json"))
            sklearn_status = "OOM_or_crashed_partial" if sklearn_res else "failed_no_data"

        spark_ran = "spark" in seed_wall_times and seed in seed_wall_times["spark"]
        sklearn_ran = "sklearn" in seed_wall_times and seed in seed_wall_times["sklearn"]

        rows.append({
            "seed": seed,
            "spark_wall_time_sec": seed_wall_times["spark"].get(seed) if spark_ran else None,
            "spark_peak_memory_mb": spark_res.get("peak_driver_memory_mb") if spark_res else None,
            "spark_status": "completed" if spark_res else ("not_run" if not spark_ran else "failed_no_data"),
            "sklearn_wall_time_sec": seed_wall_times["sklearn"].get(seed) if sklearn_ran else None,
            "sklearn_peak_memory_mb": sklearn_res.get("peak_driver_memory_mb") if sklearn_res else None,
            "sklearn_status": sklearn_status if sklearn_ran else "not_run",
            "sklearn_last_completed_step": sklearn_res.get("last_completed_step") if sklearn_res else None,
        })

    comparison_df = pd.DataFrame(rows)
    agg_dir = os.path.join(method_output_dir, "aggregated")
    os.makedirs(agg_dir, exist_ok=True)
    comparison_df.to_csv(os.path.join(agg_dir, "engine_comparison.csv"), index=False)

    has_spark_data = not comparison_df["spark_wall_time_sec"].isna().all()
    has_sklearn_data = not comparison_df["sklearn_wall_time_sec"].isna().all()

    if not has_spark_data or not has_sklearn_data:
        logger.warning("Incomplete engine comparison data - skipping plot")
        print("\n" + "=" * 100)
        print(" " * 25 + "ENGINE COMPARISON (INCOMPLETE - NEED BOTH ENGINES)")
        print("=" * 100)
        print(f"{'Seed':<8}{'Spark time(s)':>16}{'Spark peak MB':>16}{'sklearn time(s)':>18}"
              f"{'sklearn peak MB':>18}{'sklearn status':>22}")
        print("-" * 100)
        for _, r in comparison_df.iterrows():
            st = f"{r['sklearn_wall_time_sec']:.1f}" if pd.notna(r['sklearn_wall_time_sec']) else "n/a"
            sm = f"{r['sklearn_peak_memory_mb']:.0f}" if pd.notna(r['sklearn_peak_memory_mb']) else "n/a"
            spt = f"{r['spark_wall_time_sec']:.1f}" if pd.notna(r['spark_wall_time_sec']) else "n/a"
            spm = f"{r['spark_peak_memory_mb']:.0f}" if pd.notna(r['spark_peak_memory_mb']) else "n/a"
            print(f"{r['seed']:<8}{spt:>16}{spm:>16}{st:>18}{sm:>18}{r['sklearn_status']:>22}")
        print("=" * 100)
        print("\nTo get a complete comparison, run both engines together:")
        print("  python3 gbt.py --feature-method rf")
        print("=" * 100)
        return

    print("\n" + "=" * 100)
    print(" " * 25 + "SPARK vs. NO-SPARK (sklearn/pandas) ENGINE COMPARISON")
    print("=" * 100)
    print(f"{'Seed':<8}{'Spark time(s)':>16}{'Spark peak MB':>16}{'sklearn time(s)':>18}"
          f"{'sklearn peak MB':>18}{'sklearn status':>22}")
    print("-" * 100)
    for _, r in comparison_df.iterrows():
        st = f"{r['sklearn_wall_time_sec']:.1f}" if pd.notna(r['sklearn_wall_time_sec']) else "n/a"
        sm = f"{r['sklearn_peak_memory_mb']:.0f}" if pd.notna(r['sklearn_peak_memory_mb']) else "n/a"
        spt = f"{r['spark_wall_time_sec']:.1f}" if pd.notna(r['spark_wall_time_sec']) else "n/a"
        spm = f"{r['spark_peak_memory_mb']:.0f}" if pd.notna(r['spark_peak_memory_mb']) else "n/a"
        print(f"{r['seed']:<8}{spt:>16}{spm:>16}{st:>18}{sm:>18}{r['sklearn_status']:>22}")
    print("=" * 100)
    print("NOTE: if sklearn status is 'OOM_or_crashed_partial', 'sklearn peak MB' and wall time")
    print("reflect only progress up to 'sklearn_last_completed_step' before the process died.")
    print("=" * 100)

    plot_engine_comparison(comparison_df, agg_dir)
    logger.info(f"✓ Engine comparison saved to {os.path.join(agg_dir, 'engine_comparison.json')}")


def plot_engine_comparison(comparison_df: pd.DataFrame, output_dir: str):
    plot_df = comparison_df.dropna(subset=['spark_wall_time_sec', 'sklearn_wall_time_sec'])

    if len(plot_df) == 0:
        logger.warning("No complete data for engine comparison plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(plot_df))
    width = 0.35

    ax = axes[0]
    ax.bar(x - width/2, plot_df["spark_wall_time_sec"], width, label="Spark", color="#2E86AB")
    ax.bar(x + width/2, plot_df["sklearn_wall_time_sec"], width, label="sklearn (no Spark)", color="#F18F01")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["seed"].astype(str))
    ax.set_xlabel("Seed", fontsize=12)
    ax.set_ylabel("Wall time (s)", fontsize=12)
    ax.set_title("Training Time: Spark vs. No-Spark (both grid-search tuned)", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x - width/2, plot_df["spark_peak_memory_mb"], width, label="Spark", color="#2E86AB")
    ax.bar(x + width/2, plot_df["sklearn_peak_memory_mb"], width, label="sklearn (no Spark)", color="#F18F01")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["seed"].astype(str))
    ax.set_xlabel("Seed", fontsize=12)
    ax.set_ylabel("Peak driver/process memory (MB)", fontsize=12)
    ax.set_title("Peak Memory: Spark vs. No-Spark (both grid-search tuned)", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'engine_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"✓ Engine comparison plot saved to {os.path.join(output_dir, 'engine_comparison.png')}")


# =============================================================================


def discover_seed_dirs(method_output_dir: str) -> List[int]:
    if not os.path.isdir(method_output_dir):
        return []
    seeds = []
    for name in os.listdir(method_output_dir):
        if not name.startswith("seed_"):
            continue
        suffix = name[len("seed_"):]
        if not suffix.lstrip("-").isdigit():
            continue
        seed_dir = os.path.join(method_output_dir, name)
        has_metrics = os.path.exists(os.path.join(seed_dir, "metrics.json"))
        has_failed_marker = os.path.exists(os.path.join(seed_dir, "FAILED"))
        if has_metrics or has_failed_marker:
            seeds.append(int(suffix))
    return sorted(seeds)


def discover_methods(output_dir: str) -> List[str]:
    found = []
    for method in FEATURE_METHODS:
        method_dir = os.path.join(output_dir, method)
        if discover_seed_dirs(method_dir):
            found.append(method)
    return found


def _per_seed_metric_array(report: Dict, model_name: str, metric_name: str, seeds: List[int]) -> List[float]:
    values = []
    for seed in seeds:
        key = f"seed_{seed}"
        entry = report.get("per_seed", {}).get(key)
        if entry is None:
            values.append(np.nan)
            continue
        v = entry.get("metrics", {}).get(model_name, {}).get(metric_name)
        values.append(v if v is not None else np.nan)
    return values


def _paired_significance(a: List[float], b: List[float]) -> Dict[str, Any]:
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    result = {"n_paired": int(len(a)), "mean_diff": None, "ttest_p": None, "wilcoxon_p": None,
              "note": None}
    if len(a) < 2:
        result["note"] = "Fewer than 2 paired seeds available - cannot test."
        return result
    result["mean_diff"] = float(np.mean(a - b))
    try:
        t_stat, t_p = scipy_stats.ttest_rel(a, b)
        result["ttest_p"] = float(t_p)
    except Exception as e:
        result["note"] = f"t-test failed: {e}"
    if len(a) >= 5 and not np.all(a == b):
        try:
            w_stat, w_p = scipy_stats.wilcoxon(a, b)
            result["wilcoxon_p"] = float(w_p)
        except Exception as e:
            result["note"] = ((result["note"] or "") + f" wilcoxon failed: {e}").strip()
    else:
        result["note"] = ((result["note"] or "") +
                           " Wilcoxon skipped (need >=5 non-identical paired seeds).").strip()
    return result


def compile_method_comparison(output_dir: str, methods: List[str], seeds: List[int] = None):
    seeds = seeds or RANDOM_SEEDS
    comparison_rows = []
    per_method_reports = {}

    for method in methods:
        report_path = os.path.join(output_dir, method, "aggregated", "final_report.json")
        if not os.path.exists(report_path):
            logger.warning(f"No aggregated report for method={method} yet")
            continue
        with open(report_path) as f:
            report = json.load(f)
        per_method_reports[method] = report

        metric_summary = {row["Metric"]: row for row in report["metric_summary"]}
        row = {"Method": FEATURE_METHOD_LABELS.get(method, method), "method_key": method,
               "n_seeds": report["n_seeds_succeeded"]}
        for metric in ["ROC-AUC", "PR-AUC", "F1-score", "Accuracy", "Sensitivity", "Specificity", "Precision"]:
            key = f"stacking_ensemble_{metric}"
            if key in metric_summary:
                row[f"{metric}_mean"] = metric_summary[key]["Mean"]
                row[f"{metric}_std"] = metric_summary[key]["Std"]
        comparison_rows.append(row)

    if not comparison_rows:
        logger.error("No per-method aggregated reports found - nothing to compare.")
        return

    comparison_df = pd.DataFrame(comparison_rows)

    significance_results = {}
    test_metrics = ["ROC-AUC", "F1-score", "PR-AUC"]
    if "rf" in per_method_reports:
        for other in ["anova", "pca", "hybrid"]:
            if other not in per_method_reports:
                continue
            pair_key = f"rf_vs_{other}"
            significance_results[pair_key] = {}
            for metric in test_metrics:
                a = _per_seed_metric_array(per_method_reports["rf"], "stacking_ensemble", metric, seeds)
                b = _per_seed_metric_array(per_method_reports[other], "stacking_ensemble", metric, seeds)
                significance_results[pair_key][metric] = _paired_significance(a, b)

    agg_dir = os.path.join(output_dir, "aggregated")
    os.makedirs(agg_dir, exist_ok=True)
    comparison_df.to_csv(os.path.join(agg_dir, "method_comparison.csv"), index=False)

    with open(os.path.join(agg_dir, "method_comparison.json"), "w") as f:
        json.dump({
            "methods_compared": methods,
            "comparison_table": comparison_df.to_dict(orient="records"),
            "significance_tests": significance_results,
            "per_method_reports": per_method_reports,
            "generated_at": datetime.now().isoformat()
        }, f, indent=2, default=str)

    print("\n" + "=" * 100)
    print(" " * 30 + "CROSS-METHOD COMPARISON (Stacking Ensemble with GBT)")
    print("=" * 100)
    header = f"{'Method':<28} {'n':>3} {'ROC-AUC':>16} {'PR-AUC':>16} {'F1':>16} {'Accuracy':>16}"
    print(header)
    print("-" * 100)
    for _, row in comparison_df.iterrows():
        print(f"{row['Method']:<28} {int(row['n_seeds']):>3} "
              f"{row.get('ROC-AUC_mean', float('nan')):>8.4f}±{row.get('ROC-AUC_std', float('nan')):<6.4f} "
              f"{row.get('PR-AUC_mean', float('nan')):>8.4f}±{row.get('PR-AUC_std', float('nan')):<6.4f} "
              f"{row.get('F1-score_mean', float('nan')):>8.4f}±{row.get('F1-score_std', float('nan')):<6.4f} "
              f"{row.get('Accuracy_mean', float('nan')):>8.4f}±{row.get('Accuracy_std', float('nan')):<6.4f}")
    print("=" * 100)

    if significance_results:
        print("\n▶ SIGNIFICANCE TESTS (paired by seed; RF as reference)")
        print("-" * 100)
        for pair_key, metrics_dict in significance_results.items():
            print(f"  {pair_key}:")
            for metric, res in metrics_dict.items():
                tp = f"{res['ttest_p']:.4f}" if res["ttest_p"] is not None else "n/a"
                wp = f"{res['wilcoxon_p']:.4f}" if res["wilcoxon_p"] is not None else "n/a"
                md = f"{res['mean_diff']:+.4f}" if res["mean_diff"] is not None else "n/a"
                print(f"    {metric:<10} mean_diff={md:>10}  paired_t p={tp:>8}  wilcoxon p={wp:>8}"
                      f"  (n={res['n_paired']})")
        print("  NOTE: n=5 seeds is a small sample; treat p-values as indicative.")
        print("=" * 100)

    plot_method_comparison(comparison_df, agg_dir)
    logger.info(f"\n✓ Cross-method comparison saved to {os.path.join(agg_dir, 'method_comparison.json')}")


def plot_method_comparison(comparison_df: pd.DataFrame, output_dir: str):
    metrics = ["ROC-AUC", "PR-AUC", "F1-score", "Accuracy"]
    available = [m for m in metrics if f"{m}_mean" in comparison_df.columns]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(comparison_df))
    width = 0.8 / len(available)

    for i, metric in enumerate(available):
        offset = (i - len(available)/2 + 0.5) * width
        means = comparison_df[f"{metric}_mean"]
        stds = comparison_df[f"{metric}_std"]
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=metric)

    ax.set_xlabel('Feature Selection Method', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Stacking Ensemble Performance by Feature Selection Method (Mean ± SD)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Method'], rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# RESULTS 
# =============================================================================

def save_results(pipeline, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"\nSaving results to {output_dir}")

    config = {
        "feature_method": pipeline.feature_method,
        "optimal_threshold": pipeline.optimal_threshold,
        "num_selected_genes": pipeline.num_genes,
        "seed": pipeline.seed,
        "execution_times": pipeline.execution_times,
        "best_params": getattr(pipeline, "best_params", {}),
        "timestamp": datetime.now().isoformat()
    }

    results_to_save = [
        ("metrics.json", pipeline.metrics),
        ("config.json", config),
        ("selected_genes.json", pipeline.selected_genes),
    ]
    for filename, data in results_to_save:
        if data:
            with open(os.path.join(output_dir, filename), "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"  ✓ Saved {filename}")


def print_summary(pipeline):
    print("\n" + "=" * 100)
    print(" " * 25 + f"STACKING ENSEMBLE - {FEATURE_METHOD_LABELS.get(pipeline.feature_method, pipeline.feature_method).upper()} - WITH GBT - FINAL RESULTS")
    print("=" * 100)

    metrics_to_display = ['ROC-AUC', 'PR-AUC', 'Precision', 'Sensitivity', 'Specificity', 'F1-score', 'Accuracy']
    header = f"{'Model':<30}"
    for metric in metrics_to_display:
        header += f" {metric:>12}"
    print(header)
    print("-" * 100)

    if "stacking_ensemble" in pipeline.metrics:
        m = pipeline.metrics["stacking_ensemble"]
        row = "STACKING ENSEMBLE"
        for metric in metrics_to_display:
            row += f" {m.get(metric, 0):>12.4f}"
        print(row)
        print("-" * 100)

    for model_type in BASE_MODELS:
        if model_type in pipeline.metrics:
            m = pipeline.metrics[model_type]
            label = MODEL_LABELS.get(model_type, model_type).upper()
            row = label
            for metric in metrics_to_display:
                row += f" {m.get(metric, 0):>12.4f}"
            print(row)

    print("=" * 100)

    if "stacking_ensemble" in pipeline.metrics:
        ensemble = pipeline.metrics["stacking_ensemble"]
        base_candidates = [(name, pipeline.metrics[name]) for name in BASE_MODELS if name in pipeline.metrics]
        if base_candidates:
            best_base = max(base_candidates, key=lambda x: x[1].get('ROC-AUC', 0))
            if best_base[1].get('ROC-AUC', 0) > 0:
                improvement = (ensemble['ROC-AUC'] - best_base[1]['ROC-AUC']) / best_base[1]['ROC-AUC'] * 100
            else:
                improvement = 0.0
            print("\n▶ PERFORMANCE SUMMARY:")
            print(f"  Ensemble ROC-AUC: {ensemble['ROC-AUC']:.4f}")
            print(f"  Best Base Model: {best_base[0]} (ROC-AUC: {best_base[1]['ROC-AUC']:.4f})")
            print(f"  Improvement: {improvement:.1f}%")
            print(f"  Optimal Threshold: {ensemble.get('Threshold', 0.5):.4f}")
        total_time = sum(v for v in pipeline.execution_times.values() if isinstance(v, (int, float)))
        print(f"  Total Execution Time: {total_time:.1f}s ({total_time/60:.1f} min)")

        if hasattr(pipeline, 'learning_curves') and pipeline.learning_curves:
            print("\n▶ LEARNING CURVE SUMMARY (final point: train vs held-out val):")
            for model_type, lc_data in pipeline.learning_curves.items():
                if lc_data and lc_data.get('train_scores') and lc_data.get('val_scores'):
                    final_train = lc_data['train_scores'][-1]
                    final_val = lc_data['val_scores'][-1]
                    gap = final_train - final_val
                    print(f"  {MODEL_LABELS.get(model_type, model_type)}: "
                          f"train={final_train:.4f}  val={final_val:.4f}  gap={gap:+.4f}")

    print("=" * 100)


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_visualizations(pipeline, output_dir: str):
    logger.info("\nGenerating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    plot_combined_roc(pipeline, output_dir)
    plot_combined_pr(pipeline, output_dir)
    plot_confusion_matrices(pipeline, output_dir)
    plot_performance_comparison(pipeline, output_dir)
    plot_metrics_table(pipeline, output_dir)
    logger.info(f"✓ Visualizations saved to {output_dir}")


def _collect_label_prob(preds_df, cols):
    if isinstance(preds_df, pd.DataFrame):
        return {c: preds_df[c].to_numpy() for c in cols}
    rows = preds_df.select(*cols).collect()
    return {c: np.array([row[c] for row in rows]) for c in cols}


def plot_combined_roc(pipeline, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    for model_name, preds_df in pipeline.predictions.items():
        data = _collect_label_prob(preds_df, ["label", "prob_pos"])
        y_true, y_prob = data["label"], data["prob_pos"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = auc(fpr, tpr)
        label = MODEL_LABELS.get(model_name, model_name)
        color = COLORS.get(model_name, None)
        lw = 3 if model_name == 'stacking_ensemble' else 2
        ls = '-' if model_name == 'stacking_ensemble' else '--'
        ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc_val:.3f})', color=color, linewidth=lw, linestyle=ls)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - All Models (LR, RF, GBT, Stacking)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_combined_pr(pipeline, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    for model_name, preds_df in pipeline.predictions.items():
        data = _collect_label_prob(preds_df, ["label", "prob_pos"])
        y_true, y_prob = data["label"], data["prob_pos"]
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
        pr_auc_val = auc(recall_vals, precision_vals)
        label = MODEL_LABELS.get(model_name, model_name)
        color = COLORS.get(model_name, None)
        lw = 3 if model_name == 'stacking_ensemble' else 2
        ls = '-' if model_name == 'stacking_ensemble' else '--'
        ax.plot(recall_vals, precision_vals, label=f'{label} (AUC = {pr_auc_val:.3f})',
                color=color, linewidth=lw, linestyle=ls)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves (LR, RF, GBT, Stacking)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrices(pipeline, output_dir: str):
    n_models = len(pipeline.predictions)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for idx, (model_name, preds_df) in enumerate(pipeline.predictions.items()):
        if idx >= len(axes):
            break
        data = _collect_label_prob(preds_df, ["label", "pred"])
        y_true, y_pred = data["label"], data["pred"]
        cm = confusion_matrix(y_true, y_pred)
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(MODEL_LABELS.get(model_name, model_name), fontsize=12)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        metrics = pipeline.metrics.get(model_name, {})
        ax.text(0.5, -0.15, f"Acc: {metrics.get('Accuracy', 0):.3f}\nF1: {metrics.get('F1-score', 0):.3f}",
               transform=ax.transAxes, ha='center', va='top', fontsize=9)

    for idx in range(len(pipeline.predictions), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_performance_comparison(pipeline, output_dir: str):
    metrics_to_plot = ['ROC-AUC', 'PR-AUC', 'F1-score', 'Accuracy']
    data = []
    for model_name, metrics in pipeline.metrics.items():
        row = {'Model': MODEL_LABELS.get(model_name, model_name)}
        for metric in metrics_to_plot:
            row[metric] = metrics.get(metric, 0)
        data.append(row)
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.2
    for i, metric in enumerate(metrics_to_plot):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, df[metric], width, label=metric)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison (LR, RF, GBT, Stacking)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_metrics_table(pipeline, output_dir: str):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    headers = ['Model', 'ROC-AUC', 'PR-AUC', 'Sensitivity', 'Specificity', 'Precision', 'F1-score', 'Accuracy']

    table_data = []
    ensemble_metrics = pipeline.metrics.get('stacking_ensemble', {})
    row = ['STACKING ENSEMBLE']
    for header in headers[1:]:
        row.append(f"{ensemble_metrics.get(header, 0):.4f}")
    table_data.append(row)

    for model_type in BASE_MODELS:
        metrics = pipeline.metrics.get(model_type, {})
        row = [MODEL_LABELS.get(model_type, model_type).upper()]
        for header in headers[1:]:
            row.append(f"{metrics.get(header, 0):.4f}")
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', fontweight='bold')
        elif i % 2 == 1:
            cell.set_facecolor('#f5f5f5')
        if j == 0 and i > 0:
            cell.set_text_props(fontweight='bold')

    for j in range(1, len(headers)):
        col_values = [float(table_data[i][j]) for i in range(len(table_data))]
        if col_values:
            max_val = max(col_values)
            for i in range(len(table_data)):
                if float(table_data[i][j]) == max_val:
                    table.get_celld()[(i+1, j)].set_facecolor('#c7e9c0')

    plt.title(f"Model Performance Summary ({FEATURE_METHOD_LABELS.get(pipeline.feature_method, pipeline.feature_method)}) - WITH GBT",
              fontsize=16, fontweight='bold', pad=20)
    total_time = sum(v for v in pipeline.execution_times.values() if isinstance(v, (int, float)))
    ax.text(0.5, -0.05, f"Total Execution Time: {total_time:.1f}s ({total_time/60:.1f} min)",
           transform=ax.transAxes, ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Genomic stacking ensemble pipeline (with Gradient Boosting Trees)")
    parser.add_argument("--feature-method", type=str, choices=FEATURE_METHODS, default=None,
                        help="Which feature selection method this run is for: rf, anova, pca, or hybrid.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run a SINGLE (feature-method, seed) combination and exit.")
    parser.add_argument("--engine", type=str, choices=["spark", "sklearn"], default="spark",
                        help="Compute engine for a single run.")
    parser.add_argument("--with-sklearn-baseline", action="store_true",
                        help="When running all seeds, also run sklearn baseline.")
    parser.add_argument("--aggregate", action="store_true",
                        help="Compile results already on disk.")
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--output-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    if args.aggregate:
        if args.feature_method:
            method_output_dir = os.path.join(args.output_dir, args.feature_method)
            seeds_found = discover_seed_dirs(method_output_dir)
            if not seeds_found:
                logger.error(f"No seed_* result folders found under {method_output_dir}.")
                sys.exit(1)
            logger.info(f"Found results for method={args.feature_method}, seeds: {seeds_found}")
            compile_final_report(seeds_found, method_output_dir, seed_wall_times={})
            if any(os.path.exists(os.path.join(method_output_dir, f"seed_{s}", "engine_sklearn"))
                   for s in seeds_found):
                compile_engine_comparison(method_output_dir, seeds_found,
                                          {"spark": {}, "sklearn": {}})
        else:
            methods_found = discover_methods(args.output_dir)
            if not methods_found:
                logger.error(f"No feature-method result folders found.")
                sys.exit(1)
            logger.info(f"Found results for methods: {methods_found}")
            all_seeds = set()
            for method in methods_found:
                method_output_dir = os.path.join(args.output_dir, method)
                seeds_found = discover_seed_dirs(method_output_dir)
                all_seeds.update(seeds_found)
                compile_final_report(seeds_found, method_output_dir, seed_wall_times={})
            compile_method_comparison(args.output_dir, methods_found, seeds=sorted(all_seeds))
        return

    if args.seed is not None:
        if not args.feature_method:
            parser.error("--seed requires --feature-method {rf,anova,pca,hybrid}")
        logger.info(f"Worker starting: engine={args.engine} method={args.feature_method} seed={args.seed}")
        if args.engine == "sklearn":
            success = run_sklearn_seed_worker(args.feature_method, args.seed, args.data_path, args.output_dir)
        else:
            success = run_single_seed_worker(args.feature_method, args.seed, args.data_path, args.output_dir)
        sys.exit(0 if success else 1)

    if not args.feature_method:
        parser.error("Specify --feature-method {rf,anova,pca,hybrid} to run all seeds.")

    logger.info("=" * 70)
    logger.info(f"  GENOMIC STACKING ENSEMBLE PIPELINE - method={args.feature_method} (with GBT)")
    logger.info("  Gradient Boosting Trees replace Factorization Machines")
    logger.info("  sklearn engine now grid-search tuned, same param grids as Spark")
    logger.info("=" * 70)

    if args.engine == "spark" and not args.with_sklearn_baseline:
        engines = ("spark",)
    elif args.engine == "sklearn":
        engines = ("sklearn",)
    elif args.with_sklearn_baseline:
        engines = ("spark", "sklearn")
    else:
        engines = ("spark", "sklearn")

    logger.info(f"Running engines: {engines}")
    run_all_seeds_for_method(args.feature_method, RANDOM_SEEDS, args.data_path, args.output_dir, engines=engines)

    logger.info("\n" + "=" * 70)
    logger.info(f"✓ METHOD '{args.feature_method}' COMPLETED (with GBT)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()