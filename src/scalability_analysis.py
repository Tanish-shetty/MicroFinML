"""
MicroFinML - Scalability Analysis Module
Benchmarks execution time vs dataset size for both local and Spark frameworks.
Demonstrates how the system scales with increasing data volume.
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

NUMERIC_FEATURES = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
]

CATEGORICAL_FEATURES = [
    'Education', 'EmploymentType', 'MaritalStatus',
    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
]


def _build_local_pipeline():
    """Build a simple sklearn preprocessing pipeline for benchmarking."""
    numeric_transformer = SkPipeline([('scaler', StandardScaler())])
    categorical_transformer = SkPipeline([
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ], remainder='drop')
    return preprocessor


def benchmark_local_models(df_full, sample_sizes, random_state=42):
    """
    Benchmark local scikit-learn models at different data sizes.
    Returns timing results for preprocessing + training + prediction.
    """
    models = {
        'LogisticRegression': LogisticRegression(max_iter=500, random_state=random_state, n_jobs=-1),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, random_state=random_state, n_jobs=-1, eval_metric='logloss')
    }

    results = []

    for size in sample_sizes:
        actual_size = min(size, len(df_full))
        sample = df_full.sample(n=actual_size, random_state=random_state)

        X = sample.drop(columns=['Default', 'LoanID'], errors='ignore')
        y = sample['Default']

        # Split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Preprocess
        preprocessor = _build_local_pipeline()
        start = time.time()
        X_train_p = preprocessor.fit_transform(X_train)
        X_test_p = preprocessor.transform(X_test)
        preprocess_time = time.time() - start

        for model_name, model in models.items():
            # Clone model for fresh training
            from sklearn.base import clone
            m = clone(model)

            # Train
            start = time.time()
            m.fit(X_train_p, y_train)
            train_time = time.time() - start

            # Predict
            start = time.time()
            y_prob = m.predict_proba(X_test_p)[:, 1]
            predict_time = time.time() - start

            auc = roc_auc_score(y_test, y_prob)

            results.append({
                'Framework': 'scikit-learn',
                'Model': model_name,
                'DataSize': actual_size,
                'Preprocess_Time': round(preprocess_time, 4),
                'Train_Time': round(train_time, 4),
                'Predict_Time': round(predict_time, 4),
                'Total_Time': round(preprocess_time + train_time + predict_time, 4),
                'ROC_AUC': round(auc, 4)
            })
            print(f"  [Local] {model_name} @ {actual_size:,} rows: "
                  f"train={train_time:.2f}s, AUC={auc:.4f}")

    return pd.DataFrame(results)


def benchmark_spark_models(spark, filepath, sample_fractions):
    """
    Benchmark Spark MLlib models at different data fractions.
    Returns timing results for distributed processing.
    """
    from pyspark.ml.classification import (
        LogisticRegression as SparkLR,
        RandomForestClassifier as SparkRF,
        GBTClassifier as SparkGBT
    )
    from pyspark.ml.feature import (
        StringIndexer, OneHotEncoder as SparkOHE,
        VectorAssembler, StandardScaler as SparkScaler
    )
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

    schema = StructType([
        StructField("LoanID", StringType(), True),
        StructField("Age", IntegerType(), True),
        StructField("Income", IntegerType(), True),
        StructField("LoanAmount", IntegerType(), True),
        StructField("CreditScore", IntegerType(), True),
        StructField("MonthsEmployed", IntegerType(), True),
        StructField("NumCreditLines", IntegerType(), True),
        StructField("InterestRate", DoubleType(), True),
        StructField("LoanTerm", IntegerType(), True),
        StructField("DTIRatio", DoubleType(), True),
        StructField("Education", StringType(), True),
        StructField("EmploymentType", StringType(), True),
        StructField("MaritalStatus", StringType(), True),
        StructField("HasMortgage", StringType(), True),
        StructField("HasDependents", StringType(), True),
        StructField("LoanPurpose", StringType(), True),
        StructField("HasCoSigner", StringType(), True),
        StructField("Default", IntegerType(), True),
    ])

    df_full = spark.read.csv(filepath, header=True, schema=schema)
    total_count = df_full.count()

    cat_cols = ["Education", "EmploymentType", "MaritalStatus",
                "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"]
    num_cols = ["Age", "Income", "LoanAmount", "CreditScore",
                "MonthsEmployed", "NumCreditLines", "InterestRate",
                "LoanTerm", "DTIRatio"]

    spark_models = {
        'Spark_LogisticRegression': SparkLR(featuresCol='features', labelCol='Default', maxIter=50),
        'Spark_RandomForest': SparkRF(featuresCol='features', labelCol='Default', numTrees=100, maxDepth=10, seed=42),
        'Spark_GBT': SparkGBT(featuresCol='features', labelCol='Default', maxIter=50, maxDepth=6, seed=42)
    }

    results = []

    for frac in sample_fractions:
        sample_df = df_full.sample(False, frac, seed=42)
        sample_count = sample_df.count()

        train_df, test_df = sample_df.randomSplit([0.8, 0.2], seed=42)

        # Build preprocessing pipeline
        indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
        encoders = [SparkOHE(inputCol=f"{c}_idx", outputCol=f"{c}_vec") for c in cat_cols]
        assembler_inputs = num_cols + [f"{c}_vec" for c in cat_cols]
        assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features_raw")
        scaler = SparkScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
        prep_pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])

        # Preprocess
        start = time.time()
        fitted_prep = prep_pipeline.fit(train_df)
        train_processed = fitted_prep.transform(train_df)
        test_processed = fitted_prep.transform(test_df)
        # Cache for reuse
        train_processed.cache()
        test_processed.cache()
        train_processed.count()  # Force materialization
        preprocess_time = time.time() - start

        evaluator = BinaryClassificationEvaluator(
            labelCol='Default', rawPredictionCol='rawPrediction', metricName='areaUnderROC'
        )

        for model_name, model in spark_models.items():
            # Train
            start = time.time()
            fitted_model = model.fit(train_processed)
            train_time = time.time() - start

            # Predict
            start = time.time()
            predictions = fitted_model.transform(test_processed)
            predictions.count()  # Force action
            predict_time = time.time() - start

            auc = evaluator.evaluate(predictions)

            results.append({
                'Framework': 'Spark MLlib',
                'Model': model_name,
                'DataSize': sample_count,
                'Preprocess_Time': round(preprocess_time, 4),
                'Train_Time': round(train_time, 4),
                'Predict_Time': round(predict_time, 4),
                'Total_Time': round(preprocess_time + train_time + predict_time, 4),
                'ROC_AUC': round(auc, 4)
            })
            print(f"  [Spark] {model_name} @ {sample_count:,} rows: "
                  f"train={train_time:.2f}s, AUC={auc:.4f}")

        train_processed.unpersist()
        test_processed.unpersist()

    return pd.DataFrame(results)


def plot_scalability_results(results_df, save_dir='results/figures/model_comparison'):
    """Generate scalability analysis plots."""
    os.makedirs(save_dir, exist_ok=True)

    # Plot 1: Training time vs data size (Local)
    local_df = results_df[results_df['Framework'] == 'scikit-learn']
    if not local_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for model in local_df['Model'].unique():
            mdf = local_df[local_df['Model'] == model]
            ax.plot(mdf['DataSize'], mdf['Train_Time'], marker='o', linewidth=2, label=model)
        ax.set_xlabel('Dataset Size (rows)', fontsize=12)
        ax.set_ylabel('Training Time (seconds)', fontsize=12)
        ax.set_title('Scalability Analysis: Training Time vs Dataset Size\n(scikit-learn Local)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'scalability_local.png'), dpi=150, bbox_inches='tight')
        plt.close('all')

    # Plot 2: Training time vs data size (Spark)
    spark_df = results_df[results_df['Framework'] == 'Spark MLlib']
    if not spark_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for model in spark_df['Model'].unique():
            mdf = spark_df[spark_df['Model'] == model]
            ax.plot(mdf['DataSize'], mdf['Train_Time'], marker='s', linewidth=2, linestyle='--', label=model)
        ax.set_xlabel('Dataset Size (rows)', fontsize=12)
        ax.set_ylabel('Training Time (seconds)', fontsize=12)
        ax.set_title('Scalability Analysis: Training Time vs Dataset Size\n(Spark MLlib Distributed)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'scalability_spark.png'), dpi=150, bbox_inches='tight')
        plt.close('all')

    # Plot 3: Framework comparison (combined)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Training time comparison
    ax = axes[0]
    for fw in results_df['Framework'].unique():
        fw_df = results_df[results_df['Framework'] == fw]
        avg_by_size = fw_df.groupby('DataSize')['Train_Time'].mean().reset_index()
        marker = 'o' if 'scikit' in fw else 's'
        ls = '-' if 'scikit' in fw else '--'
        ax.plot(avg_by_size['DataSize'], avg_by_size['Train_Time'],
                marker=marker, linewidth=2, linestyle=ls, label=fw)
    ax.set_xlabel('Dataset Size (rows)', fontsize=12)
    ax.set_ylabel('Avg Training Time (seconds)', fontsize=12)
    ax.set_title('Training Time: Local vs Distributed', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ROC-AUC comparison
    ax = axes[1]
    for fw in results_df['Framework'].unique():
        fw_df = results_df[results_df['Framework'] == fw]
        avg_by_size = fw_df.groupby('DataSize')['ROC_AUC'].mean().reset_index()
        marker = 'o' if 'scikit' in fw else 's'
        ls = '-' if 'scikit' in fw else '--'
        ax.plot(avg_by_size['DataSize'], avg_by_size['ROC_AUC'],
                marker=marker, linewidth=2, linestyle=ls, label=fw)
    ax.set_xlabel('Dataset Size (rows)', fontsize=12)
    ax.set_ylabel('Average ROC-AUC', fontsize=12)
    ax.set_title('Model Quality: Local vs Distributed', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Scalability Analysis: scikit-learn vs Spark MLlib', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scalability_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close('all')

    print(f"Scalability plots saved to {save_dir}")


def run_full_scalability_analysis(data_path, spark=None, save_dir='results'):
    """
    Run complete scalability benchmarking.
    Tests both local and Spark at multiple data sizes.
    """
    print("=" * 70)
    print("  SCALABILITY ANALYSIS")
    print("  Benchmarking Local (scikit-learn) vs Distributed (Spark MLlib)")
    print("=" * 70)

    # Load full dataset for local benchmarks
    df_full = pd.read_csv(data_path)

    # Sample sizes for benchmarking
    local_sizes = [10000, 25000, 50000, 100000, 200000, min(255000, len(df_full))]
    spark_fractions = [0.04, 0.1, 0.2, 0.4, 0.8, 1.0]

    # Local benchmarks
    print("\n--- Local (scikit-learn) Benchmarks ---")
    local_results = benchmark_local_models(df_full, local_sizes)

    # Spark benchmarks
    all_results = local_results
    if spark:
        print("\n--- Spark MLlib Benchmarks ---")
        spark_results = benchmark_spark_models(spark, data_path, spark_fractions)
        all_results = pd.concat([local_results, spark_results], ignore_index=True)

    # Save results
    metrics_dir = os.path.join(save_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    all_results.to_csv(os.path.join(metrics_dir, 'scalability_results.csv'), index=False)
    print(f"\nScalability results saved to {metrics_dir}/scalability_results.csv")

    # Generate plots
    fig_dir = os.path.join(save_dir, 'figures', 'model_comparison')
    plot_scalability_results(all_results, fig_dir)

    # Summary table
    print("\n" + "=" * 70)
    print("SCALABILITY SUMMARY")
    print("=" * 70)
    summary = all_results.groupby(['Framework', 'Model']).agg({
        'Train_Time': ['min', 'max', 'mean'],
        'ROC_AUC': 'mean'
    }).round(4)
    print(summary.to_string())

    return all_results


if __name__ == "__main__":
    from src.spark_processing import create_spark_session

    spark = create_spark_session(app_name="MicroFinML-Scalability")
    try:
        results = run_full_scalability_analysis(
            data_path="data/raw/Loan Default.csv",
            spark=spark,
            save_dir="results"
        )
    finally:
        spark.stop()
