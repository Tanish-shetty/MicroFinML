"""
MicroFinML - Spark MLlib Model Training Module
Demonstrates distributed ML training using Spark MLlib.
Compares Spark-based models with local scikit-learn for scalability analysis.
"""

import os
import time
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.classification import (
    LogisticRegression as SparkLR,
    RandomForestClassifier as SparkRF,
    GBTClassifier as SparkGBT
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


def get_spark_models():
    """Return Spark MLlib models for training."""
    models = {
        'Spark_LogisticRegression': SparkLR(
            featuresCol='features',
            labelCol='Default',
            maxIter=100,
            regParam=0.01,
            elasticNetParam=0.0
        ),
        'Spark_RandomForest': SparkRF(
            featuresCol='features',
            labelCol='Default',
            numTrees=100,
            maxDepth=10,
            seed=42
        ),
        'Spark_GBT': SparkGBT(
            featuresCol='features',
            labelCol='Default',
            maxIter=50,
            maxDepth=6,
            stepSize=0.1,
            seed=42
        )
    }
    return models


def train_spark_model(model, train_df, model_name="Model"):
    """Train a single Spark MLlib model."""
    print(f"\nTraining {model_name} (Spark MLlib)...")
    start = time.time()
    fitted = model.fit(train_df)
    train_time = time.time() - start
    print(f"  {model_name} trained in {train_time:.2f}s")
    return fitted, train_time


def evaluate_spark_model(fitted_model, test_df, model_name="Model"):
    """Evaluate a Spark MLlib model on test data."""
    predictions = fitted_model.transform(test_df)

    # Binary classification evaluator (ROC-AUC)
    binary_eval = BinaryClassificationEvaluator(
        labelCol='Default', rawPredictionCol='rawPrediction',
        metricName='areaUnderROC'
    )
    roc_auc = binary_eval.evaluate(predictions)

    # Area under PR
    pr_eval = BinaryClassificationEvaluator(
        labelCol='Default', rawPredictionCol='rawPrediction',
        metricName='areaUnderPR'
    )
    pr_auc = pr_eval.evaluate(predictions)

    # Multiclass evaluator for accuracy, precision, recall, f1
    mc_eval = MulticlassClassificationEvaluator(
        labelCol='Default', predictionCol='prediction'
    )

    accuracy = mc_eval.evaluate(predictions, {mc_eval.metricName: 'accuracy'})
    precision = mc_eval.evaluate(predictions, {mc_eval.metricName: 'weightedPrecision'})
    recall = mc_eval.evaluate(predictions, {mc_eval.metricName: 'weightedRecall'})
    f1 = mc_eval.evaluate(predictions, {mc_eval.metricName: 'f1'})

    metrics = {
        'model_name': model_name,
        'accuracy': round(accuracy, 4),
        'weighted_precision': round(precision, 4),
        'weighted_recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4)
    }

    print(f"\n  --- {model_name} (Spark) ---")
    print(f"    Accuracy:           {metrics['accuracy']}")
    print(f"    Weighted Precision: {metrics['weighted_precision']}")
    print(f"    Weighted Recall:    {metrics['weighted_recall']}")
    print(f"    F1-Score:           {metrics['f1_score']}")
    print(f"    ROC-AUC:            {metrics['roc_auc']}")
    print(f"    PR-AUC:             {metrics['pr_auc']}")

    return metrics, predictions


def train_and_evaluate_all_spark(train_df, test_df, save_dir=None):
    """Train and evaluate all Spark MLlib models."""
    models = get_spark_models()
    all_results = {}

    for name, model in models.items():
        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")

        fitted, train_time = train_spark_model(model, train_df, name)
        metrics, predictions = evaluate_spark_model(fitted, test_df, name)
        metrics['train_time'] = round(train_time, 2)

        all_results[name] = {
            'model': fitted,
            'metrics': metrics,
            'train_time': train_time
        }

        # Save model if directory specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, name.lower())
            fitted.write().overwrite().save(model_path)
            print(f"  Model saved to {model_path}")

    return all_results


def generate_spark_comparison_table(spark_results, local_metrics_path=None):
    """
    Generate a comparison table between Spark MLlib and local scikit-learn models.
    This demonstrates scalability analysis.
    """
    # Spark results
    spark_rows = []
    for name, res in spark_results.items():
        m = res['metrics']
        spark_rows.append({
            'Framework': 'Spark MLlib',
            'Model': name.replace('Spark_', ''),
            'Accuracy': m['accuracy'],
            'Weighted_Precision': m['weighted_precision'],
            'Weighted_Recall': m['weighted_recall'],
            'F1_Score': m['f1_score'],
            'ROC_AUC': m['roc_auc'],
            'Train_Time_s': m['train_time']
        })

    spark_df = pd.DataFrame(spark_rows)

    # Load local results if available
    if local_metrics_path and os.path.exists(local_metrics_path):
        local_df = pd.read_csv(local_metrics_path, index_col=0)
        local_rows = []
        for model_name in local_df.index:
            local_rows.append({
                'Framework': 'scikit-learn (Local)',
                'Model': model_name,
                'Accuracy': local_df.loc[model_name, 'accuracy'],
                'Weighted_Precision': local_df.loc[model_name, 'precision'],
                'Weighted_Recall': local_df.loc[model_name, 'recall'],
                'F1_Score': local_df.loc[model_name, 'f1_score'],
                'ROC_AUC': local_df.loc[model_name, 'roc_auc'],
                'Train_Time_s': 0
            })
        local_pd = pd.DataFrame(local_rows)
        comparison = pd.concat([local_pd, spark_df], ignore_index=True)
    else:
        comparison = spark_df

    return comparison


def spark_feature_importance(fitted_rf_model, feature_names=None):
    """Extract feature importance from Spark Random Forest model."""
    if hasattr(fitted_rf_model, 'featureImportances'):
        importances = fitted_rf_model.featureImportances.toArray()
        if feature_names and len(feature_names) == len(importances):
            imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        else:
            imp_df = pd.DataFrame({
                'feature_index': range(len(importances)),
                'importance': importances
            }).sort_values('importance', ascending=False)
        return imp_df
    return None


if __name__ == "__main__":
    from src.spark_processing import create_spark_session, spark_preprocess_pipeline

    spark = create_spark_session()
    try:
        # Preprocess
        data = spark_preprocess_pipeline(spark, "data/raw/Loan Default.csv")

        # Select only needed columns for ML
        train_ml = data['train_df'].select("features", "Default")
        test_ml = data['test_df'].select("features", "Default")

        # Train and evaluate
        results = train_and_evaluate_all_spark(
            train_ml, test_ml,
            save_dir='models/spark'
        )

        # Comparison table
        comparison = generate_spark_comparison_table(
            results,
            local_metrics_path='results/metrics/model_performance.csv'
        )
        print("\n" + "=" * 70)
        print("FRAMEWORK COMPARISON: scikit-learn (Local) vs Spark MLlib")
        print("=" * 70)
        print(comparison.to_string(index=False))

        # Save comparison
        os.makedirs('results/metrics', exist_ok=True)
        comparison.to_csv('results/metrics/framework_comparison.csv', index=False)
        print("\nComparison saved to results/metrics/framework_comparison.csv")

    finally:
        spark.stop()
