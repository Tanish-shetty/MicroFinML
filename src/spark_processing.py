"""
MicroFinML - Spark Data Processing Module
Demonstrates distributed data processing using PySpark for Big Data Analytics.
Handles data ingestion, cleaning, transformation, and feature engineering at scale.
"""

import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType, DoubleType
)
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)
from pyspark.ml import Pipeline


def create_spark_session(app_name="MicroFinML", master="local[*]"):
    """
    Initialize a SparkSession for distributed processing.
    In production, master would point to a YARN or Spark Standalone cluster.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.ui.showConsoleProgress", "false") \
        .config("spark.driver.maxResultSize", "1g") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print(f"SparkSession created: {app_name}")
    print(f"  Spark version: {spark.version}")
    print(f"  Master: {master}")
    print(f"  Cores: {spark.sparkContext.defaultParallelism}")
    return spark


def define_schema():
    """Define explicit schema for the Loan Default dataset."""
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
    return schema


def load_data_spark(spark, filepath):
    """
    Load CSV data into a Spark DataFrame.
    In production, this would read from HDFS or a distributed file system.
    Simulates: Ingestion Layer (Sqoop/Flume → HDFS → Spark)
    """
    start = time.time()
    schema = define_schema()

    df = spark.read.csv(filepath, header=True, schema=schema)
    load_time = time.time() - start

    print(f"\nData loaded into Spark DataFrame:")
    print(f"  Records: {df.count():,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Partitions: {df.rdd.getNumPartitions()}")
    print(f"  Load time: {load_time:.2f}s")

    return df, load_time


def spark_data_profiling(df):
    """
    Perform distributed data profiling using SparkSQL.
    Demonstrates: Processing Layer capabilities.
    """
    print("\n=== Spark Data Profiling ===")

    # Basic stats
    print("\n--- Numeric Summary (via Spark) ---")
    df.select("Age", "Income", "LoanAmount", "CreditScore",
              "InterestRate", "DTIRatio").summary().show()

    # Target distribution
    print("--- Target Distribution ---")
    df.groupBy("Default").count() \
        .withColumn("percentage", F.round(F.col("count") / df.count() * 100, 2)) \
        .orderBy("Default").show()

    # Missing values check
    print("--- Missing Values ---")
    null_counts = df.select([
        F.sum(F.when(F.isnull(c), 1).otherwise(0)).alias(c)
        for c in df.columns
    ])
    null_counts.show()

    # Category distributions
    cat_cols = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]
    for col in cat_cols:
        print(f"--- {col} Distribution ---")
        df.groupBy(col).agg(
            F.count("*").alias("count"),
            F.round(F.mean("Default"), 4).alias("default_rate")
        ).orderBy(F.desc("count")).show()


def spark_feature_engineering(df):
    """
    Distributed feature engineering using Spark transformations.
    Demonstrates: Spark DataFrame API for large-scale ETL.
    """
    start = time.time()

    df = df.drop("LoanID")

    # Engineered features (computed in parallel across partitions)
    df = df.withColumn("IncomeToLoanRatio",
                       F.col("Income") / (F.col("LoanAmount") + 1))
    df = df.withColumn("LoanToIncomeRatio",
                       F.col("LoanAmount") / (F.col("Income") + 1))
    df = df.withColumn("EstMonthlyPayment",
                       F.col("LoanAmount") / (F.col("LoanTerm") + 1))
    df = df.withColumn("PaymentToIncomeRatio",
                       F.col("EstMonthlyPayment") / (F.col("Income") / 12 + 1))
    df = df.withColumn("EmploymentStability",
                       F.col("MonthsEmployed") / (F.col("Age") * 12 + 1))

    # Credit score groups
    df = df.withColumn("CreditScoreGroup",
                       F.when(F.col("CreditScore") <= 400, "Poor")
                       .when(F.col("CreditScore") <= 550, "Fair")
                       .when(F.col("CreditScore") <= 700, "Good")
                       .otherwise("Excellent"))

    # Age groups
    df = df.withColumn("AgeGroup",
                       F.when(F.col("Age") <= 25, "Young")
                       .when(F.col("Age") <= 35, "Adult")
                       .when(F.col("Age") <= 50, "MiddleAge")
                       .otherwise("Senior"))

    # Interest rate groups
    df = df.withColumn("InterestRateGroup",
                       F.when(F.col("InterestRate") <= 8, "Low")
                       .when(F.col("InterestRate") <= 15, "Medium")
                       .otherwise("High"))

    eng_time = time.time() - start
    print(f"\nFeature engineering complete (Spark):")
    print(f"  New columns: 8 engineered features")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Time: {eng_time:.2f}s")

    return df, eng_time


def build_spark_ml_pipeline(df):
    """
    Build a Spark ML preprocessing pipeline.
    Demonstrates: Distributed ML Workflow (scaling across cluster).
    """
    # Define categorical columns to index and encode
    cat_cols = ["Education", "EmploymentType", "MaritalStatus",
                "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner",
                "CreditScoreGroup", "AgeGroup", "InterestRateGroup"]

    numeric_cols = ["Age", "Income", "LoanAmount", "CreditScore",
                    "MonthsEmployed", "NumCreditLines", "InterestRate",
                    "LoanTerm", "DTIRatio", "IncomeToLoanRatio",
                    "LoanToIncomeRatio", "EstMonthlyPayment",
                    "PaymentToIncomeRatio", "EmploymentStability"]

    # Stage 1: StringIndexer for each categorical column
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in cat_cols
    ]

    # Stage 2: OneHotEncoder
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
        for c in cat_cols
    ]

    # Stage 3: Assemble all features into a single vector
    assembler_inputs = numeric_cols + [f"{c}_vec" for c in cat_cols]
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features_raw")

    # Stage 4: Scale features
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                            withStd=True, withMean=True)

    # Build pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])

    return pipeline, numeric_cols, cat_cols


def spark_preprocess_pipeline(spark, filepath):
    """
    Complete Spark preprocessing pipeline.
    Returns train/test splits and fitted pipeline.
    """
    print("=" * 60)
    print("SPARK DISTRIBUTED PREPROCESSING PIPELINE")
    print("=" * 60)

    total_start = time.time()

    # Load data
    df, load_time = load_data_spark(spark, filepath)

    # Feature engineering
    df, eng_time = spark_feature_engineering(df)

    # Build ML pipeline
    pipeline, numeric_cols, cat_cols = build_spark_ml_pipeline(df)

    # Train/test split (stratified approximation using Spark)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Fit pipeline on training data
    start = time.time()
    fitted_pipeline = pipeline.fit(train_df)
    pipe_time = time.time() - start

    # Transform
    train_processed = fitted_pipeline.transform(train_df)
    test_processed = fitted_pipeline.transform(test_df)

    total_time = time.time() - total_start

    print(f"\n=== Spark Preprocessing Summary ===")
    print(f"  Training samples: {train_processed.count():,}")
    print(f"  Test samples: {test_processed.count():,}")
    print(f"  Load time: {load_time:.2f}s")
    print(f"  Engineering time: {eng_time:.2f}s")
    print(f"  Pipeline fit time: {pipe_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")

    return {
        'train_df': train_processed,
        'test_df': test_processed,
        'fitted_pipeline': fitted_pipeline,
        'raw_train': train_df,
        'raw_test': test_df,
        'timings': {
            'load': load_time,
            'engineering': eng_time,
            'pipeline_fit': pipe_time,
            'total': total_time
        }
    }


if __name__ == "__main__":
    spark = create_spark_session()
    try:
        result = spark_preprocess_pipeline(
            spark, "data/raw/Loan Default.csv"
        )
        spark_data_profiling(result['raw_train'])
    finally:
        spark.stop()
        print("\nSparkSession stopped.")
