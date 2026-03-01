"""
MicroFinML - Complete BDA + ML Pipeline
Runs the FULL project pipeline including:
  1. Local ML Pipeline (preprocessing, training, evaluation)
  2. Spark Distributed Processing & MLlib Training
  3. Scalability Analysis (local vs Spark benchmarks)
  4. Blockchain Audit Trail Demo
  5. Literature Review Excel Generation
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

RAW_DATA = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Loan Default.csv')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


def phase1_local_ml():
    """Phase 1: Local scikit-learn + XGBoost pipeline."""
    from src.data_preprocessing import preprocess_pipeline
    from src.model_training import train_all_models
    from src.model_evaluation import full_evaluation

    print("\n" + "=" * 70)
    print("  PHASE 1: LOCAL ML PIPELINE (scikit-learn + XGBoost)")
    print("=" * 70)

    # Preprocess
    data = preprocess_pipeline(
        filepath=RAW_DATA,
        use_engineered=True,
        use_smote=True,
        save_dir=PROCESSED_DIR
    )

    # Train
    results = train_all_models(
        data['X_train'], data['y_train'],
        tune=False,
        save_dir=MODELS_DIR
    )

    # Evaluate
    models_dict = {name: res['model'] for name, res in results.items()}
    metrics_df, predictions = full_evaluation(
        models_dict,
        data['X_test'], data['y_test'],
        data['feature_names'],
        save_base_dir=RESULTS_DIR
    )

    print(f"\n  Local ML pipeline complete!")
    return data, metrics_df


def phase2_spark_pipeline():
    """Phase 2: Spark distributed processing + MLlib training."""
    from src.spark_processing import create_spark_session, spark_preprocess_pipeline
    from src.spark_ml_training import (
        train_and_evaluate_all_spark, generate_spark_comparison_table
    )

    print("\n" + "=" * 70)
    print("  PHASE 2: SPARK DISTRIBUTED PIPELINE (PySpark + Spark MLlib)")
    print("=" * 70)

    spark = create_spark_session(app_name="MicroFinML-BDA", master="local[4]")

    try:
        # Distributed preprocessing
        spark_data = spark_preprocess_pipeline(spark, RAW_DATA)

        # Select features for ML
        train_ml = spark_data['train_df'].select('features', 'Default')
        test_ml = spark_data['test_df'].select('features', 'Default')

        # Train Spark MLlib models
        spark_models_dir = os.path.join(MODELS_DIR, 'spark')
        spark_results = train_and_evaluate_all_spark(
            train_ml, test_ml, save_dir=spark_models_dir
        )

        # Compare frameworks
        local_metrics_path = os.path.join(RESULTS_DIR, 'metrics', 'model_performance.csv')
        comparison = generate_spark_comparison_table(spark_results, local_metrics_path)

        metrics_dir = os.path.join(RESULTS_DIR, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        comparison.to_csv(os.path.join(metrics_dir, 'framework_comparison.csv'), index=False)

        print(f"\n  Framework comparison saved.")
        print(comparison.to_string(index=False))

    finally:
        spark.stop()
        print("  SparkSession stopped.")

    return comparison


def phase3_scalability():
    """Phase 3: Scalability analysis benchmarks."""
    from src.spark_processing import create_spark_session
    from src.scalability_analysis import run_full_scalability_analysis

    print("\n" + "=" * 70)
    print("  PHASE 3: SCALABILITY ANALYSIS")
    print("=" * 70)

    spark = create_spark_session(app_name="MicroFinML-Scalability", master="local[4]")
    try:
        results = run_full_scalability_analysis(
            data_path=RAW_DATA,
            spark=spark,
            save_dir=RESULTS_DIR
        )
    finally:
        spark.stop()
        print("  SparkSession stopped.")

    return results


def phase4_blockchain():
    """Phase 4: Blockchain audit trail demo."""
    import joblib
    from src.blockchain_audit import LoanAuditChain
    from src.prediction import predict_single

    print("\n" + "=" * 70)
    print("  PHASE 4: BLOCKCHAIN AUDIT TRAIL")
    print("=" * 70)

    # Load model and preprocessor
    model = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
    preprocessor = joblib.load(os.path.join(PROCESSED_DIR, 'preprocessor.pkl'))

    chain = LoanAuditChain()

    # Score and record sample applications
    test_apps = [
        {'Age': 35, 'Income': 55000, 'LoanAmount': 25000, 'CreditScore': 680,
         'MonthsEmployed': 48, 'NumCreditLines': 3, 'InterestRate': 12.5,
         'LoanTerm': 36, 'DTIRatio': 0.35, 'Education': "Bachelor's",
         'EmploymentType': 'Full-time', 'MaritalStatus': 'Married',
         'HasMortgage': 'Yes', 'HasDependents': 'Yes',
         'LoanPurpose': 'Home', 'HasCoSigner': 'No'},
        {'Age': 22, 'Income': 18000, 'LoanAmount': 45000, 'CreditScore': 420,
         'MonthsEmployed': 6, 'NumCreditLines': 1, 'InterestRate': 22.0,
         'LoanTerm': 60, 'DTIRatio': 0.85, 'Education': 'High School',
         'EmploymentType': 'Part-time', 'MaritalStatus': 'Single',
         'HasMortgage': 'No', 'HasDependents': 'No',
         'LoanPurpose': 'Auto', 'HasCoSigner': 'No'},
        {'Age': 50, 'Income': 95000, 'LoanAmount': 30000, 'CreditScore': 750,
         'MonthsEmployed': 96, 'NumCreditLines': 4, 'InterestRate': 8.5,
         'LoanTerm': 24, 'DTIRatio': 0.2, 'Education': "Master's",
         'EmploymentType': 'Full-time', 'MaritalStatus': 'Married',
         'HasMortgage': 'Yes', 'HasDependents': 'Yes',
         'LoanPurpose': 'Education', 'HasCoSigner': 'Yes'},
        {'Age': 28, 'Income': 32000, 'LoanAmount': 80000, 'CreditScore': 380,
         'MonthsEmployed': 12, 'NumCreditLines': 2, 'InterestRate': 24.0,
         'LoanTerm': 48, 'DTIRatio': 0.78, 'Education': 'High School',
         'EmploymentType': 'Unemployed', 'MaritalStatus': 'Divorced',
         'HasMortgage': 'No', 'HasDependents': 'Yes',
         'LoanPurpose': 'Business', 'HasCoSigner': 'No'},
        {'Age': 42, 'Income': 75000, 'LoanAmount': 20000, 'CreditScore': 700,
         'MonthsEmployed': 60, 'NumCreditLines': 3, 'InterestRate': 10.0,
         'LoanTerm': 36, 'DTIRatio': 0.3, 'Education': "Bachelor's",
         'EmploymentType': 'Self-employed', 'MaritalStatus': 'Single',
         'HasMortgage': 'Yes', 'HasDependents': 'No',
         'LoanPurpose': 'Other', 'HasCoSigner': 'Yes'},
    ]

    print("\n  Recording loan decisions to blockchain...\n")
    for app in test_apps:
        result = predict_single(app, model, preprocessor)
        summary = {k: v for k, v in app.items()
                   if k in ['Age', 'Income', 'LoanAmount', 'CreditScore', 'LoanPurpose']}
        block = chain.add_decision(
            loan_data=summary,
            prediction=result['prediction_label'],
            risk_score=result['default_probability'],
            model_used='XGBoost'
        )
        print(f"    Block #{block.index}: {result['prediction_label']} "
              f"(risk: {result['default_probability']:.2%}) "
              f"[{block.hash[:24]}...]")

    # Validate
    is_valid, msg = chain.validate_chain()
    print(f"\n  Chain Validation: {'PASSED' if is_valid else 'FAILED'}")
    print(f"    {msg}")

    summary = chain.get_chain_summary()
    print(f"\n  Audit Summary: {summary['total_decisions']} decisions, "
          f"{summary['repay_count']} repay, {summary['default_count']} default")

    # Tamper detection
    print(f"\n  --- Tamper Detection Demo ---")
    orig = chain.chain[2].prediction
    chain.chain[2].prediction = "REPAY"
    valid, tmsg = chain.validate_chain()
    print(f"  After tampering Block #2: {'PASSED' if valid else 'FAILED'} — {tmsg}")
    chain.chain[2].prediction = orig
    chain.chain[2].hash = chain.chain[2].compute_hash()

    return chain


def phase5_literature():
    """Phase 5: Generate literature review Excel."""
    from src.create_literature_review import create_literature_review

    print("\n" + "=" * 70)
    print("  PHASE 5: LITERATURE REVIEW")
    print("=" * 70)
    create_literature_review()


def main():
    print("=" * 70)
    print("  MicroFinML — Complete BDA + ML Research Pipeline")
    print("  Data-Driven Intelligence for Sustainable Economics:")
    print("  Machine Learning for Micro-Financial Growth")
    print("=" * 70)

    # Phase 1: Local ML
    data, local_metrics = phase1_local_ml()

    # Phase 2: Spark Distributed ML
    try:
        comparison = phase2_spark_pipeline()
    except Exception as e:
        print(f"\n  [WARNING] Spark phase failed: {e}")
        print("  Spark code is ready — ensure Java 11+ and PySpark 3.5.x are installed.")
        comparison = None

    # Phase 3: Scalability Benchmarks
    try:
        scalability = phase3_scalability()
    except Exception as e:
        print(f"\n  [WARNING] Scalability phase (Spark portion) failed: {e}")
        print("  Running local-only scalability benchmarks...")
        from src.scalability_analysis import run_full_scalability_analysis
        scalability = run_full_scalability_analysis(
            data_path=RAW_DATA, spark=None, save_dir=RESULTS_DIR
        )

    # Phase 4: Blockchain Audit
    chain = phase4_blockchain()

    # Phase 5: Literature Review
    phase5_literature()

    # Final summary
    print("\n" + "=" * 70)
    print("  ALL PHASES COMPLETE!")
    print("=" * 70)
    print(f"\n  Generated outputs:")
    print(f"    data/processed/          — Preprocessed train/val/test splits")
    print(f"    models/                  — Local trained models (.pkl)")
    print(f"    models/spark/            — Spark MLlib models")
    print(f"    results/metrics/         — Performance CSVs + scalability data")
    print(f"    results/figures/         — All visualization plots")
    print(f"    reports/literature_review.xlsx — 35 categorized research papers")
    print(f"    reports/latex/main.tex   — Full 25-page Springer chapter template")
    print(f"\n  Notebooks (run interactively in Jupyter):")
    print(f"    01_data_exploration.ipynb   — EDA with visualizations")
    print(f"    02_data_preprocessing.ipynb — Feature engineering + SMOTE")
    print(f"    03_model_training.ipynb     — Model training + CV results")
    print(f"    04_model_evaluation.ipynb   — Metrics + plots + comparison")
    print(f"    05_bda_analysis.ipynb       — Spark, scalability, blockchain, impact")
    print("=" * 70)


if __name__ == "__main__":
    main()
