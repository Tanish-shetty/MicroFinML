"""
MicroFinML - Complete ML Pipeline
Run this script to execute the full pipeline:
  1. Data Preprocessing (clean, engineer, encode, split, SMOTE)
  2. Model Training (Logistic Regression, Random Forest, XGBoost)
  3. Model Evaluation (metrics, plots, comparisons)
  4. Demo Prediction (sample applicant scoring)
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import preprocess_pipeline
from src.model_training import train_all_models
from src.model_evaluation import full_evaluation
from src.prediction import predict_single


def main():
    print("=" * 70)
    print("  MicroFinML - Loan Default Prediction Pipeline")
    print("  Data-Driven Intelligence for Sustainable Economics")
    print("=" * 70)

    raw_data_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Loan Default.csv')
    processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    results_dir = os.path.join(PROJECT_ROOT, 'results')

    # =========================================================
    # PHASE 1: DATA PREPROCESSING
    # =========================================================
    print("\n" + "=" * 70)
    print("  PHASE 1: DATA PREPROCESSING")
    print("=" * 70)

    data = preprocess_pipeline(
        filepath=raw_data_path,
        use_engineered=True,
        use_smote=True,
        save_dir=processed_dir
    )

    print(f"\nPreprocessing complete!")
    print(f"  Training samples: {data['X_train'].shape[0]:,}")
    print(f"  Validation samples: {data['X_val'].shape[0]:,}")
    print(f"  Test samples: {data['X_test'].shape[0]:,}")
    print(f"  Total features: {len(data['feature_names'])}")

    # =========================================================
    # PHASE 2: MODEL TRAINING
    # =========================================================
    print("\n" + "=" * 70)
    print("  PHASE 2: MODEL TRAINING")
    print("=" * 70)

    results = train_all_models(
        data['X_train'], data['y_train'],
        tune=False,
        save_dir=models_dir
    )

    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for name, res in results.items():
        print(f"  {name:25s} | CV ROC-AUC: {res['cv_roc_auc_mean']:.4f} (+/- {res['cv_roc_auc_std']:.4f})")

    # =========================================================
    # PHASE 3: MODEL EVALUATION
    # =========================================================
    print("\n" + "=" * 70)
    print("  PHASE 3: MODEL EVALUATION")
    print("=" * 70)

    # Extract trained models
    models_dict = {name: res['model'] for name, res in results.items()}

    metrics_df, predictions = full_evaluation(
        models_dict,
        data['X_test'], data['y_test'],
        data['feature_names'],
        save_base_dir=results_dir
    )

    # =========================================================
    # PHASE 4: BEST MODEL & DEMO PREDICTION
    # =========================================================
    print("\n" + "=" * 70)
    print("  PHASE 4: BEST MODEL & DEMO PREDICTION")
    print("=" * 70)

    best_model_name = metrics_df['roc_auc'].idxmax()
    best_model = models_dict[best_model_name]
    best_metrics = metrics_df.loc[best_model_name]

    print(f"\nBest Model: {best_model_name}")
    print(f"  Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    print(f"  F1-Score:  {best_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {best_metrics['roc_auc']:.4f}")

    # Demo prediction
    preprocessor = data['preprocessor']

    sample_applicant = {
        'Age': 35,
        'Income': 55000,
        'LoanAmount': 25000,
        'CreditScore': 680,
        'MonthsEmployed': 48,
        'NumCreditLines': 3,
        'InterestRate': 12.5,
        'LoanTerm': 36,
        'DTIRatio': 0.35,
        'Education': "Bachelor's",
        'EmploymentType': 'Full-time',
        'MaritalStatus': 'Married',
        'HasMortgage': 'Yes',
        'HasDependents': 'Yes',
        'LoanPurpose': 'Home',
        'HasCoSigner': 'No'
    }

    result = predict_single(sample_applicant, best_model, preprocessor)

    print(f"\n{'='*50}")
    print("DEMO: LOAN DEFAULT PREDICTION")
    print(f"{'='*50}")
    print(f"  Applicant: Age={sample_applicant['Age']}, Income=${sample_applicant['Income']:,}, "
          f"Loan=${sample_applicant['LoanAmount']:,}")
    print(f"  Credit Score: {sample_applicant['CreditScore']}, Employment: {sample_applicant['EmploymentType']}")
    print(f"\n  Prediction:          {result['prediction_label']}")
    print(f"  Default Probability: {result['default_probability']:.2%}")
    print(f"  Repay Probability:   {result['repay_probability']:.2%}")
    print(f"  Risk Level:          {result['risk_level']}")
    print(f"{'='*50}")

    # =========================================================
    # PIPELINE COMPLETE
    # =========================================================
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n  Processed data saved to: {processed_dir}")
    print(f"  Trained models saved to: {models_dir}")
    print(f"  Results & figures saved to: {results_dir}")
    print(f"  Best model: {best_model_name} (ROC-AUC: {best_metrics['roc_auc']:.4f})")
    print(f"\n  Next steps:")
    print(f"    - Open notebooks/ for interactive exploration")
    print(f"    - Check results/figures/ for all generated plots")
    print(f"    - Check results/metrics/ for performance data")
    print(f"    - Use src/prediction.py for new loan predictions")
    print("=" * 70)


if __name__ == "__main__":
    main()
