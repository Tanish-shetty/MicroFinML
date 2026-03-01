"""
MicroFinML - Prediction Module
Provides inference pipeline for new loan applications.
"""

import pandas as pd
import numpy as np
import joblib
import os

from src.data_preprocessing import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    create_engineered_features
)


def load_model(model_path):
    """Load a trained model from disk."""
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def load_preprocessor(preprocessor_path):
    """Load the fitted preprocessor."""
    preprocessor = joblib.load(preprocessor_path)
    print(f"Preprocessor loaded from {preprocessor_path}")
    return preprocessor


def predict_single(applicant_data, model, preprocessor, use_engineered=True):
    """
    Predict default probability for a single loan applicant.

    Parameters:
        applicant_data: dict with keys matching dataset columns
        model: trained sklearn/xgboost model
        preprocessor: fitted ColumnTransformer
        use_engineered: whether to create engineered features

    Returns:
        dict with prediction (0/1), probability, and risk level
    """
    # Convert to DataFrame
    df = pd.DataFrame([applicant_data])

    # Feature engineering
    if use_engineered:
        df = create_engineered_features(df)

    # Preprocess
    X = preprocessor.transform(df)

    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    # Risk level
    default_prob = probability[1]
    if default_prob < 0.2:
        risk_level = "LOW"
    elif default_prob < 0.5:
        risk_level = "MEDIUM"
    elif default_prob < 0.75:
        risk_level = "HIGH"
    else:
        risk_level = "VERY HIGH"

    result = {
        'prediction': int(prediction),
        'prediction_label': 'DEFAULT' if prediction == 1 else 'REPAY',
        'default_probability': round(float(default_prob), 4),
        'repay_probability': round(float(probability[0]), 4),
        'risk_level': risk_level
    }

    return result


def predict_batch(df, model, preprocessor, use_engineered=True):
    """
    Predict default probability for a batch of loan applicants.

    Parameters:
        df: DataFrame with applicant data
        model: trained model
        preprocessor: fitted preprocessor
        use_engineered: whether to create engineered features

    Returns:
        DataFrame with predictions and probabilities
    """
    df_input = df.copy()

    if use_engineered:
        df_input = create_engineered_features(df_input)

    X = preprocessor.transform(df_input)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    results = df.copy()
    results['Prediction'] = predictions
    results['Prediction_Label'] = np.where(predictions == 1, 'DEFAULT', 'REPAY')
    results['Default_Probability'] = probabilities[:, 1].round(4)
    results['Repay_Probability'] = probabilities[:, 0].round(4)
    results['Risk_Level'] = pd.cut(
        probabilities[:, 1],
        bins=[0, 0.2, 0.5, 0.75, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH', 'VERY HIGH'],
        include_lowest=True
    )

    return results


def get_risk_summary(results_df):
    """Generate a risk summary from batch predictions."""
    summary = {
        'total_applications': len(results_df),
        'predicted_repay': (results_df['Prediction'] == 0).sum(),
        'predicted_default': (results_df['Prediction'] == 1).sum(),
        'default_rate': results_df['Prediction'].mean(),
        'avg_default_probability': results_df['Default_Probability'].mean(),
        'risk_distribution': results_df['Risk_Level'].value_counts().to_dict()
    }
    return summary


def demo_prediction():
    """Demo prediction with sample applicant data."""
    # Sample applicant
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

    # Load model and preprocessor
    model = load_model('models/xgboost_model.pkl')
    preprocessor = load_preprocessor('data/processed/preprocessor.pkl')

    # Predict
    result = predict_single(sample_applicant, model, preprocessor)

    print("\n" + "=" * 50)
    print("LOAN DEFAULT PREDICTION RESULT")
    print("=" * 50)
    print(f"Prediction: {result['prediction_label']}")
    print(f"Default Probability: {result['default_probability']:.2%}")
    print(f"Repay Probability: {result['repay_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print("=" * 50)

    return result


if __name__ == "__main__":
    demo_prediction()
