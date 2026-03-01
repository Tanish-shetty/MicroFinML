"""
MicroFinML - Data Preprocessing Module
Handles data loading, cleaning, encoding, scaling, and train-test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import os


# Column definitions
NUMERIC_FEATURES = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
]

CATEGORICAL_FEATURES = [
    'Education', 'EmploymentType', 'MaritalStatus',
    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
]

TARGET = 'Default'
DROP_COLUMNS = ['LoanID']


def load_data(filepath):
    """Load the loan default dataset from CSV."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df):
    """Clean the dataset - drop ID column and handle any issues."""
    df = df.copy()
    # Drop LoanID - not a predictive feature
    df = df.drop(columns=DROP_COLUMNS, errors='ignore')
    # Drop duplicates if any
    initial_rows = len(df)
    df = df.drop_duplicates()
    dropped = initial_rows - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} duplicate rows")
    return df


def get_data_summary(df):
    """Generate a summary of the dataset."""
    summary = {
        'shape': df.shape,
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'target_distribution': df[TARGET].value_counts().to_dict() if TARGET in df.columns else None
    }
    return summary


def create_engineered_features(df):
    """Create new features from existing ones."""
    df = df.copy()

    # Income to Loan Amount ratio
    df['IncomeToLoanRatio'] = df['Income'] / (df['LoanAmount'] + 1)

    # Loan Amount to Income ratio (debt burden)
    df['LoanToIncomeRatio'] = df['LoanAmount'] / (df['Income'] + 1)

    # Monthly payment estimate (simplified)
    df['EstMonthlyPayment'] = df['LoanAmount'] / (df['LoanTerm'] + 1)

    # Payment to Income ratio
    df['PaymentToIncomeRatio'] = df['EstMonthlyPayment'] / (df['Income'] / 12 + 1)

    # Credit score bins
    df['CreditScoreGroup'] = pd.cut(
        df['CreditScore'],
        bins=[0, 400, 550, 700, 850],
        labels=['Poor', 'Fair', 'Good', 'Excellent']
    )

    # Age groups
    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[0, 25, 35, 50, 70],
        labels=['Young', 'Adult', 'MiddleAge', 'Senior']
    )

    # Employment stability (months employed relative to age)
    df['EmploymentStability'] = df['MonthsEmployed'] / (df['Age'] * 12 + 1)

    # Interest rate category
    df['InterestRateGroup'] = pd.cut(
        df['InterestRate'],
        bins=[0, 8, 15, 25],
        labels=['Low', 'Medium', 'High']
    )

    return df


def build_preprocessor(numeric_features, categorical_features):
    """Build a sklearn ColumnTransformer for preprocessing."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor


def split_data(df, target_col=TARGET, test_size=0.15, val_size=0.15, random_state=42):
    """Split data into train, validation, and test sets with stratification."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Default rate - Train: {y_train.mean():.4f}, Val: {y_val.mean():.4f}, Test: {y_test.mean():.4f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE to handle class imbalance on training data only."""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"Before SMOTE: {len(y_train)} samples, Default rate: {y_train.mean():.4f}")
    print(f"After SMOTE: {len(y_resampled)} samples, Default rate: {y_resampled.mean():.4f}")
    return X_resampled, y_resampled


def preprocess_pipeline(filepath, use_engineered=True, use_smote=True, save_dir=None):
    """
    Complete preprocessing pipeline.
    Returns preprocessed data ready for model training.
    """
    # Load and clean
    df = load_data(filepath)
    df = clean_data(df)

    # Feature engineering
    if use_engineered:
        df = create_engineered_features(df)

    # Define feature lists
    numeric_feats = NUMERIC_FEATURES.copy()
    categorical_feats = CATEGORICAL_FEATURES.copy()

    if use_engineered:
        numeric_feats += [
            'IncomeToLoanRatio', 'LoanToIncomeRatio', 'EstMonthlyPayment',
            'PaymentToIncomeRatio', 'EmploymentStability'
        ]
        categorical_feats += ['CreditScoreGroup', 'AgeGroup', 'InterestRateGroup']

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Build and fit preprocessor
    preprocessor = build_preprocessor(numeric_feats, categorical_feats)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after transformation
    num_feature_names = numeric_feats
    cat_feature_names = preprocessor.named_transformers_['cat'] \
        .named_steps['onehot'].get_feature_names_out(categorical_feats).tolist()
    all_feature_names = num_feature_names + cat_feature_names

    # Convert to DataFrames
    X_train_processed = pd.DataFrame(X_train_processed, columns=all_feature_names)
    X_val_processed = pd.DataFrame(X_val_processed, columns=all_feature_names)
    X_test_processed = pd.DataFrame(X_test_processed, columns=all_feature_names)

    # Reset index for y
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Apply SMOTE on training data
    if use_smote:
        X_train_processed, y_train = apply_smote(X_train_processed, y_train)

    # Save preprocessor and processed data
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(save_dir, 'preprocessor.pkl'))
        X_train_processed.to_csv(os.path.join(save_dir, 'X_train.csv'), index=False)
        X_val_processed.to_csv(os.path.join(save_dir, 'X_val.csv'), index=False)
        X_test_processed.to_csv(os.path.join(save_dir, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(save_dir, 'y_train.csv'), index=False)
        y_val.to_csv(os.path.join(save_dir, 'y_val.csv'), index=False)
        y_test.to_csv(os.path.join(save_dir, 'y_test.csv'), index=False)
        print(f"Processed data saved to {save_dir}")

    return {
        'X_train': X_train_processed,
        'X_val': X_val_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': all_feature_names
    }


if __name__ == "__main__":
    data = preprocess_pipeline(
        filepath='data/raw/Loan Default.csv',
        use_engineered=True,
        use_smote=True,
        save_dir='data/processed'
    )
    print(f"\nFinal training features shape: {data['X_train'].shape}")
    print(f"Feature names: {data['feature_names']}")
