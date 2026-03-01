"""
MicroFinML - Model Training Module
Trains XGBoost, Random Forest, and Logistic Regression models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import joblib
import os
import time


def get_models(random_state=42):
    """Return a dictionary of models to train."""
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight='balanced',
            solver='lbfgs',
            n_jobs=-1
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=7.6,  # ratio of non-default to default (~225694/29653)
            random_state=random_state,
            eval_metric='logloss',
            n_jobs=-1
        )
    }
    return models


def get_param_grids():
    """Return hyperparameter search spaces for each model."""
    param_grids = {
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        },
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [3, 5, 10]
        },
        'XGBoost': {
            'n_estimators': [200, 300, 500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
    }
    return param_grids


def train_model(model, X_train, y_train, model_name="Model"):
    """Train a single model and return it with training time."""
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"{model_name} trained in {train_time:.2f} seconds")
    return model, train_time


def cross_validate_model(model, X_train, y_train, cv=5, scoring='roc_auc'):
    """Perform cross-validation and return scores."""
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    print(f"  CV {scoring}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def tune_hyperparameters(model, param_grid, X_train, y_train, n_iter=20, cv=3, random_state=42):
    """Tune hyperparameters using RandomizedSearchCV."""
    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring='roc_auc',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV ROC-AUC: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.best_score_


def train_all_models(X_train, y_train, tune=False, save_dir=None):
    """
    Train all models and return results.
    If tune=True, performs hyperparameter tuning (slower).
    """
    models = get_models()
    param_grids = get_param_grids()
    results = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        if tune:
            print("Tuning hyperparameters...")
            best_model, best_params, best_score = tune_hyperparameters(
                model, param_grids[name], X_train, y_train
            )
            trained_model = best_model
            train_time = 0  # included in tuning
        else:
            trained_model, train_time = train_model(model, X_train, y_train, name)

        # Cross-validation
        cv_scores = cross_validate_model(trained_model, X_train, y_train)

        results[name] = {
            'model': trained_model,
            'train_time': train_time,
            'cv_roc_auc_mean': cv_scores.mean(),
            'cv_roc_auc_std': cv_scores.std(),
            'cv_scores': cv_scores
        }

        if tune:
            results[name]['best_params'] = best_params

        # Save model
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f'{name.lower()}_model.pkl')
            joblib.dump(trained_model, model_path)
            print(f"  Model saved to {model_path}")

    return results


if __name__ == "__main__":
    # Load processed data
    processed_dir = 'data/processed'
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).squeeze()

    # Train all models
    results = train_all_models(X_train, y_train, tune=False, save_dir='models')

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for name, res in results.items():
        print(f"{name:25s} | CV ROC-AUC: {res['cv_roc_auc_mean']:.4f} (+/- {res['cv_roc_auc_std']:.4f})")
