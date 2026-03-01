"""
MicroFinML - Model Evaluation Module
Generates metrics, confusion matrices, ROC curves, and feature importance plots.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import os


plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def evaluate_model(model, X, y, model_name="Model"):
    """Evaluate a single model and return all metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1_score': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_prob),
        'avg_precision': average_precision_score(y, y_prob)
    }

    print(f"\n--- {model_name} ---")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Avg Prec:  {metrics['avg_precision']:.4f}")

    return metrics, y_pred, y_prob


def evaluate_all_models(models_dict, X, y):
    """Evaluate all models and return combined results."""
    all_metrics = []
    predictions = {}

    for name, model in models_dict.items():
        metrics, y_pred, y_prob = evaluate_model(model, X, y, name)
        all_metrics.append(metrics)
        predictions[name] = {'y_pred': y_pred, 'y_prob': y_prob}

    metrics_df = pd.DataFrame(all_metrics).set_index('model_name')
    return metrics_df, predictions


def plot_confusion_matrices(models_dict, X, y, save_dir=None):
    """Plot confusion matrices for all models side by side."""
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models_dict.items()):
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Repay', 'Default'],
            yticklabels=['Repay', 'Default']
        )
        ax.set_title(f'{name}\nConfusion Matrix', fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close('all')


def plot_roc_curves(models_dict, X, y, save_dir=None):
    """Plot ROC curves for all models on one chart."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#2196F3', '#4CAF50', '#FF5722']

    for (name, model), color in zip(models_dict.items(), colors):
        y_prob = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc = roc_auc_score(y, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close('all')


def plot_precision_recall_curves(models_dict, X, y, save_dir=None):
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#2196F3', '#4CAF50', '#FF5722']

    for (name, model), color in zip(models_dict.items(), colors):
        y_prob = model.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y, y_prob)
        ap = average_precision_score(y, y_prob)
        ax.plot(recall, precision, label=f'{name} (AP = {ap:.4f})', color=color, linewidth=2)

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'precision_recall_curves.png'), dpi=150, bbox_inches='tight')
    plt.close('all')


def plot_feature_importance(model, feature_names, model_name="Model", top_n=20, save_dir=None):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print(f"Cannot extract feature importance from {model_name}")
        return None

    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=feat_imp, x='importance', y='feature', ax=ax, palette='viridis')
    ax.set_title(f'{model_name} - Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f'{model_name.lower()}_feature_importance.png'),
            dpi=150, bbox_inches='tight'
        )
    plt.close('all')

    return feat_imp


def plot_model_comparison(metrics_df, save_dir=None):
    """Plot bar chart comparing all models across metrics."""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    plot_df = metrics_df[metrics_to_plot]

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.plot(kind='bar', ax=ax, width=0.7, edgecolor='black', linewidth=0.5)

    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close('all')


def generate_classification_reports(models_dict, X, y):
    """Generate and print classification reports for all models."""
    reports = {}
    for name, model in models_dict.items():
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, target_names=['Repay', 'Default'])
        reports[name] = report
        print(f"\n{'='*50}")
        print(f"Classification Report: {name}")
        print(f"{'='*50}")
        print(report)
    return reports


def save_metrics(metrics_df, save_path):
    """Save metrics DataFrame to CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metrics_df.to_csv(save_path)
    print(f"Metrics saved to {save_path}")


def full_evaluation(models_dict, X_test, y_test, feature_names, save_base_dir='results'):
    """Run complete evaluation pipeline."""
    fig_dir = os.path.join(save_base_dir, 'figures', 'model_comparison')
    feat_dir = os.path.join(save_base_dir, 'figures', 'feature_importance')
    metrics_dir = os.path.join(save_base_dir, 'metrics')

    # Evaluate all models
    metrics_df, predictions = evaluate_all_models(models_dict, X_test, y_test)

    # Classification reports
    generate_classification_reports(models_dict, X_test, y_test)

    # Save metrics
    save_metrics(metrics_df, os.path.join(metrics_dir, 'model_performance.csv'))

    # Plot everything
    plot_confusion_matrices(models_dict, X_test, y_test, save_dir=fig_dir)
    plot_roc_curves(models_dict, X_test, y_test, save_dir=fig_dir)
    plot_precision_recall_curves(models_dict, X_test, y_test, save_dir=fig_dir)
    plot_model_comparison(metrics_df, save_dir=fig_dir)

    # Feature importance for each model
    for name, model in models_dict.items():
        plot_feature_importance(model, feature_names, name, save_dir=feat_dir)

    return metrics_df, predictions
