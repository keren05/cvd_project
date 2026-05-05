import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import os

os.makedirs("outputs/roc", exist_ok=True)

from models import CLINICAL_FEATURES, METABOLIC_FEATURES, ECG_FEATURES, MULTIMODAL_FEATURES


def plot_roc_curves(df):
    TARGET = 'CVD_broad'
    y = df[TARGET]

    feature_sets = {
        'M1 Clinical':   CLINICAL_FEATURES,
        'M2 Metabolic':  METABOLIC_FEATURES,
        'M3 ECG':        ECG_FEATURES,
        'M4 Multimodal': MULTIMODAL_FEATURES
    }

    colors = {
        'M1 Clinical':   '#FF9800',
        'M2 Metabolic':  '#9C27B0',
        'M3 ECG':        '#2196F3',
        'M4 Multimodal': '#4CAF50'
    }

    clf = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fig, ax = plt.subplots(figsize=(9, 7))

    print("\n── ROC Curve AUC Summary ──")

    for model_name, features in feature_sets.items():
        X = df[features]

        # Collect predictions across all folds
        tprs     = []
        aucs     = []
        mean_fpr = np.linspace(0, 1, 100)

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train = X.iloc[train_idx]
            X_test  = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test  = y.iloc[test_idx]

            pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',  StandardScaler()),
                ('clf',     clf)
            ])

            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fold_auc    = auc(fpr, tpr)
            aucs.append(fold_auc)

            # Interpolate TPR at common FPR points
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        # Mean and std ROC curve
        mean_tpr        = np.mean(tprs, axis=0)
        mean_tpr[-1]    = 1.0
        mean_auc        = np.mean(aucs)
        std_auc         = np.std(aucs)
        std_tpr         = np.std(tprs, axis=0)

        print(f"  {model_name:15s} AUC = {mean_auc:.3f} (±{std_auc:.3f})")

        # Plot mean ROC
        ax.plot(
            mean_fpr, mean_tpr,
            color=colors[model_name],
            lw=2.5,
            label=f"{model_name} (AUC = {mean_auc:.3f} ±{std_auc:.3f})"
        )

        # Shaded confidence band
        ax.fill_between(
            mean_fpr,
            mean_tpr - std_tpr,
            mean_tpr + std_tpr,
            color=colors[model_name],
            alpha=0.08
        )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5,
            alpha=0.6, label='Random classifier')

    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves — CVD Risk Prediction (M1→M4)\n'
                 'Logistic Regression, 5-fold CV, N=60',
                 fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/roc/roc_curves.png', dpi=150)
    plt.close()
    print("✅ Saved: outputs/roc/roc_curves.png")


def plot_roc_best_only(df):
    """
    Clean single plot — M1 vs M4 only, for dissertation figure
    """
    TARGET = 'CVD_broad'
    y = df[TARGET]

    feature_sets = {
        'M1 Clinical (baseline)': CLINICAL_FEATURES,
        'M4 Multimodal (proposed)': MULTIMODAL_FEATURES
    }

    colors = {
        'M1 Clinical (baseline)':    '#FF9800',
        'M4 Multimodal (proposed)':  '#4CAF50'
    }

    clf = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    )

    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fig, ax = plt.subplots(figsize=(8, 7))

    for model_name, features in feature_sets.items():
        X        = df[features]
        tprs     = []
        aucs     = []
        mean_fpr = np.linspace(0, 1, 100)

        for train_idx, test_idx in cv.split(X, y):
            X_train = X.iloc[train_idx]
            X_test  = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test  = y.iloc[test_idx]

            pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',  StandardScaler()),
                ('clf',     clf)
            ])

            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            aucs.append(auc(fpr, tpr))

            interp_tpr    = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr     = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc     = np.mean(aucs)
        std_auc      = np.std(aucs)
        std_tpr      = np.std(tprs, axis=0)

        ax.plot(
            mean_fpr, mean_tpr,
            color=colors[model_name],
            lw=3,
            label=f"{model_name}\nAUC = {mean_auc:.3f} (±{std_auc:.3f})"
        )
        ax.fill_between(
            mean_fpr,
            mean_tpr - std_tpr,
            mean_tpr + std_tpr,
            color=colors[model_name],
            alpha=0.12
        )

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5,
            alpha=0.6, label='Random classifier')

    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curve: Clinical Baseline vs Multimodal Model\n'
                 'Logistic Regression, 5-fold CV, N=60',
                 fontsize=13)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/roc/roc_m1_vs_m4.png', dpi=150)
    plt.close()
    print("Saved: outputs/roc/roc_m1_vs_m4.png")