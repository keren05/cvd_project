import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os

os.makedirs("outputs/shap", exist_ok=True)

from models import MULTIMODAL_FEATURES, CLINICAL_FEATURES, ECG_FEATURES

def run_shap_analysis(df):
    TARGET = 'CVD_broad'
    y = df[TARGET]
    X = df[MULTIMODAL_FEATURES]

    # FIT PIPELINE
    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()

    X_imp    = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)
    X_scaled_df = pd.DataFrame(X_scaled, columns=MULTIMODAL_FEATURES)

    model = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    )
    model.fit(X_scaled, y)

    # ── SHAP VALUES ───────────────────────────────────────────────
    explainer   = shap.LinearExplainer(model, X_scaled_df)
    shap_values = explainer.shap_values(X_scaled_df)

    print("\n SHAP Analysis: M4 Multimodal (Logistic Regression) ")

    # ── PLOT 1: Summary Bar (mean absolute SHAP) ─────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_scaled_df,
        plot_type="bar",
        show=False,
        max_display=15
    )
    plt.title("Mean |SHAP Value| — Top 15 Features (M4 Multimodal)", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/shap/shap_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/shap/shap_bar.png")

    # ── PLOT 2: Beeswarm (impact direction) ──────────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_scaled_df,
        show=False,
        max_display=15
    )
    plt.title("SHAP Beeswarm — Feature Impact Direction (M4 Multimodal)", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/shap/shap_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/shap/shap_beeswarm.png")

    # ── PLOT 3: ECG vs Clinical feature importance comparison ─────
    shap_df = pd.DataFrame({
        'feature': MULTIMODAL_FEATURES,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    # Tag each feature by type
    ecg_only = [f for f in MULTIMODAL_FEATURES
                if f.startswith(('DS_', 'REM_', 'RS_'))]
    metabolic_only = ['HbA1c (%)', 'hba1c_cat_num', 'admission FBG (mmol/L)']

    def tag(f):
        if f in ecg_only:       return 'ECG'
        elif f in metabolic_only: return 'Metabolic'
        else:                     return 'Clinical'

    shap_df['type'] = shap_df['feature'].apply(tag)

    print("\nTop 10 features by SHAP importance:")
    print(shap_df.head(10).to_string(index=False))

    # Group by feature type
    type_summary = shap_df.groupby('type')['mean_abs_shap'].sum().reset_index()
    type_summary = type_summary.sort_values('mean_abs_shap', ascending=False)
    print("\nTotal SHAP contribution by feature type:")
    print(type_summary.to_string(index=False))

    # Plot feature type contribution
    colors = {'ECG': '#2196F3', 'Metabolic': '#4CAF50', 'Clinical': '#FF9800'}
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        type_summary['type'],
        type_summary['mean_abs_shap'],
        color=[colors[t] for t in type_summary['type']],
        alpha=0.85, edgecolor='white', linewidth=1.5
    )
    ax.set_title("Total SHAP Contribution by Feature Type", fontsize=13)
    ax.set_ylabel("Sum of Mean |SHAP Values|", fontsize=11)
    ax.set_xlabel("Feature Type", fontsize=11)
    for bar, val in zip(bars, type_summary['mean_abs_shap']):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/shap/shap_by_type.png", dpi=150)
    plt.close()
    print("Saved: outputs/shap/shap_by_type.png")

    return shap_df