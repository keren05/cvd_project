import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneOut
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs("outputs/stratified", exist_ok=True)

from models import CLINICAL_FEATURES, ECG_FEATURES, MULTIMODAL_FEATURES


def analyse_groups(df, group_col, group_label):
    """
    Reusable function — runs M1 vs M3 vs M4 for any grouping column
    """
    TARGET = 'CVD_broad'

    feature_sets = {
        'M1_Clinical':   CLINICAL_FEATURES,
        'M3_ECG':        ECG_FEATURES,
        'M4_Multimodal': MULTIMODAL_FEATURES
    }

    clf = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    )

    records = []
    print(f"\n── Stratified Analysis by {group_label} ──\n")

    for group_name in df[group_col].dropna().unique():
        group_df = df[df[group_col] == group_name]
        n_total  = len(group_df)
        n_cvd    = group_df[TARGET].sum()
        n_no_cvd = n_total - n_cvd

        print(f"{group_name}: n={n_total}, CVD={int(n_cvd)}, "
              f"no CVD={int(n_no_cvd)}")

        # Skip if not enough of both classes
        if n_cvd < 2 or n_no_cvd < 2:
            print(f"  ⚠️  Skipping — insufficient class representation\n")
            for ms in feature_sets:
                records.append({
                    'Group':      str(group_name),
                    'Group_type': group_label,
                    'Model':      ms,
                    'AUC':        np.nan,
                    'n':          n_total,
                    'n_cvd':      int(n_cvd)
                })
            continue

        y = group_df[TARGET]

        # Choose CV strategy based on group size
        if n_total <= 10:
            cv_strategy = LeaveOneOut()
            cv_label    = f"LOO-CV, n={n_total}"
        else:
            n_splits    = min(5, int(n_cvd))
            cv_strategy = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=42
            )
            cv_label = f"{n_splits}-fold"

        for model_name, features in feature_sets.items():
            X = group_df[features]

            pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',  StandardScaler()),
                ('clf',     clf)
            ])

            aucs = cross_val_score(
                pipe, X, y, cv=cv_strategy, scoring='roc_auc'
            )
            auc = aucs.mean()

            print(f"  {model_name:20s} AUC={auc:.3f} "
                  f"(±{aucs.std():.3f}, {cv_label})")

            records.append({
                'Group':      str(group_name),
                'Group_type': group_label,
                'Model':      model_name,
                'AUC':        round(auc, 3),
                'n':          n_total,
                'n_cvd':      int(n_cvd)
            })

        # Print ECG improvement
        group_records = [r for r in records if r['Group'] == str(group_name)]
        m1 = next((r['AUC'] for r in group_records
                   if r['Model'] == 'M1_Clinical'), np.nan)
        m3 = next((r['AUC'] for r in group_records
                   if r['Model'] == 'M3_ECG'), np.nan)
        if not (np.isnan(m1) or np.isnan(m3)):
            print(f"  → ECG improvement (M1→M3): +{m3 - m1:.3f}\n")
        else:
            print()

    return pd.DataFrame(records)


def plot_stratified(results_df, group_label, filename):
    """Bar chart of AUC by group for M1, M3, M4"""
    valid = results_df[results_df['AUC'].notna()]
    if len(valid) == 0:
        print(f"  ⚠️  No valid results to plot for {group_label}")
        return

    groups  = valid['Group'].unique()
    models  = ['M1_Clinical', 'M3_ECG', 'M4_Multimodal']
    x       = range(len(groups))
    width   = 0.25
    colors  = ['#FF9800', '#2196F3', '#4CAF50']

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        subset = valid[valid['Model'] == model]
        aucs   = []
        for g in groups:
            row = subset[subset['Group'] == g]
            aucs.append(row['AUC'].values[0] if len(row) > 0 else 0)

        offset = [xi + i * width for xi in x]
        ax.bar(offset, aucs, width, label=model.replace('_', ' '),
               color=colors[i], alpha=0.85)

    ax.set_xlabel(group_label, fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title(f'ECG Value by {group_label} (M1 vs M3 vs M4)', fontsize=13)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color='red', linestyle='--',
               alpha=0.4, label='Random baseline')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"outputs/stratified/{filename}", dpi=150)
    plt.close()
    print(f"✅ Saved: outputs/stratified/{filename}")


def run_stratified_analysis(df):

    all_results = []

    # ── ANALYSIS 1: Clinical HbA1c categories ────────────────────
    cat_results = analyse_groups(df, 'hba1c_cat', 'Glycaemic Category')
    all_results.append(cat_results)
    plot_stratified(cat_results, 'Glycaemic Category', 'stratified_category.png')

    # ── ANALYSIS 2: HbA1c tertiles ───────────────────────────────
    ter_results = analyse_groups(df, 'hba1c_tertile', 'HbA1c Tertile')
    all_results.append(ter_results)
    plot_stratified(ter_results, 'HbA1c Tertile', 'stratified_tertile.png')

    # ── COMBINE AND SAVE ─────────────────────────────────────────
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("outputs/stratified/stratified_results.csv", index=False)
    print("✅ Saved: outputs/stratified/stratified_results.csv")

    # ── SUMMARY TABLE ────────────────────────────────────────────
    print("\n── ECG Improvement Summary (M1→M3 AUC gain) ──")
    for _, grp in combined.groupby(['Group_type', 'Group']):
        m1 = grp[grp['Model'] == 'M1_Clinical']['AUC'].values
        m3 = grp[grp['Model'] == 'M3_ECG']['AUC'].values
        if len(m1) > 0 and len(m3) > 0 and not np.isnan(m1[0]) and not np.isnan(m3[0]):
            gain = m3[0] - m1[0]
            print(f"  {grp['Group_type'].values[0]:20s} | "
                  f"{str(grp['Group'].values[0]):15s} | "
                  f"M1={m1[0]:.3f} M3={m3[0]:.3f} gain=+{gain:.3f}")

    return combined