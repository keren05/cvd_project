import pandas as pd
from load_data import load_clinical, load_sleep
from feature_engineering import engineer_clinical, extract_rr_features
import os

if __name__ == "__main__":

    # 1. LOAD
    clinical = load_clinical()
    obj_sleep, subj_sleep = load_sleep()

    # 2. FEATURE ENGINEERING
    clinical = engineer_clinical(clinical)
    rr_features = extract_rr_features()

    # 3. MERGE
    df = clinical.merge(rr_features, on='participant_id', how='left')

    # 4. MERGE SLEEP DATA
    obj_sleep = obj_sleep.rename(columns={'number': 'participant_id'})
    subj_sleep = subj_sleep.rename(columns={'number': 'participant_id'})

    # Drop duplicate columns that already exist in clinical
    obj_sleep = obj_sleep.drop(columns=['gender', 'age', 'height（cm）', 'weight（kg）'], errors='ignore')
    subj_sleep = subj_sleep.drop(columns=['gender', 'age'], errors='ignore')

    # Drop unnamed/empty columns from objective sleep
    obj_sleep = obj_sleep.loc[:, ~obj_sleep.columns.str.startswith('Unnamed')]

    df = df.merge(obj_sleep, on='participant_id', how='left')
    df = df.merge(subj_sleep, on='participant_id', how='left')
    # ── 4b. RENAME AMBIGUOUS COLUMNS ─────────────────────────────
    df = df.rename(columns={'age_x': 'age', 'gender_x': 'gender'})

    #  5. HANDLE MISSING RR DATA
    rr_cols = [c for c in df.columns if c.startswith(('DS_','REM_','RS_'))]
    missing_before = df[rr_cols].isnull().sum().sum()
    print(f"\nMissing RR values before imputation: {missing_before}")

    # Median imputation for missing RR features
    for col in rr_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    missing_after = df[rr_cols].isnull().sum().sum()
    print(f"Missing RR values after imputation: {missing_after}")

    #  6. FINAL CHECK
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"\nTarget distribution (CVD_broad):")
    print(df['CVD_broad'].value_counts())
    print(f"\nMissing values per column (top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    #  7. SAVE CLEAN DATASET
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/clean_dataset.csv", index=False)
    print("\n✅ Clean dataset saved to outputs/clean_dataset.csv")

    # 8. RUN MODELS
    from models import run_all_models

    print("\n" + "=" * 60)
    print("RUNNING ALL MODELS (M1 → M4)")
    print("=" * 60)
    print("\nAll column names in df:")
    print(df.columns.tolist())
    results = run_all_models(df)

    print("\n── FULL RESULTS TABLE ──")
    print(results.to_string(index=False))

    results.to_csv("outputs/model_results.csv", index=False)
    print("\n Results saved to outputs/model_results.csv")

    # 9. VISUALISATIONS
    from evaluation import plot_auc_comparison, plot_best_model_metrics, print_summary

    print_summary(results)
    plot_auc_comparison(results)
    plot_best_model_metrics(results)

    # 10. SHAP ANALYSIS
    # from shap_analysis import run_shap_analysis

    # print("\n" + "=" * 60)
    # print("RUNNING SHAP INTERPRETABILITY ANALYSIS")
    # print("=" * 60)

     #shap_df = run_shap_analysis(df)

    # 11. STRATIFIED ANALYSIS
    from stratified_analysis import run_stratified_analysis

    print("\n" + "=" * 60)
    print("RUNNING STRATIFIED ANALYSIS BY GLYCAEMIC GROUP")
    print("=" * 60)

    strat_results = run_stratified_analysis(df)

    #  12. ROC CURVES
    from roc_curves import plot_roc_curves, plot_roc_best_only

    print("\n" + "=" * 60)
    print("GENERATING ROC CURVES")
    print("=" * 60)

    plot_roc_curves(df)
    plot_roc_best_only(df)