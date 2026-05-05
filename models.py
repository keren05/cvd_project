import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── FEATURE SETS FOR M1 → M4 ─────────────────────────────────────

CLINICAL_FEATURES = [
    'age', 'BMI', 'SBP (mmHg)', 'DBP (mmHg)',
    'TG (mmol/L)', 'HDL-C (mmol/L)', 'LDL-C (mmol/L)',
    'BUN (mmol/L)', 'UACR (mg/g)'
]
METABOLIC_FEATURES = CLINICAL_FEATURES + [
    'HbA1c (%)', 'hba1c_cat_num',
    'admission FBG (mmol/L)'
]

ECG_FEATURES = CLINICAL_FEATURES + [
    'DS_SDNN', 'DS_RMSSD', 'DS_mean_RR', 'DS_pNN50',
    'REM_SDNN', 'REM_RMSSD', 'REM_mean_RR', 'REM_pNN50',
    'RS_SDNN', 'RS_RMSSD', 'RS_mean_RR', 'RS_pNN50'
]

MULTIMODAL_FEATURES = METABOLIC_FEATURES + [
    'DS_SDNN', 'DS_RMSSD', 'DS_mean_RR', 'DS_pNN50',
    'REM_SDNN', 'REM_RMSSD', 'REM_mean_RR', 'REM_pNN50',
    'RS_SDNN', 'RS_RMSSD', 'RS_mean_RR', 'RS_pNN50'
]

FEATURE_SETS = {
    'M1_Clinical':    CLINICAL_FEATURES,
    'M2_Metabolic':   METABOLIC_FEATURES,
    'M3_ECG':         ECG_FEATURES,
    'M4_Multimodal':  MULTIMODAL_FEATURES
}

# CLASSIFIERS

def get_classifiers():
    return {
        'LogisticRegression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42
        ),
        'SVM': SVC(
            probability=True, class_weight='balanced', random_state=42
        ),
        'XGBoost': XGBClassifier(
            scale_pos_weight=3,  # handles imbalance (45/15=3)
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
    }

# MAIN EVALUATION FUNCTION

def run_all_models(df):
    TARGET = 'CVD_broad'
    y = df[TARGET]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['roc_auc', 'f1', 'precision', 'recall']

    results = []

    for model_name, features in FEATURE_SETS.items():
        X = df[features]

        for clf_name, clf in get_classifiers().items():
            # Pipeline: impute → scale → classify
            pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',  StandardScaler()),
                ('clf',     clf)
            ])

            scores = cross_validate(
                pipe, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=False
            )

            results.append({
                'Model':     model_name,
                'Classifier': clf_name,
                'AUC':       round(scores['test_roc_auc'].mean(), 3),
                'AUC_std':   round(scores['test_roc_auc'].std(),  3),
                'F1':        round(scores['test_f1'].mean(),       3),
                'Precision': round(scores['test_precision'].mean(),3),
                'Recall':    round(scores['test_recall'].mean(),   3),
            })

            print(f"{model_name} | {clf_name:20s} | "
                  f"AUC={scores['test_roc_auc'].mean():.3f} "
                  f"(±{scores['test_roc_auc'].std():.3f})")

    results_df = pd.DataFrame(results)
    return results_df