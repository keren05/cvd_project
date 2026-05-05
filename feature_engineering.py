import numpy as np
import scipy.io
import pandas as pd
import os

DATA_DIR = "data"

# 1. CLINICAL FEATURES
def engineer_clinical(df):
    # BMI
    df['BMI'] = df['weight'] / (df['height'] / 100) ** 2

    # HbA1c glycaemic category
    def categorise_hba1c(val):
        if val < 5.7:   return 'Normal'
        elif val < 6.5: return 'Prediabetes'
        else:           return 'Diabetes'

    df['hba1c_cat'] = df['HbA1c (%)'].apply(categorise_hba1c)
    df['hba1c_cat_num'] = df['hba1c_cat'].map(
        {'Normal': 0, 'Prediabetes': 1, 'Diabetes': 2}
    )

    # CVD target - already binary
    # Check what's in the column
    print("\nCVD column raw values:")
    print(df['Coronary artery disease and cardiac insufficiency'].value_counts(dropna=False))

    # Fill NaN with 0 (no CVD recorded = no CVD) then convert
    df['CVD'] = df['Coronary artery disease and cardiac insufficiency'].fillna(0).astype(int)

    print(f"HbA1c groups:\n{df['hba1c_cat'].value_counts()}")
    print(f"\nCVD outcome:\n{df['CVD'].value_counts()}")
    # With only 3 CVD cases, we have two options:
    # Option A: Broaden the CVD definition to include related complications
    # Option B: Use SMOTE oversampling + focus on AUC not accuracy

    # Let's check a broader CVD target
    cvd_cols = [
        'Coronary artery disease and cardiac insufficiency',
        'Lower extremity atherosclerosis or stenosis',
        'Carotid plaque'
    ]

    df['CVD_broad'] = df[cvd_cols].fillna(0).max(axis=1).astype(int)
    print(f"\nBroad CVD outcome (includes atherosclerosis + carotid plaque):")
    print(df['CVD_broad'].value_counts())
    df['hba1c_tertile'] = pd.qcut(df['HbA1c (%)'], q=3, labels=['Low', 'Mid', 'High'])
    print(df['hba1c_tertile'].value_counts())
    print(df.groupby('hba1c_tertile')['CVD_broad'].sum())

    return df

#  2. HRV FEATURES FROM RR-INTERVAL FILES
def compute_hrv(rr_array):
    """
    Given a 1D array of RR intervals (in milliseconds),
    compute time-domain HRV features.
    """
    rr = rr_array.flatten()
    if len(rr) < 2:
        return {'SDNN': np.nan, 'RMSSD': np.nan,
                'mean_RR': np.nan, 'pNN50': np.nan}

    sdnn  = np.std(rr, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    mean_rr = np.mean(rr)
    nn50  = np.sum(np.abs(np.diff(rr)) > 50)
    pnn50 = (nn50 / len(rr)) * 100

    return {
        'SDNN':    round(sdnn, 4),
        'RMSSD':   round(rmssd, 4),
        'mean_RR': round(mean_rr, 4),
        'pNN50':   round(pnn50, 4)
    }

def extract_rr_features():
    """
    Loop through all RR-interval .mat files,
    compute HRV for each sleep stage, return a DataFrame.
    """
    rr_dir = os.path.join(DATA_DIR, "RR-interval")
    records = []

    for fname in sorted(os.listdir(rr_dir)):
        if not fname.endswith(".mat"):
            continue

        participant_id = int(fname.replace(".mat", ""))
        mat = scipy.io.loadmat(os.path.join(rr_dir, fname))

        row = {'participant_id': participant_id}

        for stage in ['DS', 'REM', 'RS']:
            hrv = compute_hrv(mat[stage])
            for metric, val in hrv.items():
                row[f'{stage}_{metric}'] = val

        records.append(row)

    rr_df = pd.DataFrame(records)
    print(f"\nRR features extracted: {rr_df.shape}")
    print(rr_df.head(3))

    return rr_df

