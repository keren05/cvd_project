import pandas as pd
import scipy.io
import numpy as np
import os

DATA_DIR = "data"

def load_clinical():
    path = os.path.join(DATA_DIR, "Clinical indicators.xlsx")
    df = pd.read_excel(path)
    print(f"Clinical data loaded: {df.shape}")
    print(df.columns.tolist())
    return df

def load_sleep():
    obj = pd.read_excel(os.path.join(DATA_DIR, "Objective sleep quality.xlsx"))
    subj = pd.read_excel(os.path.join(DATA_DIR, "Subjective sleep quality.xlsx"))
    print(f"Sleep data loaded: obj={obj.shape}, subj={subj.shape}")
    return obj, subj

def inspect_single_mat(folder="ECG"):
    """
    Opens  ONE .mat file to see structure
    before processing all 60
    """
    mat_dir = os.path.join(DATA_DIR, folder)
    first_file = sorted(os.listdir(mat_dir))[0]
    path = os.path.join(mat_dir, first_file)

    mat = scipy.io.loadmat(path)

    print(f"\nFile: {first_file}")
    print(f"Keys inside: {list(mat.keys())}")

    # Print shape of each key
    for key, val in mat.items():
        if not key.startswith("__"):
            print(f"  {key}: type={type(val)}, shape={np.array(val).shape}")

    return mat

if __name__ == "__main__":
    clinical = load_clinical()
    obj_sleep, subj_sleep = load_sleep()

    print("\n--- Inspecting one ECG .mat file ---")
    ecg_mat = inspect_single_mat("ECG")

    print("\n--- Inspecting one RR-interval .mat file ---")
    rr_mat = inspect_single_mat("RR-interval")