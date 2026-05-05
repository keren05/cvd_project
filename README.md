## CVD Risk Prediction in Type 2 Diabetes
# Overview

This project uses machine learning to predict cardiovascular disease (CVD) risk in patients with Type 2 Diabetes (T2DM).
It combines clinical data, HbA1c, and ECG-derived features to evaluate whether ECG improves prediction performance.

# Project Structure
```bash
cvd_project/
├── data/
│   ├── ECG/
│   ├── RR-interval/
│   ├── Clinical indicators.xlsx
│   ├── Objective sleep quality.xlsx
│   └── Subjective sleep quality.xlsx
│
├── outputs/
│   ├── roc/
│   ├── shap/
│   ├── stratified/
│   ├── auc_comparison.png
│   ├── lr_metrics.png
│   └── clean_dataset.csv
│
├── load_data.py
├── feature_engineering.py
├── models.py
├── evaluation.py
├── roc_curves.py
├── shap_analysis.py
├── stratified_analysis.py
├── main.py
├── .gitignore
└── README.md
```
# How to Run

```bash
git clone https://github.com/your-username/cvd_project.git
cd cvd_project
pip install -r requirements.txt

python main.py
```
# Methods

- **Models:** Logistic Regression, SVM, Random Forest, XGBoost  
- **Evaluation:** 5-fold cross-validation (AUC, F1, Recall)  
- **Feature Sets:**
  - **Clinical only:** HbA1c, ECG  
  - **Full multimodal**  
- **Interpretability:** SHAP used for model explainability  

# Results
**Best model (multimodal): AUC = 0.733
**Clinical baseline: AUC = 0.696

ECG features improved prediction more than HbA1c

# Notes
**Dataset: Cheng et al. (2023)
**Small sample size (N=60)
Results are exploratory
