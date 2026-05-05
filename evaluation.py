import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # prevents display errors in PyCharm
import os

os.makedirs("outputs", exist_ok=True)

def plot_auc_comparison(results_df):
    """Bar chart of AUC across M1-M4 for each classifier"""

    fig, ax = plt.subplots(figsize=(12, 6))

    models    = results_df['Model'].unique()
    classifiers = results_df['Classifier'].unique()
    x         = range(len(models))
    width     = 0.2
    colors    = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    for i, clf in enumerate(classifiers):
        subset = results_df[results_df['Classifier'] == clf]
        aucs   = [subset[subset['Model'] == m]['AUC'].values[0] for m in models]
        stds   = [subset[subset['Model'] == m]['AUC_std'].values[0] for m in models]
        offset = [xi + i * width for xi in x]
        ax.bar(offset, aucs, width, label=clf, color=colors[i],
               yerr=stds, capsize=4, alpha=0.85)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('CVD Risk Prediction: AUC Across Feature Sets (M1→M4)', fontsize=14)
    ax.set_xticks([xi + width * 1.5 for xi in x])
    ax.set_xticklabels(['M1\nClinical', 'M2\n+Metabolic',
                        'M3\n+ECG', 'M4\nMultimodal'], fontsize=11)
    ax.set_ylim(0.4, 0.9)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/auc_comparison.png', dpi=150)
    print("✅ Saved: outputs/auc_comparison.png")
    plt.close()


def plot_best_model_metrics(results_df):
    """Grouped bar chart of AUC, F1, Precision, Recall for best classifier"""

    #  Logistic Regression (best performer)
    lr = results_df[results_df['Classifier'] == 'LogisticRegression'].copy()

    metrics = ['AUC', 'F1', 'Precision', 'Recall']
    x       = range(len(lr))
    width   = 0.2
    colors  = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        offset = [xi + i * width for xi in x]
        ax.bar(offset, lr[metric], width, label=metric,
               color=colors[i], alpha=0.85)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Logistic Regression — All Metrics Across M1→M4', fontsize=14)
    ax.set_xticks([xi + width * 1.5 for xi in x])
    ax.set_xticklabels(['M1\nClinical', 'M2\n+Metabolic',
                        'M3\n+ECG', 'M4\nMultimodal'], fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/lr_metrics.png', dpi=150)
    print("✅ Saved: outputs/lr_metrics.png")
    plt.close()


def print_summary(results_df):
    print("\n BEST MODEL PER FEATURE SET (by AUC) ")
    best = results_df.loc[results_df.groupby('Model')['AUC'].idxmax()]
    print(best[['Model', 'Classifier', 'AUC', 'AUC_std',
                'F1', 'Precision', 'Recall']].to_string(index=False))

    print("\n── AUC IMPROVEMENT M1 → M4 (Logistic Regression) ──")
    lr = results_df[results_df['Classifier'] == 'LogisticRegression']
    m1_auc = lr[lr['Model'] == 'M1_Clinical']['AUC'].values[0]
    m4_auc = lr[lr['Model'] == 'M4_Multimodal']['AUC'].values[0]
    print(f"M1 Clinical:   {m1_auc:.3f}")
    print(f"M4 Multimodal: {m4_auc:.3f}")
    print(f"Improvement:   +{m4_auc - m1_auc:.3f}")