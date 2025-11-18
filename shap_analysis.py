"""
Performs SHAP analysis: global feature importance for XGBoost, and five local case studies.
Local case selection includes:
- 2 high-confidence misclassifications (prob>0.8 or <0.2 and wrong)
- 2 model-disagreement cases (models disagree strongly)
- 1 borderline surprising case (highest absolute difference between model probs)

Outputs are saved under outputs/ (plots + CSV summaries)
"""
import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from src.utils import save_df


os.makedirs('outputs', exist_ok=True)


# load artifacts
preprocessor = joblib.load('models/preprocessor.joblib')
pca = joblib.load('models/pca.joblib')
log = joblib.load('models/logistic_pca.joblib')
xgb = joblib.load('models/xgb_prepped.joblib')


# load test set
test_df = pd.read_csv('outputs/test_set_with_preds.csv')
X_test = test_df.drop(columns=['target', 'log_proba', 'xgb_proba'])
y_test = test_df['target']


# prepare data for xgb (preprocessed matrix)
X_test_prep = preprocessor.transform(X_test)


# SHAP for XGBoost
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test_prep)


# Global importance
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X_test_prep, show=False)
plt.tight_layout()
plt.savefig('outputs/shap_summary_global.png')
plt.close()


# compute interesting local cases
xgb_proba = test_df['xgb_proba'].values
log_proba = test_df['log_proba'].values

# predictions
test_df['pred'] = (xgb_proba >= 0.5).astype(int)
test_df['wrong'] = (test_df['pred'] != y_test).astype(int)
test_df['conf'] = np.abs(xgb_proba - 0.5)

# 2 high-confidence misclassifications
high_conf_mis = (
    test_df[(test_df['wrong'] == 1) & (test_df['conf'] > 0.3)]
    .sort_values('conf', ascending=False)
    .head(2)
)

# 2 model disagreement cases
test_df['model_diff'] = np.abs(xgb_proba - log_proba)
model_disagree = (
    test_df.sort_values('model_diff', ascending=False)
    .head(2)
)

# 1 borderline surprising case
borderline = test_df.iloc[[test_df['model_diff'].idxmax()]]

# combine unique selected rows
selected_cases = pd.concat([high_conf_mis, model_disagree, borderline]).drop_duplicates()

# save
selected_cases.to_csv('outputs/shap_selected_cases.csv', index=False)


# Local SHAP plots
for idx in selected_cases.index:
    plt.figure(figsize=(7,5))
    shap.force_plot(
        explainer.expected_value,
        shap_values[idx],
        X_test_prep[idx],
        matplotlib=True,
        show=False
    )
    plt.savefig(f'outputs/shap_local_{idx}.png')
    plt.close()


print("SHAP analysis complete. Files saved in outputs/")
