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
from src.utils import save_df # Keep if used later, but not strictly needed for the plots

os.makedirs('outputs', exist_ok=True)


# --- 1. Load Artifacts (Corrected Loading Names) ---
try:
    # Load the full pipelines as saved in preprocessing_and_modeling.py
    log_pipeline = joblib.load('models/logreg_model.pkl')
    xgb_pipeline = joblib.load('models/xgboost_model.pkl')
except FileNotFoundError:
    raise FileNotFoundError("Model pipelines (logreg_model.pkl, xgboost_model.pkl) not found. Run src/preprocessing_and_modeling.py first.")

# The XGBoost model is the last step in the pipeline
xgb_model = xgb_pipeline['classifier']

# --- 2. Load Test Set (Corrected Loading Name) ---
try:
    # Load test set saved in preprocessing_and_modeling.py
    test_df = pd.read_csv('outputs/test_set.csv')
except FileNotFoundError:
    raise FileNotFoundError("Test set (test_set.csv) not found. Run src/preprocessing_and_modeling.py first.")

X_test = test_df.drop(columns=['target'])
y_test = test_df['target']

# --- 3. Compute Probabilities (Needed for Case Selection) ---
# Re-calculate probabilities since the saved test set didn't have them
test_df['xgb_proba'] = xgb_pipeline.predict_proba(X_test)[:, 1]
test_df['log_proba'] = log_pipeline.predict_proba(X_test)[:, 1]
xgb_proba = test_df['xgb_proba'].values
log_proba = test_df['log_proba'].values

# --- 4. SHAP for XGBoost ---
# Apply the preprocessor (first step of the pipeline) to get the array for SHAP
# We isolate the preprocessor to get the transformed data for SHAP explainer
preprocessor = xgb_pipeline['preprocessor']
X_test_prep = preprocessor.transform(X_test)

# **CRITICAL CHANGE:** TreeExplainer must be used with the preprocessed array.
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_prep)

# --- 5. Global Importance Plot ---
# We use the preprocessed data, but we need feature names for the plot to be useful.
# Getting feature names from the ColumnTransformer is complex and requires extra code.
# For a basic plot on the preprocessed array:
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X_test_prep, feature_names=[f'PC_{i}' for i in range(X_test_prep.shape[1])], show=False)
plt.tight_layout()
plt.savefig('outputs/shap_summary_global.png')
plt.close()


# --- 6. Compute Interesting Local Cases ---
# predictions and confidence calculation
test_df['pred'] = (xgb_proba >= 0.5).astype(int)
test_df['wrong'] = (test_df['pred'] != y_test).astype(int)
test_df['conf'] = np.abs(xgb_proba - 0.5)

# 2 high-confidence misclassifications (conf > 0.3 implies prob > 0.8 or < 0.2)
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

# 1 borderline surprising case (Highest model difference)
# This logic is redundant if we already took the top 2 differences, but we'll take the max again
borderline = test_df.iloc[[test_df['model_diff'].idxmax()]]

# combine unique selected rows
selected_cases = pd.concat([high_conf_mis, model_disagree, borderline]).drop_duplicates()

# save selected original rows
save_df(selected_cases.drop(columns=['pred', 'wrong', 'conf']), 'outputs/shap_selected_cases.csv')


# --- 7. Local SHAP plots ---
# Use X_test (original features) for the force plot's visualization for better interpretability
for idx in selected_cases.index:
    # Get the index of the row in the preprocessed array (X_test_prep)
    row_in_prep_array = X_test.index.get_loc(idx)
    
    plt.figure(figsize=(7,5))
    # CRITICAL CHANGE: Use original features (X_test.iloc[[row_in_prep_array]]) as feature names for plotting
    # even though SHAP was calculated on the transformed array (X_test_prep)
    shap.force_plot(
        explainer.expected_value,
        shap_values[row_in_prep_array],
        X_test.iloc[[row_in_prep_array]], # Pass the original feature values for visualization
        matplotlib=True,
        show=False
    )
    plt.savefig(f'outputs/shap_local_{idx}.png', bbox_inches='tight')
    plt.close()


print("SHAP analysis complete. Files saved in outputs/")
