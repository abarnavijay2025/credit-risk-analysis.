"""
Loads dataset (data/dataset.csv), preprocesses it (imputation, one-hot, scaling, PCA),
trains LogisticRegression and XGBoost, evaluates and persists models and test split.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from src.utils import load_data_if_present, save_df


# config
DATA_PATH = 'data/dataset.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODELS_DIR = 'models'
OUTPUTS_DIR = 'outputs'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# load
raw = load_data_if_present(DATA_PATH)
if raw is None:
    raise FileNotFoundError(f"No dataset found at {DATA_PATH}. Run src/data_generation.py or provide your CSV.")


# identify features
TARGET = 'target' # Corrected target name to match data_generation.py
cat_cols = raw.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [c for c in raw.columns if c not in cat_cols + [TARGET]]


# simple split
X = raw[num_cols + cat_cols]
y = raw[TARGET]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                                    stratify=y, random_state=RANDOM_STATE)


# --- Preprocessing Pipeline Definition ---
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)) # Include PCA to retain 95% of variance
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create the ColumnTransformer to apply transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='passthrough'
)


# --- Full Model Pipelines ---

# 1. Logistic Regression Pipeline (Sensitive to Scaling/PCA)
logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=RANDOM_STATE))
])

# 2. XGBoost Pipeline (Less sensitive to Scaling/PCA, but still benefits from clean data)
# Note: XGBoost is an ensemble tree model, so scaling is often optional, but the preprocessing framework is kept consistent.
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE))
])


# --- Training and Evaluation ---

print("Starting Model Training...")

# Train Logistic Regression
logreg_pipeline.fit(X_train, y_train)
logreg_preds = logreg_pipeline.predict(X_test)
logreg_proba = logreg_pipeline.predict_proba(X_test)[:, 1]

# Train XGBoost
xgb_pipeline.fit(X_train, y_train)
xgb_preds = xgb_pipeline.predict(X_test)
xgb_proba = xgb_pipeline.predict_proba(X_test)[:, 1]


# --- Model Persistence (Saving) ---

# Save the trained models
joblib.dump(logreg_pipeline, os.path.join(MODELS_DIR, 'logreg_model.pkl'))
joblib.dump(xgb_pipeline, os.path.join(MODELS_DIR, 'xgboost_model.pkl'))

# Save the test set for later SHAP analysis (combining X_test and y_test)
X_test_saved = X_test.copy()
X_test_saved[TARGET] = y_test
save_df(X_test_saved, os.path.join(OUTPUTS_DIR, 'test_set.csv'))

print("\n--- Evaluation Summary (on Test Set) ---")
print(f"Logistic Regression ROC-AUC: {roc_auc_score(y_test, logreg_proba):.4f}")
print(f"XGBoost ROC-AUC: {roc_auc_score(y_test, xgb_proba):.4f}")

print(f"\nModels saved to {MODELS_DIR}/")
print(f"Test set saved to {OUTPUTS_DIR}/test_set.csv")
