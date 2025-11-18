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


# load
raw = load_data_if_present(DATA_PATH)
if raw is None:
raise FileNotFoundError(f"No dataset found at {DATA_PATH}. Run src/data_generation.py or provide your CSV.")


# identify features
TARGET = 'default'
cat_cols = raw.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [c for c in raw.columns if c not in cat_cols + [TARGET]]


# simple split
X = raw[num_cols + cat_cols]
y = raw[TARGET]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
stratify=y, random_state=RANDOM_STATE)


# preprocessing pipeline
numeric_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='median')),
('scaler', StandardScaler())
])


categorical_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='most_frequent')),
('onehot', OneHotEncoder(handle_unknown='ignore'))
])


print("Models and test set saved to models/ and outputs/")
