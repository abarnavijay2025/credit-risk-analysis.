
# Credit Risk Analysis – Machine Learning Project

## 1. Project Overview

This project focuses on predicting credit risk using machine learning techniques.
The goal is to classify loan applicants as **low-risk** or **high-risk**, helping financial institutions make informed lending decisions.
The project includes:
* Data cleaning and preprocessing
* Exploratory data analysis (EDA)
* Model building and evaluation
* Comparison of multiple algorithms
* Final optimized model for credit risk prediction
* **Model interpretability using SHAP (Shapley Additive exPlanations)**

---

## 2. Dataset Description

The dataset (synthetic or proprietary) contains demographic, financial, and credit-related information.
Common features include:
* Age
* Income
* Employment details
* Loan amount
* Credit history
* Previous defaults

Target variable:
* **1 → High Risk** (Default)
* **0 → Low Risk** (No Default)

---

## 3. Methodology

### 3.1 Data Preprocessing

* Handled missing values (imputation)
* Encoded categorical variables (One-Hot Encoding)
* Scaled numerical features (StandardScaler)
* Dimensionality reduction using **PCA** (Principal Component Analysis)
* Balanced dataset using oversampling/undersampling (if applicable, though not in the base code)

### 3.2 Exploratory Data Analysis (EDA)

* Feature distributions
* Correlation heatmaps
* Risk patterns across variables

### 3.3 Model Development

Tested machine learning models such as:
* Logistic Regression
* Decision Tree
* Random Forest
* **XGBoost (Selected as the primary classification model)**

### 3.4 Model Evaluation

Performance measured using:
* Accuracy
* Precision, Recall, F1-Score
* **ROC-AUC Curve (Primary metric due to class imbalance)**

### 3.5 Model Interpretability (XAI) 🌟

* **Global Interpretation:** Used **SHAP Summary Plots** to identify the most important features driving the XGBoost model's predictions overall.
* **Local Interpretation:** Used **SHAP Force Plots** to explain individual predictions, specifically focusing on high-confidence misclassifications and cases with significant model disagreement.

---

## 4. Results Summary

* **Best model:** XGBoost
* **Achieved ROC-AUC:** XX% (Fill in your actual result here)
* High recall for detecting high-risk applicants
* Final models saved as `logreg_model.pkl` and `xgboost_model.pkl`

---

## 5. How to Run the Project

1.  **Clone the repository.**
2.  **Install dependencies** from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the project scripts sequentially** from the root directory:
    ```bash
    # 1. Generates data/dataset.csv
    python src/data_generation.py
    
    # 2. Preprocesses, trains models, and saves them
    python src/preprocessing_and_modeling.py
    
    # 3. Performs SHAP analysis and saves plots in outputs/
    python src/shap_analysis.py
    ```
4.  **View Results:** Check the `outputs/` directory for plots, evaluation metrics, and case summaries.

---

## 6. Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* **XGBoost**, **SHAP**
* Matplotlib, Seaborn, Plotly
* Jupyter Notebook

---

## 7. Future Improvements

* Deploy the model using Flask/Streamlit
* Implement automated hyperparameter tuning (e.g., using GridSearchCV or Optuna)
* Test deep learning models (e.g., LSTMs or CNNs)
*
