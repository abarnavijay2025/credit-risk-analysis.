
Credit Risk Analysis – Machine Learning Project

## 1. Project Overview

This project focuses on predicting credit risk using machine learning techniques.  
The goal is to classify loan applicants as **low-risk** or **high-risk**, helping financial institutions make informed lending decisions.

The project includes:
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Model building and evaluation
- Comparison of multiple algorithms
- Final optimized model for credit risk prediction

***

## 2. Dataset Description

The dataset contains demographic, financial, and credit-related information.  
Common features include:
- Age
- Income
- Employment details
- Loan amount
- Credit history
- Previous defaults

**Target variable:**
- 1 → High Risk
- 0 → Low Risk


***

## 3. Methodology

### 3.1 Data Preprocessing
- Handled missing values
- Encoded categorical variables
- Scaled numerical features
- Removed outliers
- Balanced dataset using oversampling/undersampling (if used)

### 3.2 Exploratory Data Analysis (EDA)
- Feature distributions
- Correlation heatmaps
- Risk patterns across variables

### 3.3 Model Development
Tested machine learning models such as:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine

### 3.4 Model Evaluation
Performance measured using:
- Accuracy
- Precision, Recall
- F1-Score
- ROC-AUC Curve

***

## 4. Results Summary

- **Best model:** Random Forest / XGBoost / (your chosen model)
- **Achieved accuracy:** XX%
- **High recall for detecting high-risk applicants**
- Final model saved as `trained_model.pkl`

***

## 5. How to Run the Project

**Step 1:** Install dependencies  
```bash
pip install -r requirements.txt
```

**Step 2:** Run the notebook  
Open and run:  
`notebooks/credit_risk_analysis.ipynb`

**Step 3:** Use the trained model  
Load the model in Python:
```python
import pickle
model = pickle.load(open("models/trained_model.pkl", "rb"))
```

***

## 6. Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

***

## 7. Future Improvements

- Deploy model using Flask/Streamlit
- Add hyperparameter tuning
- Add model explainability (e.g., SHAP values)
- Try deep learning models

***

## 8. Author

Ashwindaniel 
Contact: ashwindaniel2000@gmail.com  


***


Credit Risk Analysis – Machine Learning Project

## 1. Project Overview

This project focuses on predicting credit risk using machine learning techniques.  
The goal is to classify loan applicants as **low-risk** or **high-risk**, helping financial institutions make informed lending decisions.

The project includes:
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Model building and evaluation
- Comparison of multiple algorithms
- Final optimized model for credit risk prediction

***

## 2. Dataset Description

The dataset contains demographic, financial, and credit-related information.  
Common features include:
- Age
- Income
- Employment details
- Loan amount
- Credit history
- Previous defaults

**Target variable:**
- 1 → High Risk
- 0 → Low Risk


***

## 3. Methodology

### 3.1 Data Preprocessing
- Handled missing values
- Encoded categorical variables
- Scaled numerical features
- Removed outliers
- Balanced dataset using oversampling/undersampling (if used)

### 3.2 Exploratory Data Analysis (EDA)
- Feature distributions
- Correlation heatmaps
- Risk patterns across variables

### 3.3 Model Development
Tested machine learning models such as:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine

### 3.4 Model Evaluation
Performance measured using:
- Accuracy
- Precision, Recall
- F1-Score
- ROC-AUC Curve

***

## 4. Results Summary

- **Best model:** Random Forest / XGBoost / (your chosen model)
- **Achieved accuracy:** XX%
- **High recall for detecting high-risk applicants**
- Final model saved as `trained_model.pkl`

***

## 5. How to Run the Project

**Step 1:** Install dependencies  
```bash
pip install -r requirements.txt
```

**Step 2:** Run the notebook  
Open and run:  
`notebooks/credit_risk_analysis.ipynb`

**Step 3:** Use the trained model  
Load the model in Python:
```python
import pickle
model = pickle.load(open("models/trained_model.pkl", "rb"))
```

***

## 6. Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

***

## 7. Future Improvements

- Deploy model using Flask/Streamlit
- Add hyperparameter tuning
- Add model explainability (e.g., SHAP values)
- Try deep learning models

***

## 8. Author

Abarna 
Contact: abarnavijayanad@gmail.com


***

