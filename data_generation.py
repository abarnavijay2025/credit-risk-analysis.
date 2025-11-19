"""
Generates a synthetic credit-risk dataset with numeric and categorical features.
If you prefer to use a real dataset, save it as `data/dataset.csv` and skip running this script.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
# Assuming src.utils contains the save_df function
from src.utils import save_df


np.random.seed(42)


N = 20000


# Base informative features (X contains features, y contains the ideal classification target)
X, y = make_classification(n_samples=N, n_features=12, n_informative=6, n_redundant=2,
                           n_clusters_per_class=2, flip_y=0.05, class_sep=1.0, random_state=42)


df = pd.DataFrame(X, columns=[f"num_{i+1}" for i in range(X.shape[1])])
# Add synthetic demographic / financial features derived from base features


# Age: transform a numeric feature to realistic age distribution
ages = (np.abs(df['num_1']) * 10 + 30).astype(int)
ages = ages.clip(18, 85)


# Monthly income: use exponential-type mapping
income = (np.exp(df['num_2'] / 3) * 3000 + 2000).round(0)


# Existing loans
existing_loans = (np.abs(df['num_3']) * 2).astype(int)
existing_loans = existing_loans.clip(0, 10)


# Employment length categories
emp_len = np.where(df['num_4'] > 0, ' >5y', ' <=5y')


# Credit utilization (0-1)
util = (1 / (1 + np.exp(-df['num_5']))).round(3)


# Add the generated features to the dataframe
df['age'] = ages
df['monthly_income'] = income
df['existing_loans_count'] = existing_loans
df['employment_length'] = emp_len
df['credit_utilization'] = util


# --- Target Variable Generation ---
# Make higher utilization and more loans raise default probability (business logic layer)
prob_business = 0.2 + 0.4 * df['credit_utilization'] + 0.05 * df['existing_loans_count']
prob_business = np.clip(prob_business, 0.01, 0.95)

# Initialize a random state for target generation
rng = np.random.RandomState(42)

# Generate a target based on the calculated probability
target_from_prob = (rng.rand(N) < prob_business).astype(int)

# Combine the base target 'y' (which ensures separability) with the new target
# We use a weight (e.g., 0.6) to blend the two, maintaining informative structure AND business logic.
# In this corrected version, we simply use the generated 'y' from make_classification
# as the final target, as it already incorporates complex feature interactions,
# and we'll keep the new features as strong predictors.
# If you want to mix them, you could use: df['target'] = np.where(rng.rand(N) < 0.6, y, target_from_prob)

# For simplicity and robustness, let's use the classification target 'y'
# and rely on the new features to be strong, realistic predictors.
df['target'] = y

# --- Data Saving ---
# Add the final required column: the target
# df['target'] = final_y # Re-add this line if you use the blend logic above

# Now, save the dataframe using the imported utility function
save_df(df, 'data/dataset.csv')

print("Synthetic dataset successfully written to data/dataset.csv — rows:", df.shape[0])
print("Target Balance (0: Low Risk, 1: High Risk):")
print(df['target'].value_counts(normalize=True))
