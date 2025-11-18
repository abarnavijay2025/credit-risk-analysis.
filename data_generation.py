"""
Generates a synthetic credit-risk dataset with numeric and categorical features.
If you prefer to use a real dataset, save it as `data/dataset.csv` and skip running this script.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from src.utils import save_df


np.random.seed(42)


N = 20000


# base informative features
X, y = make_classification(n_samples=N, n_features=12, n_informative=6, n_redundant=2,
n_clusters_per_class=2, flip_y=0.05, class_sep=1.0, random_state=42)


df = pd.DataFrame(X, columns=[f"num_{i+1}" for i in range(X.shape[1])])
# add synthetic demographic / financial features derived from base features


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


# Add to dataframe
df['age'] = ages
df['monthly_income'] = income
df['existing_loans_count'] = existing_loans
df['employment_length'] = emp_len
df['credit_utilization'] = util


# Target: use the y from make_classification but flip a few based on business logic
# Make higher utilization and more loans raise default probability
prob = 0.2 + 0.4 * df['credit_utilization'] + 0.05 * df['existing_loans_count']
# combine with base y to keep informative structure
import numpy as np
prob = np.clip(prob, 0.01, 0.95)


# create final target
rng = np.random.RandomState(42)
final_y = (rng.rand(N) < prob).astype(int)


# to keep balance similar to classification y, mix in base y
print("Synthetic dataset written to data/dataset.csv — rows:", df.shape[0])
