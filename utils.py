import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
MODEL_DIR = os.path.join(os.getcwd(), "models")
for d in (OUTPUT_DIR, MODEL_DIR):
os.makedirs(d, exist_ok=True)




def save_df(df, path):
os.makedirs(os.path.dirname(path), exist_ok=True)
df.to_csv(path, index=False)




def load_data_if_present(path="data/dataset.csv"):
if os.path.exists(path):
return pd.read_csv(path)
return None
