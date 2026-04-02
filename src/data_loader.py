# src/data_loader.py

import pandas as pd

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.drop("label", axis=1)
    y = df["label"]
    return X, y

def load_test_data(path):
    return pd.read_csv(path)