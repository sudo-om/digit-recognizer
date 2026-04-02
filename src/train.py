# src/train.py

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import *
from data_loader import load_train_data
from preprocess import normalize
from model import get_model

def train():
    # Set experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():

        # Load data
        X, y = load_train_data(DATA_PATH)

        # Preprocess
        X = normalize(X)

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        # Model
        model = get_model(MODEL_PARAMS)

        # Train
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_val)

        acc = accuracy_score(y_val, preds)

        # Log params
        mlflow.log_params(MODEL_PARAMS)

        # Log metrics
        mlflow.log_metric("accuracy", acc)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"Validation Accuracy: {acc}")

if __name__ == "__main__":
    train()