# src/predict.py

import pandas as pd
import mlflow.sklearn

from config import TEST_PATH
from preprocess import normalize

def predict():
    model = mlflow.sklearn.load_model("runs:/<RUN_ID>/model")

    test = pd.read_csv(TEST_PATH)
    test = normalize(test)

    preds = model.predict(test)

    submission = pd.DataFrame({
        "ImageId": range(1, len(preds) + 1),
        "Label": preds
    })

    submission.to_csv("output/submission.csv", index=False)

if __name__ == "__main__":
    predict()