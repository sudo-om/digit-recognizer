# src/config.py

DATA_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"

MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
}

TEST_SIZE = 0.2
RANDOM_STATE = 42

MLFLOW_EXPERIMENT = "digit-recognizer"