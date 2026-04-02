# src/model.py

from sklearn.ensemble import RandomForestClassifier

def get_model(params):
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"]
    )
    return model