import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def train_model(X_train, y_train):
    """Trainiert das Modell mit RandomForestRegressor."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Bewertet das Modell anhand der Testdaten."""
    scores = cross_val_score(model, X_test, y_test, scoring="neg_mean_squared_error", cv=5)
    rmse_scores = np.sqrt(-scores)

    print("Model Evaluation:")
    print(f"Mean RMSE: {rmse_scores.mean()}")
    print(f"Standard Deviation: {rmse_scores.std()}")

    return rmse_scores
