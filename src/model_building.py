import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import mlflow
import os
import joblib


# =========================
# MLflow setup (CI SAFE)
# =========================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("XGBoost_Experiment")


def load_data():
    """Load data safely for CI/CD"""
    path = "data/preprocessed_data.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    return pd.read_csv(path)


def build_model():

    preprocessed_data = load_data()

    if preprocessed_data.empty:
        raise ValueError("Dataset is empty")

    # create model folder
    os.makedirs("models", exist_ok=True)

    scaler = StandardScaler()

    X = preprocessed_data.drop(columns=["binary_label"]).astype(float)
    y = preprocessed_data["binary_label"]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scaling
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

    with mlflow.start_run():

        # train
        model.fit(X_train_scaled, y_train)

        # predictions
        preds = model.predict(X_test_scaled)

        # accuracy
        acc = accuracy_score(y_test, preds)

        # logs
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_metric("accuracy", acc)

        # =========================
        # SAVE ARTIFACTS (CROSS PLATFORM SAFE)
        # =========================

        model_path = os.path.join("models", "xgb_model.json")
        scaler_path = os.path.join("models", "scaler.pkl")

        model.save_model(model_path)
        joblib.dump(scaler, scaler_path)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)

        print("✅ Model trained successfully")
        print(f"📊 Accuracy: {acc:.4f}")

        return model, acc


if __name__ == "__main__":
    build_model()