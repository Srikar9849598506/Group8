import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn
import os
import joblib


# load data
preprocessed_data = pd.read_csv(r"data/preprocessed_data.csv")

scaler = StandardScaler()


def build_model():

    if preprocessed_data is None or preprocessed_data.empty:
        print("Data not found or empty")
        return

    # create folder for saving model
    os.makedirs("models", exist_ok=True)

    # start MLflow experiment
    mlflow.set_experiment("XGBoost_Experiment")

    with mlflow.start_run():

        # features and target
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
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )

        # log params
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)

        # train
        model = xgb_model.fit(X_train_scaled, y_train)

        # predict
        preds = model.predict(X_test_scaled)

        # accuracy
        acc = accuracy_score(y_test, preds)

        # log metric
        mlflow.log_metric("accuracy", acc)

        # ---------------------------
        # 💾 SAVE MODEL LOCALLY
        # ---------------------------

        model_path = "models/xgb_model.pkl"
        scaler_path = "models/scaler.pkl"

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        print(f"Model saved at: {model_path}")
        print(f"Scaler saved at: {scaler_path}")

        # ---------------------------
        # MLflow model logging
        # ---------------------------
        mlflow.sklearn.log_model(model, "model")

        print("✅ Model trained and logged to MLflow")
        print(f"Accuracy: {acc:.4f}")

        return model


build_model()