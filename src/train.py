import pandas as pd
import pickle
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import mlflow
import mlflow.sklearn


# -----------------------------
# Create folders
# -----------------------------

os.makedirs("models", exist_ok=True)


# -----------------------------
# Load processed data
# -----------------------------

X_train = pd.read_csv(
    "data/processed/X_train.csv"
)

y_train = pd.read_csv(
    "data/processed/y_train.csv"
)


# Convert target dataframe to series

y_train = y_train.values.ravel()


# -----------------------------
# MLflow Experiment
# -----------------------------

mlflow.set_experiment(
    "Advertising Sales Prediction"
)


with mlflow.start_run():


    # -------------------------
    # Model Training
    # -------------------------

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )


    # -------------------------
    # Training Metrics
    # -------------------------

    predictions = model.predict(
        X_train
    )


    mse = mean_squared_error(
        y_train,
        predictions
    )


    r2 = r2_score(
        y_train,
        predictions
    )


    # -------------------------
    # Log Parameters
    # -------------------------

    mlflow.log_param(
        "model",
        "Linear Regression"
    )


    mlflow.log_param(
        "features",
        list(X_train.columns)
    )


    # -------------------------
    # Log Metrics
    # -------------------------

    mlflow.log_metric(
        "training_mse",
        mse
    )


    mlflow.log_metric(
        "training_r2",
        r2
    )


    # -------------------------
    # Save Model
    # -------------------------

    model_path = (
        "models/"
        "linear_regression_model.pkl"
    )


    with open(
        model_path,
        "wb"
    ) as file:

        pickle.dump(
            model,
            file
        )


    # -------------------------
    # Log Model in MLflow
    # -------------------------

    mlflow.sklearn.log_model(
        model,
        "linear_regression_model"
    )


    print(
        "Model training completed successfully!"
    )

    print(
        f"Training R2 Score: {r2}"
    )
