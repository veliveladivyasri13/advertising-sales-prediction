import pandas as pd
import pickle
import os

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import mlflow


# -----------------------------
# Load test data
# -----------------------------

X_test = pd.read_csv(
    "data/processed/X_test.csv"
)

y_test = pd.read_csv(
    "data/processed/y_test.csv"
)


# Convert target to array

y_test = y_test.values.ravel()


# -----------------------------
# Load trained model
# -----------------------------

model_path = (
    "models/"
    "linear_regression_model.pkl"
)


with open(
    model_path,
    "rb"
) as file:

    model = pickle.load(file)



# -----------------------------
# MLflow Experiment
# -----------------------------

mlflow.set_experiment(
    "Advertising Sales Prediction"
)


with mlflow.start_run(
    run_name="Model Evaluation"
):


    # -------------------------
    # Prediction
    # -------------------------

    predictions = model.predict(
        X_test
    )


    # -------------------------
    # Metrics
    # -------------------------

    mse = mean_squared_error(
        y_test,
        predictions
    )


    rmse = mse ** 0.5


    mae = mean_absolute_error(
        y_test,
        predictions
    )


    r2 = r2_score(
        y_test,
        predictions
    )


    # -------------------------
    # Log Metrics to MLflow
    # -------------------------

    mlflow.log_metric(
        "test_mse",
        mse
    )


    mlflow.log_metric(
        "test_rmse",
        rmse
    )


    mlflow.log_metric(
        "test_mae",
        mae
    )


    mlflow.log_metric(
        "test_r2",
        r2
    )


    # -------------------------
    # Save evaluation output
    # -------------------------

    os.makedirs(
        "outputs",
        exist_ok=True
    )


    results = pd.DataFrame(
        {
            "Metric": [
                "MSE",
                "RMSE",
                "MAE",
                "R2 Score"
            ],

            "Value": [
                mse,
                rmse,
                mae,
                r2
            ]
        }
    )


    results.to_csv(
        "outputs/model_evaluation.csv",
        index=False
    )


    print(
        "Model evaluation completed successfully!"
    )


    print(
        results
    )