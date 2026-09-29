import pandas as pd
import joblib
import os

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# Paths
PROCESSED_DIR = "data/processed"
MODEL_PATH = "models/linear_regression_model.pkl"
OUTPUT_DIR = "outputs"


# Load test data
def load_test_data():

    X_test = pd.read_csv(
        f"{PROCESSED_DIR}/X_test.csv"
    )

    y_test = pd.read_csv(
        f"{PROCESSED_DIR}/y_test.csv"
    )

    return X_test, y_test


# Load trained model
def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    return model


# Evaluate model
def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    }


# Save evaluation results
def save_results(results):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results_df = pd.DataFrame(
        [results]
    )

    results_df.to_csv(
        f"{OUTPUT_DIR}/model_evaluation.csv",
        index=False
    )


# Main execution
if __name__ == "__main__":

    X_test, y_test = load_test_data()

    model = load_model()

    results = evaluate_model(
        model,
        X_test,
        y_test
    )

    save_results(results)

    print("Model evaluation completed successfully.")
    print(results)