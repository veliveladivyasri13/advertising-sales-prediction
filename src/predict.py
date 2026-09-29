import pandas as pd
import pickle
import mlflow


# -----------------------------
# Load trained model
# -----------------------------

model_path = "models/linear_regression_model.pkl"


with open(model_path, "rb") as file:
    model = pickle.load(file)


# -----------------------------
# New Advertising Data
# -----------------------------

new_data = pd.DataFrame(
    {
        "TV": [230],
        "Radio": [37],
        "Newspaper": [69]
    }
)


# -----------------------------
# MLflow Prediction Run
# -----------------------------

mlflow.set_experiment(
    "Advertising Sales Prediction"
)


with mlflow.start_run(
    run_name="Sales Prediction"
):


    prediction = model.predict(
        new_data
    )


    predicted_sales = prediction[0]


    # Log input values

    mlflow.log_param(
        "TV",
        230
    )

    mlflow.log_param(
        "Radio",
        37
    )

    mlflow.log_param(
        "Newspaper",
        69
    )


    # Log prediction

    mlflow.log_metric(
        "predicted_sales",
        predicted_sales
    )


    print(
        "Predicted Sales:",
        predicted_sales
    )