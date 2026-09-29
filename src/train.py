import pandas as pd
import os
import joblib
import logging

from sklearn.linear_model import LinearRegression


# -----------------------------
# Paths
# -----------------------------

PROCESSED_DIR = "data/processed"

MODEL_DIR = "models"

MODEL_PATH = "models/linear_regression_model.pkl"

LOG_DIR = "logs"


# -----------------------------
# Logging Configuration
# -----------------------------

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename="logs/project.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -----------------------------
# Load Training Data
# -----------------------------

def load_training_data():

    X_train = pd.read_csv(
        f"{PROCESSED_DIR}/X_train.csv"
    )

    y_train = pd.read_csv(
        f"{PROCESSED_DIR}/y_train.csv"
    )

    return X_train, y_train



# -----------------------------
# Train Model
# -----------------------------

def train_model(X_train, y_train):

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    return model



# -----------------------------
# Save Model
# -----------------------------

def save_model(model):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )



# -----------------------------
# Main Execution
# -----------------------------

if __name__ == "__main__":

    logging.info(
        "Training process started"
    )


    # Load data
    X_train, y_train = load_training_data()


    logging.info(
        f"Training data loaded: {X_train.shape}"
    )


    print(
        "Training data loaded successfully."
    )

    print(
        "X_train shape:",
        X_train.shape
    )

    print(
        "y_train shape:",
        y_train.shape
    )


    # Train model
    model = train_model(
        X_train,
        y_train
    )


    logging.info(
        "Linear Regression model trained successfully"
    )


    print(
        "Model trained successfully."
    )


    # Save model
    save_model(model)


    logging.info(
        "Model saved successfully"
    )


    print(
        "Model saved successfully."
    )

    print(
        "Saved location:",
        MODEL_PATH
    )