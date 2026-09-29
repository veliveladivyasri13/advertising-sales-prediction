import pandas as pd
from sklearn.model_selection import train_test_split
import os


# Project paths
RAW_DATA_PATH = "data/raw/advertising.csv"
PROCESSED_DIR = "data/processed"


# Load dataset
def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    return df


# Split features and target
def split_data(df):

    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]

    return X, y


# Create train-test split
def create_train_test(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# Save processed data
def save_data(X_train, X_test, y_train, y_test):

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    X_train.to_csv(
        f"{PROCESSED_DIR}/X_train.csv",
        index=False
    )

    X_test.to_csv(
        f"{PROCESSED_DIR}/X_test.csv",
        index=False
    )

    y_train.to_csv(
        f"{PROCESSED_DIR}/y_train.csv",
        index=False
    )

    y_test.to_csv(
        f"{PROCESSED_DIR}/y_test.csv",
        index=False
    )


# Main execution
if __name__ == "__main__":

    df = load_data()

    X, y = split_data(df)

    X_train, X_test, y_train, y_test = create_train_test(
        X,
        y
    )

    save_data(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("Data preprocessing completed successfully.")