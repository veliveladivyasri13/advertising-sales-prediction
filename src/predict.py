import pandas as pd
import joblib


# Model path
MODEL_PATH = "models/linear_regression_model.pkl"


# Load model
def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    return model


# Make prediction
def predict_sales(model, tv, radio, newspaper):

    input_data = pd.DataFrame(
        {
            "TV": [tv],
            "Radio": [radio],
            "Newspaper": [newspaper]
        }
    )

    prediction = model.predict(
        input_data
    )

    return prediction[0][0]


# Main execution
if __name__ == "__main__":

    model = load_model()

    predicted_sales = predict_sales(
        model,
        tv=230.1,
        radio=37.8,
        newspaper=69.2
    )

    print(
        "Predicted Sales:",
        round(predicted_sales, 2)
    )