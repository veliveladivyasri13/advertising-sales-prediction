from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd


app = FastAPI(
    title="Advertising Sales Prediction API"
)


# Load trained model

with open(
    "models/linear_regression_model.pkl",
    "rb"
) as file:
    model = pickle.load(file)



class AdvertisingData(BaseModel):

    TV: float
    Radio: float
    Newspaper: float



@app.get("/")
def home():

    return {
        "message": "Advertising Sales Prediction API Running"
    }



@app.post("/predict")
def predict_sales(
    data: AdvertisingData
):

    input_data = pd.DataFrame(
        {
            "TV": [data.TV],
            "Radio": [data.Radio],
            "Newspaper": [data.Newspaper]
        }
    )


    prediction = model.predict(
        input_data
    )


    return {
        "predicted_sales": float(
            prediction[0]
        )
    }