# Put the code for your API here.
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Body
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib


# Loading the saved model
model = joblib.load('model/trained_model.pkl')

class DataItem(BaseModel):
    age: int
    workclass: str = Field(..., alias='workclass')
    fnlgt: int = Field(..., alias='fnlgt')
    education: str = Field(..., alias='education')
    education_num: int = Field(..., alias='education-num')
    marital_status: str = Field(..., alias='marital-status')
    occupation: str = Field(..., alias='occupation')
    relationship: str = Field(..., alias='relationship')
    race: str = Field(..., alias='race')
    sex: str = Field(..., alias='sex')
    capital_gain: int = Field(..., alias='capital-gain')
    capital_loss: int = Field(..., alias='capital-loss')
    hours_per_week: int = Field(..., alias='hours-per-week')
    native_country: str = Field(..., alias='native-country')

    class Config:
        json_schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to the model inference API!"}


@app.post("/predict/")
def make_prediction(item: DataItem):
    # Convert the Pydantic object to a DataFrame or suitable format for your model
    # Note: Ensure your model and necessary preprocessing steps are loaded correctly.

    data = pd.DataFrame([item.dict()])

    #data = np.array(DataItem.values())
    #print(data)

    # Perform prediction
    try:
        # Here, replace with actual prediction logic
        #prediction = model.predict([data.dict(by_alias=True)])
        #prediction = model.predict(data)
        prediction = 'test'
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

