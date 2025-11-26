from fastapi import FastAPI,Form
from pydantic import BaseModel
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from typing import Annotated

app = FastAPI(title="Housing Price Prediction API")

class HousingInput(BaseModel):
    longitude: Annotated[float,Form()]
    latitude: Annotated[float,Form()]
    housing_median_age: Annotated[float,Form()]
    total_rooms: Annotated[float,Form()]
    total_bedrooms: Annotated[float,Form()]
    population: Annotated[float,Form()]
    households: Annotated[float,Form()]
    median_income: Annotated[float,Form()]
    ocean_proximity: Annotated[str,Form()]  

df = pd.read_csv("C:/Users/margi/OneDrive/Documents/fastapi/california_housing_test.csv")
x = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

categorical_cols = ["ocean_proximity"]
numerical_cols = x.drop(columns=categorical_cols).columns.tolist()

full_pipeline = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("power", PowerTransformer(standardize=False)),
        ("scaler", StandardScaler())
    ]), numerical_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

pipeline_model = Pipeline([
    ("preprocessor", full_pipeline),
    ("regressor", LinearRegression())
])

pipeline_model.fit(x, y)



@app.post("/predict")
def predict_price(data: HousingInput):
    df_input = pd.DataFrame([data.dict()])
    prediction = pipeline_model.predict(df_input)
    return {"predicted_price": round(prediction[0], 2)}