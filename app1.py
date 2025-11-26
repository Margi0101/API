from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# -------------------------------
# 1. Load and prepare data
# -------------------------------
df = pd.read_csv(r"C:/Users/margi/OneDrive/Documents/fastapi/california_housing_test.csv")

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 2. Build and train pipeline
# -------------------------------
pipeline = make_pipeline(PowerTransformer(), StandardScaler(), LinearRegression())
pipeline.fit(X_train, y_train)

# -------------------------------
# 3. FastAPI app
# -------------------------------
app = FastAPI(title="Housing Price Prediction API")

# Request schema
class House(BaseModel):
    housing_median_age: float
    population: float
    households: float
    median_income: float
    new_room: float

# Home route
@app.get("/")
def home():
    return {"message": "Welcome to California Housing Prediction API"}

# Prediction route
@app.post("/predict")
def predict(data: House):
    input_df = pd.DataFrame([data.dict()])   # convert request to dataframe
    prediction = pipeline.predict(input_df)[0]
    return {"predicted_median_house_value": prediction}

# Evaluation route
@app.get("/evaluate")
def evaluate():
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {
        "MSE": mse,
        "MAE": mae,
        "R2_Score": r2
    }

# -------------------------------
# 1. Load and prepare data
# -------------------------------
df = pd.read_csv(r"C:\Users\margi\OneDrive\Documents\fastapi\california_housing_test.csv")

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 2. Build and train pipeline
# -------------------------------
pipeline = make_pipeline(PowerTransformer(), StandardScaler(), LinearRegression())
pipeline.fit(X_train, y_train)

# -------------------------------
# 3. FastAPI app
# -------------------------------
app = FastAPI(title="Housing Price Prediction API")

# Request schema
class House(BaseModel):
    housing_median_age: float
    population: float
    households: float
    median_income: float
    new_room: float

# Home route
@app.get("/")
def home():
    return {"message": "Welcome to California Housing Prediction API"}

# Prediction route
@app.post("/predict")
def predict(data: House):
    input_df = pd.DataFrame([data.dict()])   # convert request to dataframe
    prediction = pipeline.predict(input_df)[0]
    return {"predicted_median_house_value": prediction}

# Evaluation route
@app.get("/evaluate")
def evaluate():
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {
        "MSE": mse,
        "MAE": mae,
        "R2_Score": r2
    }
