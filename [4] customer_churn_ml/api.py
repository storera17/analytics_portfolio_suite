from __future__ import annotations
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from customer_churn_ml.model import load_model

app = FastAPI(title="Customer Churn API", version="1.0.0")
model = None


class CustomerFeatures(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0)
    contract_type: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["fiber", "dsl", "none"]
    support_tickets: int = Field(ge=0, le=20)
    late_payments: int = Field(ge=0, le=24)
    paperless_billing: int = Field(ge=0, le=1)
    senior_citizen: int = Field(ge=0, le=1)


@app.on_event("startup")
def startup_event() -> None:
    global model
    model = load_model()


@app.get("/")
def healthcheck() -> dict:
    return {"status": "ok", "message": "Customer Churn API is running."}


@app.post("/predict")
def predict(features: CustomerFeatures) -> dict:
    data = pd.DataFrame([features.model_dump()])
    probability = float(model.predict_proba(data)[:, 1][0])
    prediction = int(model.predict(data)[0])
    return {
        "prediction": prediction,
        "prediction_label": "churn" if prediction == 1 else "retain",
        "churn_probability": round(probability, 4),
    }
