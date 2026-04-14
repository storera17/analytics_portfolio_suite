from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "forecast_model.joblib"


def make_sales_data(n_days: int = 730, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    promo = rng.binomial(1, 0.18, size=n_days)
    holiday = ((dates.month == 11) & (dates.day >= 20)).astype(int) + ((dates.month == 12) & (dates.day <= 31)).astype(int)
    weekday = dates.dayofweek
    temp = 55 + 20 * np.sin(np.arange(n_days) / 365 * 2 * np.pi) + rng.normal(0, 6, size=n_days)
    trend = np.linspace(0, 18, n_days)
    seasonality = 12 * np.sin(np.arange(n_days) / 7 * 2 * np.pi)
    noise = rng.normal(0, 8, size=n_days)
    sales = 180 + trend + seasonality + 22 * promo + 35 * holiday - 0.5 * temp + noise

    return pd.DataFrame({
        "date": dates,
        "promo": promo,
        "holiday": holiday,
        "weekday": weekday,
        "month": dates.month,
        "day_of_year": dates.dayofyear,
        "temperature": temp.round(2),
        "sales": sales.round(2),
    })


def main() -> None:
    df = make_sales_data()
    X = df[["promo", "holiday", "weekday", "month", "day_of_year", "temperature"]]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Saved model to:", MODEL_PATH)
    print("Forecast metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
