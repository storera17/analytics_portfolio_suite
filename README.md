# Analytics Portfolio Suite

A multi-project portfolio repository for data analyst, data science, and business analytics applications.

This repository is intentionally structured like a small engineering workspace instead of a random notebook dump. It contains three projects:

1. `customer_churn_ml/` — binary classification pipeline with feature engineering, model training, evaluation, and FastAPI scoring.
2. `sales_forecasting_lab/` — time-series style feature engineering and regression forecasting with synthetic retail data.
3. `experiment_tracking_lab/` — MLflow experiment tracking example with multiple sklearn models.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
```

### Run project 1
```bash
python customer_churn_ml/train.py
uvicorn customer_churn_ml.api:app --reload
```

### Run project 2
```bash
python sales_forecasting_lab/train.py
```

### Run project 3
```bash
python experiment_tracking_lab/train_with_mlflow.py
mlflow ui
```

## Suggested GitHub topics

`data-analytics` `machine-learning` `fastapi` `mlflow` `forecasting` `python` `portfolio-project`

## Recommended first improvements

- Replace synthetic data with a public business dataset.
- Add screenshots of model metrics and API docs.
- Add one short case study per project under `reports/`.
- Deploy the FastAPI scoring app.
