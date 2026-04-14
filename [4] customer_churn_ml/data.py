from __future__ import annotations
import numpy as np
import pandas as pd


def make_customer_churn_data(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    tenure_months = rng.integers(1, 72, size=n_samples)
    monthly_charges = rng.normal(75, 25, size=n_samples).clip(20, 180)
    contract_type = rng.choice(["month-to-month", "one-year", "two-year"], size=n_samples, p=[0.55, 0.25, 0.20])
    internet_service = rng.choice(["fiber", "dsl", "none"], size=n_samples, p=[0.5, 0.35, 0.15])
    support_tickets = rng.poisson(1.8, size=n_samples)
    late_payments = rng.poisson(0.8, size=n_samples)
    paperless_billing = rng.choice([0, 1], size=n_samples, p=[0.35, 0.65])
    senior_citizen = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])

    raw_score = (
        1.25 * (contract_type == "month-to-month").astype(int)
        + 0.9 * (internet_service == "fiber").astype(int)
        + 0.35 * support_tickets
        + 0.5 * late_payments
        + 0.25 * paperless_billing
        - 0.025 * tenure_months
        + 0.01 * monthly_charges
        + 0.20 * senior_citizen
        - 2.3
    )
    churn_probability = 1 / (1 + np.exp(-raw_score))
    churn = rng.binomial(1, churn_probability)

    df = pd.DataFrame({
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges.round(2),
        "contract_type": contract_type,
        "internet_service": internet_service,
        "support_tickets": support_tickets,
        "late_payments": late_payments,
        "paperless_billing": paperless_billing,
        "senior_citizen": senior_citizen,
        "churn": churn,
    })
    return df
