from __future__ import annotations
from customer_churn_ml.data import make_customer_churn_data
from customer_churn_ml.model import save_model, train_model


def main() -> None:
    df = make_customer_churn_data()
    model, metrics, report = train_model(df)
    path = save_model(model)
    print("Saved model to:", path)
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print("
Classification report:
")
    print(report)


if __name__ == "__main__":
    main()
