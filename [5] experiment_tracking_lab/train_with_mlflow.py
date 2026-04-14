from __future__ import annotations
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_experiment(model_name: str) -> None:
    data = load_breast_cancer(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    if model_name == "logistic_regression":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])
        params = {"model_type": model_name, "max_iter": 1000}
    else:
        model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
        params = {"model_type": model_name, "n_estimators": 300, "max_depth": 6}

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")
        print(f"Logged run for {model_name}: {metrics}")


def main() -> None:
    mlflow.set_experiment("analytics_portfolio_suite")
    for name in ["logistic_regression", "random_forest"]:
        run_experiment(name)


if __name__ == "__main__":
    main()
