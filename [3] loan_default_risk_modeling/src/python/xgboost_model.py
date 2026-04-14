from __future__ import annotations

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocessing import make_preprocessor

try:
    from xgboost import XGBClassifier
except Exception as exc:
    raise ImportError("Install xgboost before using this module.") from exc


def make_xgboost_search(train_df):
    prep = make_preprocessor(train_df, scaling="standard")

    pipe = Pipeline(
        steps=[
            ("prep", prep),
            (
                "clf",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=123,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )

    grid = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.01, 0.1, 0.3],
        "clf__gamma": [0.0, 0.1, 0.5],
        "clf__colsample_bytree": [0.5, 0.7, 1.0],
        "clf__min_child_weight": [1, 3, 5],
        "clf__subsample": [0.5, 0.7, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    return GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=cv,
        refit=True,
        verbose=1,
    )
