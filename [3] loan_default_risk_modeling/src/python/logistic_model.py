from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocessing import make_preprocessor


def make_logistic_search(train_df):
    prep = make_preprocessor(train_df, scaling="standard")

    pipe = Pipeline(
        steps=[
            ("prep", prep),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    max_iter=5000,
                    random_state=123,
                ),
            ),
        ]
    )

    search_grid = [
        {
            "clf__penalty": ["elasticnet"],
            "clf__l1_ratio": np.linspace(0.0, 1.0, 8),
            "clf__C": np.geomspace(10.0, 1000.0, 5),
        },
        {
            "clf__penalty": ["l1"],
            "clf__C": np.geomspace(10.0, 1000.0, 10),
        },
    ]

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=123)

    return GridSearchCV(
        estimator=pipe,
        param_grid=search_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=cv,
        refit=True,
        verbose=1,
    )
