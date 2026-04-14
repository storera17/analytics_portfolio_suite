from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from preprocessing import make_preprocessor


def make_neural_search(train_df):
    prep = make_preprocessor(train_df, scaling="minmax")

    pipe = Pipeline(
        steps=[
            ("prep", prep),
            (
                "clf",
                MLPClassifier(
                    activation="logistic",
                    solver="adam",
                    max_iter=300,
                    random_state=123,
                ),
            ),
        ]
    )

    grid = {
        "clf__hidden_layer_sizes": [(n,) for n in range(1, 11)],
        "clf__alpha": np.arange(0.01, 0.16, 0.02),
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
