from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocessing import make_preprocessor


def make_logitboost_approx_search(train_df):
    prep = make_preprocessor(train_df, scaling="standard")

    pipe = Pipeline(
        steps=[
            ("prep", prep),
            ("clf", GradientBoostingClassifier(random_state=123)),
        ]
    )

    grid = {
        "clf__n_estimators": [50, 100, 200],
        "clf__learning_rate": [0.01, 0.1, 0.3],
        "clf__max_depth": [1, 2, 3],
        "clf__subsample": [0.7, 1.0],
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
