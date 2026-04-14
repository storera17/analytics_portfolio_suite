from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocessing import make_preprocessor


def make_forest_search(train_df):
    prep = make_preprocessor(train_df, scaling="standard")

    pipe = Pipeline(
        steps=[
            ("prep", prep),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=123,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    grid = {
        "clf__max_features": [4, 6, 8, 10, "sqrt"],
        "clf__min_samples_leaf": [4, 5, 8],
        "clf__min_samples_split": [10, 20, 30],
        "clf__max_depth": [None, 6, 10, 16],
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
