from __future__ import annotations

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocessing import make_preprocessor


def make_adaboost_search(train_df):
    prep = make_preprocessor(train_df, scaling="standard")
    weak_tree = DecisionTreeClassifier(max_depth=1, random_state=123)

    pipe = Pipeline(
        steps=[
            ("prep", prep),
            ("clf", AdaBoostClassifier(estimator=weak_tree, random_state=123)),
        ]
    )

    grid = {
        "clf__n_estimators": [50, 100, 200],
        "clf__learning_rate": [0.01, 0.1, 0.5, 1.0],
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
