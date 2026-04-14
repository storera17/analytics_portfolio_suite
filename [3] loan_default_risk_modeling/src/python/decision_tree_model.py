from __future__ import annotations

from itertools import product
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocessing import make_preprocessor
from metrics import oof_probabilities, find_best_cutoff, score_table


def run_tree_grid(train_df):
    x_data = train_df.drop(columns=["loan_default"])
    y_data = train_df["loan_default"]

    prep = make_preprocessor(train_df, scaling="standard")

    grid = {
        "ccp_alpha": [0.01, 0.005, 0.001, 0.0075, 0.0005, 0.00025, 0.00001, 0.000001],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "min_samples_split": [15, 25, 37, 50, 100],
        "min_samples_leaf": [10, 25, 50],
    }

    rows = []

    for params in ParameterGrid(grid):
        if params["min_samples_leaf"] > params["min_samples_split"]:
            continue

        pipe = Pipeline(
            steps=[
                ("prep", prep),
                (
                    "clf",
                    DecisionTreeClassifier(
                        criterion="gini",
                        random_state=123,
                        **params,
                    ),
                ),
            ]
        )

        oof_prob = oof_probabilities(pipe, x_data, y_data, folds=5, seed=123)
        best_cut, _ = find_best_cutoff(y_data, oof_prob)
        stats = score_table(y_data, oof_prob, best_cut)

        rows.append({
            **params,
            "best_threshold": best_cut,
            "cv_auc": stats["auc"],
            "cv_f1": stats["f1"],
        })

    return pd.DataFrame(rows).sort_values(["cv_f1", "cv_auc"], ascending=False)
