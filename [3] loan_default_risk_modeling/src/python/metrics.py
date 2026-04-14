from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

NEGATIVE_LABEL = "No"
POSITIVE_LABEL = "Yes"


def to_binary(series: pd.Series) -> pd.Series:
    return series.map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}).astype(int)


def score_table(truth: pd.Series, prob_yes: np.ndarray, threshold: float) -> dict:
    truth_num = to_binary(truth)
    pred_num = (prob_yes >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "f1": float(f1_score(truth_num, pred_num, zero_division=0)),
        "precision": float(precision_score(truth_num, pred_num, zero_division=0)),
        "recall": float(recall_score(truth_num, pred_num, zero_division=0)),
        "accuracy": float(accuracy_score(truth_num, pred_num)),
        "auc": float(roc_auc_score(truth_num, prob_yes)),
    }


def find_best_cutoff(truth: pd.Series, prob_yes: np.ndarray) -> Tuple[float, pd.DataFrame]:
    rows = []
    for cutoff in np.arange(0.01, 1.00, 0.01):
        rows.append(score_table(truth, prob_yes, float(cutoff)))

    grid = pd.DataFrame(rows)
    winner = float(grid.loc[grid["f1"].idxmax(), "threshold"])
    return winner, grid


def print_confusion(truth: pd.Series, prob_yes: np.ndarray, threshold: float) -> None:
    actual = to_binary(truth).to_numpy()
    called = (prob_yes >= threshold).astype(int)

    print("Confusion matrix:")
    print(confusion_matrix(actual, called))
    print()
    print(classification_report(actual, called, target_names=[NEGATIVE_LABEL, POSITIVE_LABEL], zero_division=0))


def oof_probabilities(base_pipeline, x_data: pd.DataFrame, y_data: pd.Series, folds: int = 5, seed: int = 123) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    y_num = to_binary(y_data)
    oof_scores = np.zeros(len(x_data), dtype=float)

    for fit_idx, val_idx in splitter.split(x_data, y_num):
        x_fit = x_data.iloc[fit_idx]
        x_val = x_data.iloc[val_idx]
        y_fit = y_data.iloc[fit_idx]

        local_pipe = clone(base_pipeline)
        local_pipe.fit(x_fit, y_fit)
        oof_scores[val_idx] = local_pipe.predict_proba(x_val)[:, 1]

    return oof_scores
