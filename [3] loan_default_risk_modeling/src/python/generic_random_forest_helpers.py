from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_TARGET = "target"
DEFAULT_POSITIVE = "Yes"
DEFAULT_NEGATIVE = "No"


@dataclass
class ForestRunResult:
    best_params: Dict[str, object]
    best_threshold: float
    cv_auc: float
    cv_f1: float
    holdout_auc: Optional[float]
    holdout_f1: Optional[float]
    fitted_model: Pipeline
    summary_table: pd.DataFrame


def load_data(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)

    raise ValueError("Unsupported file type. Use CSV, Parquet, or Pickle.")


def normalize_target_labels(
    data: pd.DataFrame,
    target_col: str = DEFAULT_TARGET,
    positive_label: str = DEFAULT_POSITIVE,
    negative_label: str = DEFAULT_NEGATIVE,
) -> pd.DataFrame:
    output = data.copy()
    output[target_col] = output[target_col].astype(str)
    output[target_col] = pd.Categorical(
        output[target_col],
        categories=[negative_label, positive_label],
        ordered=True,
    )
    return output


def make_balanced_subset(
    data: pd.DataFrame,
    target_col: str = DEFAULT_TARGET,
    positive_label: str = DEFAULT_POSITIVE,
    negative_label: str = DEFAULT_NEGATIVE,
    seed: int = 123,
) -> pd.DataFrame:
    positive_rows = data.loc[data[target_col] == positive_label]
    negative_rows = data.loc[data[target_col] == negative_label]

    if len(positive_rows) == 0 or len(negative_rows) == 0:
        raise ValueError("Both classes must be present to perform undersampling.")

    minority_n = min(len(positive_rows), len(negative_rows))

    positive_draw = positive_rows.sample(n=minority_n, random_state=seed, replace=False)
    negative_draw = negative_rows.sample(n=minority_n, random_state=seed, replace=False)

    combined = pd.concat([positive_draw, negative_draw], axis=0)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_feature_types(data: pd.DataFrame, target_col: str = DEFAULT_TARGET) -> Tuple[List[str], List[str]]:
    predictors = data.drop(columns=[target_col], errors="ignore")
    numeric_cols = predictors.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in predictors.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols


def make_preprocessor(data: pd.DataFrame, target_col: str = DEFAULT_TARGET) -> ColumnTransformer:
    numeric_cols, categorical_cols = split_feature_types(data, target_col=target_col)

    numeric_pipeline = Pipeline(
        steps=[
            ("fill_num", SimpleImputer(strategy="median")),
            ("scale_num", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("fill_cat", SimpleImputer(strategy="most_frequent")),
            ("encode_cat", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def to_binary(
    series: pd.Series,
    positive_label: str = DEFAULT_POSITIVE,
    negative_label: str = DEFAULT_NEGATIVE,
) -> pd.Series:
    return series.map({negative_label: 0, positive_label: 1}).astype(int)


def confusion_summary(
    truth: pd.Series,
    pred: pd.Series,
    positive_label: str = DEFAULT_POSITIVE,
) -> Dict[str, object]:
    truth_num = to_binary(truth, positive_label=positive_label)
    pred_num = to_binary(pred, positive_label=positive_label)

    cm = confusion_matrix(truth_num, pred_num)
    report = classification_report(
        truth_num,
        pred_num,
        target_names=["negative_class", "positive_class"],
        zero_division=0,
        output_dict=True,
    )

    return {
        "confusion_matrix": cm,
        "report": report,
    }


def print_confusion(
    truth: pd.Series,
    pred: pd.Series,
    positive_label: str = DEFAULT_POSITIVE,
) -> None:
    result = confusion_summary(truth, pred, positive_label=positive_label)
    print("Confusion matrix:")
    print(result["confusion_matrix"])
    print()
    print(classification_report(
        to_binary(truth, positive_label=positive_label),
        to_binary(pred, positive_label=positive_label),
        target_names=["negative_class", "positive_class"],
        zero_division=0,
    ))


def threshold_to_class(
    prob_yes: np.ndarray,
    threshold: float,
    positive_label: str = DEFAULT_POSITIVE,
    negative_label: str = DEFAULT_NEGATIVE,
) -> pd.Series:
    out = np.where(prob_yes >= threshold, positive_label, negative_label)
    return pd.Series(out, dtype="object")


def find_best_f1_threshold(
    truth: pd.Series,
    prob_yes: np.ndarray,
    positive_label: str = DEFAULT_POSITIVE,
    negative_label: str = DEFAULT_NEGATIVE,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, pd.DataFrame]:
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    truth_num = to_binary(truth, positive_label=positive_label, negative_label=negative_label)
    rows = []

    for threshold in thresholds:
        pred_num = (prob_yes >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "f1": float(f1_score(truth_num, pred_num, zero_division=0)),
                "precision": float(precision_score(truth_num, pred_num, zero_division=0)),
                "recall": float(recall_score(truth_num, pred_num, zero_division=0)),
                "accuracy": float(accuracy_score(truth_num, pred_num)),
                "auc": float(roc_auc_score(truth_num, prob_yes)),
            }
        )

    curve = pd.DataFrame(rows)
    best_threshold = float(curve.loc[curve["f1"].idxmax(), "threshold"])
    return best_threshold, curve


def out_of_fold_probabilities(
    pipeline: Pipeline,
    x_data: pd.DataFrame,
    y_data: pd.Series,
    folds: int = 5,
    seed: int = 123,
) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    y_num = to_binary(y_data)
    oof = np.zeros(len(x_data), dtype=float)

    for fit_idx, val_idx in splitter.split(x_data, y_num):
        x_fit = x_data.iloc[fit_idx]
        x_val = x_data.iloc[val_idx]
        y_fit = y_data.iloc[fit_idx]

        local_model = clone(pipeline)
        local_model.fit(x_fit, y_fit)
        oof[val_idx] = local_model.predict_proba(x_val)[:, 1]

    return oof


def make_forest_search(
    train_data: pd.DataFrame,
    target_col: str = DEFAULT_TARGET,
    n_trees: int = 300,
    folds: int = 5,
    seed: int = 123,
) -> GridSearchCV:
    prep = make_preprocessor(train_data, target_col=target_col)

    pipeline = Pipeline(
        steps=[
            ("prep", prep),
            (
                "forest",
                RandomForestClassifier(
                    n_estimators=n_trees,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    grid = {
        "forest__max_features": [4, 6, 8, 10, "sqrt"],
        "forest__min_samples_leaf": [4, 5, 8],
        "forest__min_samples_split": [10, 20, 30],
        "forest__max_depth": [None, 6, 10, 16],
    }

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    return GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=cv,
        refit=True,
        verbose=1,
    )


def fit_generic_forest_workflow(
    train_data: pd.DataFrame,
    holdout_data: Optional[pd.DataFrame] = None,
    target_col: str = DEFAULT_TARGET,
    positive_label: str = DEFAULT_POSITIVE,
    negative_label: str = DEFAULT_NEGATIVE,
    n_trees: int = 300,
    folds: int = 5,
    seed: int = 123,
) -> ForestRunResult:
    x_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]

    searcher = make_forest_search(
        train_data=train_data,
        target_col=target_col,
        n_trees=n_trees,
        folds=folds,
        seed=seed,
    )
    searcher.fit(x_train, y_train)

    best_model = searcher.best_estimator_

    cv_prob = out_of_fold_probabilities(best_model, x_train, y_train, folds=folds, seed=seed)
    best_threshold, curve = find_best_f1_threshold(
        y_train,
        cv_prob,
        positive_label=positive_label,
        negative_label=negative_label,
    )

    cv_summary = curve.loc[curve["threshold"] == best_threshold].reset_index(drop=True)
    cv_auc = float(cv_summary.loc[0, "auc"])
    cv_f1 = float(cv_summary.loc[0, "f1"])

    holdout_auc = None
    holdout_f1 = None

    if holdout_data is not None:
        x_hold = holdout_data.drop(columns=[target_col])
        y_hold = holdout_data[target_col]
        hold_prob = best_model.predict_proba(x_hold)[:, 1]
        hold_pred = threshold_to_class(
            hold_prob,
            best_threshold,
            positive_label=positive_label,
            negative_label=negative_label,
        )

        holdout_auc = float(roc_auc_score(to_binary(y_hold), hold_prob))
        holdout_f1 = float(f1_score(to_binary(y_hold), to_binary(hold_pred), zero_division=0))

    return ForestRunResult(
        best_params=searcher.best_params_,
        best_threshold=best_threshold,
        cv_auc=cv_auc,
        cv_f1=cv_f1,
        holdout_auc=holdout_auc,
        holdout_f1=holdout_f1,
        fitted_model=best_model,
        summary_table=cv_summary,
    )


def extract_feature_importance(fitted_model: Pipeline) -> pd.DataFrame:
    forest = fitted_model.named_steps["forest"]
    prep = fitted_model.named_steps["prep"]
    feature_names = prep.get_feature_names_out()

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": forest.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return importance.reset_index(drop=True)


def select_top_features(
    importance_table: pd.DataFrame,
    min_importance: float = 0.0,
    top_n: Optional[int] = None,
) -> List[str]:
    chosen = importance_table.loc[importance_table["importance"] > min_importance, "feature"].tolist()
    if top_n is not None:
        chosen = chosen[:top_n]
    return chosen
