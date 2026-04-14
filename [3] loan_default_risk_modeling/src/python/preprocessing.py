from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

TARGET_NAME = "loan_default"
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"
LABEL_ORDER = [NEGATIVE_LABEL, POSITIVE_LABEL]


def load_table(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)

    raise ValueError(
        f"Unsupported file type: {suffix}. Convert RDS files to CSV, Parquet, or Pickle first."
    )


def normalize_target(frame: pd.DataFrame, target_col: str = TARGET_NAME) -> pd.DataFrame:
    copy_df = frame.copy()
    copy_df[target_col] = copy_df[target_col].astype(str)
    copy_df[target_col] = pd.Categorical(copy_df[target_col], categories=LABEL_ORDER, ordered=True)
    return copy_df


def to_binary(series: pd.Series) -> pd.Series:
    return series.map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}).astype(int)


def make_balanced_subset(frame: pd.DataFrame, target_col: str = TARGET_NAME, seed: int = 123) -> pd.DataFrame:
    yes_block = frame.loc[frame[target_col] == POSITIVE_LABEL]
    no_block = frame.loc[frame[target_col] == NEGATIVE_LABEL]

    if len(yes_block) == 0 or len(no_block) == 0:
        raise ValueError("Both classes must be present for undersampling.")

    minority_size = min(len(yes_block), len(no_block))
    yes_draw = yes_block.sample(n=minority_size, random_state=seed, replace=False)
    no_draw = no_block.sample(n=minority_size, random_state=seed, replace=False)

    mixed = pd.concat([yes_draw, no_draw], axis=0)
    return mixed.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_feature_types(frame: pd.DataFrame, target_col: str = TARGET_NAME) -> Tuple[List[str], List[str]]:
    predictors = frame.drop(columns=[target_col], errors="ignore")
    numeric_cols = predictors.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in predictors.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def make_preprocessor(
    frame: pd.DataFrame,
    target_col: str = TARGET_NAME,
    scaling: str = "standard",
) -> ColumnTransformer:
    num_cols, cat_cols = split_feature_types(frame, target_col=target_col)
    scaler = MinMaxScaler() if scaling == "minmax" else StandardScaler()

    numeric_flow = Pipeline(
        steps=[
            ("fill_num", SimpleImputer(strategy="median")),
            ("scale_num", scaler),
        ]
    )

    categorical_flow = Pipeline(
        steps=[
            ("fill_cat", SimpleImputer(strategy="most_frequent")),
            ("encode_cat", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numbers", numeric_flow, num_cols),
            ("categories", categorical_flow, cat_cols),
        ],
        remainder="drop",
    )
