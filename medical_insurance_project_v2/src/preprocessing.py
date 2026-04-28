"""
preprocessing.py
----------------
Feature engineering + sklearn ColumnTransformer pipeline.

Domain features added:
  smoker_bmi   - smokers with high BMI cost the most
  age_smoker   - age impact is amplified for smokers
  bmi_category - clinical underweight/normal/overweight/obese buckets
  age_group    - young/adult/senior buckets
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC = ["age", "bmi", "children", "smoker_bmi", "age_smoker"]
CATEGORICAL = ["sex", "smoker", "region", "bmi_category", "age_group"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Pure function - returns a new dataframe."""
    df = df.copy()
    smoker_num = (df["smoker"].str.lower() == "yes").astype(int)
    df["smoker_bmi"] = smoker_num * df["bmi"]
    df["age_smoker"] = smoker_num * df["age"]

    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"]
    ).astype(str)

    df["age_group"] = pd.cut(
        df["age"], bins=[0, 30, 50, 120],
        labels=["young", "adult", "senior"]
    ).astype(str)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Numeric -> StandardScaler ; Categorical -> OneHotEncoder."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def split_xy(df: pd.DataFrame):
    """Returns (X, y) where X has engineered features, y is `charges`."""
    df = add_features(df)
    return df[NUMERIC + CATEGORICAL], df["charges"]
