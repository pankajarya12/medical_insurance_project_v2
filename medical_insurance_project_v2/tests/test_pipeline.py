"""Smoke tests for preprocessing + data loading."""
import pandas as pd
from src.preprocessing import add_features, split_xy, NUMERIC, CATEGORICAL

def _df():
    return pd.DataFrame([{"age":30,"sex":"male","bmi":28.5,"children":1,
                          "smoker":"no","region":"southeast","charges":3000}])

def test_add_features_columns():
    out = add_features(_df())
    for col in ["smoker_bmi","age_smoker","bmi_category","age_group"]:
        assert col in out.columns

def test_split_xy_shapes():
    X, y = split_xy(_df())
    assert list(X.columns) == NUMERIC + CATEGORICAL
    assert len(X) == len(y) == 1
