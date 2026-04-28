"""
data_loader.py
--------------
Loads CSV datasets safely with schema validation (Pydantic).

Why? A real ML pipeline must reject malformed user uploads early -
otherwise garbage rows crash training deep inside sklearn.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator
from .paths import DATA_DIR, UPLOAD_DIR, safe_path, DEFAULT_DATASET

REQUIRED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]


class InsuranceRow(BaseModel):
    """Schema for a single row of insurance data."""
    age: int = Field(ge=0, le=120)
    sex: str
    bmi: float = Field(gt=0, lt=100)
    children: int = Field(ge=0, le=20)
    smoker: str
    region: str
    charges: float = Field(ge=0)

    @field_validator("sex")
    @classmethod
    def _sex(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"male", "female"}: raise ValueError("sex must be male/female")
        return v

    @field_validator("smoker")
    @classmethod
    def _smoker(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"yes", "no"}: raise ValueError("smoker must be yes/no")
        return v


def _resolve_dataset(name: Optional[str]) -> Path:
    """Pick correct directory for given filename - never escape sandbox."""
    name = name or DEFAULT_DATASET
    # Try uploads first, then default data dir
    for base in (UPLOAD_DIR, DATA_DIR):
        try:
            p = safe_path(base, name)
            if p.exists(): return p
        except ValueError:
            continue
    raise FileNotFoundError(f"Dataset not found: {name}")


def load_dataset(name: Optional[str] = None, validate: bool = True) -> pd.DataFrame:
    """
    Load a CSV dataset by filename. `validate=True` runs Pydantic schema check
    on a sample (first 50 rows) to catch obvious format errors quickly.
    """
    path = _resolve_dataset(name)
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].dropna()

    if validate:
        errors = []
        for i, row in df.head(50).iterrows():
            try: InsuranceRow(**row.to_dict())
            except ValidationError as e:
                errors.append(f"row {i}: {e.errors()[0]['msg']}")
        if errors:
            raise ValueError("Schema validation failed:\n" + "\n".join(errors[:5]))

    return df.reset_index(drop=True)


def save_uploaded_csv(file_bytes: bytes, filename: str) -> Path:
    """
    Persist a user-uploaded CSV into UPLOAD_DIR using a SAFE filename.
    Returns the saved path.
    """
    # sanitise filename
    safe_name = Path(filename).name           # strip directories
    if not safe_name.lower().endswith(".csv"):
        raise ValueError("Only .csv files are allowed")
    if len(safe_name) > 100:
        raise ValueError("Filename too long")

    target = safe_path(UPLOAD_DIR, safe_name)
    target.write_bytes(file_bytes)

    # Quick validation - reject if not parseable
    try:
        load_dataset(safe_name, validate=True)
    except Exception as e:
        target.unlink(missing_ok=True)
        raise ValueError(f"Uploaded file rejected: {e}")
    return target
