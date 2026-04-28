"""
paths.py
--------
Centralised, HARDENED path handling for the entire project.

Why a dedicated module?
- Avoids hard-coded "../data/file.csv" strings scattered across files.
- Prevents path-traversal attacks (`../../etc/passwd`) when users upload files.
- Makes the project portable (works locally, in Docker, on any OS).

Every other module imports from here instead of building paths manually.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present (silent if missing)
load_dotenv()

# Project root = parent of /src
ROOT: Path = Path(__file__).resolve().parent.parent

# Configurable directories (overridable via .env)
DATA_DIR:    Path = ROOT / os.getenv("DATA_DIR", "data")
MODELS_DIR:  Path = ROOT / os.getenv("MODELS_DIR", "models")
REPORTS_DIR: Path = ROOT / os.getenv("REPORTS_DIR", "reports")
EDA_DIR:     Path = ROOT / "app" / "eda_images"
UPLOAD_DIR:  Path = DATA_DIR / "uploads"

# Ensure all dirs exist
for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, EDA_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_DATASET = os.getenv("DEFAULT_DATASET", "medical_insurance.csv")


def safe_path(base: Path, user_path: str | os.PathLike) -> Path:
    """
    Resolve `user_path` inside `base` and refuse to escape it.

    Blocks classic path-traversal: '../../etc/passwd', absolute paths,
    symlink tricks, NUL bytes, etc.  Raises ValueError on violation.
    """
    base = base.resolve()
    candidate = (base / user_path).resolve()
    if not str(candidate).startswith(str(base) + os.sep) and candidate != base:
        raise ValueError(f"Unsafe path detected: {user_path!r} escapes {base}")
    return candidate


def list_datasets() -> list[str]:
    """Return all .csv files available in the data dir (recursive, 1 level)."""
    files = list(DATA_DIR.glob("*.csv")) + list(UPLOAD_DIR.glob("*.csv"))
    return sorted({f.name for f in files})
