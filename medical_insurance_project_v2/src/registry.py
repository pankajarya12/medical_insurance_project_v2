"""
registry.py
-----------
Lightweight Model Registry on top of joblib + JSON metadata.

Each registered model lives in:  models/registry/<name>__v<version>/
                                    ├── model.pkl
                                    └── meta.json

A pointer file `models/registry/active.json` records which version is
currently "production" - used by the Streamlit app for predictions.
"""
from __future__ import annotations
import json, time, uuid
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any
import joblib
from .paths import MODELS_DIR

REG_DIR = MODELS_DIR / "registry"
REG_DIR.mkdir(parents=True, exist_ok=True)
ACTIVE_PTR = REG_DIR / "active.json"


@dataclass
class ModelCard:
    name: str
    version: int
    algorithm: str
    metrics: dict[str, float]
    params: dict[str, Any]
    feature_count: int
    rows_trained_on: int
    dataset: str
    created_at: str
    model_id: str

    def to_dict(self) -> dict: return asdict(self)


def _next_version(name: str) -> int:
    existing = [p for p in REG_DIR.glob(f"{name}__v*") if p.is_dir()]
    return 1 + max((int(p.name.split("v")[-1]) for p in existing), default=0)


def register(model: Any, *, name: str, algorithm: str, metrics: dict,
             params: dict, feature_count: int, rows: int, dataset: str) -> ModelCard:
    """Persist a trained model + metadata. Returns the new ModelCard."""
    version = _next_version(name)
    folder = REG_DIR / f"{name}__v{version}"
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / "model.pkl")

    card = ModelCard(
        name=name, version=version, algorithm=algorithm,
        metrics={k: float(v) for k, v in metrics.items()},
        params={k: str(v) for k, v in params.items()},
        feature_count=feature_count, rows_trained_on=rows,
        dataset=dataset,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        model_id=uuid.uuid4().hex[:12],
    )
    (folder / "meta.json").write_text(json.dumps(card.to_dict(), indent=2))

    # First model auto-promoted to active
    if not ACTIVE_PTR.exists():
        set_active(name, version)
    return card


def list_models() -> list[ModelCard]:
    cards = []
    for folder in sorted(REG_DIR.glob("*__v*")):
        meta = folder / "meta.json"
        if meta.exists():
            cards.append(ModelCard(**json.loads(meta.read_text())))
    return cards


def load(name: str, version: int):
    return joblib.load(REG_DIR / f"{name}__v{version}" / "model.pkl")


def set_active(name: str, version: int) -> None:
    folder = REG_DIR / f"{name}__v{version}"
    if not folder.exists():
        raise FileNotFoundError(folder)
    ACTIVE_PTR.write_text(json.dumps({"name": name, "version": version}))


def get_active() -> tuple[str, int] | None:
    if not ACTIVE_PTR.exists(): return None
    d = json.loads(ACTIVE_PTR.read_text())
    return d["name"], d["version"]


def load_active():
    info = get_active()
    if not info: raise RuntimeError("No active model. Train one first.")
    return load(*info), info


def delete(name: str, version: int) -> None:
    import shutil
    folder = REG_DIR / f"{name}__v{version}"
    if folder.exists(): shutil.rmtree(folder)
    info = get_active()
    if info and info == (name, version):
        ACTIVE_PTR.unlink(missing_ok=True)
