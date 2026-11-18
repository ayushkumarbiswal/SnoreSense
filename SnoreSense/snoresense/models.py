"""Model registry and prediction (Module 2 — Ajit Kumar Pradhan: US-03, 10, 11)."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from . import cnn
from .config import MODELS_DIR

MODEL_LABELS = {
    "rf": "Random Forest (handcrafted features)",
    "svm": "SVM (handcrafted features)",
    "cnn": "CNN (log-mel spectrogram)",
    "ensemble": "Ensemble (average of all three)",
}


def registry_path() -> Path:
    return MODELS_DIR / "registry.json"


def load_registry() -> dict:
    return json.loads(registry_path().read_text())


def save_registry(reg: dict) -> None:
    registry_path().write_text(json.dumps(reg, indent=1))


def load_metrics() -> dict:
    p = MODELS_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


class ModelBundle:
    """The three trained models of one registry version."""

    def __init__(self, version: dict):
        self.version = version
        self.id = version["id"]
        self.decision_model = version.get("decision_model", "ensemble")
        f = version["files"]
        self.rf = joblib.load(MODELS_DIR / f["rf"])
        self.svm = joblib.load(MODELS_DIR / f["svm"])
        self.cnn_params = cnn.load(MODELS_DIR / f["cnn"])

    @classmethod
    def active(cls) -> "ModelBundle":
        reg = load_registry()
        v = next(v for v in reg["versions"] if v["id"] == reg["active"])
        return cls(v)

    def predict(self, feats: np.ndarray, mel: np.ndarray) -> dict[str, float]:
        x = feats.reshape(1, -1)
        out = {
            "rf": float(self.rf.predict_proba(x)[0, 1]),
            "svm": float(self.svm.predict_proba(x)[0, 1]),
            "cnn": cnn.predict_proba(self.cnn_params, mel),
        }
        out["ensemble"] = float(np.mean([out["rf"], out["svm"], out["cnn"]]))
        out["decision"] = out[self.decision_model]
        return out


def register_version(rf, svm, test_metrics: dict, n_train: int, n_extra: int, base: dict, note: str) -> dict:
    """Save a retrained candidate version (RF + SVM retrained, CNN carried over from `base`)."""
    reg = load_registry()
    major, minor = base["id"][1:].split(".")
    existing = {v["id"] for v in reg["versions"]}
    n = int(minor) + 1
    while f"v{major}.{n}" in existing:
        n += 1
    vid = f"v{major}.{n}"
    folder = MODELS_DIR / "versions" / vid
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, folder / "rf.joblib", compress=3)
    joblib.dump(svm, folder / "svm.joblib", compress=3)
    version = {
        "id": vid, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": "candidate",
        "description": note,
        "files": {"rf": f"versions/{vid}/rf.joblib", "svm": f"versions/{vid}/svm.joblib", "cnn": base["files"]["cnn"]},
        "decision_model": base.get("decision_model", "ensemble"),
        "train_windows": n_train, "extra_clips": n_extra, "test": test_metrics,
    }
    reg["versions"].append(version)
    save_registry(reg)
    return version


def set_active(version_id: str, decision_model: str | None = None) -> None:
    reg = load_registry()
    for v in reg["versions"]:
        if v["id"] == version_id:
            v["status"] = "active"
            if decision_model:
                v["decision_model"] = decision_model
        elif v["status"] == "active":
            v["status"] = "previous"
    reg["active"] = version_id
    save_registry(reg)


def delete_version(version_id: str) -> None:
    reg = load_registry()
    if version_id == reg["active"] or version_id == "v1.0":
        raise ValueError("The active version and the base version can't be deleted.")
    reg["versions"] = [v for v in reg["versions"] if v["id"] != version_id]
    save_registry(reg)
    shutil.rmtree(MODELS_DIR / "versions" / version_id, ignore_errors=True)
