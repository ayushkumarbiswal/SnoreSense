"""Train and evaluate all models (Module 2 — Ajit Kumar Pradhan: US-03, 04, 10, 11).

    python -m snoresense.train            # build dataset (cached), train RF + SVM + CNN, evaluate
    python -m snoresense.train --fast     # smaller dataset, for a quick check

Outputs (models/):
    rf.joblib, svm.joblib, cnn.npz    trained models
    metrics.json                      test metrics, confusion matrices, ROC curves, noise-robustness,
                                      feature importance — shown on the admin "Model comparison" page
    registry.json                     model versions (admin "Model management" page)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from multiprocessing import Pool

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import cnn
from .audio import preprocess
from .config import CACHE_DIR, MODELS_DIR, SR, WINDOW_SEC
from .features import FEATURE_NAMES, detect_events, handcrafted_features, log_mel

_POOLS = None


def _init_worker(folds):
    global _POOLS
    from . import dataset
    _POOLS = dataset.load_pools(folds)


def _make_example(args):
    label, seed, snr = args
    from . import dataset
    rng = np.random.default_rng(seed)
    y = dataset.make_window(rng, _POOLS, label, WINDOW_SEC, snr)
    return window_inputs(y)


def window_inputs(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw 30 s window → (handcrafted feature vector, log-mel). Same path as inference."""
    y, _ = preprocess(y, trim=False)
    events, _, _ = detect_events(y)
    return handcrafted_features(y, events=events), log_mel(y)


def build_split(name: str, folds, n_per_class: int, seed: int, snr=None, workers: int = 2):
    path = CACHE_DIR / f"{name}.npz"
    if path.exists():
        d = np.load(path)
        if len(d["y"]) == 2 * n_per_class:
            return d["X"], d["M"].astype(np.float32), d["y"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    labels = np.array([0] * n_per_class + [1] * n_per_class)
    tasks = [(int(l), seed * 100_000 + i, snr) for i, l in enumerate(labels)]
    t0 = time.time()
    with Pool(workers, initializer=_init_worker, initargs=(folds,)) as pool:
        out = pool.map(_make_example, tasks, chunksize=8)
    X = np.stack([o[0] for o in out])
    T = min(o[1].shape[1] for o in out)
    M = np.stack([o[1][:, :T] for o in out]).astype(np.float16)
    np.savez_compressed(path, X=X, M=M, y=labels)
    print(f"  {name}: {len(labels)} windows in {time.time() - t0:.0f} s")
    return X, M.astype(np.float32), labels


def evaluate(y_true, prob) -> dict:
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_true, prob)
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int))
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),         # sensitivity
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "roc": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()},
    }


def cnn_predict(params, M) -> np.ndarray:
    return np.array([cnn.predict_proba(params, m) for m in M])


def train_classic(X, y, extra_X=None, extra_y=None):
    if extra_X is not None and len(extra_X):
        X, y = np.vstack([X, extra_X]), np.concatenate([y, extra_y])
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                n_jobs=-1, random_state=42).fit(X, y)
    svm = make_pipeline(StandardScaler(), SVC(C=3.0, gamma="scale", probability=True, random_state=42)).fit(X, y)
    return rf, svm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--skip-cnn", action="store_true", help="reuse models/cnn.npz (log-mel input unchanged)")
    args = ap.parse_args()
    from .dataset import TEST_FOLDS, TRAIN_FOLDS

    n_train, n_test, n_rob = (150, 60, 30) if args.fast else (750, 250, 100)
    print("Building dataset (real ESC-50 sounds → breathing patterns → phone-recording conditions)")
    X_tr, M_tr, y_tr = build_split("train", TRAIN_FOLDS, n_train, seed=1)
    X_te, M_te, y_te = build_split("test", TEST_FOLDS, n_test, seed=2)
    robust_sets = {snr: build_split(f"robust_snr{snr}", TEST_FOLDS, n_rob, seed=10 + snr, snr=float(snr))
                   for snr in (0, 5, 10, 20, 30)}

    print("Training Random Forest and SVM on handcrafted features")
    t0 = time.time()
    rf, svm = train_classic(X_tr, y_tr)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    cv_rf = cross_val_score(RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=0),
                            X_tr, y_tr, cv=cv, scoring="f1")
    cv_svm = cross_val_score(make_pipeline(StandardScaler(), SVC(C=3.0, gamma="scale")), X_tr, y_tr, cv=cv, scoring="f1")
    t_classic = time.time() - t0

    hist_path = MODELS_DIR / "cnn_history.json"
    if args.skip_cnn:
        print("Reusing trained CNN (models/cnn.npz)")
        params = cnn.load(MODELS_DIR / "cnn.npz")
        saved = json.loads(hist_path.read_text())
        history, t_cnn = saved["history"], saved["seconds"]
    else:
        print("Training CNN on log-mel spectrograms (JAX)")
        t0 = time.time()
        rng = np.random.default_rng(0)
        va = rng.random(len(y_tr)) < 0.12
        params, history = cnn.train(M_tr[~va], y_tr[~va], M_tr[va], y_tr[va], epochs=args.epochs)
        t_cnn = time.time() - t0
        MODELS_DIR.mkdir(exist_ok=True)
        hist_path.write_text(json.dumps({"history": history, "seconds": round(t_cnn, 1)}))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(rf, MODELS_DIR / "rf.joblib", compress=3)
    joblib.dump(svm, MODELS_DIR / "svm.joblib", compress=3)
    np.savez_compressed(MODELS_DIR / "cnn.npz", **params)
    # feature tables let the admin page retrain RF/SVM without the raw dataset (US-11)
    np.savez_compressed(MODELS_DIR / "train_features.npz", X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te)

    print("Evaluating on held-out test set (fold 5 sound sources)")
    p = {"rf": rf.predict_proba(X_te)[:, 1], "svm": svm.predict_proba(X_te)[:, 1], "cnn": cnn_predict(params, M_te)}
    p["ensemble"] = (p["rf"] + p["svm"] + p["cnn"]) / 3
    test = {k: evaluate(y_te, v) for k, v in p.items()}
    robustness = {k: {} for k in p}
    for snr, (Xr, Mr, yr) in robust_sets.items():
        pr = {"rf": rf.predict_proba(Xr)[:, 1], "svm": svm.predict_proba(Xr)[:, 1], "cnn": cnn_predict(params, Mr)}
        pr["ensemble"] = (pr["rf"] + pr["svm"] + pr["cnn"]) / 3
        for k, v in pr.items():
            e = evaluate(yr, v)
            robustness[k][str(snr)] = {m: e[m] for m in ("accuracy", "f1", "recall", "specificity", "roc_auc")}
    imp = sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda t: -t[1])

    for k, v in test.items():
        print(f"  {k:9s} acc {v['accuracy']:.3f}  F1 {v['f1']:.3f}  sens {v['recall']:.3f}  spec {v['specificity']:.3f}  AUC {v['roc_auc']:.3f}")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics = {
        "created": now,
        "dataset": {
            "source": "ESC-50 (real snoring, breathing, coughing and bedroom-background recordings), "
                      "arranged into normal vs apnea-like breathing patterns under phone-recording conditions",
            "window_sec": WINDOW_SEC, "sample_rate": SR,
            "train_windows": int(len(y_tr)), "test_windows": int(len(y_te)),
            "train_folds": list(TRAIN_FOLDS), "test_folds": list(TEST_FOLDS),
            "robustness_windows_per_snr": int(2 * n_rob),
        },
        "test": test,
        "robustness": robustness,
        "cv_f1": {"rf": [float(x) for x in cv_rf], "svm": [float(x) for x in cv_svm]},
        "feature_importance": [[n, float(v)] for n, v in imp],
        "cnn_history": history,
        "train_seconds": {"classic": round(t_classic, 1), "cnn": round(t_cnn, 1)},
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=1))
    registry = {
        "active": "v1.0",
        "versions": [{
            "id": "v1.0", "created": now, "status": "active",
            "description": "Initial release: RF + SVM (handcrafted features) and CNN (log-mel), ensemble of all three",
            "files": {"rf": "rf.joblib", "svm": "svm.joblib", "cnn": "cnn.npz"},
            "decision_model": "ensemble",
            "train_windows": int(len(y_tr)), "extra_clips": 0,
            "test": {k: {m: v[m] for m in ("accuracy", "f1", "recall", "specificity", "roc_auc")} for k, v in test.items()},
        }],
    }
    (MODELS_DIR / "registry.json").write_text(json.dumps(registry, indent=1))
    print("Saved models and metrics to", MODELS_DIR)


if __name__ == "__main__":
    main()
