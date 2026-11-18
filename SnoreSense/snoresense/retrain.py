"""Retrain the feature-based models with newly uploaded, labelled recordings (US-11)."""
from __future__ import annotations

import numpy as np

from .audio import load_audio, preprocess
from .config import MODELS_DIR, SR, WINDOW_HOP_SEC, WINDOW_SEC
from .features import detect_events, handcrafted_features
from .models import load_registry, register_version


def clip_windows(path: str) -> list[np.ndarray]:
    """Feature vectors for every 30 s window of a labelled clip (same path as inference)."""
    y, _ = preprocess(load_audio(path))
    win, hop = int(WINDOW_SEC * SR), int(WINDOW_HOP_SEC * SR)
    spans = [(0, len(y))] if len(y) <= win else [(s, s + win) for s in range(0, len(y) - win + 1, hop)]
    out = []
    for a, b in spans:
        seg = y[a:b]
        if len(seg) < 15 * SR:
            continue
        ev, _, _ = detect_events(seg)
        out.append(handcrafted_features(seg, events=ev))
    return out


def retrain(clips: list[dict], note: str = "", log=lambda *_: None) -> dict:
    """clips: [{"path": ..., "label": 0/1}]. Returns the new candidate registry version."""
    from .train import evaluate, train_classic

    base = np.load(MODELS_DIR / "train_features.npz")
    X_tr, y_tr, X_te, y_te = base["X_tr"], base["y_tr"], base["X_te"], base["y_te"]
    extra_X, extra_y = [], []
    for c in clips:
        try:
            feats = clip_windows(c["path"])
        except Exception as e:                      # unreadable file: skip, report
            log(f"skipped {c['path']}: {e}")
            continue
        extra_X += feats
        extra_y += [c["label"]] * len(feats)
        log(f"{c.get('filename', c['path'])}: {len(feats)} windows")
    log("training Random Forest and SVM…")
    rf, svm = train_classic(X_tr, y_tr, np.array(extra_X) if extra_X else None, np.array(extra_y) if extra_y else None)
    p_rf, p_svm = rf.predict_proba(X_te)[:, 1], svm.predict_proba(X_te)[:, 1]
    test = {}
    for k, p in (("rf", p_rf), ("svm", p_svm)):
        e = evaluate(y_te, p)
        test[k] = {m: e[m] for m in ("accuracy", "f1", "recall", "specificity", "roc_auc")}
    reg = load_registry()
    base_v = next(v for v in reg["versions"] if v["id"] == reg["active"])
    test["cnn"] = base_v["test"].get("cnn", {})
    note = note or (f"Retrained RF + SVM with {len(clips)} uploaded clip(s) ({len(extra_X)} windows); "
                    f"CNN carried over from {base_v['id']}")
    return register_version(rf, svm, test, int(len(y_tr) + len(extra_y)), len(clips), base_v, note)
