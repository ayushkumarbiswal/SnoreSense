"""End-to-end screening pipeline: raw audio → risk category (the shared API layer the UI calls)."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .audio import preprocess, quality_check
from .config import (APNEA_PAUSE_SEC, FLAG_THRESHOLD, LABELS, MIN_ANALYSIS_SEC, RECORDING_FLAG_FRACTION, SR,
                     WINDOW_HOP_SEC, WINDOW_SEC)
from .features import detect_events, handcrafted_features, log_mel, pauses_from_events
from .models import ModelBundle


def _windows(n: int, sr: int = SR) -> list[tuple[int, int]]:
    win, hop = int(WINDOW_SEC * sr), int(WINDOW_HOP_SEC * sr)
    if n <= win:
        return [(0, n)]
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] + win < n:                     # cover the tail with a final, end-aligned window
        starts.append(n - win)
    return [(s, s + win) for s in starts]


def _envelope(y: np.ndarray, points: int = 1500, sr: int = SR) -> dict:
    """Min/max envelope for plotting long waveforms quickly."""
    n = max(1, len(y) // points)
    m = len(y) // n
    blocks = y[: m * n].reshape(m, n)
    t = (np.arange(m) * n + n / 2) / sr
    return {"t": t.round(3).tolist(), "min": blocks.min(1).round(4).tolist(), "max": blocks.max(1).round(4).tolist()}


def analyze(y_raw: np.ndarray, bundle: ModelBundle, progress=None) -> dict:
    """Run the full screening. `progress(step, fraction)` is called for UI updates."""
    step = progress or (lambda *_: None)

    step("Checking recording quality", 0.05)
    q = quality_check(y_raw)
    result = {"quality": asdict(q), "model_version": bundle.id, "decision_model": bundle.decision_model}
    if not q.ok:
        return {**result, "status": "rejected", "issues": q.issues}

    step("Reducing background noise and trimming silence at the edges", 0.2)
    y, offset = preprocess(y_raw)
    duration = len(y) / SR
    if duration < MIN_ANALYSIS_SEC:
        return {**result, "status": "rejected",
                "issues": ["After removing silence at the start and end, less than "
                           f"{MIN_ANALYSIS_SEC:.0f} s of breathing sounds remain. Please record for longer."]}

    step("Detecting breathing and snoring sounds", 0.35)
    events, db, thr = detect_events(y)
    if len(events) < 3:
        return {**result, "status": "rejected",
                "issues": ["We couldn't find clear breathing or snoring sounds in this recording. "
                           "Place the phone about 1 metre from the sleeper's head and try again."]}
    pauses = pauses_from_events(events)

    windows = []
    spans = _windows(len(y))
    for i, (a, b) in enumerate(spans):
        step(f"Extracting features and running models (window {i + 1} of {len(spans)})", 0.4 + 0.55 * i / len(spans))
        seg = y[a:b]
        seg_events, _, _ = detect_events(seg)
        probs = bundle.predict(handcrafted_features(seg, events=seg_events), log_mel(seg))
        windows.append({"start": a / SR, "end": b / SR, **probs, "flagged": probs["decision"] >= FLAG_THRESHOLD})

    flagged = [w for w in windows if w["flagged"]]
    frac = len(flagged) / len(windows)
    label = int(frac >= RECORDING_FLAG_FRACTION) if len(windows) > 1 else int(windows[0]["flagged"])
    if label:
        confidence = float(np.mean([w["decision"] for w in flagged]))
    else:
        confidence = float(np.mean([1 - w["decision"] for w in windows]))
    hours = duration / 3600
    longest = max((b - a for a, b in pauses), default=0.0)
    summary = {
        "duration_sec": round(duration, 1),
        "n_events": len(events),
        "events_per_min": round(len(events) / (duration / 60), 1),
        "n_pauses": len(pauses),
        "longest_pause_sec": round(longest, 1),
        "pauses_per_hour": round(len(pauses) / hours, 1),
        "flagged_fraction": round(frac, 3),
        "n_windows": len(windows),
        "mean_probs": {k: round(float(np.mean([w[k] for w in windows])), 3) for k in ("rf", "svm", "cnn", "ensemble")},
    }
    step("Done", 1.0)
    return {
        **result,
        "status": "ok",
        "label": label,
        "label_text": LABELS[label],
        "confidence": round(confidence, 3),
        "summary": summary,
        "windows": windows,
        "events": [{"start": round(e.start, 2), "end": round(e.end, 2), "peak_db": round(e.peak_db, 1)} for e in events],
        "pauses": [{"start": round(a, 2), "end": round(b, 2)} for a, b in pauses],
        "trim_offset_sec": round(offset, 2),
        "envelope": _envelope(y),
        "pause_threshold_sec": APNEA_PAUSE_SEC,
    }


EXPLANATIONS = {
    0: ("Your breathing sounds followed a steady rhythm. We did not find repeated long pauses "
        "(10 seconds or more) followed by loud recovery snores — the acoustic pattern linked with "
        "sleep apnea. Regular snoring on its own is common and usually harmless."),
    1: ("Parts of this recording show an irregular pattern: breathing sounds stop for 10 seconds or "
        "longer and then restart with a loud snore or gasp, or breathing becomes very shallow for a "
        "while. This pattern is statistically associated with sleep apnea, but it can also be caused "
        "by recording issues (the sleeper turning away, background noise). It is a reason to talk to a "
        "doctor, not a diagnosis."),
}

NEXT_STEPS = {
    0: ["Keep screening occasionally — patterns can change with weight, alcohol or sleeping position.",
        "If you feel very sleepy during the day, wake up gasping, or someone notices you stop breathing, "
        "see a doctor even if this result is normal."],
    1: ["Book an appointment with a doctor or sleep specialist and show them this result and history.",
        "A sleep study (polysomnography or a home sleep test) is the only way to diagnose sleep apnea.",
        "Record another night to check whether the pattern repeats — a single recording can be misleading.",
        "Seek urgent care if breathing pauses come with chest pain, severe breathlessness or confusion."],
}
