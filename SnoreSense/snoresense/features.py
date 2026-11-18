"""Breathing-event detection and feature extraction (Module 1 → Module 2 interface).

The preprocessing module outputs a standardised feature vector (for Random Forest / SVM)
and a log-mel spectrogram (for the CNN); both feed the classification module.
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .audio import frame_db
from .config import APNEA_PAUSE_SEC, CNN_HOP, CNN_N_MELS, FRAME_HOP, N_FFT, N_MFCC, SR

FPS = SR / FRAME_HOP


# --------------------------------------------------------------------------- events
@dataclass
class Event:
    start: float
    end: float
    peak_db: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def band_db(y: np.ndarray, sr: int = SR, smooth_sec: float = 0.16) -> np.ndarray:
    """Frame energy (dB) in the 80–2500 Hz band where snoring and breathing energy lies.

    Band-limiting ignores much of the hiss from fans, rain or traffic, which makes event and
    pause detection far more reliable on noisy phone recordings than broadband loudness.
    """
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=FRAME_HOP)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    db = 10 * np.log10(S[(f >= 80) & (f <= 2500)].sum(axis=0) + 1e-10)
    k = max(1, int(smooth_sec * FPS))
    return np.convolve(db, np.ones(k) / k, mode="same")


def detect_events(y: np.ndarray, sr: int = SR) -> tuple[list[Event], np.ndarray, float]:
    """Find breathing / snoring sound events with an adaptive, hysteresis threshold.

    Returns (events, frame-level dB envelope, threshold in dB).
    """
    if len(y) < N_FFT * 2:
        return [], np.array([-120.0]), -120.0
    db_s = band_db(y, sr)
    floor, loud = np.percentile(db_s, 10), np.percentile(db_s, 97)
    span = max(loud - floor, 1e-3)
    on = floor + max(6.0, 0.4 * span)
    off = floor + max(4.0, 0.3 * span)
    active = np.zeros(len(db_s), dtype=bool)
    state = False
    for i, v in enumerate(db_s):
        state = v > on if not state else v > off
        active[i] = state
    # merge gaps shorter than 0.3 s, drop blips shorter than 0.2 s
    events: list[Event] = []
    idx = np.flatnonzero(np.diff(np.concatenate([[0], active.astype(int), [0]])))
    segs = list(zip(idx[::2], idx[1::2]))
    merged: list[list[int]] = []
    for s, e in segs:
        if merged and (s - merged[-1][1]) / FPS < 0.3:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    for s, e in merged:
        dur = (e - s) / FPS
        if dur < 0.2:
            continue
        events.append(Event(start=s / FPS, end=e / FPS, peak_db=float(db_s[s:e].max())))
    return events, db_s, float(on)


def pauses_from_events(events: list[Event], min_sec: float = APNEA_PAUSE_SEC) -> list[tuple[float, float]]:
    """Silent gaps between consecutive events lasting at least `min_sec`."""
    out = []
    for a, b in zip(events, events[1:]):
        if b.start - a.end >= min_sec:
            out.append((a.end, b.start))
    return out


# --------------------------------------------------------------------------- handcrafted features
FEATURE_NAMES: list[str] = (
    [f"mfcc{i}_mean" for i in range(1, N_MFCC + 1)]
    + [f"mfcc{i}_std" for i in range(1, N_MFCC + 1)]
    + ["centroid_mean", "centroid_std", "bandwidth_mean", "rolloff_mean", "flatness_mean",
       "zcr_mean", "zcr_std", "rms_db_std"]
    + ["events_per_min", "event_dur_mean", "ioi_mean", "ioi_cv", "max_gap_sec", "n_pauses_10s",
       "pause_time_frac", "activity_frac", "event_loudness_std", "post_pause_loudness",
       "periodicity", "loudness_range", "longest_low_span_sec"]
    + [f"quiet{int(t * 100)}_{k}" for t in (0.2, 0.35, 0.5) for k in ("longest_sec", "runs_10s", "frac")]
    + ["env_periodicity", "env_dynamic_range_db"]
)
QUIET_LEVELS = (0.2, 0.35, 0.5)


def _runs(mask: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return (idx[1::2] - idx[::2]) / FPS if len(idx) else np.array([0.0])


def envelope_features(y: np.ndarray, sr: int = SR) -> list[float]:
    """Segmentation-free pause statistics on the band-limited loudness envelope.

    The envelope is rescaled to [0, 1] between its 5th and 98th percentile; for several "quiet"
    levels we measure the longest quiet run, the number of quiet runs ≥ 10 s and the fraction of
    time spent quiet. These stay reliable even when individual events are hard to separate.
    """
    db = band_db(y, sr, smooth_sec=0.5)
    lo, hi = np.percentile(db, 5), np.percentile(db, 98)
    r = (db - lo) / (hi - lo + 1e-9)
    out: list[float] = []
    for t in QUIET_LEVELS:
        runs = _runs(r < t)
        out += [float(runs.max()), float((runs >= APNEA_PAUSE_SEC).sum()), float((r < t).mean())]
    x = r - r.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0] + 1e-9
    a, b = int(2 * FPS), min(int(8 * FPS), len(ac) - 1)
    out += [float(ac[a:b].max()) if b > a else 0.0, float(hi - lo)]
    return out


def _periodicity(active: np.ndarray) -> float:
    """Strength of the breathing rhythm: peak autocorrelation of the activity signal at 2–8 s lags."""
    x = active.astype(float) - active.mean()
    if np.allclose(x, 0):
        return 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0] + 1e-9
    lo, hi = int(2 * FPS), min(int(8 * FPS), len(ac) - 1)
    return float(ac[lo:hi].max()) if hi > lo else 0.0


def _longest_low_span(events: list[Event], total: float, drop_db: float = 10.0) -> float:
    """Longest stretch with no breathing sound louder than (typical level - drop_db).

    Covers both silent pauses (apnea-like) and runs of very shallow breathing (hypopnea-like).
    """
    if not events:
        return total
    ref = np.median([e.peak_db for e in events])
    strong = [e for e in events if e.peak_db >= ref - drop_db]
    if not strong:
        return total
    edges = [0.0] + [x for e in strong for x in (e.start, e.end)] + [total]
    return float(max(edges[i + 1] - edges[i] for i in range(0, len(edges) - 1, 2)))


def handcrafted_features(y: np.ndarray, sr: int = SR, events: list[Event] | None = None) -> np.ndarray:
    """Fixed-length feature vector (len(FEATURE_NAMES)) for one analysis window."""
    if events is None:
        events, db, _ = detect_events(y, sr)
    else:
        db = frame_db(y)
    dur_min = max(len(y) / sr / 60, 1e-6)

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=FRAME_HOP)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=40)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)
    # weight spectral statistics towards sound (not silence) frames
    loud = db > np.percentile(db, 60)
    loud = loud[: S.shape[1]] if loud.sum() > 3 else np.ones(S.shape[1], dtype=bool)
    cent = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(S=np.sqrt(S), sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(S=np.sqrt(S), sr=sr)[0]
    flat = librosa.feature.spectral_flatness(S=np.sqrt(S))[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=FRAME_HOP)[0]
    n = min(len(loud), len(cent), len(zcr))
    loud = loud[:n]

    spectral = [
        cent[:n][loud].mean() / 1000, cent[:n][loud].std() / 1000, bw[:n][loud].mean() / 1000,
        roll[:n][loud].mean() / 1000, flat[:n][loud].mean(), zcr[:n][loud].mean(), zcr[:n][loud].std(),
        float(np.std(db)),
    ]

    # ---- event / pause statistics (the clinically motivated part)
    total = len(y) / sr
    onsets = np.array([e.start for e in events])
    ioi = np.diff(onsets) if len(onsets) > 1 else np.array([])
    gaps = [b.start - a.end for a, b in zip(events, events[1:])]
    if events:
        gaps_all = [events[0].start] + gaps + [total - events[-1].end]
    else:
        gaps_all = [total]
    pauses = [g for g in gaps if g >= APNEA_PAUSE_SEC]
    peaks = np.array([e.peak_db for e in events]) if events else np.array([-80.0])
    post_pause = 0.0
    if pauses:
        # loudness of the first sound after each long pause relative to the median event
        rel = [events[i + 1].peak_db - np.median(peaks) for i, g in enumerate(gaps) if g >= APNEA_PAUSE_SEC]
        post_pause = float(np.mean(rel))
    active = np.zeros(int(total * FPS) + 1, dtype=bool)
    for e in events:
        active[int(e.start * FPS):int(e.end * FPS) + 1] = True
    temporal = [
        len(events) / dur_min,
        float(np.mean([e.duration for e in events])) if events else 0.0,
        float(ioi.mean()) if len(ioi) else total,
        float(ioi.std() / (ioi.mean() + 1e-9)) if len(ioi) > 1 else 0.0,
        float(max(gaps_all)),
        float(len(pauses)) / (total / 30.0),           # pauses per 30 s
        float(sum(pauses) / total),
        float(active.mean()),
        float(peaks.std()) if len(peaks) > 1 else 0.0,
        post_pause,
        _periodicity(active),
        float(np.percentile(peaks, 90) - np.percentile(peaks, 10)) if len(peaks) > 1 else 0.0,
        _longest_low_span(events, total),
    ]
    feats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), spectral, temporal,
                            envelope_features(y, sr)]).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- CNN input
def log_mel(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Log-mel spectrogram (CNN_N_MELS × T), scaled to roughly [-1, 1] relative to the clip's peak."""
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT * 2, hop_length=CNN_HOP,
                                       n_mels=CNN_N_MELS, fmin=50, fmax=sr // 2)
    db = librosa.power_to_db(m, ref=np.max, top_db=80.0)      # 0 dB = loudest bin, floor -80
    return ((db + 40.0) / 40.0).astype(np.float32)
