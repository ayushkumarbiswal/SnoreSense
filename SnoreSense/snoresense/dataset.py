"""Training data: real snoring/breathing recordings arranged into night-time breathing patterns.

Source audio: ESC-50 (Piczak, 2015; CC BY-NC 3.0) — real snoring, breathing and coughing
recordings plus bedroom-type background sounds (rain, wind, crickets, clock, traffic).
Public apnea-annotated audio (e.g. PSG-Audio) needs registration and is not bundled, so each
training window is assembled from individual real snore/breath sounds following the
breathing patterns that define the two classes:

* Normal   — regular breathing/snoring rhythm (3–5 s cycle), natural jitter, occasional
             sub-threshold gaps (< 10 s), position-change loudness drift, coughs.
* Irregular — obstructive-apnea-like cycles: crescendo snoring → breathing pause of 10–25 s
             (apnea) or 10–20 s of very shallow breathing (hypopnea) → loud recovery
             snore / gasp cluster.

Every window then goes through simulated consumer-phone recording conditions: room reverb,
phone-microphone band-limiting, background noise at a random SNR, random gain.
ESC-50 folds keep train and test sound sources separate (folds 1–4 train, fold 5 test).

`python -m snoresense.dataset download` fetches the ESC-50 clips used (~150 MB).
"""
from __future__ import annotations

import argparse
import csv
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from .audio import frame_db, load_audio
from .config import ESC50_DIR, SR

ESC50_RAW = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
EVENT_CATS = ["snoring", "breathing", "coughing"]
AMBIENT_CATS = ["rain", "wind", "crickets", "clock_tick", "engine"]
TRAIN_FOLDS, TEST_FOLDS = (1, 2, 3, 4), (5,)


# --------------------------------------------------------------------------- download
def download(verbose: bool = True) -> None:
    (ESC50_DIR / "audio").mkdir(parents=True, exist_ok=True)
    meta_path = ESC50_DIR / "esc50.csv"
    if not meta_path.exists():
        urllib.request.urlretrieve(f"{ESC50_RAW}/meta/esc50.csv", meta_path)
    rows = [r for r in csv.DictReader(open(meta_path)) if r["category"] in EVENT_CATS + AMBIENT_CATS]
    todo = [r["filename"] for r in rows if not (ESC50_DIR / "audio" / r["filename"]).exists()]
    if verbose:
        print(f"ESC-50: {len(rows)} clips needed, {len(todo)} to download")

    def fetch(name):
        urllib.request.urlretrieve(f"{ESC50_RAW}/audio/{name}", ESC50_DIR / "audio" / name)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(fetch, todo))
    if verbose:
        print("done")


def _meta() -> list[dict]:
    return list(csv.DictReader(open(ESC50_DIR / "esc50.csv")))


# --------------------------------------------------------------------------- source pools
@dataclass
class Pools:
    snore: list[np.ndarray]
    breath: list[np.ndarray]
    cough: list[np.ndarray]
    ambient: list[np.ndarray]


def _cut_events(y: np.ndarray, top_db: float = 20.0, pad: float = 0.12) -> list[np.ndarray]:
    db = np.convolve(frame_db(y), np.ones(5) / 5, mode="same")
    act = db > db.max() - top_db
    idx = np.flatnonzero(np.diff(np.concatenate([[0], act.astype(int), [0]])))
    fps = SR / 512
    out = []
    for s, e in zip(idx[::2], idx[1::2]):
        dur = (e - s) / fps
        if not 0.25 <= dur <= 3.5:
            continue
        a, b = max(0, int((s / fps - pad) * SR)), min(len(y), int((e / fps + pad) * SR))
        seg = y[a:b].copy()
        fade = min(int(0.05 * SR), len(seg) // 4)
        seg[:fade] *= np.linspace(0, 1, fade)
        seg[-fade:] *= np.linspace(1, 0, fade)
        seg /= np.max(np.abs(seg)) + 1e-9                     # each event peak-normalised to 0 dBFS
        out.append(seg.astype(np.float32))
    return out


def load_pools(folds) -> Pools:
    folds = {str(f) for f in folds}
    rows = [r for r in _meta() if r["fold"] in folds]
    get = lambda cat: [load_audio(ESC50_DIR / "audio" / r["filename"]) for r in rows if r["category"] == cat]
    snore = [e for y in get("snoring") for e in _cut_events(y)]
    breath = [e for y in get("breathing") for e in _cut_events(y)]
    cough = [e for y in get("coughing") for e in _cut_events(y, top_db=15)]
    ambient = [y / (np.sqrt(np.mean(y ** 2)) + 1e-9) for c in AMBIENT_CATS for y in get(c)
               if np.sqrt(np.mean(y ** 2)) > 1e-4]
    return Pools(snore, breath, cough, ambient)


# --------------------------------------------------------------------------- recording conditions
def _colored_noise(rng, n, exponent):
    f = np.fft.rfftfreq(n)
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.maximum(f, 1 / n) ** (exponent / 2)
    x = np.fft.irfft(spec, n)
    return (x / (np.sqrt(np.mean(x ** 2)) + 1e-9)).astype(np.float32)


def background(rng, n, pools: Pools) -> np.ndarray:
    bed = _colored_noise(rng, n, rng.choice([1.0, 1.5, 2.0]))          # pink / brown room tone
    if pools.ambient and rng.random() < 0.7:
        amb = pools.ambient[rng.integers(len(pools.ambient))]
        reps = int(np.ceil(n / len(amb))) + 1
        amb = np.tile(amb, reps)[rng.integers(len(amb)):][:n]
        w = rng.uniform(0.3, 1.0)
        bed = np.sqrt(1 - w ** 2) * bed + w * amb
    return (bed / (np.sqrt(np.mean(bed ** 2)) + 1e-9)).astype(np.float32)


def room_and_phone(rng, y: np.ndarray) -> np.ndarray:
    if rng.random() < 0.6:                                            # small-room reverb
        rt60 = rng.uniform(0.15, 0.6)
        t = np.arange(int(rt60 * SR)) / SR
        ir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
        ir[0] = 1.0 / rng.uniform(0.3, 1.0)
        y = fftconvolve(y, ir / np.max(np.abs(ir)), mode="full")[: len(y)]
    hp = rng.uniform(80, 250)                                         # phone microphone band-limit
    lp = rng.uniform(3500, 7500)
    y = sosfilt(butter(2, [hp, lp], btype="band", fs=SR, output="sos"), y)
    return y.astype(np.float32)


def finish(rng, clean: np.ndarray, pools: Pools, snr_db: float | None = None) -> np.ndarray:
    """Add room/phone colouration, background noise at `snr_db`, random gain."""
    y = room_and_phone(rng, clean)
    db = frame_db(y)
    loud = db > db.max() - 15                      # frames belonging to the breathing sounds
    sig_rms = float(np.sqrt(np.mean((10 ** (db[loud] / 20)) ** 2)))
    snr = rng.uniform(5, 30) if snr_db is None else snr_db
    noise = background(rng, len(y), pools) * sig_rms / 10 ** (snr / 20)
    y = y + noise
    if rng.random() < 0.2:
        y = np.tanh(2.0 * y / (np.max(np.abs(y)) + 1e-9)) / np.tanh(2.0)  # mild phone AGC/saturation
    peak = rng.uniform(-28, -4)
    y = y / (np.max(np.abs(y)) + 1e-9) * 10 ** (peak / 20)
    return y.astype(np.float32)


# --------------------------------------------------------------------------- breathing patterns
def _place(buf, ev, t, gain_db):
    a = int(t * SR)
    if a >= len(buf) or a + len(ev) <= 0:
        return
    s0 = max(0, -a)
    b = min(len(buf), a + len(ev))
    buf[max(a, 0):b] += ev[s0:s0 + b - max(a, 0)] * 10 ** (gain_db / 20)


class Sleeper:
    """One simulated sleeper: a fixed breathing period and snore/breath mix."""

    def __init__(self, rng, pools: Pools):
        self.rng, self.pools = rng, pools
        self.period = rng.uniform(3.0, 5.0)
        self.jitter = rng.uniform(0.05, 0.14)
        self.p_snore = rng.uniform(0.55, 1.0)
        # each sleeper uses a small personal subset of snore sounds (a voice)
        k = min(len(pools.snore), int(rng.integers(3, 8)))
        self.snores = [pools.snore[i] for i in rng.choice(len(pools.snore), k, replace=False)]
        self.breath_gain = rng.uniform(-14, -6)

    def breath(self, buf, t, gain=0.0, force_snore=None):
        r = self.rng
        snore = (r.random() < self.p_snore) if force_snore is None else force_snore
        if snore:
            ev, g = self.snores[r.integers(len(self.snores))], gain + r.normal(0, 1.5)
        else:
            ev, g = self.pools.breath[r.integers(len(self.pools.breath))], gain + self.breath_gain + r.normal(0, 1.5)
        _place(buf, ev, t, g)
        return t + self.period * (1 + r.normal(0, self.jitter))


def compose_normal(rng, pools: Pools, length: float, start: float | None = None) -> tuple[np.ndarray, list]:
    buf = np.zeros(int(length * SR), dtype=np.float32)
    s = Sleeper(rng, pools)
    t = -rng.uniform(0, s.period) if start is None else start
    drift = rng.normal(0, 3) if rng.random() < 0.25 else 0.0
    gap_at = rng.uniform(0, length) if rng.random() < 0.35 else None
    cough_at = rng.uniform(0, length) if rng.random() < 0.15 else None
    while t < length:
        g = drift * (t / length)
        t = s.breath(buf, t, g)
        if gap_at is not None and t > gap_at:                      # sub-threshold pause (< 10 s)
            t += rng.uniform(2.0, 9.0 - s.period)
            gap_at = None
        if cough_at is not None and t > cough_at and pools.cough:
            _place(buf, pools.cough[rng.integers(len(pools.cough))], t - 1.0, rng.uniform(-6, 2))
            cough_at = None
    return buf, []


def compose_irregular(rng, pools: Pools, length: float, require_event: bool = True) -> tuple[np.ndarray, list]:
    """Returns (audio, ground-truth events [(start, end, kind)])."""
    for _ in range(20):
        buf = np.zeros(int(length * SR), dtype=np.float32)
        s = Sleeper(rng, pools)
        t = -rng.uniform(0, 15)
        events = []
        while t < length:
            k = int(rng.integers(2, 6))                               # snoring run before the event
            crescendo = rng.random() < 0.6
            for i in range(k):
                t = s.breath(buf, t, gain=(4.0 * i / max(k - 1, 1) - 2.0) if crescendo else 0.0, force_snore=True if crescendo else None)
            if rng.random() < 0.7:                                    # obstructive apnea: silence
                dur = rng.uniform(10, 25)
                if rng.random() < 0.3:                                # faint effort sounds
                    tt = t + s.period
                    while tt < t + dur - 1:
                        _place(buf, pools.breath[rng.integers(len(pools.breath))], tt, rng.uniform(-30, -24))
                        tt += s.period * rng.uniform(0.8, 1.3)
                events.append((t, t + dur, "apnea"))
                t += dur
            else:                                                     # hypopnea: very shallow breathing
                dur = rng.uniform(10, 20)
                end = t + dur
                events.append((t, end, "hypopnea"))
                while t < end:
                    t = s.breath(buf, t, gain=rng.uniform(-22, -15), force_snore=False)
            # recovery: gasp + loud snores at a faster rate
            if rng.random() < 0.5:
                _place(buf, pools.breath[rng.integers(len(pools.breath))], t, rng.uniform(4, 10))
                t += s.period * 0.6
            for _ in range(int(rng.integers(1, 4))):
                _place(buf, s.snores[rng.integers(len(s.snores))], t, rng.uniform(5, 12))
                t += s.period * rng.uniform(0.55, 0.8)
        inside = [e for e in events if e[0] >= 0 and e[1] <= length]
        if inside or not require_event:
            return buf, events
    return buf, events


def make_window(rng, pools: Pools, label: int, length: float = 30.0, snr_db: float | None = None) -> np.ndarray:
    clean, _ = compose_irregular(rng, pools, length) if label == 1 else compose_normal(rng, pools, length)
    return finish(rng, clean, pools, snr_db)


def make_night(rng, pools: Pools, plan: list[tuple[str, float]], snr_db: float | None = None) -> tuple[np.ndarray, list]:
    """A longer recording made of segments, e.g. [("normal", 60), ("irregular", 90)]."""
    parts, events, t0 = [], [], 0.0
    for kind, dur in plan:
        if kind == "normal":
            y, ev = compose_normal(rng, pools, dur)
        else:
            y, ev = compose_irregular(rng, pools, dur)
        parts.append(y)
        events += [(a + t0, b + t0, k) for a, b, k in ev if a >= 0 and b <= dur]
        t0 += dur
    return finish(rng, np.concatenate(parts), pools, snr_db), events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download"])
    args = ap.parse_args()
    if args.cmd == "download":
        download()


if __name__ == "__main__":
    main()
