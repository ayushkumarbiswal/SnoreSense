"""Audio loading and preprocessing (Module 1 — Ayush Kumar Biswal: US-01, 02, 08, 09).

Everything downstream works on 16 kHz mono float32 audio in [-1, 1].
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import librosa
import numpy as np
import soundfile as sf

from .config import FRAME_HOP, MIN_ANALYSIS_SEC, N_FFT, SR


# --------------------------------------------------------------------------- loading
def load_audio(source, sr: int = SR) -> np.ndarray:
    """Load a path, bytes or file-like object (wav / flac / ogg / mp3) as mono float32 at `sr`."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        y, file_sr = sf.read(source, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    except Exception:
        # Fallback for formats libsndfile can't read (librosa uses audioread/ffmpeg if present).
        if hasattr(source, "seek"):
            source.seek(0)
        y, file_sr = librosa.load(source, sr=None, mono=True)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr, res_type="soxr_hq")
    return np.ascontiguousarray(y, dtype=np.float32)


def to_wav_bytes(y: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# --------------------------------------------------------------------------- helpers
def frame_db(y: np.ndarray, hop: int = FRAME_HOP, n_fft: int = N_FFT) -> np.ndarray:
    """Short-time RMS level in dBFS, one value per hop."""
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop, center=True)[0]
    return 20 * np.log10(np.maximum(rms, 1e-6))


# --------------------------------------------------------------------------- quality gate (US-09)
@dataclass
class QualityReport:
    ok: bool
    duration_sec: float
    level_dbfs: float
    peak_dbfs: float
    clipped_pct: float
    snr_db: float
    issues: list[str] = field(default_factory=list)


def quality_check(y: np.ndarray, sr: int = SR) -> QualityReport:
    """Reject recordings we can't analyse reliably instead of guessing (US-09)."""
    duration = len(y) / sr
    issues: list[str] = []
    if duration < MIN_ANALYSIS_SEC:
        issues.append(f"Recording is too short ({duration:.0f} s). Please record at least "
                      f"{MIN_ANALYSIS_SEC:.0f} seconds — longer recordings give more reliable results.")
    db = frame_db(y) if len(y) > N_FFT else np.array([-120.0])
    level = float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)) if len(y) else -120.0
    peak = float(20 * np.log10(np.max(np.abs(y)) + 1e-9)) if len(y) else -120.0
    clipped = float(np.mean(np.abs(y) >= 0.999) * 100) if len(y) else 0.0
    # SNR estimate: loud frames (95th pct) vs. noise floor (10th pct)
    snr = float(np.percentile(db, 95) - np.percentile(db, 10))
    if peak < -45:
        issues.append("Recording is too quiet — no clear breathing or snoring sounds. Place the phone "
                      "closer to the sleeper (about 1 metre) and try again.")
    if clipped > 0.5:
        issues.append("Recording is distorted (too loud / clipping). Move the phone further away and try again.")
    if peak >= -45 and snr < 8:
        issues.append("Background noise is too strong relative to the breathing sounds "
                      f"(estimated SNR {snr:.0f} dB). Turn off fans/TV if possible and record again.")
    return QualityReport(ok=not issues, duration_sec=duration, level_dbfs=level, peak_dbfs=peak,
                         clipped_pct=clipped, snr_db=snr, issues=issues)


# --------------------------------------------------------------------------- preprocessing (US-08)
def reduce_noise(y: np.ndarray, sr: int = SR, strength: float = 1.0) -> np.ndarray:
    """Spectral-gating noise reduction.

    The noise profile is estimated per frequency bin from the quietest 15% of frames
    (stationary background such as fans, hum, room tone); bins that don't rise clearly
    above it are attenuated with a smooth mask.
    """
    if len(y) < N_FFT * 2:
        return y
    S = librosa.stft(y, n_fft=N_FFT, hop_length=FRAME_HOP)
    mag, phase = np.abs(S), np.angle(S)
    frame_energy = mag.sum(axis=0)
    quiet = frame_energy <= np.percentile(frame_energy, 15)
    noise_mu = mag[:, quiet].mean(axis=1, keepdims=True)
    noise_sd = mag[:, quiet].std(axis=1, keepdims=True)
    thresh = noise_mu + 1.5 * noise_sd * strength
    mask = 1.0 / (1.0 + np.exp(-(mag - thresh) / (noise_sd + 1e-8)))     # soft gate in [0, 1]
    # smooth the mask over time and frequency to avoid musical noise
    from scipy.ndimage import uniform_filter
    mask = uniform_filter(mask, size=(3, 5))
    mask = 0.1 + 0.9 * mask                                                 # max ~20 dB attenuation
    out = librosa.istft(mag * mask * np.exp(1j * phase), hop_length=FRAME_HOP, length=len(y))
    return out.astype(np.float32)


def trim_edges(y: np.ndarray, sr: int = SR, top_db: float = 35.0, margin_sec: float = 1.0) -> tuple[np.ndarray, float]:
    """Trim leading/trailing silence only.

    Pauses *inside* the recording are kept on purpose — long breathing pauses are exactly
    the pattern the screening looks for. Returns (trimmed audio, seconds removed at start).
    """
    if len(y) == 0:
        return y, 0.0
    _, (start, end) = librosa.effects.trim(y, top_db=top_db, frame_length=N_FFT, hop_length=FRAME_HOP)
    margin = int(margin_sec * sr)
    start, end = max(0, start - margin), min(len(y), end + margin)
    return y[start:end], start / sr


def normalize(y: np.ndarray, target_peak_db: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(y)) + 1e-9
    return (y * (10 ** (target_peak_db / 20) / peak)).astype(np.float32)


def preprocess(y: np.ndarray, sr: int = SR, trim: bool = True) -> tuple[np.ndarray, float]:
    """Preprocessing chain shared by training and inference: DC removal, noise reduction,
    edge trimming (whole recordings only — training windows keep their exact length), peak
    normalisation. Returns (audio, seconds trimmed from the start)."""
    y = y - np.mean(y)                      # remove DC offset
    y = reduce_noise(y, sr)
    offset = 0.0
    if trim:
        y, offset = trim_edges(y, sr)
    return normalize(y), offset
