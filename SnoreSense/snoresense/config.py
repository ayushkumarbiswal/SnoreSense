"""Shared constants and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SAMPLES_DIR = ROOT / "samples"
ESC50_DIR = DATA_DIR / "esc50"          # downloaded source clips (not committed)
CACHE_DIR = DATA_DIR / "cache"          # generated training windows/features (not committed)
UPLOADS_DIR = DATA_DIR / "uploads"      # admin-uploaded labelled clips for retraining
DB_PATH = DATA_DIR / "snoresense.db"

SR = 16_000                 # all audio is resampled to 16 kHz mono
WINDOW_SEC = 30.0           # analysis window length
WINDOW_HOP_SEC = 15.0       # 50% overlap between windows
MIN_ANALYSIS_SEC = 15.0     # shorter recordings are rejected by the quality check

# Feature extraction
FRAME_HOP = 512             # 32 ms hop for envelope / handcrafted features
N_FFT = 1024
N_MFCC = 13
# CNN input
CNN_N_MELS = 64
CNN_HOP = 1024              # 64 ms hop -> ~469 frames per 30 s window

APNEA_PAUSE_SEC = 10.0      # clinical convention: a breathing pause of >= 10 s
FLAG_THRESHOLD = 0.5        # window probability above which a window is flagged
RECORDING_FLAG_FRACTION = 0.25  # recording is "irregular" if >= 25% of windows are flagged

LABELS = {0: "Normal snoring pattern", 1: "Irregular pattern detected"}
