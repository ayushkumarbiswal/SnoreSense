"""Run one short analysis so librosa/numba compile and cache their functions now,
instead of making the first screening in the app slow. Called by start.bat / start.sh after install."""
import time

from .analysis import analyze
from .audio import load_audio
from .config import SAMPLES_DIR, SR
from .models import ModelBundle

if __name__ == "__main__":
    t = time.time()
    y = load_audio(SAMPLES_DIR / "sample_1_regular_snoring.flac")[: 20 * SR]
    analyze(y, ModelBundle.active())
    print(f"[setup] models ready ({time.time() - t:.0f} s)")
