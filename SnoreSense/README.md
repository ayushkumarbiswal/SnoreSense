# SnoreSense

**Audio-Based Screening Tool for Early Detection of Sleep Apnea Risk Patterns**
Major Project-I · Department of CSE, GIET University, Gunupur

Team: Ayush Kumar Biswal · Ajit Kumar Pradhan · K. Dinesh Kumar Patro

SnoreSense analyses a sleep recording (recorded live in the browser or uploaded) and flags the
acoustic pattern associated with obstructive sleep apnea: breathing sounds that stop for 10 seconds
or more and restart with a loud snore or gasp. It is a **first-level screening aid, not a diagnosis**.

---

## Run it (one command)

**Needs:** Python 3.10–3.13 (3.12 recommended). Nothing else.

| Windows | macOS / Linux / Git Bash |
|---|---|
| double-click **`start.bat`** | `bash start.sh` |

The script creates a virtual environment, installs the packages (first run: a few minutes),
and opens **http://localhost:8501**. Later starts take a few seconds. Close the window or press
Ctrl+C to stop.

The trained models, sample recordings and demo accounts are included — **nothing is trained on the day**.

### Demo accounts

Use the **Continue as …** buttons on the sign-in page, or sign in with password `demo123`:

| Role | Email | What to show |
|---|---|---|
| Home user | `alex@snoresense.demo` | screening, result, history, consent, data export |
| Caregiver | `priya@snoresense.demo` | family profiles (her father's history is flagged repeatedly) |
| Administrator | `admin@snoresense.demo` | usage analytics, model versions & retraining, model comparison |

To reset everything to the demo state before the review: `start.bat reset` (or `bash start.sh reset`).

---

## Suggested demo flow (≈ 6 minutes)

1. **Sign in as Home user → Screen my sleep.** First use asks for consent (US-13).
2. **Sample recordings → "2 · Pause-interrupted snoring" → Analyse.** Walk through the progress
   steps (quality check → noise reduction → event detection → features → models).
3. **Result page:** risk category + confidence, explanation (US-04), waveform with the detected
   ≥ 10 s pauses (red) and flagged windows (amber), per-model scores, next steps and disclaimer (US-12).
4. Run **"1 · Regular snoring"** for contrast, and **"4 · Poor-quality recording"** to show the app
   asking to re-record instead of guessing (US-09). Optionally **record live** with the microphone (US-01).
5. **History:** trend of pauses per hour; select a row to view or delete it (US-05).
6. **Sign out → Caregiver:** Family profiles; history filtered per family member (US-06, US-07).
7. **Sign out → Administrator:** dashboard and usage analytics (US-10); Model management — versions,
   decision model, upload labelled clips and retrain (US-11); **Model comparison** — RF vs SVM vs CNN,
   ROC, confusion matrices, feature importance and **noise robustness** (the project's novelty).

---

## How it works

```
audio (mic / upload)
  └─► Module 1 · preprocessing (snoresense/audio.py, features.py)          Ayush — US-01, 02, 08, 09
        resample 16 kHz mono → quality gate (length, level, clipping, SNR)
        → spectral-gating noise reduction → trim leading/trailing silence
        → breathing/snore event detection (adaptive hysteresis threshold) → pauses ≥ 10 s
        → per 30 s window (50 % overlap): 47 handcrafted features + 64-band log-mel spectrogram
  └─► Module 2 · classification (snoresense/models.py, cnn.py, train.py)   Ajit — US-03, 04, 10, 11
        Random Forest + SVM on handcrafted features, CNN on log-mel, ensemble average
        → window probabilities → recording flagged if ≥ 25 % of windows are irregular
  └─► Module 3 · interface & data (app.py, snoresense/ui/, db.py)          Dinesh — US-05, 06, 07, 12, 13
        Streamlit UI, SQLite: users, family profiles, consent, history; audio stored only on opt-in
```

**Handcrafted features:** MFCC 1–13 (mean, std), spectral centroid / bandwidth / roll-off /
flatness, zero-crossing rate, loudness variability, and pause statistics — breathing sounds per
minute, inter-onset interval variability, longest gap, pauses ≥ 10 s, longest low-breathing span
(hypopnea-like), loudness of the first sound after a pause, rhythm periodicity.

**CNN:** 3 conv blocks on the log-mel spectrogram, then dilated temporal convolutions with a ~20 s
receptive field (long enough to "see" a 10 s pause), global pooling, dense layer. Trained with JAX;
inference is plain NumPy, so the app needs no deep-learning framework.

## Data — please read

* **Source audio:** [ESC-50](https://github.com/karolpiczak/ESC-50) (Piczak, 2015; CC BY-NC 3.0) —
  real recordings of snoring, breathing and coughing, plus bedroom background sounds (rain, wind,
  crickets, clock, traffic).
* Public audio with apnea annotations (e.g. PSG-Audio) requires registration and was not available
  for this prototype. Training windows are therefore **assembled from the real snore and breath
  sounds** following the two breathing patterns: regular rhythm (normal) vs. crescendo snoring →
  10–25 s pause or shallow breathing → loud recovery snore (apnea-like). Each window then gets
  simulated phone-recording conditions: room reverb, phone-microphone band-limiting, background
  noise at 5–30 dB SNR, random gain.
* Train and test use **different sound sources** (ESC-50 folds 1–4 vs fold 5), and the sample
  recordings are built from the test fold only.
* The reported accuracy therefore measures how well each approach separates these breathing
  patterns under realistic recording conditions — **it is not clinical accuracy**. The next step is
  retraining on polysomnography-labelled recordings (the admin page and `python -m snoresense.train`
  are built for that).

## Retraining

* **In the app (Administrator → Model management):** upload labelled recordings, click *Retrain* —
  Random Forest and SVM are refitted on the original features plus the new clips and saved as a
  candidate version you can activate.
* **Full retrain incl. CNN:**
  ```
  pip install -r requirements-train.txt
  python -m snoresense.dataset download     # ESC-50 clips used (~150 MB)
  python -m snoresense.train                # ~20 min on a laptop CPU
  python -m snoresense.demo build           # regenerate samples + demo history
  ```

## Project layout

```
app.py                     Streamlit entry point (role-based navigation)
snoresense/
  audio.py                 loading, quality check, noise reduction, trimming
  features.py              event/pause detection, handcrafted features, log-mel
  cnn.py                   CNN (JAX training, NumPy inference)
  models.py                model registry, prediction, versions
  analysis.py              end-to-end screening pipeline used by the UI
  dataset.py               training-data construction from ESC-50
  train.py                 train + evaluate RF, SVM, CNN; writes models/metrics.json
  retrain.py               in-app retraining with uploaded labelled clips
  db.py                    SQLite: users, profiles, consent, screenings
  demo.py                  sample recordings + demo accounts/history
  ui/                      sign-in, user pages, admin pages
models/                    trained models, metrics, registry
samples/                   demo recordings (held-out test audio)
data/                      SQLite database (created on first start), seed data
start.bat / start.sh       one-command start
```

## References

* K. J. Piczak, "ESC: Dataset for Environmental Sound Classification," ACM Multimedia, 2015.
* G. Korompili et al., "PSG-Audio, a scored polysomnography dataset with simultaneous audio
  recordings for sleep apnea studies," Scientific Data 8, 197 (2021).
* B. McFee et al., "librosa: Audio and Music Signal Analysis in Python," SciPy, 2015.
* American Academy of Sleep Medicine scoring rules: apnea = airflow cessation ≥ 10 s.
