"""Demo content: sample recordings and demo accounts with screening history.

    python -m snoresense.demo build     # (maintainers) regenerate samples/ and data/seed.json
                                        # from held-out ESC-50 audio using the trained models

The app calls `seed_database()` automatically on first start, so a fresh copy is demo-ready.
All seeded results are real outputs of the screening pipeline on generated recordings; only the
users and dates are demo values.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import soundfile as sf

from . import db
from .config import DATA_DIR, SAMPLES_DIR, SR

SEED_PATH = DATA_DIR / "seed.json"
DEMO_PASSWORD = "demo123"

DEMO_USERS = [
    {"email": "alex@snoresense.demo", "name": "Alex Sharma", "role": "home", "profiles": []},
    {"email": "priya@snoresense.demo", "name": "Priya Nair", "role": "caregiver",
     "profiles": [("Ramesh Nair", "Parent"), ("Sunita Nair", "Parent")]},
    {"email": "admin@snoresense.demo", "name": "Admin", "role": "admin", "profiles": []},
]

SAMPLES = [
    {"file": "sample_1_regular_snoring.flac", "title": "1 · Regular snoring (2.5 min)",
     "description": "Steady snoring every few seconds in a quiet bedroom — expected: Normal.",
     "plan": [("normal", 150)], "snr": 22, "seed": 501},
    {"file": "sample_2_irregular_pattern.flac", "title": "2 · Pause-interrupted snoring (2.5 min)",
     "description": "Snoring that repeatedly stops for 10–25 s and restarts with a loud snore or gasp — "
                    "expected: Irregular pattern.",
     "plan": [("normal", 30), ("irregular", 120)], "snr": 20, "seed": 502},
    {"file": "sample_3_noisy_phone_recording.flac", "title": "3 · Noisy phone recording, mixed night (3 min)",
     "description": "Phone on a bedside table with fan/rain noise; regular snoring that turns irregular "
                    "in the second half — shows how flagged segments are localised.",
     "plan": [("normal", 90), ("irregular", 90)], "snr": 9, "seed": 503},
    {"file": "sample_4_poor_quality.flac", "title": "4 · Poor-quality recording (20 s)",
     "description": "Mostly fan noise with barely audible breathing — expected: the app asks you to record "
                    "again instead of guessing (US-09).",
     "plan": "noise", "snr": None, "seed": 504},
]

# (user email, profile name or None for self, days ago, kind) — kind: normal / irregular / mixed
HISTORY = (
    [("alex@snoresense.demo", None, d, k) for d, k in
     [(13, "normal"), (11, "normal"), (9, "normal"), (6, "mixed"), (4, "normal"), (1, "normal")]]
    + [("priya@snoresense.demo", None, d, "normal") for d in (12, 5)]
    + [("priya@snoresense.demo", "Ramesh Nair", d, k) for d, k in
       [(14, "mixed"), (10, "irregular"), (7, "irregular"), (3, "irregular"), (2, "mixed")]]
    + [("priya@snoresense.demo", "Sunita Nair", d, "normal") for d in (8, 2)]
)
N_COMMUNITY = 14


def _night(rng, pools, kind: str):
    from .dataset import make_night
    dur = float(rng.choice([90, 120, 150, 180]))
    plan = {"normal": [("normal", dur)], "irregular": [("normal", 30), ("irregular", dur - 30)],
            "mixed": [("normal", dur / 2), ("irregular", dur / 2)]}[kind]
    y, _ = make_night(rng, pools, plan, snr_db=float(rng.uniform(8, 28)))
    return y


def build():
    """Regenerate sample recordings and seed screenings (needs ESC-50 + trained models)."""
    from .analysis import analyze
    from .dataset import TEST_FOLDS, load_pools, make_night
    from .models import ModelBundle

    pools = load_pools(TEST_FOLDS)                    # held-out sound sources only
    bundle = ModelBundle.active()
    SAMPLES_DIR.mkdir(exist_ok=True)
    manifest = []
    for s in SAMPLES:
        rng = np.random.default_rng(s["seed"])
        if s["plan"] == "noise":
            n = int(20 * SR)
            from .dataset import compose_normal
            clean, _ = compose_normal(rng, pools, 20)
            from .dataset import _colored_noise
            fan = _colored_noise(rng, n, 1.0)                     # steady fan noise
            hum = 0.3 * np.sin(2 * np.pi * 50 * np.arange(n) / SR)  # mains hum
            y = 0.12 * clean / (np.abs(clean).max() + 1e-9) + fan + hum
            y = (y / np.abs(y).max() * 10 ** (-38 / 20)).astype(np.float32)
        else:
            y, _ = make_night(rng, pools, s["plan"], snr_db=s["snr"])
        sf.write(SAMPLES_DIR / s["file"], y, SR, format="FLAC")
        r = analyze(y, bundle)
        outcome = r.get("label_text") or "rejected: " + "; ".join(r.get("issues", []))
        print(f"{s['file']}: {outcome} ({r.get('confidence', '')})")
        manifest.append({k: s[k] for k in ("file", "title", "description")})
    (SAMPLES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))

    rng = np.random.default_rng(2026)
    seeds = []
    jobs = list(HISTORY)
    for i in range(N_COMMUNITY):                       # anonymous community activity for admin analytics
        for _ in range(int(rng.integers(1, 5))):
            kind = rng.choice(["normal", "irregular", "mixed"], p=[0.68, 0.17, 0.15])
            jobs.append((f"user{i + 1:02d}@snoresense.demo", None, int(rng.integers(0, 15)), str(kind)))
    for email, prof, days, kind in jobs:
        y = _night(rng, pools, kind)
        r = analyze(y, bundle)
        if r["status"] != "ok":
            continue
        community = email.startswith("user")
        keep = {k: r[k] for k in ("label", "label_text", "confidence", "summary", "model_version", "decision_model")}
        if not community:
            keep.update({k: r[k] for k in ("windows", "events", "pauses", "envelope", "quality", "trim_offset_sec")})
        seeds.append({"email": email, "profile": prof, "days_ago": days, "hour": int(rng.integers(5, 9)),
                      "source": str(rng.choice(["record", "upload"], p=[0.6, 0.4])), "kind": kind, "result": keep})
        print(f"  {email:28s} {prof or '':12s} {kind:9s} → {r['label_text']} ({r['confidence']:.2f})")
    SEED_PATH.write_text(json.dumps({"samples": manifest, "screenings": seeds}))
    print(f"wrote {len(seeds)} seeded screenings to {SEED_PATH}")


def seed_database() -> None:
    """Create demo accounts and history (runs once, on an empty database)."""
    data = json.loads(SEED_PATH.read_text()) if SEED_PATH.exists() else {"screenings": []}
    ids = {}
    for u in DEMO_USERS:
        uid = db.create_user(u["email"], u["name"], u["role"], DEMO_PASSWORD, demo=True)
        ids[u["email"]] = uid
        for name, rel in u["profiles"]:
            db.add_profile(uid, name, rel)
        if u["role"] != "admin":
            db.set_consent(uid, True, True, False)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    for s in data["screenings"]:
        email = s["email"]
        if email not in ids:
            n = email.split("@")[0].replace("user", "")
            ids[email] = db.create_user(email, f"Community user {n}", "home", DEMO_PASSWORD + n, demo=True)
            db.set_consent(ids[email], True, True, False)
        uid = ids[email]
        profs = db.profiles(uid)
        prof = next((p for p in profs if (p["name"] == s["profile"] if s["profile"] else p["relation"] == "Self")), profs[0])
        created = (now - timedelta(days=s["days_ago"])).replace(hour=s["hour"]).isoformat()
        r = s["result"]
        if "envelope" not in r:                      # community rows: summary only
            r = {**r, "windows": [], "events": [], "pauses": [], "envelope": {"t": [], "min": [], "max": []}}
        fname = f"night-{created[:10]}.wav" if s["source"] == "upload" else "live-recording.wav"
        db.save_screening(uid, prof["id"], s["source"], fname, r, None, created)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
