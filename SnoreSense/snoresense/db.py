"""SQLite storage for users, family profiles, consent and screening history
(Module 3 — K. Dinesh Kumar Patro: US-05, 06, 07, 12, 13).

Privacy by design: raw audio is only stored when the user explicitly opts in; otherwise only the
result and summary numbers are kept. Every row is tied to a user-controlled profile and can be
deleted on request.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import DB_PATH, DATA_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('home', 'caregiver', 'admin')),
    salt TEXT NOT NULL, password_hash TEXT NOT NULL, created TEXT NOT NULL, demo INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL, relation TEXT NOT NULL DEFAULT 'Self', created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS consent (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    analyse INTEGER NOT NULL DEFAULT 0, store_history INTEGER NOT NULL DEFAULT 0,
    store_audio INTEGER NOT NULL DEFAULT 0, updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS screenings (
    id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    created TEXT NOT NULL, source TEXT NOT NULL, filename TEXT,
    duration_sec REAL, label INTEGER, confidence REAL, flagged_fraction REAL,
    n_pauses INTEGER, longest_pause_sec REAL, pauses_per_hour REAL, events_per_min REAL,
    model_version TEXT, decision_model TEXT, details TEXT, audio BLOB
);
CREATE TABLE IF NOT EXISTS training_clips (
    id INTEGER PRIMARY KEY, created TEXT NOT NULL, filename TEXT NOT NULL, label INTEGER NOT NULL,
    path TEXT NOT NULL, added_by INTEGER REFERENCES users(id) ON DELETE SET NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init() -> bool:
    """Create tables. Returns True if the database was empty (so demo data should be seeded)."""
    with connect() as con:
        con.executescript(SCHEMA)
        return con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0


# --------------------------------------------------------------------------- users
def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()


def create_user(email: str, name: str, role: str, password: str, demo: bool = False) -> int:
    salt = secrets.token_hex(16)
    with connect() as con:
        cur = con.execute("INSERT INTO users (email, name, role, salt, password_hash, created, demo) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?)",
                          (email.lower().strip(), name.strip(), role, salt, _hash(password, salt), now(), int(demo)))
        uid = cur.lastrowid
        if role != "admin":
            con.execute("INSERT INTO profiles (user_id, name, relation, created) VALUES (?, ?, 'Self', ?)",
                        (uid, name.strip(), now()))
        con.execute("INSERT INTO consent (user_id, updated) VALUES (?, ?)", (uid, now()))
    return uid


def authenticate(email: str, password: str) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
    if row and secrets.compare_digest(row["password_hash"], _hash(password, row["salt"])):
        return dict(row)
    return None


def get_user(uid: int) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
    return dict(row) if row else None


def delete_user(uid: int) -> None:
    with connect() as con:
        con.execute("DELETE FROM users WHERE id = ?", (uid,))


# --------------------------------------------------------------------------- profiles (US-06, 07)
def profiles(uid: int) -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT p.*, (SELECT COUNT(*) FROM screenings s WHERE s.profile_id = p.id) AS n_screenings "
            "FROM profiles p WHERE user_id = ? ORDER BY relation != 'Self', id", (uid,))]


def add_profile(uid: int, name: str, relation: str) -> None:
    with connect() as con:
        con.execute("INSERT INTO profiles (user_id, name, relation, created) VALUES (?, ?, ?, ?)",
                    (uid, name.strip(), relation, now()))


def delete_profile(uid: int, profile_id: int) -> None:
    with connect() as con:
        con.execute("DELETE FROM profiles WHERE id = ? AND user_id = ? AND relation != 'Self'", (profile_id, uid))


# --------------------------------------------------------------------------- consent (US-13)
def get_consent(uid: int) -> dict:
    with connect() as con:
        row = con.execute("SELECT * FROM consent WHERE user_id = ?", (uid,)).fetchone()
    return dict(row) if row else {"analyse": 0, "store_history": 0, "store_audio": 0}


def set_consent(uid: int, analyse: bool, store_history: bool, store_audio: bool) -> None:
    with connect() as con:
        con.execute("INSERT INTO consent (user_id, analyse, store_history, store_audio, updated) VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(user_id) DO UPDATE SET analyse = excluded.analyse, "
                    "store_history = excluded.store_history, store_audio = excluded.store_audio, updated = excluded.updated",
                    (uid, int(analyse), int(store_history and analyse), int(store_audio and store_history and analyse), now()))
        if not store_audio:
            con.execute("UPDATE screenings SET audio = NULL WHERE user_id = ?", (uid,))


# --------------------------------------------------------------------------- screenings (US-05)
def save_screening(uid: int, profile_id: int, source: str, filename: str, result: dict,
                   audio: bytes | None = None, created: str | None = None) -> int:
    s = result["summary"]
    details = {k: result[k] for k in ("windows", "events", "pauses", "envelope", "summary", "quality",
                                      "trim_offset_sec", "label_text") if k in result}
    with connect() as con:
        cur = con.execute(
            "INSERT INTO screenings (user_id, profile_id, created, source, filename, duration_sec, label, confidence, "
            "flagged_fraction, n_pauses, longest_pause_sec, pauses_per_hour, events_per_min, model_version, "
            "decision_model, details, audio) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (uid, profile_id, created or now(), source, filename, s["duration_sec"], result["label"],
             result["confidence"], s["flagged_fraction"], s["n_pauses"], s["longest_pause_sec"],
             s["pauses_per_hour"], s["events_per_min"], result["model_version"], result["decision_model"],
             json.dumps(details), audio))
        return cur.lastrowid


def screenings(uid: int | None = None, profile_id: int | None = None, with_details: bool = False) -> list[dict]:
    cols = "s.*" if with_details else ("s.id, s.user_id, s.profile_id, s.created, s.source, s.filename, s.duration_sec, "
                                       "s.label, s.confidence, s.flagged_fraction, s.n_pauses, s.longest_pause_sec, "
                                       "s.pauses_per_hour, s.events_per_min, s.model_version, s.decision_model, "
                                       "s.audio IS NOT NULL AS has_audio")
    q, args = f"SELECT {cols}, p.name AS profile_name, p.relation FROM screenings s JOIN profiles p ON p.id = s.profile_id WHERE 1=1", []
    if uid is not None:
        q += " AND s.user_id = ?"
        args.append(uid)
    if profile_id is not None:
        q += " AND s.profile_id = ?"
        args.append(profile_id)
    with connect() as con:
        return [dict(r) for r in con.execute(q + " ORDER BY s.created DESC", args)]


def get_screening(uid: int, sid: int) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT s.*, p.name AS profile_name FROM screenings s JOIN profiles p ON p.id = s.profile_id "
                          "WHERE s.id = ? AND s.user_id = ?", (sid, uid)).fetchone()
    return dict(row) if row else None


def delete_screening(uid: int, sid: int) -> None:
    with connect() as con:
        con.execute("DELETE FROM screenings WHERE id = ? AND user_id = ?", (sid, uid))


def delete_all_data(uid: int) -> None:
    with connect() as con:
        con.execute("DELETE FROM screenings WHERE user_id = ?", (uid,))
        con.execute("DELETE FROM profiles WHERE user_id = ? AND relation != 'Self'", (uid,))


def export_user(uid: int) -> dict:
    user = get_user(uid)
    rows = screenings(uid, with_details=True)
    for r in rows:
        r.pop("audio", None)
        r["details"] = json.loads(r["details"]) if r.get("details") else None
    return {
        "exported": now(),
        "user": {k: user[k] for k in ("email", "name", "role", "created")},
        "consent": {k: v for k, v in get_consent(uid).items() if k != "user_id"},
        "profiles": [{k: p[k] for k in ("name", "relation", "created")} for p in profiles(uid)],
        "screenings": rows,
    }


# --------------------------------------------------------------------------- admin (US-10, 11) — anonymised only
def usage_rows() -> list[dict]:
    """Screening rows without any personal identifiers, for aggregated analytics."""
    with connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT s.created, s.source, s.duration_sec, s.label, s.confidence, s.pauses_per_hour, "
            "s.model_version, s.user_id AS uid FROM screenings s")]


def user_counts() -> dict:
    with connect() as con:
        rows = con.execute("SELECT role, COUNT(*) FROM users GROUP BY role").fetchall()
    return {r[0]: r[1] for r in rows}


def add_training_clip(filename: str, label: int, path: str, uid: int) -> None:
    with connect() as con:
        con.execute("INSERT INTO training_clips (created, filename, label, path, added_by) VALUES (?, ?, ?, ?, ?)",
                    (now(), filename, label, path, uid))


def training_clips() -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM training_clips ORDER BY id DESC")]


def delete_training_clip(cid: int) -> str | None:
    with connect() as con:
        row = con.execute("SELECT path FROM training_clips WHERE id = ?", (cid,)).fetchone()
        con.execute("DELETE FROM training_clips WHERE id = ?", (cid,))
    return row["path"] if row else None
