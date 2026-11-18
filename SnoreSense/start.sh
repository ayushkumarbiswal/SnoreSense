#!/usr/bin/env bash
# SnoreSense — one-command start (macOS / Linux / Git Bash on Windows).
#   bash start.sh          install what's missing and start the app
#   bash start.sh reset    also reset the database to the demo state
# Opens http://localhost:8501. Ctrl+C stops the app.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
PORT=8501

echo "== SnoreSense =="
PY=""
for cand in python3.12 python3.11 python3.13 python3.10 python3 python "py -3.12" "py -3"; do
  if $cand -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,14) else 1)" >/dev/null 2>&1; then
    PY="$cand"; break
  fi
done
[ -n "$PY" ] || { echo "[error] Python 3.10–3.13 not found (https://www.python.org/downloads/)"; exit 1; }
echo "[python] $($PY --version)"

if [ -x .venv/Scripts/python.exe ]; then VENV_PY=.venv/Scripts/python.exe; else VENV_PY=.venv/bin/python; fi
if [ ! -x "$VENV_PY" ] || ! "$VENV_PY" -c "import sys" >/dev/null 2>&1; then
  echo "[setup] creating virtual environment (first run only)..."
  rm -rf .venv
  $PY -m venv .venv
  if [ -x .venv/Scripts/python.exe ]; then VENV_PY=.venv/Scripts/python.exe; else VENV_PY=.venv/bin/python; fi
fi

if ! cmp -s requirements.txt .venv/requirements.installed; then
  echo "[setup] installing packages (first run takes a few minutes)..."
  "$VENV_PY" -m pip install --disable-pip-version-check -q --upgrade pip
  "$VENV_PY" -m pip install --disable-pip-version-check -q -r requirements.txt
  cp requirements.txt .venv/requirements.installed
  echo "[setup] preparing the models (one-time, about 30 seconds)..."
  "$VENV_PY" -m snoresense.warmup
else
  echo "[setup] packages already installed"
fi

if [ "${1:-}" = "reset" ] && [ -f data/snoresense.db ]; then
  rm -f data/snoresense.db
  echo "[reset] database cleared — demo accounts and history will be re-created"
fi

port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1; }
port_busy $PORT && PORT=8502
port_busy $PORT && PORT=8503

URL="http://localhost:$PORT"
echo
echo "== Starting SnoreSense on $URL =="
echo "  Demo logins: use the 'Continue as ...' buttons on the sign-in page"
echo "  Ctrl+C to stop."
(
  sleep 6
  if command -v cmd.exe >/dev/null 2>&1; then cmd.exe /c start "" "$URL" >/dev/null 2>&1
  elif command -v open >/dev/null 2>&1; then open "$URL" >/dev/null 2>&1
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1
  fi
) &
exec "$VENV_PY" -m streamlit run app.py --server.port "$PORT"
