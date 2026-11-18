@echo off
REM ============================================================
REM  SnoreSense - one-click start (Windows)
REM
REM  Double-click this file, or run from a terminal in this folder:
REM     start.bat          install what's missing and start the app
REM     start.bat reset    also reset the database to the demo state
REM
REM  The app opens in your browser at http://localhost:8501
REM  Close this window (or press Ctrl+C) to stop it.
REM ============================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
set "PORT=8501"

echo.
echo == SnoreSense ==
echo.

REM ---------- find Python 3.10 - 3.14 (3.12 preferred) ----------
set "PY="
for %%V in (3.12 3.11 3.13 3.10 3.14) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,14) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo [error] Python 3.10 - 3.13 was not found.
  echo         Install Python 3.12 from https://www.python.org/downloads/
  echo         and tick "Add python.exe to PATH" during installation.
  goto :fail
)
echo [python] using:
%PY% --version

REM ---------- virtual environment ----------
if exist "%VENV_PY%" (
  "%VENV_PY%" -c "import sys" >nul 2>&1 || (
    echo [setup] existing environment is broken - recreating it...
    rmdir /s /q .venv
  )
)
if not exist "%VENV_PY%" (
  echo [setup] creating virtual environment ^(first run only^)...
  %PY% -m venv .venv
  if errorlevel 1 goto :fail
)

REM ---------- dependencies (re-installed only when requirements.txt changes) ----------
set "STAMP=.venv\requirements.installed"
set "NEED_INSTALL=1"
if exist "%STAMP%" (
  fc /b requirements.txt "%STAMP%" >nul 2>&1 && set "NEED_INSTALL=0"
)
if "!NEED_INSTALL!"=="1" (
  echo [setup] installing packages ^(first run takes a few minutes^)...
  "%VENV_PY%" -m pip install --disable-pip-version-check -q --upgrade pip
  "%VENV_PY%" -m pip install --disable-pip-version-check -q -r requirements.txt
  if errorlevel 1 (
    echo [error] package installation failed - check your internet connection and try again.
    goto :fail
  )
  copy /y requirements.txt "%STAMP%" >nul
  echo [setup] preparing the models ^(one-time, about 30 seconds^)...
  "%VENV_PY%" -m snoresense.warmup
) else (
  echo [setup] packages already installed
)

REM ---------- optional reset ----------
if /i "%~1"=="reset" (
  if exist data\snoresense.db (
    del /q data\snoresense.db
    echo [reset] database cleared - demo accounts and history will be re-created
  )
)

REM ---------- pick a free port ----------
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul 2>&1 && set "PORT=8502"
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul 2>&1 && set "PORT=8503"

echo.
echo == Starting SnoreSense on http://localhost:%PORT% ==
echo   Demo logins: use the "Continue as ..." buttons on the sign-in page
echo   (or alex@snoresense.demo / priya@snoresense.demo / admin@snoresense.demo, password demo123)
echo   Close this window or press Ctrl+C to stop.
echo.
start "" /b cmd /c "ping -n 8 127.0.0.1 >nul & start http://localhost:%PORT%"
"%VENV_PY%" -m streamlit run app.py --server.port %PORT%
exit /b 0

:fail
echo.
echo Startup did not complete. See the message above.
pause
exit /b 1
