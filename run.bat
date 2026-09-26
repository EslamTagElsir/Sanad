@echo off
setlocal EnableExtensions
title Sanad
cd /d "%~dp0"

echo ============================================
echo   Sanad
echo ============================================
echo.

rem ---- 1) WSL project path, computed from this file's location ----
set "PROJECT_DIR="
for /f "usebackq delims=" %%i in (`wsl -e wslpath -a "%~dp0."`) do set "PROJECT_DIR=%%i"
if not defined PROJECT_DIR goto :no_wsl
set "VENV_PY=$HOME/agentassist_venv/bin/python3"

wsl -e bash -c "test -x %VENV_PY%"
if errorlevel 1 goto :no_venv

rem ---- 2) Dependencies: install requirements.txt only if something is missing ----
echo [1/4] Checking dependencies...
wsl -e bash -c "%VENV_PY% -c 'import fastapi, uvicorn, pymupdf, sentence_transformers, sklearn, openai, jwt, bcrypt, sendgrid, dotenv' 2>/dev/null"
if not errorlevel 1 goto :deps_ok
echo       Missing packages - installing requirements.txt, this can take a few minutes...
wsl -e bash -c "cd '%PROJECT_DIR%' && %VENV_PY% -m pip install -q -r requirements.txt"
if errorlevel 1 goto :deps_failed
:deps_ok

rem ---- 3) Stop a previous instance of THIS project, then pick a free port ----
echo [2/4] Stopping any previous instance...
wsl -e bash -c "pkill -f 'uvicorn [s]tage4_production' ; exit 0"

set "PORT="
for /f "usebackq delims=" %%p in (`powershell -NoProfile -Command "$p = 8000; while ($p -le 8050 -and (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)) { $p++ }; $p"`) do set "PORT=%%p"
if not defined PORT set "PORT=8000"
if "%PORT%"=="8000" goto :port_ok
set "PORT_OWNER=another program"
for /f "usebackq delims=" %%o in (`powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if ($c) { (Get-Process -Id $c.OwningProcess).ProcessName }"`) do set "PORT_OWNER=%%o"
echo       Port 8000 is already used by %PORT_OWNER% - using port %PORT% instead.
:port_ok

rem ---- 4) Start the server in its own window; the window stays open if it crashes ----
set "INSTANCE_ID=%RANDOM%%RANDOM%%RANDOM%"
echo [3/4] Starting the server on port %PORT% (first start loads the AI model, ~30-60 s)...
start "Sanad Server - port %PORT% (close to stop)" wsl -e bash -c "cd '%PROJECT_DIR%' && mkdir -p logs && export PYTHONUNBUFFERED=1 SANAD_INSTANCE_ID=%INSTANCE_ID% SANAD_PUBLIC_URL=http://localhost:%PORT% && %VENV_PY% -m uvicorn stage4_production.service:app --host 0.0.0.0 --port %PORT% 2>&1 | tee logs/server.log; echo; echo '=== Server stopped. Full log: logs/server.log ==='; read -p 'Press Enter to close this window...'"

rem ---- 5) Wait until /health answers with OUR instance id (up to 3 minutes) ----
rem     127.0.0.1 not localhost: localhost tries IPv6 ::1 first, which can hang here until timeout.
echo [4/4] Waiting for the server to be ready...
set /a waited=0
:waitloop
ping -n 3 127.0.0.1 >nul
set /a waited+=2
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 'http://127.0.0.1:%PORT%/health'; if ($r.Content -match '%INSTANCE_ID%') { exit 0 } else { exit 2 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 goto :ready
if %waited% LSS 6 goto :waitloop
wsl -e bash -c "pgrep -f 'uvicorn [s]tage4_production' >/dev/null"
if errorlevel 1 goto :crashed
if %waited% GEQ 180 goto :timeout
goto :waitloop

:ready
echo.
echo Server is up after about %waited% s. Opening http://localhost:%PORT%/app/
start "" "http://localhost:%PORT%/app/"
echo.
echo The server keeps running in the other window.
echo To stop it: close that window, or run stop.bat.
ping -n 4 127.0.0.1 >nul
exit /b 0

:crashed
echo.
echo ERROR: the server stopped during startup. Last lines of logs\server.log:
echo --------------------------------------------
wsl -e bash -c "cd '%PROJECT_DIR%' && tail -n 25 logs/server.log"
echo --------------------------------------------
pause
exit /b 1

:timeout
echo.
echo ERROR: the server did not answer within 3 minutes. Check the server window or logs\server.log.
pause
exit /b 1

:no_wsl
echo ERROR: WSL is not available. Install WSL, or run the server manually.
pause
exit /b 1

:no_venv
echo ERROR: Python environment not found in WSL at ~/agentassist_venv
echo Create it with:  python3 -m venv ~/agentassist_venv  then run this file again.
pause
exit /b 1

:deps_failed
echo ERROR: installing requirements.txt failed. See the messages above.
pause
exit /b 1
