@echo off
setlocal

title Agent-Assist Copilot

set "PROJECT_DIR=/mnt/c/Users/Lenovo/OneDrive/Desktop/New folder (2)/agent-assist-copilot"
set "VENV_PY=~/agentassist_venv/bin/python3"

echo ============================================
echo   Agent-Assist Copilot
echo ============================================
echo.
echo Stopping any previous server instance...
wsl -e bash -c "pkill -f 'uvicorn stage4_production' 2>/dev/null; exit 0"

echo Starting the server (WSL)...
start "Agent-Assist Copilot Server (close this window to stop)" wsl -e bash -c "cd '%PROJECT_DIR%' && %VENV_PY% -m uvicorn stage4_production.service:app --host 0.0.0.0 --port 8000"

echo Waiting for the server to be ready...
set /a attempts=0

:waitloop
set /a attempts+=1
powershell -NoProfile -Command "try { $c = New-Object Net.Sockets.TcpClient; $c.Connect('localhost', 8000); $c.Close(); exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    if %attempts% GEQ 30 (
        echo.
        echo Server did not start within 30 seconds. Check the server window for errors.
        pause
        exit /b 1
    )
    timeout /t 1 /nobreak >nul
    goto waitloop
)

echo Server is up. Opening the app...
start "" "http://localhost:8000/app/"

echo.
echo Done. The server keeps running in the other window.
echo To stop it later, either close that window or run stop.bat.
timeout /t 3 /nobreak >nul
