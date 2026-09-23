@echo off
title Agent-Assist Copilot - Stop
echo Stopping Agent-Assist Copilot server...
wsl -e bash -c "pkill -f 'uvicorn stage4_production' 2>/dev/null; exit 0"
echo Done.
timeout /t 2 /nobreak >nul
