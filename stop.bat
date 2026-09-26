@echo off
title Sanad - Stop
echo Stopping Sanad server...
rem "[s]tage4" so pkill does not match (and kill) its own shell command line
wsl -e bash -c "pkill -f 'uvicorn [s]tage4_production' && echo Stopped. || echo No running server found."
ping -n 3 127.0.0.1 >nul
