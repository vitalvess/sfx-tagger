@echo off
chcp 65001 >nul
title SFX Tagger
cd /d "%~dp0"
set "VPY=%LOCALAPPDATA%\SFXTagger\venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo  A app ainda nao esta instalada. Corre primeiro "instalar_windows.bat".
  pause
  exit /b 1
)
"%VPY%" run.py
if errorlevel 1 pause
