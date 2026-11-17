@echo off
rem LocalChat with phone access (PWA) ON. Pairing code: open http://127.0.0.1:8000/phone on this PC.
setlocal
cd /d "%~dp0"
set "LOCALCHAT_REMOTE=1"
set "LOCALCHAT_DATA_DIR=%~dp0data"
if not exist "%LOCALCHAT_DATA_DIR%" mkdir "%LOCALCHAT_DATA_DIR%"
if not exist "%~dp0LocalChat.exe" (
  echo LocalChat.exe was not found. Please extract the complete Portable ZIP.
  pause
  exit /b 1
)
start "" "%~dp0LocalChat.exe"
