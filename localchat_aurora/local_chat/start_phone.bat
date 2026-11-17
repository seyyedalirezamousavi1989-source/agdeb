@echo off
rem Starts LocalChat with phone access (PWA) turned ON. The pairing code is shown only on this PC at http://127.0.0.1:8000/phone
setlocal
cd /d "%~dp0"
set "LOCALCHAT_REMOTE=1"
call "%~dp0start.bat"
