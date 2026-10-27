@echo off
REM Double-click to install DOMINION (Basic: signal mode, no GPU).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_basic.ps1"
echo.
pause
