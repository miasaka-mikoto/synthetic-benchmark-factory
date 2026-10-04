@echo off
setlocal
rem Thin wrapper so users can double-click the build from Explorer.
where powershell.exe >nul 2>nul
if errorlevel 1 (
  echo PowerShell was not found. Please run build_windows.ps1 in PowerShell.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_windows.ps1" %*
exit /b %errorlevel%
