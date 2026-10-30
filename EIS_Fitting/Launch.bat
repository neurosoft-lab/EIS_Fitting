@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp003_Bin\launch.ps1" %*
exit /b %errorlevel%
