@echo off
title VSCode Launcher

:: Navigate to the script's directory
cd /d "%~dp0"

:: Start VSCode and immediately detach
start "" /b code .

:: Force immediate exit
exit