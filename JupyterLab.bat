@echo off
title Jupyter Launcher
cd /d "%~dp0"

:: Try the standard command
jupyter lab
if %errorlevel% equ 0 goto end

:: If that fails, try running as a module via 'python'
python -m jupyter lab
if %errorlevel% equ 0 goto end

:: If that fails, try the Windows 'py' launcher
py -m jupyter lab
if %errorlevel% equ 0 goto end

echo.
echo [ERROR] Could not find Jupyter. 
echo Attempting to install it now...
py -m pip install jupyterlab
py -m jupyter lab

:end
pause