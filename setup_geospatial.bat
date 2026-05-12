rem ===========================================================
rem Install Software/Tools for Geospatial Data Science Projects
rem Python | PyCharm-Community | VS Code | Spyder | Git | DBeaver-Community | Docker-Desktop | QGIS | ESA-SNAP
rem ===========================================================

@echo off
:: Check for administrative permissions
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Administrative permissions confirmed.
) else (
    echo ERROR: Run this script as Administrator.
    pause
    exit
)

:: chocolatey - The Package Manager for Windows
echo Installing Chocolatey...
powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org'))"
SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"

echo Installing Software Packages...

:: Development & Python
choco install -y python
choco install -y pycharm-community
choco install -y vscode
choco install -y spyder

:: Tools
choco install -y git
choco install -y dbeaver-community
choco install -y docker-desktop

:: GIS
choco install -y qgis
choco install -y esa-snap

echo All software installed successfully!
pause

:: Notes:
:: 1. Right-click install_dev.bat -> Run as Administrator.
:: 2. Key Information:
:: 2.1 Method: Uses "choco install -y" for silent, automatic installation.
:: 2.2 Requirement: Must be run with Administrator privileges.
:: 2.3 Result: Installs the latest stable versions of all requested tools.
:: 3. The script includes a check to ensure it is run as administrator.
:: 4. The -y flag allows all installations to proceed without user confirmation.
:: 5. You may need to log out and log back in for Git/Python PATH changes to take effect.