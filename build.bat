@echo off
setlocal

REM Wildfire Burn Severity Predictor - Windows build script.
REM Double-click this file on a Windows machine to build
REM dist\WildfireSeverity.exe via PyInstaller.

cd /d "%~dp0"

echo ================================================================
echo   Building WildfireSeverity.exe
echo ================================================================
echo.

set "PYTHON_CMD="
where py >nul 2>&1 && set "PYTHON_CMD=py -3"
if not defined PYTHON_CMD (
    where python >nul 2>&1 && set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo ERROR: Python is not installed or not on your PATH.
    echo Install Python 3.10 or newer from:
    echo   https://www.python.org/downloads/windows/
    echo and check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

echo Using Python:
%PYTHON_CMD% --version
echo.

if not exist "build_venv\Scripts\python.exe" (
    echo Creating build virtual environment...
    %PYTHON_CMD% -m venv build_venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo.
)

echo Installing dependencies ^(this can take several minutes on first run^)...
"build_venv\Scripts\python.exe" -m pip install --upgrade pip
"build_venv\Scripts\python.exe" -m pip install -r requirements.txt
"build_venv\Scripts\python.exe" -m pip install pyinstaller
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
echo.

echo Building executable with PyInstaller...
"build_venv\Scripts\python.exe" -m PyInstaller wildfire.spec --clean --noconfirm
if errorlevel 1 (
    echo ERROR: PyInstaller build failed.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo   Build complete.
echo   Hand this single file to your teacher:
echo     %CD%\dist\WildfireSeverity.exe
echo ================================================================
echo.
pause
endlocal
