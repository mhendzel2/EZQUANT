@echo off
REM ============================================================================
REM Nuclei Segmentation App - Startup Script
REM ============================================================================
REM This script activates the Python virtual environment and launches the
REM Nuclei Segmentation Application.
REM
REM Prerequisites:
REM   - Run install.bat first to create venv and install dependencies
REM
REM Usage:
REM   Double-click start.bat or run from command prompt
REM ============================================================================

setlocal enabledelayedexpansion

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo ========================================================================
    echo  ERROR: Virtual environment not found!
    echo ========================================================================
    echo.
    echo Please run 'install.bat' first to install the application.
    echo.
    pause
    exit /b 1
)

REM Check if main.py exists
if not exist "main.py" (
    echo.
    echo ========================================================================
    echo  ERROR: main.py not found!
    echo ========================================================================
    echo.
    echo Please make sure you're running this script from the application directory.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo  Nuclei Segmentation App - Starting...
echo ========================================================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate virtual environment
    echo Try reinstalling with install.bat
    echo.
    pause
    exit /b 1
)

REM Display Python version
echo Virtual environment activated.
python --version
echo.

REM Check for GPU
echo Checking hardware...
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" 2>nul
if errorlevel 1 (
    echo Note: Unable to detect GPU status
)
echo.

REM Launch application
echo Launching Nuclei Segmentation App...
echo.
echo ========================================================================
echo.

python main.py

REM Check exit code
if errorlevel 1 (
    echo.
    echo ========================================================================
    echo  Application exited with errors
    echo ========================================================================
    echo.
    echo If you encounter issues:
    echo   1. Make sure all dependencies are installed: install.bat
    echo   2. Check the error messages above
    echo   3. Try running: python main.py
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo  Application closed normally
echo ========================================================================
echo.
