@echo off
REM ============================================================================
REM Nuclei Segmentation App - Installation Script
REM ============================================================================
REM This script creates a Python virtual environment and installs all
REM dependencies required to run the application from source.
REM
REM Requirements:
REM   - Python 3.9, 3.10, or 3.11 (64-bit)
REM   - pip (included with Python)
REM   - ~5 GB free disk space
REM
REM Usage:
REM   Double-click install.bat or run from command prompt
REM ============================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ========================================================================
echo  EZQUANT - Installation
echo ========================================================================
echo.

REM Set preferred Python version (change this if needed)
set PREFERRED_PYTHON=py -3.10

REM Check if Python launcher is available and preferred version exists
echo [1/6] Checking Python installation...
%PREFERRED_PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.10 not found via Python Launcher, checking for other versions...
    
    REM Try standard python command
    python --version >nul 2>&1
    if errorlevel 1 (
        echo.
        echo ERROR: Python is not installed or not in PATH
        echo.
        echo Please install Python 3.10 from:
        echo   https://www.python.org/downloads/
        echo.
        echo Make sure to check "Add Python to PATH" during installation!
        echo.
        pause
        exit /b 1
    )
    set PYTHON_CMD=python
) else (
    echo Found Python 3.10 via Python Launcher
    set PYTHON_CMD=%PREFERRED_PYTHON%
)

REM Get Python version
for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Using Python version: %PYTHON_VERSION%

REM Extract major and minor version
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

REM Check Python version (must be 3.9, 3.10, or 3.11)
if not "%PYTHON_MAJOR%"=="3" (
    echo.
    echo ERROR: Python 3.x is required, found Python %PYTHON_VERSION%
    echo Please install Python 3.9, 3.10, or 3.11
    echo.
    pause
    exit /b 1
)

if %PYTHON_MINOR% LSS 9 (
    echo.
    echo WARNING: Python 3.9+ is recommended, found Python %PYTHON_VERSION%
    echo Some features may not work correctly.
    echo.
    choice /M "Continue anyway"
    if errorlevel 2 exit /b 1
)

if %PYTHON_MINOR% GTR 11 (
    echo.
    echo WARNING: Python 3.9-3.11 is recommended, found Python %PYTHON_VERSION%
    echo Compatibility with Python 3.12+ is not guaranteed.
    echo.
    choice /M "Continue anyway"
    if errorlevel 2 exit /b 1
)

echo Python version OK: %PYTHON_VERSION%
echo.

REM Check if venv already exists
if exist "venv" (
    echo.
    echo WARNING: Virtual environment 'venv' already exists!
    echo.
    choice /M "Delete and recreate"
    if errorlevel 2 (
        echo Installation cancelled.
        pause
        exit /b 0
    )
    echo Removing old virtual environment...
    rmdir /s /q venv
    if exist "venv" (
        echo.
        echo ERROR: Failed to remove existing virtual environment.
        echo Please close any terminal/editor using venv and try again.
        echo.
        pause
        exit /b 1
    )
)

REM Create virtual environment with specific Python version
echo [2/6] Creating virtual environment with Python %PYTHON_VERSION%...
echo Using Python: 
%PYTHON_CMD% -c "import sys; print(f'  Executable: {sys.executable}'); print(f'  Version: {sys.version}')"
echo.
%PYTHON_CMD% -m venv venv
if errorlevel 1 (
    echo.
    echo ERROR: Failed to create virtual environment
    echo Make sure you have venv module installed:
    echo   python -m pip install --user virtualenv
    echo.
    pause
    exit /b 1
)
echo Virtual environment created: venv\
echo Python %PYTHON_VERSION% will be used in this environment.
echo.

REM Activate virtual environment
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate virtual environment
    echo.
    pause
    exit /b 1
)
echo Virtual environment activated.
set VENV_PYTHON=venv\Scripts\python.exe
set VENV_PIP=%VENV_PYTHON% -m pip
echo.

REM Upgrade pip
echo [4/6] Upgrading pip, setuptools, and wheel...
%VENV_PIP% install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo.
    echo WARNING: Failed to upgrade pip
    echo Continuing with existing version...
    echo.
)
echo.

REM Install PyTorch with CUDA support (if available)
echo [5/6] Installing PyTorch...
echo.
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No NVIDIA GPU detected - installing CPU-only version
    echo Note: Segmentation will be slower without GPU
    echo.
    %VENV_PIP% install torch torchvision --index-url https://download.pytorch.org/whl/cpu
) else (
    set GPU_NAME=
    set USE_NIGHTLY=0
    for /f "usebackq delims=" %%G in (`nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul`) do (
        if not defined GPU_NAME set GPU_NAME=%%G
    )
    if not defined GPU_NAME set GPU_NAME=Unknown NVIDIA GPU

    echo NVIDIA GPU detected: !GPU_NAME!
    echo !GPU_NAME! | findstr /I /C:"RTX 50" >nul
    if not errorlevel 1 set USE_NIGHTLY=1

    echo This may take several minutes...
    if "!USE_NIGHTLY!"=="1" (
        echo Installing PyTorch Nightly with CUDA 12.8 for RTX 50-series (sm_120) support
        %VENV_PIP% install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
    ) else (
        echo Installing stable PyTorch with CUDA 12.4
        %VENV_PIP% install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    )
)

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install PyTorch
    echo Please check your internet connection and try again.
    echo.
    pause
    exit /b 1
)
echo PyTorch installed successfully.
echo.

REM Install application dependencies
echo [6/6] Installing application dependencies...
echo This may take 10-15 minutes depending on your connection...
echo.

set REQUIREMENTS_FILE=requirements.txt
if not exist "%REQUIREMENTS_FILE%" (
    if exist "requirements_updated.txt" (
        echo WARNING: requirements.txt not found. Using requirements_updated.txt instead.
        set REQUIREMENTS_FILE=requirements_updated.txt
    ) else (
        echo ERROR: No requirements file found.
        echo Expected requirements.txt or requirements_updated.txt in this folder.
        echo.
        pause
        exit /b 1
    )
)

set TEMP_REQUIREMENTS=%TEMP%\ezquant_requirements_no_torch_%RANDOM%%RANDOM%.txt
findstr /V /I /R /C:"^torch" /C:"^torchvision" /C:"^torchaudio" "%REQUIREMENTS_FILE%" > "%TEMP_REQUIREMENTS%"
if not exist "%TEMP_REQUIREMENTS%" (
    echo.
    echo ERROR: Failed to prepare requirements list for installation.
    echo.
    pause
    exit /b 1
)

%VENV_PIP% install -r "%TEMP_REQUIREMENTS%"
set PIP_INSTALL_RC=%ERRORLEVEL%
del /q "%TEMP_REQUIREMENTS%" >nul 2>&1
if not "%PIP_INSTALL_RC%"=="0" (
    echo.
    echo ERROR: Failed to install dependencies
    echo Please check the error messages above.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo  Installation Complete!
echo ========================================================================
echo.
echo Virtual environment: venv\
echo Python version: %PYTHON_VERSION%
echo.
echo Next steps:
echo   1. Run 'start.bat' to launch the application
echo   2. Or manually: venv\Scripts\activate.bat, then python main.py
echo.
echo Optional - Download Cellpose models (recommended):
echo   venv\Scripts\activate.bat
echo   python -c "from cellpose import models; models.Cellpose(gpu=False, model_type='nuclei')"
echo.
echo For GPU acceleration, keep NVIDIA drivers up to date and install CUDA Toolkit when needed:
echo   https://developer.nvidia.com/cuda-downloads
echo   (RTX 50-series uses PyTorch nightly CUDA 12.8 wheels in this installer)
echo.
echo ========================================================================
echo.
pause
