@echo off
REM ============================================================================
REM PyTorch Upgrade Script for RTX 50-series GPU Support
REM ============================================================================
REM This script upgrades PyTorch to a version compatible with CUDA sm_120
REM (RTX 5050 and newer GPUs)
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo  PyTorch Upgrade for RTX 50-series GPU Support
echo ========================================================================
echo.
echo This will upgrade PyTorch to support CUDA capability sm_120
echo Required for: RTX 5050, RTX 5060, RTX 5070, RTX 5080, RTX 5090
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first to create the environment.
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate virtual environment
    echo.
    pause
    exit /b 1
)
echo.

REM Check for NVIDIA GPU
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: No NVIDIA GPU detected
    echo This upgrade is only needed for NVIDIA RTX 50-series GPUs
    echo.
    choice /M "Continue anyway"
    if errorlevel 2 exit /b 0
)
echo NVIDIA GPU detected.
echo.

REM Show current PyTorch version
echo Current PyTorch installation:
python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>nul
if errorlevel 1 (
    echo   PyTorch not currently installed
)
echo.

echo Upgrading to PyTorch with CUDA 12.4 support...
echo This may take several minutes...
echo.

REM Uninstall current PyTorch
echo Step 1: Removing old PyTorch installation...
pip uninstall -y torch torchvision
echo.

REM Install new PyTorch with CUDA 12.4
echo Step 2: Installing PyTorch with CUDA 12.4...
pip install torch>=2.5.0 torchvision>=0.20.0 --index-url https://download.pytorch.org/whl/cu124

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install PyTorch
    echo Please check your internet connection and try again.
    echo.
    pause
    exit /b 1
)
echo.

REM Verify installation
echo ========================================================================
echo Verifying installation...
echo ========================================================================
echo.

python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); print(); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

if errorlevel 1 (
    echo.
    echo ERROR: PyTorch installation verification failed
    echo.
    pause
    exit /b 1
)
echo.

REM Test CUDA capability
echo Testing GPU compatibility...
python -c "import torch; import warnings; warnings.filterwarnings('ignore'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f'Device: {device}'); test = torch.zeros(1).to(device); print('✓ GPU test successful!' if device.type == 'cuda' else '⚠ Running on CPU')"
echo.

echo ========================================================================
echo  Upgrade Complete!
echo ========================================================================
echo.
echo PyTorch has been upgraded to support CUDA 12.4 (sm_120)
echo.
echo IMPORTANT: You must have NVIDIA CUDA Toolkit 12.4 installed:
echo   Download from: https://developer.nvidia.com/cuda-downloads
echo.
echo After installing CUDA 12.4, your RTX 5050 should work with GPU acceleration!
echo.
echo ========================================================================
echo.
pause
