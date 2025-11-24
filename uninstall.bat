@echo off
REM ============================================================================
REM Nuclei Segmentation App - Uninstall Script
REM ============================================================================
REM This script removes the virtual environment and optionally cleans up
REM cached models and temporary files.
REM
REM Usage:
REM   Double-click uninstall.bat or run from command prompt
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo  Nuclei Segmentation App - Uninstall
echo ========================================================================
echo.

REM Check if venv exists
if not exist "venv" (
    echo No virtual environment found.
    echo Nothing to uninstall.
    echo.
    pause
    exit /b 0
)

echo This will remove the virtual environment and optionally clean up:
echo   - Virtual environment (venv\)
echo   - Cached models (~/.cellpose/models/)
echo   - Temporary files (build\, dist\, __pycache__\)
echo.
choice /M "Continue with uninstall"
if errorlevel 2 (
    echo Uninstall cancelled.
    pause
    exit /b 0
)

echo.
echo [1/3] Removing virtual environment...
rmdir /s /q venv
if exist "venv" (
    echo WARNING: Could not fully remove venv\ - you may need to delete it manually
) else (
    echo Virtual environment removed.
)
echo.

REM Ask about cleaning temp files
echo [2/3] Clean temporary files?
echo This includes: __pycache__\, build\, dist\, *.pyc
echo.
choice /M "Clean temporary files"
if not errorlevel 2 (
    echo Cleaning temporary files...
    for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
    if exist "build" rmdir /s /q build
    if exist "dist" rmdir /s /q dist
    del /s /q *.pyc >nul 2>&1
    echo Temporary files cleaned.
) else (
    echo Skipped.
)
echo.

REM Ask about cleaning cached models
echo [3/3] Remove cached Cellpose models?
echo Location: %USERPROFILE%\.cellpose\models\
echo Size: ~200-500 MB
echo Note: Models will be re-downloaded if you reinstall
echo.
choice /M "Remove cached models"
if not errorlevel 2 (
    if exist "%USERPROFILE%\.cellpose" (
        echo Removing Cellpose cache...
        rmdir /s /q "%USERPROFILE%\.cellpose"
        echo Cellpose cache removed.
    ) else (
        echo No Cellpose cache found.
    )
) else (
    echo Skipped.
)
echo.

echo ========================================================================
echo  Uninstall Complete
echo ========================================================================
echo.
echo The application source code has been preserved.
echo To reinstall, run: install.bat
echo.
pause
