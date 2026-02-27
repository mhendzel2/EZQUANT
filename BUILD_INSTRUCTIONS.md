# Build Instructions for Nuclei Segmentation Application

This guide explains how to build a standalone Windows executable and installer for the Nuclei Segmentation Application.

---

## Prerequisites

### Required Software
1. **Python 3.9+** (tested with 3.9, 3.10, 3.11)
2. **PyInstaller 6.0+**
   ```bash
   pip install pyinstaller>=6.0
   ```
3. **Inno Setup 6.0+** (for creating installer)
   - Download from: https://jrsoftware.org/isinfo.php
   - Install to default location

4. **All application dependencies**
   ```bash
  pip install -r requirements.txt
   ```

### Optional (Recommended)
- **UPX** (for executable compression)
  - Download from: https://upx.github.io/
  - Extract upx.exe to a directory in your PATH

- **Application Icon** (icon.ico)
  - 256x256 PNG/ICO file
  - Place in root directory before building

---

## Step 1: Prepare Environment

### 1.1 Create Clean Virtual Environment
```powershell
# Navigate to project directory
cd C:\NucleiSegmentationApp

# Create virtual environment
python -m venv venv_build

# Activate it
.\venv_build\Scripts\Activate.ps1

# Install all dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install pyinstaller>=6.0
```

### 1.2 Verify Installation
```powershell
# Test that app runs
python main.py

# Check PyInstaller
pyinstaller --version
```

### 1.3 Download Pre-trained Models (Optional)
```python
# Run this to cache Cellpose models
python -c "from cellpose import models; models.Cellpose(gpu=False, model_type='nuclei')"

# Models will be stored in:
# Windows: C:\Users\<username>\.cellpose\models\
```

---

## Step 2: Build Executable

### 2.1 Create Application Icon (Optional)
If you don't have `icon.ico`, create a simple one:

```python
# create_icon.py
from PIL import Image, ImageDraw, ImageFont

# Create 256x256 icon with blue nucleus
img = Image.new('RGB', (256, 256), color='white')
draw = ImageDraw.Draw(img)

# Draw nucleus-like circle
draw.ellipse([40, 40, 216, 216], fill='#2E86AB', outline='#0F4C75', width=8)
draw.ellipse([100, 100, 156, 156], fill='#6C9BD1', outline='#2E86AB', width=4)

# Save as ICO
img.save('icon.ico', format='ICO', sizes=[(256, 256)])
print("Created icon.ico")
```

Run it:
```powershell
python create_icon.py
```

### 2.2 Run PyInstaller
```powershell
# Clean previous builds
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

# Build application
pyinstaller build.spec

# This will take 10-30 minutes depending on your system
# PyInstaller will:
# 1. Analyze dependencies
# 2. Collect all required files
# 3. Bundle Python interpreter
# 4. Create executable in dist/NucleiSegmentationApp/
```

### 2.3 Verify Build
```powershell
# Navigate to output directory
cd dist\NucleiSegmentationApp

# Test the executable
.\NucleiSegmentationApp.exe

# Check folder size (should be ~2-4 GB with all dependencies)
```

---

## Step 3: Test Standalone Build

### 3.1 Test on Build Machine
```powershell
# Run from dist folder
cd dist\NucleiSegmentationApp
.\NucleiSegmentationApp.exe

# Test key functionality:
# - Application launches without errors
# - Can create new project
# - Can load TIFF file
# - Segmentation works (if GPU available)
# - All tabs are accessible
# - Can export results
```

### 3.2 Check for Missing Dependencies
If you encounter errors:

```powershell
# Run with console window to see errors
pyinstaller build.spec --console

# Check logs
Get-Content build\NucleiSegmentationApp\warn-NucleiSegmentationApp.txt
```

Common issues:
- **Missing DLLs**: Add to `binaries` in build.spec
- **Missing modules**: Add to `hiddenimports` in build.spec
- **Missing data files**: Add to `datas` in build.spec

---

## Step 4: Create Installer

### 4.1 Create License File
Create `LICENSE.txt` in the root directory:

```
MIT License

Copyright (c) 2025 Research Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 4.2 Create Quick Start Guide
Create `docs/QUICK_START.md`:

```markdown
# Quick Start Guide

## First Launch

1. Run **Nuclei Segmentation App** from Start Menu or Desktop
2. Click **File → New Project** to create a new project
3. Load a TIFF image with **File → Open Image**

## Basic Workflow

1. **Segment**: Click "Run Segmentation" in Segmentation tab
2. **Review**: Check Quality Control panel for issues
3. **Correct**: Use manual tools to fix errors (split/merge/delete)
4. **Measure**: Go to Analysis tab and click "Extract Measurements"
5. **Visualize**: View charts in Visualization tab
6. **Export**: Save results as CSV/Excel

## GPU Acceleration

If you have an NVIDIA GPU with CUDA:
- Ensure CUDA Toolkit 11.8+ is installed
- GPU will be automatically detected and used

## Support

- Documentation: `docs/` folder in installation directory
- GitHub: https://github.com/yourlab/nuclei-segmentation
- Email: support@yourlab.edu
```

### 4.3 Compile Installer
```powershell
# Navigate back to project root
cd C:\NucleiSegmentationApp

# Compile with Inno Setup (GUI)
# 1. Open Inno Setup Compiler
# 2. File → Open → Select installer.iss
# 3. Build → Compile
# Or use command line:
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss

# Output will be in Output/NucleiSegmentationApp_v1.0.0_Setup.exe
```

---

## Step 5: Test Installer

### 5.1 Test Installation
```powershell
# Run the installer
.\Output\NucleiSegmentationApp_v1.0.0_Setup.exe

# Test on a clean Windows 10/11 machine if possible
# - Install to default location
# - Launch from Start Menu
# - Test core functionality
# - Uninstall cleanly
```

### 5.2 Test Upgrade
```powershell
# Install version 1.0.0
# Modify version in build.spec and installer.iss
# Rebuild and install - should detect and upgrade
```

---

## Step 6: Distribution

### 6.1 Prepare Release Package
```powershell
# Create distribution folder
New-Item -ItemType Directory -Path Release -Force

# Copy installer
Copy-Item "Output\NucleiSegmentationApp_v1.0.0_Setup.exe" -Destination "Release\"

# Copy documentation
Copy-Item README.md -Destination Release\
Copy-Item -Recurse docs -Destination Release\

# Create checksums
CertUtil -hashfile "Release\NucleiSegmentationApp_v1.0.0_Setup.exe" SHA256 > Release\SHA256SUMS.txt
```

### 6.2 Upload to Distribution Platform
- **GitHub Releases**: Upload installer + checksums
- **Institutional Server**: FTP/web hosting
- **Package Manager**: Consider Chocolatey for Windows

---

## Troubleshooting

### Build Fails: Missing Module
**Error:** `ModuleNotFoundError: No module named 'X'`

**Solution:**
```python
# Add to hiddenimports in build.spec
hiddenimports=[
    'X',
    'X.submodule',
]
```

### Build Fails: Missing DLL
**Error:** `ImportError: DLL load failed: The specified module could not be found.`

**Solution:**
```python
# Add to binaries in build.spec
binaries=[
    ('path/to/library.dll', '.'),
]
```

### Executable Size Too Large
**Problem:** dist folder is >5 GB

**Solution:**
```python
# Add to excludes in build.spec
excludes=[
    'matplotlib',  # If not needed
    'test',
    'unittest',
    'IPython',
]
```

Use UPX compression:
```powershell
pyinstaller build.spec --upx-dir="C:\path\to\upx"
```

### Application Crashes on Startup
**Problem:** No error message, immediate crash

**Solution:**
```powershell
# Build with console to see errors
pyinstaller build.spec --console

# Check Windows Event Viewer
# - Windows Logs → Application
# - Look for application errors
```

### GPU Not Detected
**Problem:** Application doesn't use GPU

**Solution:**
- Ensure CUDA Toolkit is installed on target machine
- Check torch.cuda.is_available() in bundled Python
- May need to bundle CUDA DLLs manually

---

## Build Size Optimization

### Expected Sizes
- **Executable only**: ~100 MB
- **Full distribution**: 2-4 GB (includes PyTorch, Cellpose, etc.)
- **Installer**: 1-2 GB (compressed)

### Reduce Size
1. **Exclude unnecessary packages**:
   ```python
   excludes=['matplotlib', 'IPython', 'jupyter']
   ```

2. **Strip binaries** (reduces ~20%):
   ```python
   strip=True  # In EXE() configuration
   ```

3. **Use UPX compression** (reduces ~30%):
   ```powershell
   pyinstaller build.spec --upx-dir="C:\upx"
   ```

4. **Single-file mode** (slower startup):
   ```python
   exe = EXE(
       ...
       one_file=True,  # Single .exe instead of folder
   )
   ```

---

## Continuous Integration (CI)

### GitHub Actions Example
```yaml
# .github/workflows/build.yml
name: Build Windows Installer

on:
  release:
    types: [created]

jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pyinstaller
      
      - name: Build executable
        run: pyinstaller build.spec
      
      - name: Setup Inno Setup
        run: |
          choco install innosetup -y
      
      - name: Build installer
        run: iscc installer.iss
      
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: installer
          path: Output/*.exe
```

---

## Code Signing (Optional)

For production distribution, consider code signing:

```powershell
# Obtain code signing certificate
# Sign executable
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com NucleiSegmentationApp.exe

# Sign installer
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com NucleiSegmentationApp_v1.0.0_Setup.exe
```

Benefits:
- Removes Windows SmartScreen warnings
- Verifies publisher identity
- Required for some institutional deployments

---

## Appendix: Build Checklist

- [ ] All dependencies installed
- [ ] Application runs from source
- [ ] Icon created (icon.ico)
- [ ] Version info updated (version_info.txt)
- [ ] License file created (LICENSE.txt)
- [ ] Documentation complete
- [ ] PyInstaller build succeeds
- [ ] Executable tested standalone
- [ ] Inno Setup installer created
- [ ] Installer tested on clean machine
- [ ] Release notes written
- [ ] Checksums generated
- [ ] Distribution package created
- [ ] (Optional) Code signed

---

## Support

For build issues:
- Check PyInstaller documentation: https://pyinstaller.org/
- Inno Setup help: https://jrsoftware.org/ishelp/
- Project issues: https://github.com/yourlab/nuclei-segmentation/issues

**Last Updated:** November 22, 2025  
**Version:** 1.0.0
