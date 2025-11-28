# PyTorch Upgrade for RTX 50-Series GPU Support

## Problem
If you see this warning when running segmentation:
```
NVIDIA GeForce RTX 5050 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

Your GPU is too new for the old PyTorch version (2.0.x with CUDA 11.8).

## Solution

### Step 1: Download CUDA 12.4
**IMPORTANT: You must install CUDA 12.4 first!**

1. Go to: https://developer.nvidia.com/cuda-downloads
2. Select:
   - Operating System: Windows
   - Architecture: x86_64
   - Version: Your Windows version
   - Installer Type: exe (network) - recommended
3. Download and install
4. Reboot your computer

### Step 2: Upgrade PyTorch

#### For Existing Installations:
Double-click: `upgrade_pytorch.bat`

Or manually:
```powershell
# Activate virtual environment
venv\Scripts\activate

# Upgrade PyTorch
pip uninstall -y torch torchvision
pip install torch>=2.5.0 torchvision>=0.20.0 --index-url https://download.pytorch.org/whl/cu124
```

#### For Fresh Installations:
Just run `install.bat` - it will automatically install the correct version.

### Step 3: Verify
After upgrade, run the application and check:
- No more sm_120 warning
- Status bar shows "GPU: Available"
- Segmentation is much faster

## What Changed

### Updated Versions
- **PyTorch**: 2.0.x → 2.5.0+
- **CUDA**: 11.8 → 12.4
- **Torchvision**: 0.15.0 → 0.20.0+

### Why This Matters
- RTX 5050 (and all RTX 50-series) use CUDA compute capability sm_120
- PyTorch 2.0.x only supports up to sm_90
- PyTorch 2.5+ adds sm_120 support
- CUDA 12.4 is required for sm_120

### Performance Impact
**Before (CPU mode):**
- ~30-60 seconds per image
- High CPU usage
- Warning messages

**After (GPU mode):**
- ~5-10 seconds per image
- Low CPU usage
- No warnings
- 3-6x faster!

## Troubleshooting

### "Failed to install PyTorch"
- Check internet connection
- Try again - downloads are large (~2GB)
- Use network installer for CUDA (not local)

### "CUDA not found"
- Did you install CUDA 12.4?
- Did you reboot after installing CUDA?
- Check: `nvidia-smi` in command prompt

### Still showing CPU mode
1. Verify CUDA installed: `nvidia-smi`
2. Verify PyTorch sees GPU:
   ```powershell
   venv\Scripts\activate
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. Should print `True`

### "DLL load failed" errors
- Reboot after installing CUDA 12.4
- Ensure no other CUDA versions interfering
- Reinstall CUDA 12.4

## Version Compatibility

| GPU Series | CUDA Capability | PyTorch Version | CUDA Version |
|------------|-----------------|-----------------|--------------|
| RTX 30-series | sm_86 | 2.0+ | 11.8+ |
| RTX 40-series | sm_89 | 2.0+ | 11.8+ |
| RTX 50-series | sm_120 | **2.5+** | **12.4+** |

## Additional Resources

- PyTorch Installation: https://pytorch.org/get-started/locally/
- CUDA Downloads: https://developer.nvidia.com/cuda-downloads
- CUDA Compute Capability: https://developer.nvidia.com/cuda-gpus

## Questions?

Check `upgrade_pytorch.bat` output for detailed GPU information after upgrade.
