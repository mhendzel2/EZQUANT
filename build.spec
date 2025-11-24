# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Nuclei Segmentation Application
Builds a standalone Windows executable with all dependencies bundled.

Usage:
    pyinstaller build.spec

Requirements:
    - PyInstaller 6.0+
    - All dependencies from requirements_updated.txt installed
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

# Application metadata
APP_NAME = 'NucleiSegmentationApp'
APP_VERSION = '1.0.0'
APP_AUTHOR = 'Research Lab'
APP_DESCRIPTION = 'AI-powered nuclei segmentation and analysis tool'

# Get the application directory
block_cipher = None
app_dir = os.path.abspath('.')

# Collect all submodules for key packages
cellpose_submodules = collect_submodules('cellpose')
sam_submodules = collect_submodules('segment_anything')
skimage_submodules = collect_submodules('skimage')
torch_submodules = collect_submodules('torch')
plotly_submodules = collect_submodules('plotly')

# Collect data files for packages that need them
cellpose_datas = collect_data_files('cellpose', include_py_files=True)
sam_datas = collect_data_files('segment_anything', include_py_files=True)
plotly_datas = collect_data_files('plotly')
skimage_datas = collect_data_files('skimage')

# Collect dynamic libraries (DLLs)
torch_binaries = collect_dynamic_libs('torch')

# Application-specific data files
app_datas = [
    ('plugins', 'plugins'),  # Include all plugin files
    ('docs', 'docs'),  # Include documentation
    ('README.md', '.'),
]

# Add icon if it exists
if os.path.exists('icon.ico'):
    app_datas.append(('icon.ico', '.'))

# Hidden imports that PyInstaller might miss
hiddenimports = [
    # Core packages
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'PySide6.QtWebEngineWidgets',
    'PySide6.QtWebEngineCore',
    'pyqtgraph',
    'pyqtgraph.opengl',
    
    # Scientific stack
    'numpy',
    'numpy.core._methods',
    'numpy.lib.format',
    'scipy',
    'scipy.special',
    'scipy.special.cython_special',
    'pandas',
    'pandas._libs.tslibs.timedeltas',
    
    # Image processing
    'tifffile',
    'imagecodecs',
    'imagecodecs._shared',
    'PIL',
    'PIL.Image',
    'cv2',
    
    # Machine learning
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torchvision',
    'cellpose',
    'cellpose.models',
    'cellpose.dynamics',
    'segment_anything',
    
    # Scikit packages
    'sklearn',
    'sklearn.mixture',
    'sklearn.preprocessing',
    'sklearn.decomposition',
    'skimage',
    'skimage.io',
    'skimage.measure',
    'skimage.feature',
    'skimage.filters',
    'skimage.morphology',
    'skimage.segmentation',
    
    # Plotting
    'plotly',
    'plotly.graph_objs',
    'plotly.express',
    'kaleido',
    
    # Data formats
    'openpyxl',
    'openpyxl.cell',
    'openpyxl.styles',
    'xlsxwriter',
    
    # Other
    'certifi',
    'charset_normalizer',
] + cellpose_submodules + sam_submodules + skimage_submodules + torch_submodules + plotly_submodules

# Analysis configuration
a = Analysis(
    ['main.py'],
    pathex=[app_dir],
    binaries=torch_binaries,
    datas=app_datas + cellpose_datas + sam_datas + plotly_datas + skimage_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'matplotlib',  # We use Plotly instead
        'IPython',
        'jupyter',
        'notebook',
        'tkinter',
        'test',
        'unittest',
        'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove duplicate files
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Hide console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None,
    version='version_info.txt' if os.path.exists('version_info.txt') else None,
)

# Collect everything into distribution folder
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)
