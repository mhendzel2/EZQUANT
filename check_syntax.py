"""
Syntax check for all Python files
Compiles all modules to verify syntax correctness
"""

import py_compile
from pathlib import Path
import sys

# Project structure
project_root = Path(".")

print("=" * 70)
print("NUCLEI SEGMENTATION APP - SYNTAX VALIDATION")
print("=" * 70)

files_to_check = [
    # Core modules
    "main.py",
    "core/project_data.py",
    "core/image_io.py",
    "core/segmentation.py",
    "core/segmentation_backend.py",
    "core/segmentation_3d.py",
    "core/cellpose3_restoration.py",
    "core/active_learning.py",
    "core/quality_control.py",
    "core/measurements.py",
    "core/plugin_loader.py",
    
    # Workers
    "workers/segmentation_worker.py",
    
    # GUI modules
    "gui/main_window.py",
    "gui/image_viewer.py",
    "gui/segmentation_panel.py",
    "gui/qc_panel.py",
    "gui/manual_correction_tools.py",
    "gui/analysis_panel.py",
    "gui/visualization_panel.py",
    "gui/quality_dashboard.py",
    "gui/batch_processing.py",
    
    # Plugins
    "plugins/plugin_template.py",
    "plugins/examples/texture_analysis.py",
    "plugins/examples/radial_profile.py",
]

passed = 0
failed = 0
errors = []

print("\nChecking Python syntax...\n")

for file_path in files_to_check:
    full_path = project_root / file_path
    
    if not full_path.exists():
        print(f"⚠️  SKIP: {file_path} (not found)")
        continue
    
    try:
        py_compile.compile(str(full_path), doraise=True)
        print(f"✓ PASS: {file_path}")
        passed += 1
    except py_compile.PyCompileError as e:
        print(f"✗ FAIL: {file_path}")
        print(f"   Error: {e}")
        failed += 1
        errors.append((file_path, str(e)))

print("\n" + "=" * 70)
print("SYNTAX CHECK SUMMARY")
print("=" * 70)
print(f"Total files: {passed + failed}")
print(f"Passed: {passed} ✓")
print(f"Failed: {failed} ✗")

if failed == 0:
    print("\n🎉 ALL FILES HAVE VALID PYTHON SYNTAX! 🎉")
    print("\nAll modules compiled successfully.")
    print("The application structure is ready.")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements_updated.txt")
    print("2. Run the application: python main.py")
else:
    print(f"\n⚠️  {failed} file(s) have syntax errors:")
    for file_path, error in errors:
        print(f"  - {file_path}")

print("=" * 70)
