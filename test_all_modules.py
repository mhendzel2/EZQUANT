"""
Comprehensive test script for all modules
Tests imports and basic functionality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("NUCLEI SEGMENTATION APP - MODULE TEST")
print("=" * 70)

# Test counter
tests_passed = 0
tests_failed = 0

def test_module(module_name, test_func):
    """Test a module and report results"""
    global tests_passed, tests_failed
    try:
        print(f"\n[Testing] {module_name}...", end=" ")
        test_func()
        print("✓ PASS")
        tests_passed += 1
    except Exception as e:
        print(f"✗ FAIL: {e}")
        tests_failed += 1

# === Core Modules ===
print("\n" + "=" * 70)
print("CORE MODULES")
print("=" * 70)

def test_project_data():
    from core.project_data import Project, ImageData
    project = Project("test_project")
    assert project.name == "test_project"
    assert project.storage_backend == "json"

def test_image_io():
    from core.image_io import TIFFLoader
    loader = TIFFLoader()
    assert loader is not None

def test_segmentation():
    from core.segmentation import SegmentationEngine
    engine = SegmentationEngine()
    assert engine is not None

def test_quality_control():
    from core.quality_control import QualityControl
    qc = QualityControl()
    assert qc is not None
    assert len(qc.phase_labels) == 3

def test_measurements():
    from core.measurements import MeasurementEngine
    engine = MeasurementEngine()
    assert engine is not None
    categories = engine.get_available_categories()
    assert 'basic_shape' in categories

def test_plugin_loader():
    from core.plugin_loader import PluginLoader
    loader = PluginLoader()
    assert loader is not None

test_module("project_data", test_project_data)
test_module("image_io", test_image_io)
test_module("segmentation", test_segmentation)
test_module("quality_control", test_quality_control)
test_module("measurements", test_measurements)
test_module("plugin_loader", test_plugin_loader)

# === Workers ===
print("\n" + "=" * 70)
print("WORKER THREADS")
print("=" * 70)

def test_segmentation_worker():
    from workers.segmentation_worker import SegmentationWorker, DiameterEstimationWorker
    # Just test imports
    assert SegmentationWorker is not None
    assert DiameterEstimationWorker is not None

test_module("segmentation_worker", test_segmentation_worker)

# === GUI Modules ===
print("\n" + "=" * 70)
print("GUI MODULES")
print("=" * 70)

def test_main_window():
    from gui.main_window import MainWindow
    # Just test import (don't instantiate without QApplication)
    assert MainWindow is not None

def test_image_viewer():
    from gui.image_viewer import ImageViewer
    assert ImageViewer is not None

def test_segmentation_panel():
    from gui.segmentation_panel import SegmentationPanel
    assert SegmentationPanel is not None

def test_qc_panel():
    from gui.qc_panel import QCPanel, ParameterSuggestionDialog
    assert QCPanel is not None
    assert ParameterSuggestionDialog is not None

def test_manual_correction_tools():
    from gui.manual_correction_tools import (CorrectionToolbar, ManualCorrectionManager,
                                             SplitCommand, MergeCommand, DeleteCommand, AddCommand)
    assert CorrectionToolbar is not None
    assert ManualCorrectionManager is not None

def test_analysis_panel():
    from gui.analysis_panel import AnalysisPanel
    assert AnalysisPanel is not None

def test_visualization_panel():
    from gui.visualization_panel import VisualizationPanel
    assert VisualizationPanel is not None

def test_quality_dashboard():
    from gui.quality_dashboard import QualityDashboard
    assert QualityDashboard is not None

def test_batch_processing():
    from gui.batch_processing import BatchProcessingDialog, BatchProcessingWorker
    assert BatchProcessingDialog is not None
    assert BatchProcessingWorker is not None

test_module("main_window", test_main_window)
test_module("image_viewer", test_image_viewer)
test_module("segmentation_panel", test_segmentation_panel)
test_module("qc_panel", test_qc_panel)
test_module("manual_correction_tools", test_manual_correction_tools)
test_module("analysis_panel", test_analysis_panel)
test_module("visualization_panel", test_visualization_panel)
test_module("quality_dashboard", test_quality_dashboard)
test_module("batch_processing", test_batch_processing)

# === Plugins ===
print("\n" + "=" * 70)
print("PLUGIN SYSTEM")
print("=" * 70)

def test_plugin_template():
    from plugins.plugin_template import MeasurementPlugin, TemplatePlugin, IntensityGradientPlugin
    assert MeasurementPlugin is not None
    template = TemplatePlugin()
    assert template.get_name() == "Template Plugin"

def test_texture_analysis():
    from plugins.examples.texture_analysis import TextureAnalysisPlugin
    plugin = TextureAnalysisPlugin()
    assert plugin.get_name() == "Texture Analysis"

def test_radial_profile():
    from plugins.examples.radial_profile import RadialProfilePlugin
    plugin = RadialProfilePlugin()
    assert plugin.get_name() == "Radial Profile"

test_module("plugin_template", test_plugin_template)
test_module("texture_analysis", test_texture_analysis)
test_module("radial_profile", test_radial_profile)

# === Main Entry Point ===
print("\n" + "=" * 70)
print("MAIN APPLICATION")
print("=" * 70)

def test_main():
    import main
    # Just test that main module imports
    assert main is not None

test_module("main", test_main)

# === Summary ===
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"Total tests: {tests_passed + tests_failed}")
print(f"Passed: {tests_passed} ✓")
print(f"Failed: {tests_failed} ✗")

if tests_failed == 0:
    print("\n🎉 ALL TESTS PASSED! 🎉")
    print("\nThe application is ready to run.")
    print("To start the application, run: python main.py")
else:
    print(f"\n⚠️  {tests_failed} test(s) failed. Please review the errors above.")

print("=" * 70)
