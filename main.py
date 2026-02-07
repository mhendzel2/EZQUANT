"""
Nuclei Segmentation & Analysis Application
Main entry point for the desktop application
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt, QTimer
import torch
import PIL.Image

# Disable DecompressionBombError for large images
PIL.Image.MAX_IMAGE_PIXELS = None

from gui.main_window import MainWindow


# Configuration file path for persistent settings
CONFIG_PATH = Path.home() / ".ezquant_config.json"


def load_config() -> dict:
    """Load application configuration from persistent file"""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load config file: {e}")
    return {}


def save_config(cfg: dict) -> bool:
    """Save application configuration to persistent file"""
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)
        return True
    except IOError as e:
        print(f"Warning: Could not save config file: {e}")
        return False


def check_gpu_availability():
    """Check for CUDA-capable GPU and return status info"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return True, f"GPU: {gpu_name} ({gpu_memory:.1f} GB)"
    else:
        return False, "CPU mode (No CUDA-capable GPU detected)"


def show_gpu_dialog(parent, gpu_available: bool, gpu_info: str) -> QMessageBox:
    """Display GPU detection information.

    This is intentionally non-blocking and parented to the main window to avoid
    startup "hang" behavior (e.g., hidden/off-screen modal dialogs).
    """
    msg_box = QMessageBox(parent)

    if gpu_available:
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("GPU Detected")
        msg_box.setText("CUDA-capable GPU detected!")
        msg_box.setInformativeText(
            f"{gpu_info}\n\n"
            "Segmentation will run significantly faster using GPU acceleration."
        )
    else:
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("GPU Not Detected")
        msg_box.setText("No CUDA-capable GPU detected.")
        msg_box.setInformativeText(
            f"{gpu_info}\n\n"
            "Segmentation will run in CPU mode (slower).\n\n"
            "For optimal performance, install:\n"
            "• NVIDIA GPU with 8GB+ VRAM\n"
            "• CUDA Toolkit 11.8 or newer\n"
            "• PyTorch with CUDA support\n\n"
            "Visit: https://pytorch.org/get-started/locally/"
        )

    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.setWindowModality(Qt.WindowModality.NonModal)
    msg_box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    msg_box.open()

    return msg_box


def main():
    """Main application entry point"""
    # Set high DPI scaling (only the policy, other attributes are deprecated in Qt6)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("Nuclei Segmentation & Analysis")
    app.setOrganizationName("NucleiSegApp")
    
    # Load persistent configuration
    cfg = load_config()
    
    # Check GPU availability
    gpu_available, gpu_info = check_gpu_availability()
    
    # Create and show main window
    window = MainWindow(gpu_available=gpu_available, gpu_info=gpu_info)
    window.showMaximized()

    # Show GPU dialog only on first run (non-blocking + parented to main window)
    if cfg.get('show_gpu_dialog', True):
        def _show_startup_gpu_dialog():
            window._startup_gpu_msgbox = show_gpu_dialog(window, gpu_available, gpu_info)
            # Update config to not show dialog again
            cfg['show_gpu_dialog'] = False
            cfg['gpu_enabled'] = gpu_available
            cfg['first_run_date'] = str(Path(__file__).stat().st_mtime)
            save_config(cfg)

        QTimer.singleShot(0, _show_startup_gpu_dialog)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
