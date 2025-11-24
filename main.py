"""
Nuclei Segmentation & Analysis Application
Main entry point for the desktop application
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt
import torch

from gui.main_window import MainWindow


def check_gpu_availability():
    """Check for CUDA-capable GPU and return status info"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return True, f"GPU: {gpu_name} ({gpu_memory:.1f} GB)"
    else:
        return False, "CPU mode (No CUDA-capable GPU detected)"


def show_gpu_dialog(gpu_available, gpu_info):
    """Display GPU detection information on first run"""
    if gpu_available:
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("GPU Detected")
        msg_box.setText("CUDA-capable GPU detected!")
        msg_box.setInformativeText(
            f"{gpu_info}\n\n"
            "Segmentation will run significantly faster using GPU acceleration."
        )
        msg_box.exec()
    else:
        msg_box = QMessageBox()
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
        msg_box.exec()


def main():
    """Main application entry point"""
    # Set high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    app = QApplication(sys.argv)
    app.setApplicationName("Nuclei Segmentation & Analysis")
    app.setOrganizationName("NucleiSegApp")
    
    # Check GPU availability
    gpu_available, gpu_info = check_gpu_availability()
    
    # Create and show main window
    window = MainWindow(gpu_available=gpu_available, gpu_info=gpu_info)
    window.show()
    
    # Show GPU dialog on first run (can be disabled in settings)
    # TODO: Check if first run from settings
    show_gpu_dialog(gpu_available, gpu_info)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
