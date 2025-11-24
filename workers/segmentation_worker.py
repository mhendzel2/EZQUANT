"""
Worker thread for running segmentation without blocking the UI
"""

from PySide6.QtCore import QThread, Signal
import numpy as np
from typing import Dict, Tuple, Optional

from core.segmentation import SegmentationEngine


class SegmentationWorker(QThread):
    """
    Background worker for running segmentation
    """
    
    # Signals
    finished = Signal(np.ndarray, dict)  # masks, results_dict
    error = Signal(str)  # error message
    progress = Signal(int)  # progress percentage (if applicable)
    status = Signal(str)  # status message
    
    def __init__(self, 
                 image: np.ndarray,
                 parameters: Dict,
                 gpu_available: bool = False):
        super().__init__()
        
        self.image = image
        self.parameters = parameters
        self.gpu_available = gpu_available
        self._is_cancelled = False
    
    def run(self):
        """Run the segmentation in background thread"""
        try:
            self.status.emit("Initializing segmentation engine...")
            
            # Create segmentation engine
            engine = SegmentationEngine(gpu_available=self.gpu_available)
            
            # Get engine type
            engine_type = self.parameters.get('engine', 'cellpose')
            
            if engine_type == 'cellpose':
                self._run_cellpose(engine)
            elif engine_type == 'sam':
                self._run_sam(engine)
            else:
                raise ValueError(f"Unknown engine type: {engine_type}")
            
        except Exception as e:
            import traceback
            error_msg = f"Segmentation error: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)
    
    def _run_cellpose(self, engine: SegmentationEngine):
        """Run Cellpose segmentation"""
        self.status.emit("Running Cellpose segmentation...")
        
        masks, info = engine.segment_cellpose(
            image=self.image,
            model_name=self.parameters.get('model_name', 'nuclei'),
            diameter=self.parameters.get('diameter'),
            flow_threshold=self.parameters.get('flow_threshold', 0.4),
            cellprob_threshold=self.parameters.get('cellprob_threshold', 0.0),
            do_3d=self.parameters.get('do_3d', False),
            channels=self.parameters.get('channels', [0, 0])
        )
        
        if self._is_cancelled:
            return
        
        self.status.emit("Segmentation complete!")
        self.finished.emit(masks, info)
    
    def _run_sam(self, engine: SegmentationEngine):
        """Run SAM segmentation"""
        self.status.emit("Running SAM segmentation...")
        
        # For 3D images, process middle slice or first slice
        if self.image.ndim == 4:
            # Take middle Z slice
            z_mid = self.image.shape[0] // 2
            image_2d = self.image[z_mid]
        else:
            image_2d = self.image
        
        model_type = self.parameters.get('model_type', 'vit_h')
        automatic = self.parameters.get('automatic', True)
        
        try:
            masks, info = engine.segment_sam(
                image=image_2d,
                model_type=model_type,
                automatic=automatic
            )
            
            # If original was 3D, expand mask to 3D (replicate to all slices for now)
            if self.image.ndim == 4:
                n_slices = self.image.shape[0]
                masks_3d = np.zeros((n_slices,) + masks.shape, dtype=masks.dtype)
                for i in range(n_slices):
                    masks_3d[i] = masks
                masks = masks_3d
                info['note'] = 'Applied 2D segmentation to all slices'
            
            if self._is_cancelled:
                return
            
            self.status.emit("Segmentation complete!")
            self.finished.emit(masks, info)
            
        except FileNotFoundError as e:
            self.error.emit(
                "SAM model checkpoint not found.\n\n"
                "Please download the model checkpoint and place it in the 'models/' directory.\n"
                "Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints\n\n"
                f"Error: {str(e)}"
            )
        except ImportError as e:
            self.error.emit(
                "SAM (segment-anything) not installed.\n\n"
                "Install with: pip install segment-anything\n\n"
                f"Error: {str(e)}"
            )
    
    def cancel(self):
        """Cancel the segmentation"""
        self._is_cancelled = True
        self.status.emit("Cancelling...")


class DiameterEstimationWorker(QThread):
    """
    Background worker for estimating cell diameter
    """
    
    # Signals
    finished = Signal(float)  # estimated diameter
    error = Signal(str)
    status = Signal(str)
    
    def __init__(self, image: np.ndarray, gpu_available: bool = False):
        super().__init__()
        self.image = image
        self.gpu_available = gpu_available
    
    def run(self):
        """Run diameter estimation"""
        try:
            self.status.emit("Estimating cell diameter...")
            
            engine = SegmentationEngine(gpu_available=self.gpu_available)
            diameter = engine.estimate_diameter(self.image)
            
            self.status.emit(f"Estimated diameter: {diameter:.1f} pixels")
            self.finished.emit(diameter)
            
        except Exception as e:
            self.error.emit(f"Diameter estimation error: {str(e)}")
