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
            elif engine_type == 'allen':
                self._run_allen(engine)
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

    def _run_allen(self, engine: SegmentationEngine):
        """Run Allen Segmenter segmentation"""
        self.status.emit("Running Allen Segmenter...")
        
        masks, info = engine.segment_allen(
            image=self.image,
            mode=self.parameters.get('mode', 'auto'),
            structure_id=self.parameters.get('structure_id'),
            workflow_id=self.parameters.get('workflow_id'),
            config=self.parameters.get('config')
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
                masks = np.repeat(masks[np.newaxis, ...], n_slices, axis=0)
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
    
    def __init__(self, image: np.ndarray, gpu_available: bool = False, model_name: str = 'nuclei'):
        super().__init__()
        self.image = image
        self.gpu_available = gpu_available
        self.model_name = model_name
    
    def run(self):
        """Run diameter estimation"""
        try:
            self.status.emit("Estimating cell diameter...")
            
            engine = SegmentationEngine(gpu_available=self.gpu_available)
            diameter = engine.estimate_diameter(self.image, model_name=self.model_name)
            
            self.status.emit(f"Estimated diameter: {diameter:.1f} pixels")
            self.finished.emit(diameter)
            
        except Exception as e:
            self.error.emit(f"Diameter estimation error: {str(e)}")


class BatchSegmentationWorker(QThread):
    """
    Background worker for batch segmentation of multiple images
    """
    
    # Signals
    progress = Signal(int, int)  # current, total
    image_finished = Signal(int, np.ndarray, dict)  # index, masks, results
    all_finished = Signal(dict)  # summary statistics
    error = Signal(int, str)  # image index, error message
    status = Signal(str)
    
    def __init__(self,
                 images_data: list,  # List of (index, filepath, ImageData) tuples
                 parameters: Dict,
                 gpu_available: bool = False):
        super().__init__()
        
        self.images_data = images_data
        self.parameters = parameters
        self.gpu_available = gpu_available
        self._is_cancelled = False
    
    def run(self):
        """Run batch segmentation"""
        total = len(self.images_data)
        successful = 0
        failed = 0
        total_nuclei = 0
        
        try:
            # Create segmentation engine once
            self.status.emit("Initializing segmentation engine...")
            engine = SegmentationEngine(gpu_available=self.gpu_available)
            
            engine_type = self.parameters.get('engine', 'cellpose')
            
            for i, (img_index, filepath, img_data) in enumerate(self.images_data):
                if self._is_cancelled:
                    self.status.emit("Batch segmentation cancelled")
                    break
                
                self.progress.emit(i + 1, total)
                self.status.emit(f"Processing {img_data.filename} ({i+1}/{total})...")
                
                try:
                    # Load image
                    from core.image_io import TIFFLoader
                    
                    if filepath.lower().endswith('.nd'):
                        image, metadata = TIFFLoader.load_metamorph_nd(filepath, stage=0, timepoint=0)
                    elif filepath.lower().endswith('.vsi'):
                        image, metadata = TIFFLoader.load_vsi(filepath, scene=0)
                    elif filepath.lower().endswith('.lif'):
                        image, metadata = TIFFLoader.load_lif(filepath, scene=0)
                    else:
                        image, metadata = TIFFLoader.load_tiff(filepath)
                    
                    # Run segmentation
                    if engine_type == 'cellpose':
                        masks, info = self._segment_cellpose(engine, image)
                    elif engine_type == 'sam':
                        masks, info = self._segment_sam(engine, image)
                    else:
                        raise ValueError(f"Unknown engine: {engine_type}")
                    
                    # Emit success
                    nucleus_count = info.get('nucleus_count', 0)
                    total_nuclei += nucleus_count
                    successful += 1
                    
                    self.image_finished.emit(img_index, masks, info)
                    
                except Exception as e:
                    failed += 1
                    error_msg = f"Failed to segment {img_data.filename}: {str(e)}"
                    self.error.emit(img_index, error_msg)
            
            # Send summary
            summary = {
                'total': total,
                'successful': successful,
                'failed': failed,
                'total_nuclei': total_nuclei,
                'cancelled': self._is_cancelled
            }
            
            self.all_finished.emit(summary)
            
        except Exception as e:
            self.status.emit(f"Batch segmentation error: {str(e)}")
    
    def _segment_cellpose(self, engine, image) -> Tuple[np.ndarray, Dict]:
        """Segment with Cellpose"""
        # Extract parameters
        model_name = self.parameters.get('model_name', 'nuclei')
        diameter = self.parameters.get('diameter')
        flow_threshold = self.parameters.get('flow_threshold', 0.4)
        cellprob_threshold = self.parameters.get('cellprob_threshold', 0.0)
        do_3d = self.parameters.get('do_3d', False)
        channels = self.parameters.get('channels', [0, 0])
        
        # Run segmentation - properly unpack the (masks, info) tuple
        masks, info = engine.segment_cellpose(
            image=image,
            model_name=model_name,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            do_3d=do_3d,
            channels=channels
        )
        
        # Return the masks and info from the engine directly
        # This ensures consistency between single and batch segmentation
        return masks, info
    
    def _segment_sam(self, engine, image) -> Tuple[np.ndarray, Dict]:
        """Segment with SAM"""
        model_type = self.parameters.get('model_type', 'vit_h')
        automatic = self.parameters.get('automatic', True)
        
        # Run segmentation - properly unpack the (masks, info) tuple
        masks, info = engine.segment_sam(
            image=image,
            model_type=model_type,
            automatic=automatic
        )
        
        # Return the masks and info from the engine directly
        # This ensures consistency between single and batch segmentation
        return masks, info
    
    def cancel(self):
        """Cancel batch segmentation"""
        self._is_cancelled = True

