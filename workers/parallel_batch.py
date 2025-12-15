"""
Parallel batch processing utilities for high-throughput analysis
Uses multiprocessing for CPU-bound tasks to bypass Python's GIL
"""

import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import time
import traceback
from dataclasses import dataclass


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing"""
    max_workers: Optional[int] = None  # None = auto-detect
    use_gpu: bool = False
    gpu_devices: Optional[List[int]] = None  # List of GPU device IDs to use
    segmentation_engine: str = 'cellpose'
    segmentation_params: Optional[Dict] = None
    run_qc: bool = True
    run_measurements: bool = True
    measurement_config: Optional[Dict] = None
    cache_enabled: bool = True
    resume_from_checkpoint: bool = True
    checkpoint_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = max(1, mp.cpu_count() - 1)
        if self.segmentation_params is None:
            self.segmentation_params = {}
        if self.measurement_config is None:
            self.measurement_config = {}
        if self.gpu_devices is None:
            self.gpu_devices = [0]


def _process_single_image(args: Tuple) -> Dict:
    """
    Process a single image - designed to run in a separate process.
    This function is the worker for multiprocessing.
    
    Args:
        args: Tuple of (image_path, config_dict, worker_id)
        
    Returns:
        Dict with processing results
    """
    image_path, config_dict, worker_id = args
    
    result = {
        'image_path': str(image_path),
        'image_name': Path(image_path).name,
        'status': 'pending',
        'worker_id': worker_id,
        'start_time': time.time(),
        'end_time': None,
        'error': None,
        'n_nuclei': 0,
        'masks': None,
        'measurements': None,
        'qc_results': None
    }
    
    try:
        # Import inside function to avoid serialization issues
        from core.image_io import TIFFLoader
        from core.segmentation import SegmentationEngine
        from core.quality_control import QualityControl
        from core.measurements import MeasurementEngine
        
        # Determine GPU usage for this worker
        use_gpu = config_dict.get('use_gpu', False)
        gpu_devices = config_dict.get('gpu_devices', [0])
        
        # Assign GPU based on worker ID (round-robin)
        if use_gpu and gpu_devices:
            import torch
            gpu_idx = worker_id % len(gpu_devices)
            device_id = gpu_devices[gpu_idx]
            torch.cuda.set_device(device_id)
        
        # Load image
        filepath = str(image_path)
        if filepath.lower().endswith('.nd'):
            image, metadata = TIFFLoader.load_metamorph_nd(filepath, stage=0, timepoint=0)
        elif filepath.lower().endswith('.vsi'):
            image, metadata = TIFFLoader.load_vsi(filepath, scene=0)
        elif filepath.lower().endswith('.lif'):
            image, metadata = TIFFLoader.load_lif(filepath, scene=0)
        else:
            image, metadata = TIFFLoader.load_tiff(filepath)
        
        # Run segmentation
        engine = SegmentationEngine(gpu_available=use_gpu)
        seg_engine = config_dict.get('segmentation_engine', 'cellpose')
        seg_params = config_dict.get('segmentation_params', {})
        
        if seg_engine == 'cellpose':
            masks, seg_info = engine.segment_cellpose(
                image=image,
                model_name=seg_params.get('model_name', 'nuclei'),
                diameter=seg_params.get('diameter'),
                flow_threshold=seg_params.get('flow_threshold', 0.4),
                cellprob_threshold=seg_params.get('cellprob_threshold', 0.0),
                do_3d=seg_params.get('do_3d', False),
                channels=seg_params.get('channels', [0, 0])
            )
        elif seg_engine == 'sam':
            masks, seg_info = engine.segment_sam(
                image=image,
                model_type=seg_params.get('model_type', 'vit_h'),
                automatic=seg_params.get('automatic', True)
            )
        elif seg_engine == 'allen':
            masks, seg_info = engine.segment_allen(
                image=image,
                mode=seg_params.get('mode', 'auto'),
                structure_id=seg_params.get('structure_id'),
                workflow_id=seg_params.get('workflow_id')
            )
        else:
            raise ValueError(f"Unknown segmentation engine: {seg_engine}")
        
        result['n_nuclei'] = seg_info.get('nucleus_count', 0)
        result['segmentation_info'] = seg_info
        # Note: We don't store masks directly in result to avoid memory issues
        # Instead, return metadata and let caller handle storage
        
        # Run QC if enabled
        if config_dict.get('run_qc', True) and masks is not None:
            qc = QualityControl()
            # Simplified QC for batch processing
            qc_results = {
                'nucleus_count': result['n_nuclei'],
                'flagged_count': 0,
                'flagged_percentage': 0.0
            }
            result['qc_results'] = qc_results
        
        # Run measurements if enabled
        if config_dict.get('run_measurements', True) and masks is not None:
            extractor = MeasurementEngine()
            measurement_config = config_dict.get('measurement_config', {})
            
            # Get intensity images from the loaded image
            intensity_images = {}
            channel_names = metadata.get('channel_names', [])
            
            if image.ndim == 4:  # 3D multichannel
                for i, ch_name in enumerate(channel_names):
                    intensity_images[ch_name] = np.max(image[:, i, :, :], axis=0)
            elif image.ndim == 3:  # 2D multichannel or 3D single
                if len(channel_names) > 0:
                    for i, ch_name in enumerate(channel_names):
                        if i < image.shape[0]:
                            intensity_images[ch_name] = image[i]
                else:
                    intensity_images['Channel_0'] = image
            else:  # 2D
                intensity_images['Channel_0'] = image
            
            # Extract measurements
            measurements_df = extractor.extract_measurements(
                masks=masks,
                intensity_images=intensity_images,
                pixel_size=metadata.get('pixel_size')
            )
            
            # Convert to dict for serialization
            result['measurements'] = measurements_df.to_dict() if measurements_df is not None else None
        
        result['status'] = 'complete'
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = f"{str(e)}\n{traceback.format_exc()}"
    
    result['end_time'] = time.time()
    result['processing_time'] = result['end_time'] - result['start_time']
    
    return result


class ParallelBatchProcessor:
    """
    High-throughput batch processor using multiprocessing.
    
    This class implements parallel processing strategies for batch segmentation
    and analysis to maximize CPU/GPU utilization.
    """
    
    def __init__(self, config: Optional[BatchProcessingConfig] = None):
        """
        Initialize the parallel batch processor.
        
        Args:
            config: BatchProcessingConfig with processing parameters
        """
        self.config = config or BatchProcessingConfig()
        self.results: List[Dict] = []
        self.progress_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None
        self._cancelled = False
    
    def set_progress_callback(self, callback: Callable[[int, int], None]):
        """Set callback for progress updates (current, total)"""
        self.progress_callback = callback
    
    def set_status_callback(self, callback: Callable[[str], None]):
        """Set callback for status messages"""
        self.status_callback = callback
    
    def _emit_progress(self, current: int, total: int):
        """Emit progress update"""
        if self.progress_callback:
            self.progress_callback(current, total)
    
    def _emit_status(self, message: str):
        """Emit status message"""
        if self.status_callback:
            self.status_callback(message)
    
    def process_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Process a batch of images in parallel using multiprocessing.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of result dictionaries for each image
        """
        self.results = []
        self._cancelled = False
        total = len(image_paths)
        
        if total == 0:
            return self.results
        
        self._emit_status(f"Starting batch processing of {total} images with {self.config.max_workers} workers...")
        
        # Prepare arguments for workers
        config_dict = {
            'use_gpu': self.config.use_gpu,
            'gpu_devices': self.config.gpu_devices,
            'segmentation_engine': self.config.segmentation_engine,
            'segmentation_params': self.config.segmentation_params,
            'run_qc': self.config.run_qc,
            'run_measurements': self.config.run_measurements,
            'measurement_config': self.config.measurement_config
        }
        
        # Ensure max_workers is set
        max_workers = self.config.max_workers or max(1, mp.cpu_count() - 1)
        
        worker_args = [
            (path, config_dict, i % max_workers)
            for i, path in enumerate(image_paths)
        ]
        
        # Use ProcessPoolExecutor for better exception handling
        completed = 0
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(_process_single_image, args): args[0]
                    for args in worker_args
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_path):
                    if self._cancelled:
                        self._emit_status("Batch processing cancelled")
                        break
                    
                    try:
                        result = future.result(timeout=600)  # 10 minute timeout per image
                        self.results.append(result)
                        
                        completed += 1
                        self._emit_progress(completed, total)
                        
                        status = "✓" if result['status'] == 'complete' else "✗"
                        self._emit_status(
                            f"{status} {result['image_name']}: "
                            f"{result['n_nuclei']} nuclei ({result.get('processing_time', 0):.1f}s)"
                        )
                        
                    except Exception as e:
                        image_path = future_to_path[future]
                        self.results.append({
                            'image_path': str(image_path),
                            'image_name': Path(image_path).name,
                            'status': 'error',
                            'error': str(e),
                            'n_nuclei': 0
                        })
                        completed += 1
                        self._emit_progress(completed, total)
        
        except Exception as e:
            self._emit_status(f"Batch processing error: {e}")
        
        return self.results
    
    def cancel(self):
        """Cancel batch processing"""
        self._cancelled = True
    
    def get_summary(self) -> Dict:
        """Get summary statistics of batch processing"""
        if not self.results:
            return {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'total_nuclei': 0,
                'avg_processing_time': 0.0
            }
        
        successful = sum(1 for r in self.results if r.get('status') == 'complete')
        failed = sum(1 for r in self.results if r.get('status') == 'error')
        total_nuclei = sum(r.get('n_nuclei', 0) for r in self.results)
        
        processing_times = [r.get('processing_time', 0) for r in self.results if r.get('processing_time')]
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        
        return {
            'total': len(self.results),
            'successful': successful,
            'failed': failed,
            'total_nuclei': total_nuclei,
            'avg_processing_time': avg_time,
            'cancelled': self._cancelled
        }


class BatchStateManager:
    """
    Manages batch processing state for pause/resume functionality.
    Stores progress in JSON/SQLite for crash recovery.
    """
    
    def __init__(self, state_file: Optional[str] = None):
        """
        Initialize state manager.
        
        Args:
            state_file: Path to state file (JSON)
        """
        self.state_file = Path(state_file) if state_file else None
        self.state = {
            'started_at': None,
            'completed_images': [],
            'pending_images': [],
            'failed_images': [],
            'last_updated': None
        }
    
    def save_state(self):
        """Save current state to file"""
        if self.state_file is None:
            return
        
        import json
        self.state['last_updated'] = time.time()
        
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def load_state(self) -> bool:
        """Load state from file. Returns True if state was loaded."""
        if self.state_file is None or not self.state_file.exists():
            return False
        
        import json
        try:
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
            return True
        except Exception:
            return False
    
    def mark_completed(self, image_path: str, result: Dict):
        """Mark an image as completed"""
        self.state['completed_images'].append({
            'path': image_path,
            'result': result,
            'completed_at': time.time()
        })
        if image_path in self.state['pending_images']:
            self.state['pending_images'].remove(image_path)
        self.save_state()
    
    def mark_failed(self, image_path: str, error: str):
        """Mark an image as failed"""
        self.state['failed_images'].append({
            'path': image_path,
            'error': error,
            'failed_at': time.time()
        })
        if image_path in self.state['pending_images']:
            self.state['pending_images'].remove(image_path)
        self.save_state()
    
    def get_pending_images(self) -> List[str]:
        """Get list of images that still need processing"""
        return self.state.get('pending_images', [])
    
    def clear_state(self):
        """Clear all state"""
        self.state = {
            'started_at': None,
            'completed_images': [],
            'pending_images': [],
            'failed_images': [],
            'last_updated': None
        }
        if self.state_file and self.state_file.exists():
            self.state_file.unlink()
