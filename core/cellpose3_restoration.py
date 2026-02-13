"""
Cellpose3 restoration mode integration.

Cellpose3 targets restoration for segmentation in degraded microscopy,
directly aligned with undergraduate-lab usability: fewer knobs, fewer catastrophic failures.

This module provides a wrapper around Cellpose3's restoration features
to improve segmentation quality on noisy or degraded images.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class Cellpose3RestorationEngine:
    """
    Wrapper for Cellpose3 with restoration mode enabled.
    
    Cellpose3 includes denoising and restoration capabilities that can
    significantly improve segmentation quality on degraded images, which
    is common in undergraduate lab settings.
    """
    
    def __init__(self, gpu_available: bool = False):
        """
        Initialize Cellpose3 restoration engine.
        
        Args:
            gpu_available: Whether GPU is available for acceleration
        """
        self.gpu_available = gpu_available
        self.device = 'cuda' if gpu_available else 'cpu'
        self._model = None
        self._model_name = None
    
    def _load_model(self, model_name: str = 'nuclei', restore_type: str = 'denoise_cyto3'):
        """
        Load Cellpose3 model with restoration enabled.
        
        Args:
            model_name: Base model name ('nuclei', 'cyto3', etc.)
            restore_type: Type of restoration to apply
                         Options: 'denoise_cyto3', 'deblur_cyto3', None
        """
        try:
            from cellpose import models
            
            # Check if we need to reload
            if self._model is None or self._model_name != model_name:
                logger.info(f"Loading Cellpose3 model: {model_name} with restoration: {restore_type}")
                
                # Cellpose3 models support restoration via restore_type parameter
                self._model = models.Cellpose(
                    gpu=self.gpu_available,
                    model_type=model_name,
                    restore_type=restore_type  # Cellpose3 restoration feature
                )
                self._model_name = model_name
                
                logger.info(f"Cellpose3 model loaded successfully on {self.device}")
        
        except ImportError as e:
            logger.error("Cellpose3 not installed or version too old")
            raise ImportError(
                "Cellpose3 (>=3.0) is required for restoration mode. "
                "Install with: pip install cellpose>=3.0"
            ) from e
        except TypeError as e:
            # restore_type might not be supported in older versions
            logger.warning(
                f"restore_type parameter not supported: {e}. "
                "You may need to upgrade to Cellpose 3.0+. "
                "Falling back to standard mode."
            )
            from cellpose import models
            self._model = models.Cellpose(
                gpu=self.gpu_available,
                model_type=model_name
            )
            self._model_name = model_name
    
    def segment_with_restoration(self,
                                image: np.ndarray,
                                model_name: str = 'nuclei',
                                restoration_mode: str = 'auto',
                                diameter: Optional[float] = None,
                                flow_threshold: float = 0.4,
                                cellprob_threshold: float = 0.0,
                                do_3d: bool = False,
                                **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Segment with Cellpose3 restoration mode.
        
        Args:
            image: Input image (2D or 3D)
            model_name: Cellpose model name
            restoration_mode: Restoration mode to apply
                            'auto': Automatically detect best restoration
                            'denoise': Apply denoising restoration
                            'deblur': Apply deblurring restoration
                            'none': No restoration
            diameter: Expected cell diameter in pixels
            flow_threshold: Flow error threshold
            cellprob_threshold: Cell probability threshold
            do_3d: Whether to use 3D segmentation
            **kwargs: Additional parameters passed to Cellpose
            
        Returns:
            Tuple of (masks, info_dict)
        """
        import time
        start_time = time.time()
        
        # Determine restoration type based on mode
        restore_type = self._get_restore_type(image, restoration_mode)
        
        # Load model with appropriate restoration
        self._load_model(model_name, restore_type)
        
        # Prepare channels
        channels = kwargs.get('channels', [0, 0])
        
        # Run segmentation with restoration
        logger.info(
            f"Running Cellpose3 segmentation with restoration mode: {restore_type}"
        )
        
        try:
            masks, flows, styles, diams = self._model.eval(
                image,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                do_3d=do_3d,
                channels=channels,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Cellpose3 segmentation failed: {e}")
            raise
        
        end_time = time.time()
        
        # Calculate statistics
        nucleus_count = len(np.unique(masks)) - 1  # Exclude background
        
        # Assess restoration quality
        restoration_metrics = self._assess_restoration_quality(image, flows)
        
        info = {
            'model_name': model_name,
            'restoration_mode': restore_type,
            'nucleus_count': nucleus_count,
            'diameter_used': diams if diameter is None else diameter,
            'flow_threshold': flow_threshold,
            'cellprob_threshold': cellprob_threshold,
            'do_3d': do_3d,
            'processing_time': end_time - start_time,
            'restoration_metrics': restoration_metrics,
            'gpu_used': self.gpu_available
        }
        
        return masks, info
    
    def _get_restore_type(self, image: np.ndarray, mode: str) -> Optional[str]:
        """
        Determine the appropriate restoration type based on image quality.
        
        Args:
            image: Input image
            mode: Restoration mode ('auto', 'denoise', 'deblur', 'none')
            
        Returns:
            Restoration type string or None
        """
        if mode == 'none':
            return None
        elif mode == 'denoise':
            return 'denoise_cyto3'
        elif mode == 'deblur':
            return 'deblur_cyto3'
        elif mode == 'auto':
            # Auto-detect based on image quality metrics
            snr = self._estimate_snr(image)
            blur_score = self._estimate_blur(image)
            
            logger.info(f"Image quality - SNR: {snr:.2f}, Blur score: {blur_score:.4f}")
            
            # Decision logic
            if snr < 10:  # Low SNR indicates noisy image
                return 'denoise_cyto3'
            elif blur_score > 0.1:  # High blur score
                return 'deblur_cyto3'
            else:
                return None  # Image quality is acceptable
        else:
            logger.warning(f"Unknown restoration mode: {mode}, using default")
            return 'denoise_cyto3'
    
    def _estimate_snr(self, image: np.ndarray) -> float:
        """
        Estimate signal-to-noise ratio of the image.
        
        Args:
            image: Input image
            
        Returns:
            Estimated SNR
        """
        # Simple SNR estimation: mean / std
        img_float = image.astype(np.float32)
        
        # Use middle slices if 3D to avoid edge effects
        if img_float.ndim == 3:
            mid_z = img_float.shape[0] // 2
            img_float = img_float[mid_z]
        
        mean_signal = np.mean(img_float)
        std_noise = np.std(img_float)
        
        if std_noise == 0:
            return float('inf')
        
        return mean_signal / std_noise
    
    def _estimate_blur(self, image: np.ndarray) -> float:
        """
        Estimate blur level using Laplacian variance.
        
        Args:
            image: Input image
            
        Returns:
            Blur score (higher = more blurred)
        """
        from scipy import ndimage
        
        # Use middle slice if 3D
        img = image
        if img.ndim == 3:
            mid_z = img.shape[0] // 2
            img = img[mid_z]
        
        # Compute Laplacian
        laplacian = ndimage.laplace(img.astype(np.float32))
        
        # Variance of Laplacian (lower = more blurred)
        variance = np.var(laplacian)
        
        # Normalize and invert so higher score = more blur
        # This is a heuristic - adjust based on your images
        max_expected_variance = 1000.0
        blur_score = 1.0 - min(variance / max_expected_variance, 1.0)
        
        return blur_score
    
    def _assess_restoration_quality(self, 
                                   image: np.ndarray,
                                   flows: Any) -> Dict[str, float]:
        """
        Assess quality of restoration and segmentation.
        
        Args:
            image: Original input image
            flows: Flow output from Cellpose
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Flow quality metrics
        if flows is not None and len(flows) > 0:
            flow_magnitude = flows[0]  # Flow field magnitude
            
            if isinstance(flow_magnitude, np.ndarray):
                metrics['mean_flow_magnitude'] = float(np.mean(np.abs(flow_magnitude)))
                metrics['max_flow_magnitude'] = float(np.max(np.abs(flow_magnitude)))
        
        # Image quality metrics
        metrics['snr'] = self._estimate_snr(image)
        metrics['blur_score'] = self._estimate_blur(image)
        
        return metrics


def should_use_restoration(image: np.ndarray,
                          snr_threshold: float = 10.0,
                          blur_threshold: float = 0.1) -> Tuple[bool, str]:
    """
    Determine if restoration mode should be used based on image quality.
    
    Args:
        image: Input image
        snr_threshold: SNR below which to recommend denoising
        blur_threshold: Blur score above which to recommend deblurring
        
    Returns:
        Tuple of (should_use_restoration, recommended_mode)
    """
    engine = Cellpose3RestorationEngine()
    
    snr = engine._estimate_snr(image)
    blur_score = engine._estimate_blur(image)
    
    if snr < snr_threshold and blur_score > blur_threshold:
        return True, 'denoise'  # Noisy and blurred
    elif snr < snr_threshold:
        return True, 'denoise'  # Primarily noisy
    elif blur_score > blur_threshold:
        return True, 'deblur'  # Primarily blurred
    else:
        return False, 'none'  # Good quality
