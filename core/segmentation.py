"""
Core segmentation module wrapping Cellpose and SAM APIs
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import torch
import sys
import os
from core.segmentation_backend import get_segmenter_backend, SegmentationBackend

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# Module-level cache for Allen backend instances
_allen_backend_cache: Dict[str, SegmentationBackend] = {}


def get_cached_allen_backend(mode: str) -> SegmentationBackend:
    """
    Get a cached Allen segmenter backend instance.
    Caching reduces initialization overhead for repeated segmentation calls.
    
    Args:
        mode: 'classic', 'ml', or 'auto'
        
    Returns:
        Cached SegmentationBackend instance
    """
    if mode not in _allen_backend_cache:
        _allen_backend_cache[mode] = get_segmenter_backend(mode)
    return _allen_backend_cache[mode]


def clear_allen_backend_cache():
    """Clear the Allen backend cache to free memory"""
    global _allen_backend_cache
    _allen_backend_cache.clear()


class SegmentationEngine:
    """
    Wrapper for Cellpose and SAM segmentation models
    """
    
    def __init__(self, gpu_available: bool = False):
        self.gpu_available = gpu_available
        self.device = 'cuda' if gpu_available else 'cpu'
        self._configure_torch_runtime()
        
        # Model instances (lazy loaded)
        self._cellpose_model = None
        self._cellpose_model_type = None  # Track currently loaded model type
        self._sam_predictor = None
        self._sam_model_type = None  # Track currently loaded SAM model type
        self._sam_mask_generator = None  # Cache SAM automatic mask generator
        self._sam_generator_model_type = None
        
        # Available models
        self.cellpose_models = ['nuclei', 'cyto', 'cyto2', 'cyto3', 'cyto_sam']
        self.sam_models = ['vit_h', 'vit_l', 'vit_b']
        
        # Allen Segmenter Backend (cached)
        self._allen_backend_mode = None

    def _configure_torch_runtime(self):
        """Configure safe runtime defaults for faster GPU inference."""
        if not self.gpu_available:
            return

        try:
            # cuDNN autotuner improves throughput for repeated conv workloads.
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends.cudnn, "allow_tf32"):
                    torch.backends.cudnn.allow_tf32 = True

            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True

            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception:
            # Performance tuning is best-effort and should never break startup.
            pass

    @staticmethod
    def _compute_label_area_stats(labels: np.ndarray) -> Tuple[int, float, float]:
        """
        Compute nucleus count, median area, and CV of areas from a labeled mask.

        This avoids repeated per-label full-image scans by using bincount where
        possible (integer label maps).
        """
        if labels is None or labels.size == 0:
            return 0, 0.0, 0.0

        # Fast path for integer/boolean label maps
        if labels.dtype.kind in ('b', 'u', 'i'):
            labels_int = labels.astype(np.int64, copy=False)
            max_label = int(labels_int.max()) if labels_int.size else 0
            if max_label <= 0:
                return 0, 0.0, 0.0

            counts = np.bincount(labels_int.ravel(), minlength=max_label + 1)
            areas = counts[1:]  # exclude background
            areas = areas[areas > 0]
        else:
            # Fallback for non-integer outputs
            unique_labels, counts = np.unique(labels, return_counts=True)
            valid = unique_labels > 0
            if not np.any(valid):
                return 0, 0.0, 0.0
            areas = counts[valid]

        nucleus_count = int(areas.size)
        if nucleus_count == 0:
            return 0, 0.0, 0.0

        median_area = float(np.median(areas))
        mean_area = float(np.mean(areas))
        cv_area = float(np.std(areas) / mean_area * 100.0) if mean_area > 0 else 0.0

        return nucleus_count, median_area, cv_area

    def _get_sam_mask_generator(self):
        """Get or create a cached SAM automatic mask generator for current model."""
        from segment_anything import SamAutomaticMaskGenerator

        if (
            self._sam_mask_generator is None
            or self._sam_generator_model_type != self._sam_model_type
        ):
            self._sam_mask_generator = SamAutomaticMaskGenerator(self._sam_predictor.model)
            self._sam_generator_model_type = self._sam_model_type

        return self._sam_mask_generator

    def segment_allen(self,
                      image: np.ndarray,
                      mode: str = 'auto',
                      structure_id: Optional[str] = None,
                      workflow_id: Optional[str] = None,
                      config: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Perform segmentation using Allen Cell & Structure Segmenter
        
        Uses caching to avoid reinitializing the backend for repeated calls
        with the same mode, significantly improving throughput.
        
        Args:
            image: Image array (Z, Y, X) or (C, Z, Y, X)
            mode: 'classic', 'ml', or 'auto'
            structure_id: Structure ID (e.g., 'LAMP1', 'DNA')
            workflow_id: Workflow ID (for ML models)
            config: Configuration dictionary
            
        Returns:
            tuple: (masks, info_dict)
        """
        import time
        start_time = time.time()
        
        # Use cached backend for better throughput
        backend = get_cached_allen_backend(mode)
        
        # Run segmentation
        masks = backend.segment(
            volume=image,
            structure_id=structure_id,
            workflow_id=workflow_id,
            config=config
        )
        
        end_time = time.time()
        
        # Calculate basic stats
        object_count, _, _ = self._compute_label_area_stats(masks)
        
        # Get versions
        versions = {}
        try:
            import aicssegmentation
            versions['aicssegmentation'] = aicssegmentation.__version__
        except ImportError:
            pass
            
        try:
            import aicsmlsegment
            versions['aicsmlsegment'] = aicsmlsegment.__version__
        except ImportError:
            pass
            
        try:
            import torch
            versions['torch'] = torch.__version__
        except ImportError:
            pass

        info = {
            'backend': mode,
            'structure_id': structure_id,
            'workflow_id': workflow_id,
            'object_count': object_count,
            'processing_time': end_time - start_time,
            'versions': versions
        }
        
        return masks, info

    def segment_cellpose(self,
                        image: np.ndarray,
                        model_name: str = 'nuclei',
                        diameter: Optional[float] = None,
                        flow_threshold: float = 0.4,
                        cellprob_threshold: float = 0.0,
                        do_3d: bool = False,
                        channels: List[int] = [0, 0]) -> Tuple[np.ndarray, Dict]:
        """
        Perform segmentation using Cellpose
        
        Args:
            image: Image array with shape (Z, C, Y, X), (C, Y, X), (Z, Y, X), or (Y, X)
            model_name: Cellpose model name ('nuclei', 'cyto3', etc.)
            diameter: Expected cell diameter in pixels (None for auto-detect)
            flow_threshold: Flow error threshold (0-3, higher = fewer masks)
            cellprob_threshold: Cell probability threshold (-6 to 6, higher = fewer masks)
            do_3d: Whether to use 3D segmentation
            channels: [cytoplasm_channel, nucleus_channel], use [0,0] for grayscale
        
        Returns:
            tuple: (masks, info_dict)
                masks: Labeled mask array (same spatial dims as input)
                info_dict: Dictionary with segmentation statistics
                
        Raises:
            ValueError: If parameters are invalid
            TypeError: If image type is invalid
        """
        from cellpose import models
        import time
        
        start_time = time.time()
        
        # === Input Validation ===
        
        # Validate image type and shape
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be a numpy array, got {type(image).__name__}")
        
        if image.ndim < 2 or image.ndim > 4:
            raise ValueError(
                f"Image must have 2-4 dimensions, got {image.ndim}. "
                f"Expected shapes: (Y, X), (C, Y, X), (Z, Y, X), or (Z, C, Y, X)"
            )
        
        if image.size == 0:
            raise ValueError("Image is empty (size=0)")
        
        # Validate model name
        if model_name not in self.cellpose_models:
            raise ValueError(
                f"Unknown Cellpose model: '{model_name}'. "
                f"Available models: {', '.join(self.cellpose_models)}"
            )
        
        # Validate diameter
        if diameter is not None:
            if not isinstance(diameter, (int, float)):
                raise TypeError(f"Diameter must be a number, got {type(diameter).__name__}")
            if diameter <= 0:
                raise ValueError(f"Diameter must be positive, got {diameter}")
        
        # Validate thresholds
        if not 0 <= flow_threshold <= 3:
            raise ValueError(f"flow_threshold must be between 0 and 3, got {flow_threshold}")
        
        if not -6 <= cellprob_threshold <= 6:
            raise ValueError(f"cellprob_threshold must be between -6 and 6, got {cellprob_threshold}")
        
        # Validate channels
        if not isinstance(channels, (list, tuple)) or len(channels) != 2:
            raise ValueError(f"channels must be a list/tuple of 2 integers, got {channels}")
        
        # Warn about 3D mode limitations
        if do_3d and image.ndim < 3:
            import warnings
            warnings.warn(
                "do_3d=True but image has less than 3 dimensions. "
                "2D segmentation will be performed.",
                UserWarning
            )
        
        # === Model Initialization ===
        
        # Initialize model if needed (using instance variable to track model type)
        if self._cellpose_model is None or self._cellpose_model_type != model_name:
            self._cellpose_model = models.CellposeModel(
                gpu=self.gpu_available,
                model_type=model_name
            )
            self._cellpose_model_type = model_name
        
        # Prepare image
        imgs = self._prepare_image_for_cellpose(image, do_3d=do_3d)
        
        # Run segmentation
        masks, flows, styles, diams = self._cellpose_model.eval(
            imgs,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            do_3D=do_3d,
            normalize=True
        )
        
        end_time = time.time()
        
        # Calculate statistics
        nucleus_count, median_area, cv_area = self._compute_label_area_stats(masks)
        
        info = {
            'model_name': model_name,
            'diameter': float(diams) if diameter is None else float(diameter),
            'flow_threshold': flow_threshold,
            'cellprob_threshold': cellprob_threshold,
            'do_3d': do_3d,
            'nucleus_count': nucleus_count,
            'median_area': median_area,
            'cv_area': cv_area,
            'processing_time': end_time - start_time,
        }
        
        return masks, info
    
    def segment_sam(self,
                   image: np.ndarray,
                   model_type: str = 'vit_h',
                   points: Optional[np.ndarray] = None,
                   boxes: Optional[np.ndarray] = None,
                   automatic: bool = True,
                   process_3d_per_slice: bool = False,
                   random_seed: Optional[int] = 42) -> Tuple[np.ndarray, Dict]:
        """
        Perform segmentation using SAM (Segment Anything Model)
        
        Note: SAM is inherently a 2D model. For 3D/4D images, by default only
        the middle slice is processed. Set process_3d_per_slice=True to process
        each slice independently.
        
        Args:
            image: Image array with shape (Z, C, Y, X), (C, Y, X), (Z, Y, X), or (Y, X)
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            points: Point prompts with shape (N, 2) as (x, y)
            boxes: Box prompts with shape (N, 4) as (x1, y1, x2, y2)
            automatic: If True, use automatic mask generation
            process_3d_per_slice: If True and image is 3D/4D, process each Z-slice
                                 independently. If False, only middle slice is used.
            random_seed: Random seed for reproducibility (default: 42)
        
        Returns:
            tuple: (masks, info_dict)
                masks: Labeled mask array
                info_dict: Dictionary with segmentation statistics
                
        Raises:
            ImportError: If segment-anything is not installed
            FileNotFoundError: If SAM checkpoint file is missing
            ValueError: If parameters are invalid
            TypeError: If image type is invalid
        """
        try:
            from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError("segment-anything not installed. Install with: pip install segment-anything")
        
        import time
        
        # === Input Validation ===
        
        # Validate image type and shape
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be a numpy array, got {type(image).__name__}")
        
        if image.ndim < 2 or image.ndim > 4:
            raise ValueError(
                f"Image must have 2-4 dimensions, got {image.ndim}. "
                f"Expected shapes: (Y, X), (C, Y, X), (Z, Y, X), or (Z, C, Y, X)"
            )
        
        if image.size == 0:
            raise ValueError("Image is empty (size=0)")
        
        # Validate points if provided
        if points is not None:
            if not isinstance(points, np.ndarray):
                raise TypeError(f"points must be a numpy array, got {type(points).__name__}")
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError(f"points must have shape (N, 2), got {points.shape}")
        
        # Validate boxes if provided
        if boxes is not None:
            if not isinstance(boxes, np.ndarray):
                raise TypeError(f"boxes must be a numpy array, got {type(boxes).__name__}")
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError(f"boxes must have shape (N, 4), got {boxes.shape}")
        
        # Validate automatic mode vs prompts
        if not automatic and points is None and boxes is None:
            raise ValueError(
                "When automatic=False, either points or boxes must be provided"
            )
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        start_time = time.time()
        
        # Validate model type
        if model_type not in self.sam_models:
            raise ValueError(
                f"Unknown SAM model: '{model_type}'. "
                f"Available models: {', '.join(self.sam_models)}"
            )
        
        # Load model if needed (using instance variable to track model type)
        if self._sam_predictor is None or self._sam_model_type != model_type:
            # Note: Model checkpoint path needs to be configured
            checkpoint_path = get_resource_path(f"models/sam_{model_type}.pth")
            
            # Validate checkpoint file exists before attempting to load
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"SAM model checkpoint not found at: {checkpoint_path}\n\n"
                    f"Please download the '{model_type}' checkpoint from:\n"
                    f"https://github.com/facebookresearch/segment-anything#model-checkpoints\n\n"
                    f"Save it as: {checkpoint_path}"
                )
            
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            if self.gpu_available:
                sam.to(device='cuda')
            self._sam_predictor = SamPredictor(sam)
            self._sam_model_type = model_type
            self._sam_mask_generator = None
            self._sam_generator_model_type = None
        
        # Determine if we have a 3D/4D image
        is_volumetric = image.ndim >= 3 and (
            (image.ndim == 3 and image.shape[0] > 3) or  # (Z, Y, X) with Z > 3
            image.ndim == 4  # (Z, C, Y, X)
        )
        
        if is_volumetric and process_3d_per_slice:
            # Process each Z-slice independently
            masks, info = self._segment_sam_3d_per_slice(
                image, model_type, automatic, start_time
            )
        else:
            # Standard 2D processing (or middle slice of 3D)
            masks, info = self._segment_sam_2d(
                image, model_type, points, boxes, automatic, start_time, is_volumetric
            )
        
        return masks, info
    
    def _segment_sam_2d(self,
                        image: np.ndarray,
                        model_type: str,
                        points: Optional[np.ndarray],
                        boxes: Optional[np.ndarray],
                        automatic: bool,
                        start_time: float,
                        is_volumetric: bool) -> Tuple[np.ndarray, Dict]:
        """Helper method for 2D SAM segmentation"""
        import time
        
        # Prepare image (SAM expects RGB)
        img_rgb = self._prepare_image_for_sam(image)
        
        self._sam_predictor.set_image(img_rgb)
        
        if automatic:
            # Automatic mask generation
            masks_list = self._get_sam_mask_generator().generate(img_rgb)
            
            # Convert to labeled mask
            masks = self._combine_sam_masks(masks_list, img_rgb.shape[:2])
        else:
            # Prompt-based segmentation
            if points is None and boxes is None:
                raise ValueError("Either points or boxes must be provided for non-automatic segmentation")
            
            masks, scores, logits = self._sam_predictor.predict(
                point_coords=points,
                point_labels=np.ones(len(points)) if points is not None else None,
                box=boxes,
                multimask_output=True
            )
            
            # Use best mask
            masks = masks[np.argmax(scores)]
        
        end_time = time.time()
        
        # Calculate statistics
        nucleus_count, median_area, cv_area = self._compute_label_area_stats(masks)
        
        # Add warning if volumetric data was processed as single slice
        note = None
        if is_volumetric:
            note = ("SAM is a 2D model. Only the middle Z-slice was processed. "
                   "Set process_3d_per_slice=True to segment each slice independently.")
        
        info = {
            'model_name': f'SAM_{model_type}',
            'automatic': automatic,
            'nucleus_count': nucleus_count,
            'median_area': median_area,
            'cv_area': cv_area,
            'processing_time': end_time - start_time,
        }
        
        if note:
            info['note'] = note
        
        return masks, info
    
    def _segment_sam_3d_per_slice(self,
                                   image: np.ndarray,
                                   model_type: str,
                                   automatic: bool,
                                   start_time: float) -> Tuple[np.ndarray, Dict]:
        """
        Process 3D/4D image with SAM slice-by-slice
        
        Args:
            image: 3D (Z, Y, X) or 4D (Z, C, Y, X) image array
            model_type: SAM model type
            automatic: Whether to use automatic mask generation
            start_time: Start time for timing
            
        Returns:
            tuple: (3D masks, info_dict)
        """
        import time
        
        # Determine Z dimension and prepare slices
        if image.ndim == 4:
            # (Z, C, Y, X) - take first channel for each slice
            n_slices = image.shape[0]
            output_shape = (n_slices, image.shape[2], image.shape[3])
        else:
            # (Z, Y, X)
            n_slices = image.shape[0]
            output_shape = image.shape
        
        # Initialize 3D mask array
        masks_3d = np.zeros(output_shape, dtype=np.int32)
        
        # Track statistics across slices
        all_areas = []
        total_nuclei = 0
        current_max_label = 0

        mask_generator = self._get_sam_mask_generator() if automatic else None
        
        for z_idx in range(n_slices):
            if image.ndim == 4:
                slice_2d = image[z_idx, 0]
            else:
                slice_2d = image[z_idx]

            # Prepare slice for SAM
            img_rgb = self._prepare_image_for_sam(slice_2d)
            
            self._sam_predictor.set_image(img_rgb)
            
            if automatic:
                masks_list = mask_generator.generate(img_rgb)
                slice_masks = self._combine_sam_masks(masks_list, img_rgb.shape[:2])
            else:
                # For per-slice automatic processing, we don't support prompts
                slice_masks = np.zeros(img_rgb.shape[:2], dtype=np.int32)
            
            # Relabel to ensure unique labels across slices
            if slice_masks.max() > 0:
                # Offset labels to be unique across all slices
                slice_masks_offset = slice_masks.astype(np.int32, copy=True)
                positive = slice_masks_offset > 0
                slice_masks_offset[positive] += current_max_label
                current_max_label = int(slice_masks_offset.max())
                masks_3d[z_idx] = slice_masks_offset

                # Collect area stats with vectorized counts
                counts = np.bincount(slice_masks.ravel())
                slice_areas = counts[1:]
                slice_areas = slice_areas[slice_areas > 0]
                if slice_areas.size > 0:
                    all_areas.extend(slice_areas.tolist())
                    total_nuclei += int(slice_areas.size)
        
        end_time = time.time()
        
        # Calculate statistics
        if all_areas:
            areas_arr = np.asarray(all_areas, dtype=np.float64)
            median_area = float(np.median(areas_arr))
            mean_area = float(np.mean(areas_arr))
            cv_area = float(np.std(areas_arr) / mean_area * 100.0) if mean_area > 0 else 0.0
        else:
            median_area = 0.0
            cv_area = 0.0
        
        info = {
            'model_name': f'SAM_{model_type}',
            'automatic': automatic,
            'nucleus_count': total_nuclei,
            'median_area': median_area,
            'cv_area': cv_area,
            'processing_time': end_time - start_time,
            'n_slices_processed': n_slices,
            'note': f'Processed {n_slices} slices independently. Labels are unique per slice.'
        }
        
        return masks_3d, info
    
    def _prepare_image_for_cellpose(self, image: np.ndarray, do_3d: bool = False) -> np.ndarray:
        """Prepare image for Cellpose (expects channels-last or grayscale)"""
        if image.ndim == 2:
            # Single channel 2D: (Y, X)
            return image
        elif image.ndim == 3:
            # Could be (C, Y, X) or (Z, Y, X)
            # If C <= 3, treat as channels and transpose to (Y, X, C)
            if image.shape[0] <= 3:
                return np.transpose(image, (1, 2, 0))
            else:
                # Likely Z-stack, take middle slice
                return image[image.shape[0] // 2]
        elif image.ndim == 4:
            # (Z, C, Y, X)
            if do_3d:
                # Cellpose expects (Z, Y, X) for single channel 3D
                # or (Z, C, Y, X) / (Z, Y, X, C) depending on channel settings
                # Standardize to (Z, Y, X, C)
                return np.transpose(image, (0, 2, 3, 1))
            else:
                # Take middle slice for 2D preview/segmentation
                z_mid = image.shape[0] // 2
                slice_img = image[z_mid]  # (C, Y, X)
                if slice_img.shape[0] <= 3:
                    return np.transpose(slice_img, (1, 2, 0))
                else:
                    return slice_img[0]  # Take first channel
        
        return image
    
    def _prepare_image_for_sam(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for SAM (expects RGB uint8)"""
        if image.ndim == 2:
            # Grayscale to RGB
            img_rgb = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3:
            if image.shape[0] <= 3:
                # (C, Y, X) -> (Y, X, C)
                img_rgb = np.transpose(image, (1, 2, 0))
                if img_rgb.shape[-1] == 1:
                    img_rgb = np.repeat(img_rgb, 3, axis=-1)
                elif img_rgb.shape[-1] == 2:
                    # Pad to 3 channels
                    img_rgb = np.concatenate([img_rgb, img_rgb[:, :, :1]], axis=-1)
            else:
                # Z-stack, take middle
                img_rgb = np.stack([image[image.shape[0]//2]] * 3, axis=-1)
        else:
            img_rgb = image
        
        # Normalize to uint8
        if img_rgb.dtype != np.uint8:
            img_min = img_rgb.min()
            img_max = img_rgb.max()
            if img_max > img_min:
                img_rgb = ((img_rgb - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_rgb = np.zeros_like(img_rgb, dtype=np.uint8)
        
        return img_rgb
    
    def _combine_sam_masks(self, masks_list: List[Dict], shape: Tuple[int, int]) -> np.ndarray:
        """Combine SAM automatic masks into labeled mask array"""
        labeled_mask = np.zeros(shape, dtype=np.int32)
        
        # Sort by area (largest first) to handle overlaps
        masks_list = sorted(masks_list, key=lambda x: x['area'], reverse=True)
        
        label = 1
        for mask_dict in masks_list:
            mask = mask_dict['segmentation']
            # Only assign to unlabeled pixels
            labeled_mask[mask & (labeled_mask == 0)] = label
            label += 1
        
        return labeled_mask
    
    def get_available_models(self, engine: str = 'cellpose') -> List[str]:
        """
        Get list of available models
        
        Args:
            engine: 'cellpose' or 'sam'
        
        Returns:
            List of model names
        """
        if engine == 'cellpose':
            return self.cellpose_models
        elif engine == 'sam':
            return self.sam_models
        else:
            return []
    
    def estimate_diameter(self, image: np.ndarray, model_name: str = 'nuclei') -> float:
        """
        Estimate cell diameter using a quick segmentation pass.
        
        In Cellpose 4.0+, the SizeModel (sz) was removed. This method runs
        a quick segmentation with a default diameter and then calculates
        the median diameter from the resulting masks.
        
        Args:
            image: Image array
            model_name: Model type hint ('nuclei' or 'cyto') for default diameter
        
        Returns:
            Estimated diameter in pixels
        """
        from cellpose import models, utils
        
        # Default diameters based on model type (from Cellpose defaults)
        default_diameters = {
            'nuclei': 17.0,
            'cyto': 30.0,
            'cyto2': 30.0,
            'cyto3': 30.0,
            'cyto_sam': 30.0
        }
        
        # Get default diameter for initial pass
        initial_diameter = default_diameters.get(model_name, 30.0)
        
        # Ensure model is loaded
        if self._cellpose_model is None:
            self._cellpose_model = models.CellposeModel(gpu=self.gpu_available)
        
        imgs = self._prepare_image_for_cellpose(image)
        
        try:
            # Run a quick segmentation with default diameter
            masks, flows, styles = self._cellpose_model.eval(
                imgs,
                diameter=initial_diameter,
                flow_threshold=0.4,
                cellprob_threshold=0.0
            )
            
            # Calculate diameter from masks if any cells were found
            if masks is not None and np.any(masks > 0):
                median_diam, diams = utils.diameters(masks)
                if median_diam > 0:
                    return float(median_diam)
            
            # Fall back to default if no cells found
            return initial_diameter
            
        except Exception as e:
            # If estimation fails, return a reasonable default
            print(f"Diameter estimation warning: {e}, using default {initial_diameter}")
            return initial_diameter
