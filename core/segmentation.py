"""
Core segmentation module wrapping Cellpose and SAM APIs
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import torch


class SegmentationEngine:
    """
    Wrapper for Cellpose and SAM segmentation models
    """
    
    def __init__(self, gpu_available: bool = False):
        self.gpu_available = gpu_available
        self.device = 'cuda' if gpu_available else 'cpu'
        
        # Model instances (lazy loaded)
        self._cellpose_model = None
        self._sam_predictor = None
        
        # Available models
        self.cellpose_models = ['nuclei', 'cyto', 'cyto2', 'cyto3', 'cyto_sam']
        self.sam_models = ['vit_h', 'vit_l', 'vit_b']
    
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
        """
        from cellpose import models
        import time
        
        start_time = time.time()
        
        # Initialize model if needed
        if self._cellpose_model is None or self._cellpose_model.pretrained_model != model_name:
            self._cellpose_model = models.CellposeModel(
                gpu=self.gpu_available,
                model_type=model_name
            )
        
        # Prepare image
        imgs = self._prepare_image_for_cellpose(image)
        
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
        unique_labels = np.unique(masks)
        nucleus_count = len(unique_labels) - 1  # Exclude background
        
        if nucleus_count > 0:
            areas = []
            for label in unique_labels:
                if label == 0:
                    continue
                areas.append(np.sum(masks == label))
            
            median_area = float(np.median(areas))
            cv_area = float(np.std(areas) / np.mean(areas) * 100) if np.mean(areas) > 0 else 0.0
        else:
            median_area = 0.0
            cv_area = 0.0
        
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
                   automatic: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Perform segmentation using SAM (Segment Anything Model)
        
        Args:
            image: Image array with shape (C, Y, X) or (Y, X)
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            points: Point prompts with shape (N, 2) as (x, y)
            boxes: Box prompts with shape (N, 4) as (x1, y1, x2, y2)
            automatic: If True, use automatic mask generation
        
        Returns:
            tuple: (masks, info_dict)
                masks: Labeled mask array
                info_dict: Dictionary with segmentation statistics
        """
        try:
            from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError("segment-anything not installed. Install with: pip install segment-anything")
        
        import time
        
        start_time = time.time()
        
        # Load model if needed
        if self._sam_predictor is None:
            # Note: Model checkpoint path needs to be configured
            checkpoint_path = f"models/sam_{model_type}.pth"
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            if self.gpu_available:
                sam.to(device='cuda')
            self._sam_predictor = SamPredictor(sam)
        
        # Prepare image (SAM expects RGB)
        img_rgb = self._prepare_image_for_sam(image)
        
        self._sam_predictor.set_image(img_rgb)
        
        if automatic:
            # Automatic mask generation
            mask_generator = SamAutomaticMaskGenerator(self._sam_predictor.model)
            masks_list = mask_generator.generate(img_rgb)
            
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
        unique_labels = np.unique(masks)
        nucleus_count = len(unique_labels) - 1
        
        if nucleus_count > 0:
            areas = []
            for label in unique_labels:
                if label == 0:
                    continue
                areas.append(np.sum(masks == label))
            
            median_area = float(np.median(areas))
            cv_area = float(np.std(areas) / np.mean(areas) * 100) if np.mean(areas) > 0 else 0.0
        else:
            median_area = 0.0
            cv_area = 0.0
        
        info = {
            'model_name': f'SAM_{model_type}',
            'automatic': automatic,
            'nucleus_count': nucleus_count,
            'median_area': median_area,
            'cv_area': cv_area,
            'processing_time': end_time - start_time,
        }
        
        return masks, info
    
    def _prepare_image_for_cellpose(self, image: np.ndarray) -> np.ndarray:
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
            # (Z, C, Y, X) - take middle Z slice and transpose C
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
    
    def estimate_diameter(self, image: np.ndarray) -> float:
        """
        Estimate cell diameter using Cellpose's size estimation
        
        Args:
            image: Image array
        
        Returns:
            Estimated diameter in pixels
        """
        from cellpose import models
        
        if self._cellpose_model is None:
            self._cellpose_model = models.CellposeModel(
                gpu=self.gpu_available,
                model_type='nuclei'
            )
        
        imgs = self._prepare_image_for_cellpose(image)
        
        # Use Cellpose's size estimation
        diameter, _ = self._cellpose_model.sz.eval(imgs)
        
        return float(diameter)
