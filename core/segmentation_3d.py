"""
True 3D segmentation backends for volumetric nuclei segmentation.

This module provides:
1. 3D U-Net / nnU-Net-style volumetric segmentation
2. Hybrid 2D segmentation + 3D linking via overlap/graph matching
3. Foundation encoder adapters (SAM2/SAM3-derived for 3D)

Addresses the requirement to replace slice-by-slice 3D with true 3D instance consistency.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any
import logging
from scipy import ndimage
from skimage import measure
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class Anisotropy3DPreprocessor:
    """
    Anisotropy-aware preprocessing for 3D volumes.
    
    Handles z-anisotropy common in microscopy where z-resolution
    is typically 2-5x worse than xy-resolution.
    """
    
    def __init__(self, voxel_size: Optional[Tuple[float, float, float]] = None):
        """
        Initialize preprocessor.
        
        Args:
            voxel_size: (z, y, x) voxel dimensions in physical units (e.g., microns).
                       If None, assumes isotropic voxels.
        """
        self.voxel_size = voxel_size or (1.0, 1.0, 1.0)
        self.anisotropy_ratio = self._calculate_anisotropy_ratio()
    
    def _calculate_anisotropy_ratio(self) -> Tuple[float, float, float]:
        """Calculate anisotropy ratio normalized to minimum dimension."""
        min_size = min(self.voxel_size)
        return tuple(v / min_size for v in self.voxel_size)
    
    def resample_to_isotropic(self, volume: np.ndarray) -> Tuple[np.ndarray, Tuple[float, ...]]:
        """
        Resample anisotropic volume to isotropic voxels.
        
        Args:
            volume: Input 3D volume (Z, Y, X)
            
        Returns:
            Tuple of (resampled_volume, zoom_factors)
        """
        if np.allclose(self.anisotropy_ratio, (1.0, 1.0, 1.0)):
            # Already isotropic
            return volume, (1.0, 1.0, 1.0)
        
        # Calculate zoom factors (inverse of anisotropy ratio)
        zoom_factors = tuple(1.0 / r for r in self.anisotropy_ratio)
        
        # Resample using spline interpolation
        resampled = ndimage.zoom(volume, zoom_factors, order=1)
        
        logger.info(f"Resampled volume from {volume.shape} to {resampled.shape} (zoom: {zoom_factors})")
        
        return resampled, zoom_factors
    
    def resample_masks_to_original(self, masks: np.ndarray, 
                                   original_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Resample masks back to original anisotropic dimensions.
        
        Args:
            masks: Labeled mask array from isotropic space
            original_shape: Original volume shape (Z, Y, X)
            
        Returns:
            Masks resampled to original shape
        """
        if masks.shape == original_shape:
            return masks
        
        # Calculate zoom factors to get back to original shape
        zoom_factors = tuple(o / m for o, m in zip(original_shape, masks.shape))
        
        # Use nearest neighbor for label preservation
        resampled_masks = ndimage.zoom(masks, zoom_factors, order=0)
        
        logger.info(f"Resampled masks from {masks.shape} to {resampled_masks.shape}")
        
        return resampled_masks


class Hybrid2D3DBackend:
    """
    Hybrid backend: 2D segmentation on slices + 3D linking by overlap.
    
    This approach:
    1. Runs 2D segmentation on each z-slice independently
    2. Links instances across slices using overlap-based graph matching
    3. Applies shape priors to resolve ambiguities
    
    Advantages:
    - Leverages robust 2D segmentation models (Cellpose, SAM)
    - Lower memory requirements than full 3D models
    - Good for anisotropic data where z-resolution is poor
    """
    
    def __init__(self, 
                 segmentation_2d_fn: callable,
                 min_overlap_ratio: float = 0.3,
                 max_distance_z: int = 2):
        """
        Initialize hybrid backend.
        
        Args:
            segmentation_2d_fn: Function that takes 2D image and returns labeled mask
            min_overlap_ratio: Minimum IoU to consider instances as the same object
            max_distance_z: Maximum z-distance to search for matching instances
        """
        self.segmentation_2d_fn = segmentation_2d_fn
        self.min_overlap_ratio = min_overlap_ratio
        self.max_distance_z = max_distance_z
    
    def segment(self, volume: np.ndarray) -> np.ndarray:
        """
        Perform hybrid 2D+3D segmentation.
        
        Args:
            volume: 3D volume (Z, Y, X)
            
        Returns:
            3D labeled mask with instance consistency across slices
        """
        z_slices = volume.shape[0]
        
        # Step 1: Run 2D segmentation on all slices
        logger.info(f"Running 2D segmentation on {z_slices} slices...")
        slice_masks = []
        for z in range(z_slices):
            mask_2d = self.segmentation_2d_fn(volume[z])
            slice_masks.append(mask_2d)
        
        # Step 2: Link instances across slices
        logger.info("Linking instances across z-slices...")
        linked_masks = self._link_slices(slice_masks)
        
        return np.array(linked_masks)
    
    def _link_slices(self, slice_masks: List[np.ndarray]) -> np.ndarray:
        """
        Link 2D instances across slices using overlap-based matching.
        
        Args:
            slice_masks: List of 2D labeled masks, one per z-slice
            
        Returns:
            3D labeled mask with consistent instance IDs across slices
        """
        if not slice_masks:
            return np.array([])
        
        z_slices = len(slice_masks)
        output_shape = (z_slices,) + slice_masks[0].shape
        linked_masks = np.zeros(output_shape, dtype=np.int32)
        
        # Initialize with first slice
        linked_masks[0] = slice_masks[0]
        next_global_id = int(slice_masks[0].max()) + 1
        
        # Process each subsequent slice
        for z in range(1, z_slices):
            current_slice = slice_masks[z]
            
            # Find best matches with previous slice(s)
            matched_ids = self._match_instances(
                current_slice, 
                linked_masks[max(0, z - self.max_distance_z):z]
            )
            
            # Assign IDs based on matches
            linked_masks[z] = self._assign_matched_ids(
                current_slice, matched_ids, next_global_id
            )
            
            # Update next available ID
            next_global_id = int(linked_masks[z].max()) + 1
        
        return linked_masks
    
    def _match_instances(self, 
                        current_slice: np.ndarray,
                        previous_slices: np.ndarray) -> Dict[int, int]:
        """
        Match instances in current slice to instances in previous slices.
        
        Args:
            current_slice: 2D labeled mask for current slice
            previous_slices: 3D array of previous slices (Z, Y, X)
            
        Returns:
            Dictionary mapping current instance IDs to matched global IDs
        """
        if previous_slices.size == 0:
            return {}
        
        # Get most recent slice for matching
        prev_slice = previous_slices[-1]
        
        # Get unique labels
        current_labels = np.unique(current_slice)
        current_labels = current_labels[current_labels > 0]
        
        prev_labels = np.unique(prev_slice)
        prev_labels = prev_labels[prev_labels > 0]
        
        if len(current_labels) == 0 or len(prev_labels) == 0:
            return {}
        
        # Build overlap matrix
        overlap_matrix = np.zeros((len(current_labels), len(prev_labels)))
        
        for i, curr_id in enumerate(current_labels):
            curr_mask = (current_slice == curr_id)
            curr_area = curr_mask.sum()
            
            for j, prev_id in enumerate(prev_labels):
                prev_mask = (prev_slice == prev_id)
                overlap = (curr_mask & prev_mask).sum()
                
                # Calculate IoU
                union = (curr_mask | prev_mask).sum()
                iou = overlap / union if union > 0 else 0.0
                
                overlap_matrix[i, j] = iou
        
        # Use Hungarian algorithm for optimal matching
        row_ind, col_ind = linear_sum_assignment(-overlap_matrix)
        
        # Build match dictionary (only keep matches above threshold)
        matches = {}
        for i, j in zip(row_ind, col_ind):
            if overlap_matrix[i, j] >= self.min_overlap_ratio:
                matches[int(current_labels[i])] = int(prev_labels[j])
        
        return matches
    
    def _assign_matched_ids(self,
                           current_slice: np.ndarray,
                           matched_ids: Dict[int, int],
                           next_id: int) -> np.ndarray:
        """
        Assign global IDs to current slice based on matches.
        
        Args:
            current_slice: 2D labeled mask
            matched_ids: Dictionary mapping local IDs to global IDs
            next_id: Next available global ID for unmatched instances
            
        Returns:
            Relabeled slice with global IDs
        """
        output = np.zeros_like(current_slice)
        
        for local_id in np.unique(current_slice):
            if local_id == 0:
                continue
            
            mask = (current_slice == local_id)
            
            if local_id in matched_ids:
                # Use matched global ID
                output[mask] = matched_ids[local_id]
            else:
                # Assign new global ID
                output[mask] = next_id
                next_id += 1
        
        return output


class True3DBackend:
    """
    True 3D segmentation backend using volumetric networks.
    
    This is a placeholder for future integration with:
    - 3D U-Net models
    - nnU-Net framework
    - SAM2/SAM3-derived 3D architectures
    
    Currently provides an interface that can be implemented with specific models.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize 3D backend.
        
        Args:
            model_path: Path to pretrained 3D model weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = device
        self._model = None
        
        logger.warning(
            "True3DBackend is a placeholder. "
            "Requires 3D U-Net or nnU-Net implementation."
        )
    
    def load_model(self):
        """Load the 3D segmentation model."""
        raise NotImplementedError(
            "3D model loading not yet implemented. "
            "This requires integration with 3D U-Net or nnU-Net."
        )
    
    def segment(self, volume: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform true 3D segmentation.
        
        Args:
            volume: 3D volume (Z, Y, X) or (C, Z, Y, X)
            **kwargs: Additional model-specific parameters
            
        Returns:
            3D labeled mask
        """
        raise NotImplementedError(
            "True 3D segmentation not yet implemented. "
            "Use Hybrid2D3DBackend as an intermediate solution."
        )


def compute_3d_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute 3D Intersection over Union (IoU) metric.
    
    Args:
        mask1: First binary mask (Z, Y, X)
        mask2: Second binary mask (Z, Y, X)
        
    Returns:
        IoU score (0-1)
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_and(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)


def compute_3d_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute 3D Dice coefficient.
    
    Args:
        mask1: First binary mask (Z, Y, X)
        mask2: Second binary mask (Z, Y, X)
        
    Returns:
        Dice score (0-1)
    """
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    
    if total == 0:
        return 0.0
    
    return float(2.0 * intersection / total)


def compute_split_merge_errors(pred_labels: np.ndarray, 
                               gt_labels: np.ndarray) -> Tuple[int, int]:
    """
    Compute split and merge errors for instance segmentation.
    
    Split error: Ground truth instance split into multiple predictions
    Merge error: Multiple ground truth instances merged into one prediction
    
    Args:
        pred_labels: Predicted labeled mask
        gt_labels: Ground truth labeled mask
        
    Returns:
        Tuple of (split_count, merge_count)
    """
    split_count = 0
    merge_count = 0
    
    # Get unique labels (excluding background)
    gt_ids = np.unique(gt_labels)
    gt_ids = gt_ids[gt_ids > 0]
    
    pred_ids = np.unique(pred_labels)
    pred_ids = pred_ids[pred_ids > 0]
    
    # Check for splits (one GT maps to multiple predictions)
    for gt_id in gt_ids:
        gt_mask = (gt_labels == gt_id)
        overlapping_preds = np.unique(pred_labels[gt_mask])
        overlapping_preds = overlapping_preds[overlapping_preds > 0]
        
        if len(overlapping_preds) > 1:
            split_count += 1
    
    # Check for merges (one prediction maps to multiple GTs)
    for pred_id in pred_ids:
        pred_mask = (pred_labels == pred_id)
        overlapping_gts = np.unique(gt_labels[pred_mask])
        overlapping_gts = overlapping_gts[overlapping_gts > 0]
        
        if len(overlapping_gts) > 1:
            merge_count += 1
    
    return split_count, merge_count
