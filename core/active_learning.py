"""
Active learning loop for continuous model improvement.

This module implements:
1. Uncertainty estimation for selecting nuclei to correct
2. Correction tracking system
3. Model adapter fine-tuning infrastructure
4. Versioning and reproducibility tracking

The active learning loop continuously selects "uncertain" nuclei for manual correction
and feeds corrected masks back into a small adapter fine-tune.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import time
from pathlib import Path
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class UncertaintyEstimator:
    """
    Estimate uncertainty in segmentation predictions.
    
    Uses multiple signals to identify nuclei that may benefit from manual correction:
    - Flow field uncertainty (from Cellpose)
    - Boundary confidence
    - Shape irregularity
    - Size outliers
    """
    
    @staticmethod
    def estimate_cellpose_uncertainty(masks: np.ndarray,
                                     flows: Any,
                                     cellprob: Optional[np.ndarray] = None) -> Dict[int, float]:
        """
        Estimate uncertainty for each nucleus from Cellpose outputs.
        
        Args:
            masks: Labeled mask array
            flows: Flow field from Cellpose (list of arrays)
            cellprob: Cell probability map
            
        Returns:
            Dictionary mapping nucleus ID to uncertainty score (0-1, higher = more uncertain)
        """
        uncertainties = {}
        
        # Get unique nucleus IDs
        nucleus_ids = np.unique(masks)
        nucleus_ids = nucleus_ids[nucleus_ids > 0]
        
        for nuc_id in nucleus_ids:
            nuc_mask = (masks == nuc_id)
            uncertainty_scores = []
            
            # 1. Flow field uncertainty
            if flows is not None and len(flows) > 0:
                flow_magnitude = flows[0]
                if isinstance(flow_magnitude, np.ndarray):
                    # Get flow magnitude at nucleus boundary
                    from scipy import ndimage
                    eroded = ndimage.binary_erosion(nuc_mask)
                    boundary = nuc_mask & ~eroded
                    
                    if boundary.sum() > 0:
                        boundary_flow = np.abs(flow_magnitude[boundary])
                        # Low flow magnitude at boundary = uncertain
                        mean_boundary_flow = np.mean(boundary_flow)
                        flow_uncertainty = 1.0 - min(mean_boundary_flow, 1.0)
                        uncertainty_scores.append(flow_uncertainty)
            
            # 2. Cell probability uncertainty
            if cellprob is not None:
                nuc_prob = cellprob[nuc_mask]
                if len(nuc_prob) > 0:
                    mean_prob = np.mean(nuc_prob)
                    # Low probability = uncertain
                    prob_uncertainty = 1.0 - mean_prob
                    uncertainty_scores.append(prob_uncertainty)
            
            # 3. Shape irregularity
            shape_uncertainty = UncertaintyEstimator._compute_shape_irregularity(nuc_mask)
            uncertainty_scores.append(shape_uncertainty)
            
            # Combine uncertainties (average)
            if uncertainty_scores:
                uncertainties[int(nuc_id)] = float(np.mean(uncertainty_scores))
            else:
                uncertainties[int(nuc_id)] = 0.5  # Default medium uncertainty
        
        return uncertainties
    
    @staticmethod
    def _compute_shape_irregularity(mask: np.ndarray) -> float:
        """
        Compute shape irregularity score.
        
        Args:
            mask: Binary mask of single nucleus
            
        Returns:
            Irregularity score (0-1, higher = more irregular)
        """
        from skimage import measure
        
        try:
            # Get region properties
            props = measure.regionprops(mask.astype(int))[0]
            
            # Solidity: area / convex area (lower = more irregular)
            solidity = props.solidity
            irregularity = 1.0 - solidity
            
            # Also consider eccentricity
            eccentricity = props.eccentricity
            
            # Combine (weight solidity more)
            score = 0.7 * irregularity + 0.3 * eccentricity
            
            return float(np.clip(score, 0, 1))
        
        except (IndexError, ValueError):
            return 0.5  # Default medium score if measurement fails
    
    @staticmethod
    def select_top_uncertain(uncertainties: Dict[int, float],
                           n: int = 10) -> List[int]:
        """
        Select top N most uncertain nuclei.
        
        Args:
            uncertainties: Dictionary mapping nucleus ID to uncertainty score
            n: Number of nuclei to select
            
        Returns:
            List of nucleus IDs sorted by uncertainty (descending)
        """
        sorted_items = sorted(uncertainties.items(), key=lambda x: x[1], reverse=True)
        return [nuc_id for nuc_id, _ in sorted_items[:n]]


class CorrectionTracker:
    """
    Track manual corrections for active learning.
    
    Stores:
    - Original predictions
    - User corrections
    - Correction metadata (timestamp, user, image context)
    - Quality metrics
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize correction tracker.
        
        Args:
            storage_path: Path to store correction data (JSON format)
        """
        self.storage_path = storage_path or Path("./corrections")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.corrections: List[Dict[str, Any]] = []
        self._load_corrections()
    
    def _load_corrections(self):
        """Load existing corrections from disk."""
        corrections_file = self.storage_path / "corrections.json"
        if corrections_file.exists():
            try:
                with open(corrections_file, 'r') as f:
                    self.corrections = json.load(f)
                logger.info(f"Loaded {len(self.corrections)} corrections from {corrections_file}")
            except Exception as e:
                logger.error(f"Failed to load corrections: {e}")
                self.corrections = []
    
    def _save_corrections(self):
        """Save corrections to disk."""
        corrections_file = self.storage_path / "corrections.json"
        try:
            with open(corrections_file, 'w') as f:
                json.dump(self.corrections, f, indent=2)
            logger.info(f"Saved {len(self.corrections)} corrections to {corrections_file}")
        except Exception as e:
            logger.error(f"Failed to save corrections: {e}")
    
    def add_correction(self,
                      image_id: str,
                      nucleus_id: int,
                      original_mask: np.ndarray,
                      corrected_mask: np.ndarray,
                      correction_type: str,
                      metadata: Optional[Dict[str, Any]] = None):
        """
        Record a manual correction.
        
        Args:
            image_id: Unique identifier for the image
            nucleus_id: ID of the corrected nucleus
            original_mask: Original segmentation mask
            corrected_mask: User-corrected mask
            correction_type: Type of correction ('split', 'merge', 'delete', 'add', 'modify')
            metadata: Additional metadata (user, timestamp, etc.)
        """
        # Create unique correction ID
        correction_id = self._generate_correction_id(image_id, nucleus_id)
        
        # Calculate correction metrics
        metrics = self._compute_correction_metrics(original_mask, corrected_mask)
        
        # Save masks to disk
        mask_path = self.storage_path / f"{correction_id}_masks.npz"
        np.savez_compressed(
            mask_path,
            original=original_mask,
            corrected=corrected_mask
        )
        
        # Build correction record
        correction_record = {
            'correction_id': correction_id,
            'image_id': image_id,
            'nucleus_id': int(nucleus_id),
            'correction_type': correction_type,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'mask_path': str(mask_path),
            'metadata': metadata or {}
        }
        
        self.corrections.append(correction_record)
        self._save_corrections()
        
        logger.info(f"Recorded correction {correction_id}: {correction_type}")
    
    def _generate_correction_id(self, image_id: str, nucleus_id: int) -> str:
        """Generate unique correction ID."""
        timestamp = int(time.time() * 1000)
        content = f"{image_id}_{nucleus_id}_{timestamp}"
        hash_digest = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"corr_{hash_digest}"
    
    def _compute_correction_metrics(self,
                                   original: np.ndarray,
                                   corrected: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics comparing original and corrected masks.
        
        Args:
            original: Original mask
            corrected: Corrected mask
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # IoU
        intersection = np.logical_and(original, corrected).sum()
        union = np.logical_or(original, corrected).sum()
        metrics['iou'] = float(intersection / union) if union > 0 else 0.0
        
        # Dice
        total = original.sum() + corrected.sum()
        metrics['dice'] = float(2.0 * intersection / total) if total > 0 else 0.0
        
        # Area change
        original_area = original.sum()
        corrected_area = corrected.sum()
        if original_area > 0:
            metrics['area_change_ratio'] = float(corrected_area / original_area)
        else:
            metrics['area_change_ratio'] = 0.0
        
        return metrics
    
    def get_corrections_for_training(self,
                                    min_corrections: int = 10) -> List[Dict[str, Any]]:
        """
        Get corrections suitable for model fine-tuning.
        
        Args:
            min_corrections: Minimum number of corrections needed
            
        Returns:
            List of correction records
        """
        if len(self.corrections) < min_corrections:
            logger.warning(
                f"Only {len(self.corrections)} corrections available, "
                f"need at least {min_corrections} for training"
            )
            return []
        
        return self.corrections


class ModelAdapter:
    """
    Model adapter for fine-tuning with user corrections.
    
    This is a lightweight adapter that can be fine-tuned on small amounts
    of correction data without requiring full model retraining.
    
    Currently a placeholder for future implementation with:
    - LoRA-style adapters
    - Small correction-specific layers
    - Transfer learning from base models
    """
    
    def __init__(self, base_model_name: str = 'nuclei', device: str = 'cuda'):
        """
        Initialize model adapter.
        
        Args:
            base_model_name: Name of base segmentation model
            device: Device for training ('cuda' or 'cpu')
        """
        self.base_model_name = base_model_name
        self.device = device
        self.adapter_weights = None
        self.version = 1
        
        logger.warning(
            "ModelAdapter is a placeholder. "
            "Full implementation requires adapter architecture definition."
        )
    
    def fine_tune(self,
                 corrections: List[Dict[str, Any]],
                 epochs: int = 10,
                 batch_size: int = 4) -> Dict[str, Any]:
        """
        Fine-tune adapter on correction data.
        
        Args:
            corrections: List of correction records
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Fine-tuning adapter on {len(corrections)} corrections...")
        
        # TODO: Implement actual fine-tuning
        # This would involve:
        # 1. Load base model
        # 2. Add adapter layers
        # 3. Prepare training data from corrections
        # 4. Run training loop
        # 5. Save adapter weights
        
        raise NotImplementedError(
            "Model adapter fine-tuning not yet implemented. "
            "This requires defining adapter architecture and training loop."
        )
    
    def save_adapter(self, path: Path):
        """Save adapter weights."""
        raise NotImplementedError("Adapter saving not yet implemented")
    
    def load_adapter(self, path: Path):
        """Load adapter weights."""
        raise NotImplementedError("Adapter loading not yet implemented")


class ActiveLearningManager:
    """
    Manage the active learning workflow.
    
    Coordinates:
    - Uncertainty estimation
    - Correction tracking
    - Model fine-tuning
    - Version management
    """
    
    def __init__(self,
                 storage_path: Optional[Path] = None,
                 min_corrections_for_training: int = 20):
        """
        Initialize active learning manager.
        
        Args:
            storage_path: Path to store active learning data
            min_corrections_for_training: Minimum corrections needed before fine-tuning
        """
        self.storage_path = storage_path or Path("./active_learning")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.min_corrections_for_training = min_corrections_for_training
        
        # Initialize components
        self.uncertainty_estimator = UncertaintyEstimator()
        self.correction_tracker = CorrectionTracker(self.storage_path / "corrections")
        self.model_adapter = None  # Lazy initialization
        
        # Training schedule
        self.last_training_time: Optional[datetime] = None
        self.training_scheduled = False
    
    def identify_uncertain_nuclei(self,
                                 masks: np.ndarray,
                                 flows: Any,
                                 cellprob: Optional[np.ndarray] = None,
                                 top_n: int = 10) -> List[int]:
        """
        Identify most uncertain nuclei for manual review.
        
        Args:
            masks: Segmentation masks
            flows: Flow field from segmentation
            cellprob: Cell probability map
            top_n: Number of nuclei to select
            
        Returns:
            List of nucleus IDs ranked by uncertainty
        """
        uncertainties = self.uncertainty_estimator.estimate_cellpose_uncertainty(
            masks, flows, cellprob
        )
        
        uncertain_nuclei = self.uncertainty_estimator.select_top_uncertain(
            uncertainties, n=top_n
        )
        
        logger.info(f"Identified {len(uncertain_nuclei)} uncertain nuclei for review")
        
        return uncertain_nuclei
    
    def record_correction(self,
                         image_id: str,
                         nucleus_id: int,
                         original_mask: np.ndarray,
                         corrected_mask: np.ndarray,
                         correction_type: str,
                         metadata: Optional[Dict[str, Any]] = None):
        """
        Record a user correction.
        
        Args:
            image_id: Image identifier
            nucleus_id: Nucleus ID
            original_mask: Original segmentation
            corrected_mask: Corrected segmentation
            correction_type: Type of correction
            metadata: Additional metadata
        """
        self.correction_tracker.add_correction(
            image_id=image_id,
            nucleus_id=nucleus_id,
            original_mask=original_mask,
            corrected_mask=corrected_mask,
            correction_type=correction_type,
            metadata=metadata
        )
        
        # Check if we should schedule training
        self._check_training_schedule()
    
    def _check_training_schedule(self):
        """Check if model should be retrained based on corrections."""
        num_corrections = len(self.correction_tracker.corrections)
        
        if num_corrections >= self.min_corrections_for_training:
            if not self.training_scheduled:
                logger.info(
                    f"Accumulated {num_corrections} corrections. "
                    f"Model retraining recommended."
                )
                self.training_scheduled = True
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained."""
        return self.training_scheduled
    
    def trigger_retraining(self) -> Dict[str, Any]:
        """
        Trigger model adapter retraining.
        
        Returns:
            Training results
        """
        if not self.should_retrain():
            logger.warning("Retraining not recommended yet (insufficient corrections)")
            return {'status': 'skipped', 'reason': 'insufficient_corrections'}
        
        # Get corrections for training
        corrections = self.correction_tracker.get_corrections_for_training(
            min_corrections=self.min_corrections_for_training
        )
        
        if not corrections:
            return {'status': 'failed', 'reason': 'no_corrections_available'}
        
        # Initialize adapter if needed
        if self.model_adapter is None:
            self.model_adapter = ModelAdapter()
        
        # Run fine-tuning (currently not implemented)
        try:
            results = self.model_adapter.fine_tune(corrections)
            self.last_training_time = datetime.now()
            self.training_scheduled = False
            
            logger.info("Model adapter retrained successfully")
            return {'status': 'success', 'results': results}
        
        except NotImplementedError:
            logger.warning("Model adapter fine-tuning not yet implemented")
            return {'status': 'not_implemented'}
