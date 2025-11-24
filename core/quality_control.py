"""
Quality control module for analyzing segmentation quality
Includes DNA intensity distribution analysis and outlier detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.mixture import GaussianMixture
from scipy import stats


class QualityControl:
    """
    Quality control system for segmentation validation
    """
    
    def __init__(self):
        self.phase_boundaries: List[float] = []
        self.phase_labels: List[str] = ['G1', 'S', 'G2/M']
        self.gmm_model: Optional[GaussianMixture] = None
    
    def analyze_dna_intensity(self,
                             masks: np.ndarray,
                             dna_image: np.ndarray,
                             n_phases: int = 3,
                             percentile_threshold: Tuple[float, float] = (5, 95)) -> Dict:
        """
        Analyze DNA intensity distribution for cell cycle analysis
        
        Args:
            masks: Labeled segmentation masks
            dna_image: DNA channel intensity image
            n_phases: Number of cell cycle phases to detect (2-5)
            percentile_threshold: (low, high) percentiles for outlier detection
        
        Returns:
            Dictionary with analysis results
        """
        from skimage.measure import regionprops
        
        # Extract mean DNA intensity per nucleus
        if masks.ndim == 3:
            # 3D mask, analyze middle slice or all slices
            z_mid = masks.shape[0] // 2
            mask_slice = masks[z_mid]
            if dna_image.ndim == 4:
                dna_slice = dna_image[z_mid]
                if dna_slice.ndim == 3:
                    dna_slice = dna_slice[0]  # First channel
            elif dna_image.ndim == 3:
                dna_slice = dna_image[z_mid] if dna_image.shape[0] > 1 else dna_image[0]
            else:
                dna_slice = dna_image
        else:
            mask_slice = masks
            if dna_image.ndim == 3:
                dna_slice = dna_image[0]  # First channel
            else:
                dna_slice = dna_image
        
        # Get region properties
        regions = regionprops(mask_slice, intensity_image=dna_slice)
        
        if len(regions) == 0:
            return {
                'nucleus_count': 0,
                'intensities': np.array([]),
                'flagged_nuclei': [],
                'phase_boundaries': [],
                'phase_assignments': {},
                'error': 'No nuclei detected'
            }
        
        # Extract intensities
        intensities = np.array([r.intensity_mean for r in regions])
        nucleus_ids = np.array([r.label for r in regions])
        
        # Fit Gaussian Mixture Model for cell cycle phases
        phase_assignments = {}
        if n_phases >= 2 and len(intensities) >= n_phases * 3:  # Need enough data
            try:
                self.gmm_model = GaussianMixture(
                    n_components=n_phases,
                    covariance_type='full',
                    random_state=42
                )
                
                # Fit model
                intensities_reshaped = intensities.reshape(-1, 1)
                self.gmm_model.fit(intensities_reshaped)
                
                # Predict phases
                phase_labels = self.gmm_model.predict(intensities_reshaped)
                
                # Get phase boundaries (at means)
                means = self.gmm_model.means_.flatten()
                sorted_means = np.sort(means)
                
                # Calculate boundaries between phases
                self.phase_boundaries = []
                for i in range(len(sorted_means) - 1):
                    boundary = (sorted_means[i] + sorted_means[i+1]) / 2
                    self.phase_boundaries.append(float(boundary))
                
                # Assign phase labels based on intensity
                for nuc_id, intensity, phase in zip(nucleus_ids, intensities, phase_labels):
                    phase_assignments[int(nuc_id)] = int(phase)
                
            except Exception as e:
                print(f"GMM fitting failed: {e}")
                self.phase_boundaries = []
        else:
            # Use simple percentile-based boundaries
            if n_phases == 2:
                self.phase_boundaries = [float(np.median(intensities))]
            elif n_phases == 3:
                self.phase_boundaries = [
                    float(np.percentile(intensities, 33.3)),
                    float(np.percentile(intensities, 66.6))
                ]
        
        # Detect outliers using percentile method
        low_threshold = np.percentile(intensities, percentile_threshold[0])
        high_threshold = np.percentile(intensities, percentile_threshold[1])
        
        # Flag nuclei outside expected range
        flagged_nuclei = []
        for nuc_id, intensity in zip(nucleus_ids, intensities):
            if intensity < low_threshold or intensity > high_threshold:
                flagged_nuclei.append({
                    'nucleus_id': int(nuc_id),
                    'intensity': float(intensity),
                    'reason': 'below_threshold' if intensity < low_threshold else 'above_threshold',
                    'z_score': float((intensity - np.mean(intensities)) / np.std(intensities))
                })
        
        # Calculate statistics
        results = {
            'nucleus_count': len(regions),
            'intensities': intensities,
            'nucleus_ids': nucleus_ids,
            'mean_intensity': float(np.mean(intensities)),
            'median_intensity': float(np.median(intensities)),
            'std_intensity': float(np.std(intensities)),
            'cv_intensity': float(np.std(intensities) / np.mean(intensities) * 100) if np.mean(intensities) > 0 else 0,
            'min_intensity': float(np.min(intensities)),
            'max_intensity': float(np.max(intensities)),
            'flagged_nuclei': flagged_nuclei,
            'flagged_count': len(flagged_nuclei),
            'flagged_percentage': len(flagged_nuclei) / len(regions) * 100,
            'phase_boundaries': self.phase_boundaries,
            'phase_assignments': phase_assignments,
            'low_threshold': float(low_threshold),
            'high_threshold': float(high_threshold),
        }
        
        return results
    
    def suggest_parameters(self,
                          qc_results: Dict,
                          current_params: Dict,
                          confirmed_error_rate: float) -> Dict:
        """
        Suggest segmentation parameter adjustments based on QC results
        
        Args:
            qc_results: Quality control analysis results
            current_params: Current segmentation parameters
            confirmed_error_rate: Percentage of confirmed segmentation errors
        
        Returns:
            Dictionary with suggested parameters and reasoning
        """
        suggestions = {
            'should_rerun': False,
            'changes': [],
            'new_params': current_params.copy()
        }
        
        # Check if error rate exceeds threshold
        if confirmed_error_rate > 5.0:
            suggestions['should_rerun'] = True
            
            # Analyze flagged nuclei patterns
            flagged = qc_results.get('flagged_nuclei', [])
            
            if not flagged:
                return suggestions
            
            # Count reasons
            below_count = sum(1 for f in flagged if f['reason'] == 'below_threshold')
            above_count = sum(1 for f in flagged if f['reason'] == 'above_threshold')
            
            # Get current parameters
            diameter = current_params.get('diameter', 30)
            flow_threshold = current_params.get('flow_threshold', 0.4)
            cellprob_threshold = current_params.get('cellprob_threshold', 0.0)
            
            # Many small/dim objects suggests under-segmentation or debris
            if below_count > above_count and below_count > len(flagged) * 0.6:
                # Increase thresholds to be more stringent
                new_cellprob = min(cellprob_threshold + 0.2, 3.0)
                suggestions['new_params']['cellprob_threshold'] = new_cellprob
                suggestions['changes'].append(
                    f"Increase cell probability threshold: {cellprob_threshold:.1f} → {new_cellprob:.1f} "
                    f"(many small/dim objects detected - likely over-segmentation or debris)"
                )
                
                # Optionally increase diameter
                if diameter and diameter < 40:
                    new_diameter = diameter * 1.2
                    suggestions['new_params']['diameter'] = new_diameter
                    suggestions['changes'].append(
                        f"Increase diameter: {diameter:.1f} → {new_diameter:.1f} pixels "
                        f"(small objects suggest diameter too small)"
                    )
            
            # Many large/bright objects suggests over-segmentation
            elif above_count > below_count and above_count > len(flagged) * 0.6:
                # Decrease thresholds to be less stringent
                new_cellprob = max(cellprob_threshold - 0.2, -3.0)
                suggestions['new_params']['cellprob_threshold'] = new_cellprob
                suggestions['changes'].append(
                    f"Decrease cell probability threshold: {cellprob_threshold:.1f} → {new_cellprob:.1f} "
                    f"(large/bright objects suggest under-segmentation)"
                )
                
                new_flow = max(flow_threshold - 0.1, 0.1)
                suggestions['new_params']['flow_threshold'] = new_flow
                suggestions['changes'].append(
                    f"Decrease flow threshold: {flow_threshold:.1f} → {new_flow:.1f} "
                    f"(allow more flexible boundaries)"
                )
            
            # Mixed issues
            else:
                # General adjustment - slightly increase stringency
                new_cellprob = min(cellprob_threshold + 0.1, 3.0)
                suggestions['new_params']['cellprob_threshold'] = new_cellprob
                suggestions['changes'].append(
                    f"Adjust cell probability threshold: {cellprob_threshold:.1f} → {new_cellprob:.1f} "
                    f"(mixed quality issues detected)"
                )
        
        return suggestions
    
    def update_phase_boundaries(self, boundaries: List[float]):
        """Manually update phase boundaries"""
        self.phase_boundaries = boundaries
    
    def assign_phases_with_boundaries(self,
                                     intensities: np.ndarray,
                                     boundaries: List[float],
                                     labels: List[str]) -> Dict[int, str]:
        """
        Assign cell cycle phases based on manual boundaries
        
        Args:
            intensities: Array of DNA intensities
            boundaries: List of boundary values (sorted)
            labels: Phase labels for each region
        
        Returns:
            Dictionary mapping indices to phase labels
        """
        assignments = {}
        
        for i, intensity in enumerate(intensities):
            # Find which region the intensity falls into
            phase_idx = 0
            for boundary in boundaries:
                if intensity >= boundary:
                    phase_idx += 1
                else:
                    break
            
            # Ensure phase_idx is within bounds
            phase_idx = min(phase_idx, len(labels) - 1)
            assignments[i] = labels[phase_idx]
        
        return assignments
