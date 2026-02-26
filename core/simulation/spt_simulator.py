"""
SPT (Single Particle Tracking) Simulator.

Generates realistic particle trajectories with various motion modes.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from .diffusion_3d import Particle3DSimulator, DiffusionParameters


@dataclass
class SPTParameters:
    """
    SPT experimental parameters.
    
    Attributes:
        n_trajectories: Number of trajectories to simulate
        trajectory_length: Number of frames per trajectory
        frame_interval: Time between frames (s)
        localization_precision: Positional uncertainty (µm)
        detection_probability: Probability of detecting particle per frame
    """
    n_trajectories: int = 100
    trajectory_length: int = 100
    frame_interval: float = 0.033  # s (30 fps)
    localization_precision: float = 0.020  # µm (20 nm)
    detection_probability: float = 0.95


class SPTSimulator:
    """
    Single particle tracking simulator with realistic noise and gaps.
    
    Features:
    - Multiple diffusion modes (normal, anomalous, confined, directed)
    - Localization error
    - Missing detections (blinking)
    - Track analysis tools
    """
    
    def __init__(
        self,
        diffusion_params: DiffusionParameters,
        spt_params: SPTParameters,
        geometry=None
    ):
        """
        Initialize SPT simulator.
        
        Args:
            diffusion_params: Particle diffusion parameters
            spt_params: SPT experimental parameters
            geometry: Compartment geometry
        """
        self.diffusion_params = diffusion_params
        self.spt_params = spt_params
        self.geometry = geometry
        
        self.particle_sim = Particle3DSimulator(diffusion_params, geometry)
    
    def simulate_tracks(
        self,
        permeability: float = 0.0,
        binding_rate: float = 0.0,
        unbinding_rate: float = 0.0,
        directed_velocity: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Simulate multiple SPT trajectories.
        
        Args:
            permeability: Membrane permeability
            binding_rate: Binding rate constant
            unbinding_rate: Unbinding rate constant
            directed_velocity: Optional drift velocity (vx, vy, vz) in µm/s
            
        Returns:
            Dictionary with trajectory data and analysis
        """
        trajectories = []
        
        for track_idx in range(self.spt_params.n_trajectories):
            # Random initial position
            if self.geometry is not None:
                # Start in nucleoplasm
                initial_pos = self._generate_random_position()
            else:
                initial_pos = np.random.randn(3) * 2.0
            
            # Simulate trajectory
            trajectory = self._simulate_single_track(
                initial_pos,
                permeability,
                binding_rate,
                unbinding_rate,
                directed_velocity
            )
            
            trajectories.append(trajectory)
        
        # Analyze trajectories
        analysis = self._analyze_trajectories(trajectories)
        
        return {
            'trajectories': trajectories,
            'analysis': analysis,
            'parameters': {
                'D': self.diffusion_params.D,
                'alpha': self.diffusion_params.alpha,
                'permeability': permeability,
                'binding_rate': binding_rate,
                'unbinding_rate': unbinding_rate
            }
        }
    
    def _generate_random_position(self) -> np.ndarray:
        """Generate random position in nuclear compartment."""
        max_attempts = 1000
        
        for _ in range(max_attempts):
            pos = np.random.uniform(
                self.geometry.nuclear_envelope.center - self.geometry.nuclear_envelope.radii,
                self.geometry.nuclear_envelope.center + self.geometry.nuclear_envelope.radii
            )
            
            if self.geometry.nuclear_envelope.is_inside(pos):
                return pos
        
        # Fallback to center
        return self.geometry.nuclear_envelope.center
    
    def _simulate_single_track(
        self,
        initial_pos: np.ndarray,
        permeability: float,
        binding_rate: float,
        unbinding_rate: float,
        directed_velocity: Optional[np.ndarray]
    ) -> Dict:
        """
        Simulate single particle trajectory with experimental noise.
        
        Returns:
            Dictionary with positions, times, and metadata
        """
        # Calculate total simulation time
        total_time = self.spt_params.trajectory_length * self.spt_params.frame_interval
        n_steps = int(total_time / self.diffusion_params.dt)
        
        # Adjust diffusion parameters for this simulation
        sim_params = DiffusionParameters(
            D=self.diffusion_params.D,
            alpha=self.diffusion_params.alpha,
            dt=self.diffusion_params.dt,
            n_steps=n_steps,
            temperature=self.diffusion_params.temperature,
            viscosity=self.diffusion_params.viscosity
        )
        
        temp_sim = Particle3DSimulator(sim_params, self.geometry)
        
        # Simulate ground truth trajectory
        positions, times, metadata = temp_sim.simulate_trajectory(
            initial_pos,
            permeability=permeability,
            binding_rate=binding_rate,
            unbinding_rate=unbinding_rate
        )
        
        # Add directed motion if specified
        if directed_velocity is not None:
            for i in range(len(positions)):
                positions[i] += directed_velocity * times[i]
        
        # Sample at frame intervals
        frame_indices = np.arange(
            0,
            n_steps,
            int(self.spt_params.frame_interval / self.diffusion_params.dt)
        )
        
        sampled_positions = positions[frame_indices]
        sampled_times = times[frame_indices]
        
        # Add localization noise
        noise = np.random.randn(*sampled_positions.shape) * self.spt_params.localization_precision
        noisy_positions = sampled_positions + noise
        
        # Simulate detection gaps (blinking)
        detected = np.random.rand(len(sampled_positions)) < self.spt_params.detection_probability
        
        # Keep only detected positions
        detected_positions = noisy_positions[detected]
        detected_times = sampled_times[detected]
        detected_indices = np.where(detected)[0]
        
        return {
            'positions': detected_positions,
            'times': detected_times,
            'frame_indices': detected_indices,
            'ground_truth_positions': sampled_positions,
            'ground_truth_times': sampled_times,
            'n_detected': len(detected_positions),
            'n_frames': len(sampled_positions),
            'metadata': metadata
        }
    
    def _analyze_trajectories(self, trajectories: List[Dict]) -> Dict:
        """
        Analyze ensemble of trajectories.
        
        Computes:
        - Mean squared displacement (MSD)
        - Diffusion coefficient distribution
        - Alpha (anomalous exponent) distribution
        - Confinement analysis
        
        Returns:
            Dictionary with analysis results
        """
        all_msds = []
        D_estimates = []
        alpha_estimates = []
        track_lengths = []
        
        for traj in trajectories:
            positions = traj['positions']
            times = traj['times']
            
            if len(positions) < 5:
                continue  # Skip short tracks
            
            # Compute MSD for this trajectory
            msd = self._compute_track_msd(positions, times)
            all_msds.append(msd)
            
            # Estimate diffusion parameters
            D_est, alpha_est = self._estimate_parameters_from_msd(msd, times)
            D_estimates.append(D_est)
            alpha_estimates.append(alpha_est)
            
            track_lengths.append(len(positions))
        
        # Ensemble-averaged MSD
        # Find common maximum lag
        min_length = min(len(msd) for msd in all_msds) if all_msds else 0
        
        if min_length > 0:
            ensemble_msd = np.mean([msd[:min_length] for msd in all_msds], axis=0)
            msd_std = np.std([msd[:min_length] for msd in all_msds], axis=0)
        else:
            ensemble_msd = np.array([])
            msd_std = np.array([])
        
        return {
            'n_tracks': len(trajectories),
            'n_analyzed': len(D_estimates),
            'mean_track_length': np.mean(track_lengths) if track_lengths else 0,
            'D_estimates': np.array(D_estimates),
            'D_mean': np.mean(D_estimates) if D_estimates else np.nan,
            'D_std': np.std(D_estimates) if D_estimates else np.nan,
            'alpha_estimates': np.array(alpha_estimates),
            'alpha_mean': np.mean(alpha_estimates) if alpha_estimates else np.nan,
            'alpha_std': np.std(alpha_estimates) if alpha_estimates else np.nan,
            'ensemble_msd': ensemble_msd,
            'msd_std': msd_std
        }
    
    def _compute_track_msd(
        self,
        positions: np.ndarray,
        times: np.ndarray
    ) -> np.ndarray:
        """
        Compute MSD for a single trajectory.
        
        Uses overlapping time windows for better statistics.
        """
        n_points = len(positions)
        max_lag = min(n_points // 4, 50)  # Use up to 1/4 of track length
        
        msd = np.zeros(max_lag)
        counts = np.zeros(max_lag)
        
        for lag in range(1, max_lag):
            for i in range(n_points - lag):
                displacement = positions[i + lag] - positions[i]
                msd[lag] += np.sum(displacement**2)
                counts[lag] += 1
        
        # Average
        valid_indices = counts > 0
        msd[valid_indices] /= counts[valid_indices]
        
        return msd
    
    def _estimate_parameters_from_msd(
        self,
        msd: np.ndarray,
        times: np.ndarray
    ) -> Tuple[float, float]:
        """
        Estimate D and alpha from MSD curve.
        
        Returns:
            D_estimate, alpha_estimate
        """
        if len(msd) < 5:
            return np.nan, np.nan
        
        # Use first 20 points for fitting
        n_fit = min(20, len(msd))
        
        # For anomalous diffusion: MSD = 6*D*t^alpha (3D)
        # Log-log fit: log(MSD) = log(6*D) + alpha*log(t)
        
        dt = self.spt_params.frame_interval
        t_values = np.arange(1, n_fit) * dt
        msd_values = msd[1:n_fit]
        
        # Remove zeros
        valid = msd_values > 0
        if np.sum(valid) < 3:
            return np.nan, np.nan
        
        t_values = t_values[valid]
        msd_values = msd_values[valid]
        
        # Log-log fit
        log_t = np.log(t_values)
        log_msd = np.log(msd_values)
        
        coeffs = np.polyfit(log_t, log_msd, 1)
        alpha_est = coeffs[0]
        log_6D = coeffs[1]
        
        D_est = np.exp(log_6D) / 6.0
        
        return D_est, alpha_est
    
    def export_tracks_to_trackmate_format(self, trajectories: List[Dict]) -> str:
        """
        Export tracks in TrackMate-compatible CSV format.
        
        Returns:
            CSV string
        """
        lines = ["TRACK_ID,FRAME,POSITION_X,POSITION_Y,POSITION_Z"]
        
        for track_id, traj in enumerate(trajectories):
            positions = traj['positions']
            frame_indices = traj['frame_indices']
            
            for frame, pos in zip(frame_indices, positions):
                lines.append(
                    f"{track_id},{frame},{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}"
                )
        
        return "\n".join(lines)
