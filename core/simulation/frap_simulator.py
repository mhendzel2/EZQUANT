"""
FRAP (Fluorescence Recovery After Photobleaching) Simulator.

Simulates photobleaching and recovery in 3D compartments.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from .diffusion_3d import Particle3DSimulator, DiffusionParameters


@dataclass
class FRAPParameters:
    """
    FRAP experimental parameters.
    
    Attributes:
        bleach_center: Center of bleach region (x, y, z) in µm
        bleach_radius: Radius of bleach region in µm
        bleach_depth: Depth of bleach (0-1, 1=complete)
        pre_bleach_time: Time before bleaching (s)
        post_bleach_time: Time after bleaching (s)
        n_particles: Number of particles to simulate
        imaging_interval: Time between imaging frames (s)
    """
    bleach_center: np.ndarray
    bleach_radius: float = 1.0  # µm
    bleach_depth: float = 0.8
    pre_bleach_time: float = 5.0  # s
    post_bleach_time: float = 60.0  # s
    n_particles: int = 1000
    imaging_interval: float = 0.5  # s


class FRAPSimulator:
    """
    3D FRAP simulator with realistic photobleaching dynamics.
    
    Features:
    - 3D diffusion of fluorescent particles
    - Photobleaching with Gaussian profile
    - Recovery curve generation
    - Comparison to analytical solutions
    """
    
    def __init__(
        self,
        diffusion_params: DiffusionParameters,
        frap_params: FRAPParameters,
        geometry=None
    ):
        """
        Initialize FRAP simulator.
        
        Args:
            diffusion_params: Particle diffusion parameters
            frap_params: FRAP experimental parameters
            geometry: Compartment geometry
        """
        self.diffusion_params = diffusion_params
        self.frap_params = frap_params
        self.geometry = geometry
        
        # Create particle simulator
        self.particle_sim = Particle3DSimulator(diffusion_params, geometry)
    
    def simulate_frap_experiment(
        self,
        permeability: float = 0.0,
        binding_rate: float = 0.0,
        unbinding_rate: float = 0.0,
        use_2d_projection: bool = False
    ) -> Dict:
        """
        Simulate complete FRAP experiment.
        
        Args:
            permeability: Membrane permeability
            binding_rate: Binding rate constant
            unbinding_rate: Unbinding rate constant
            use_2d_projection: If True, project to 2D (demonstrates bias)
            
        Returns:
            Dictionary with recovery curves and metadata
        """
        n_particles = self.frap_params.n_particles
        
        # Generate initial particle positions
        # Uniform distribution in nucleus
        if self.geometry is not None:
            initial_positions = self._generate_initial_positions(n_particles)
        else:
            # Simple cubic volume
            initial_positions = np.random.randn(n_particles, 3) * 2.0
        
        # Simulate pre-bleach equilibrium
        pre_bleach_steps = int(self.frap_params.pre_bleach_time / self.diffusion_params.dt)
        
        # For computational efficiency, assume pre-bleach equilibrium
        # In full version, would simulate to equilibrium
        
        # Apply photobleaching
        bleached = self._apply_photobleaching(initial_positions)
        
        # Simulate recovery
        post_bleach_steps = int(self.frap_params.post_bleach_time / self.diffusion_params.dt)
        imaging_steps = int(self.frap_params.imaging_interval / self.diffusion_params.dt)
        
        recovery_curve = []
        time_points = []
        all_trajectories = []
        
        # Simulate each particle
        for particle_idx in range(n_particles):
            # Adjust diffusion steps for recovery period
            recovery_params = DiffusionParameters(
                D=self.diffusion_params.D,
                alpha=self.diffusion_params.alpha,
                dt=self.diffusion_params.dt,
                n_steps=post_bleach_steps,
                temperature=self.diffusion_params.temperature,
                viscosity=self.diffusion_params.viscosity
            )
            
            # Create temporary simulator for this trajectory
            temp_sim = Particle3DSimulator(recovery_params, self.geometry)
            
            positions, times, _ = temp_sim.simulate_trajectory(
                initial_positions[particle_idx],
                permeability=permeability,
                binding_rate=binding_rate,
                unbinding_rate=unbinding_rate
            )
            
            all_trajectories.append({
                'positions': positions,
                'bleached': bleached[particle_idx],
                'initial_pos': initial_positions[particle_idx]
            })
        
        # Compute recovery curve at imaging intervals
        n_frames = int(post_bleach_steps / imaging_steps)
        
        for frame_idx in range(n_frames):
            step_idx = frame_idx * imaging_steps
            time = frame_idx * self.frap_params.imaging_interval
            
            # Compute fluorescence in bleach region
            fluorescence = self._compute_fluorescence_in_region(
                all_trajectories,
                step_idx,
                use_2d_projection
            )
            
            recovery_curve.append(fluorescence)
            time_points.append(time)
        
        # Normalize recovery curve
        recovery_curve = np.array(recovery_curve)
        f_initial = recovery_curve[0] if len(recovery_curve) > 0 else 0
        f_final = recovery_curve[-1] if len(recovery_curve) > 0 else 1
        f_prebleach = 1.0  # Normalized
        
        if f_final - f_initial > 1e-6:
            normalized_recovery = (recovery_curve - f_initial) / (f_final - f_initial)
        else:
            normalized_recovery = recovery_curve
        
        # Fit recovery curve to extract parameters
        fit_params = self._fit_recovery_curve(time_points, normalized_recovery)
        
        # Compare 3D vs 2D if requested
        comparison = None
        if use_2d_projection:
            # Also compute 3D version for comparison
            recovery_3d = []
            for frame_idx in range(n_frames):
                step_idx = frame_idx * imaging_steps
                fluorescence_3d = self._compute_fluorescence_in_region(
                    all_trajectories, step_idx, use_2d_projection=False
                )
                recovery_3d.append(fluorescence_3d)
            
            recovery_3d = np.array(recovery_3d)
            if f_final - f_initial > 1e-6:
                normalized_recovery_3d = (recovery_3d - f_initial) / (f_final - f_initial)
            else:
                normalized_recovery_3d = recovery_3d
            
            comparison = {
                'recovery_2d': normalized_recovery,
                'recovery_3d': normalized_recovery_3d,
                'bias': np.mean(np.abs(normalized_recovery - normalized_recovery_3d))
            }
        
        return {
            'time': np.array(time_points),
            'recovery': normalized_recovery,
            'raw_fluorescence': recovery_curve,
            'fit_params': fit_params,
            'trajectories': all_trajectories[:10],  # Save first 10 for visualization
            'n_particles': n_particles,
            'comparison_2d_3d': comparison
        }
    
    def _generate_initial_positions(self, n_particles: int) -> np.ndarray:
        """Generate random initial positions within nuclear envelope."""
        positions = []
        
        while len(positions) < n_particles:
            # Generate random position in bounding box
            pos = np.random.uniform(
                self.geometry.nuclear_envelope.center - self.geometry.nuclear_envelope.radii,
                self.geometry.nuclear_envelope.center + self.geometry.nuclear_envelope.radii
            )
            
            # Check if inside nucleus
            if self.geometry.nuclear_envelope.is_inside(pos):
                positions.append(pos)
        
        return np.array(positions)
    
    def _apply_photobleaching(self, positions: np.ndarray) -> np.ndarray:
        """
        Apply photobleaching with Gaussian profile.
        
        Returns:
            Array of bleaching factors (0=fully bleached, 1=unbleached)
        """
        center = self.frap_params.bleach_center
        radius = self.frap_params.bleach_radius
        depth = self.frap_params.bleach_depth
        
        # Compute distance from bleach center
        distances = np.linalg.norm(positions - center, axis=1)
        
        # Gaussian bleaching profile
        bleach_profile = 1.0 - depth * np.exp(-(distances**2) / (2 * radius**2))
        
        return bleach_profile
    
    def _compute_fluorescence_in_region(
        self,
        trajectories: List[Dict],
        step_idx: int,
        use_2d_projection: bool
    ) -> float:
        """
        Compute total fluorescence in bleach region.
        
        Args:
            trajectories: List of particle trajectories
            step_idx: Time step index
            use_2d_projection: Use 2D or 3D region
            
        Returns:
            Total fluorescence
        """
        center = self.frap_params.bleach_center
        radius = self.frap_params.bleach_radius
        
        total_fluorescence = 0.0
        n_in_region = 0
        
        for traj in trajectories:
            if step_idx >= len(traj['positions']):
                continue
            
            pos = traj['positions'][step_idx]
            
            # Check if in bleach region
            if use_2d_projection:
                # 2D projection (x, y only) - demonstrates bias
                distance = np.linalg.norm(pos[:2] - center[:2])
            else:
                # Full 3D distance
                distance = np.linalg.norm(pos - center)
            
            if distance <= radius:
                # Add fluorescence (affected by bleaching)
                total_fluorescence += traj['bleached']
                n_in_region += 1
        
        # Normalize by region volume
        if n_in_region > 0:
            return total_fluorescence / n_in_region
        return 0.0
    
    def _fit_recovery_curve(
        self,
        time: np.ndarray,
        recovery: np.ndarray
    ) -> Dict:
        """
        Fit FRAP recovery curve to exponential model.
        
        Model: F(t) = F_∞ * (1 - exp(-t/τ))
        
        Returns:
            Dictionary with fitted parameters
        """
        if len(time) < 3:
            return {'tau': np.nan, 'mobile_fraction': np.nan, 'D_fit': np.nan}
        
        # Simple exponential fit
        from scipy.optimize import curve_fit
        
        def recovery_model(t, mobile_fraction, tau):
            return mobile_fraction * (1 - np.exp(-t / tau))
        
        try:
            popt, _ = curve_fit(
                recovery_model,
                time,
                recovery,
                p0=[0.8, 5.0],
                bounds=([0, 0.1], [1.0, 100.0])
            )
            
            mobile_fraction, tau = popt
            
            # Estimate diffusion coefficient from tau and geometry
            # For circular bleach region: τ ≈ r²/(4*D)
            r_bleach = self.frap_params.bleach_radius
            D_fit = r_bleach**2 / (4 * tau)
            
            return {
                'tau': tau,
                'mobile_fraction': mobile_fraction,
                'D_fit': D_fit
            }
        except:
            return {'tau': np.nan, 'mobile_fraction': np.nan, 'D_fit': np.nan}
