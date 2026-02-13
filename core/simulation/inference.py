"""
Parameter Inference using Simulation-Based Inference (SBI) and Bayesian Calibration.

Implements:
- ABC (Approximate Bayesian Computation)
- Summary statistics for FRAP/SPT data
- Posterior distribution estimation
- Parameter recovery validation
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Callable
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import differential_evolution


@dataclass
class PriorDistribution:
    """
    Prior distribution for parameters.
    
    Attributes:
        name: Parameter name
        distribution: Distribution type ('uniform', 'lognormal', 'normal')
        params: Distribution parameters (e.g., [min, max] for uniform)
    """
    name: str
    distribution: str
    params: list
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from prior distribution."""
        if self.distribution == 'uniform':
            return np.random.uniform(self.params[0], self.params[1], n_samples)
        elif self.distribution == 'lognormal':
            return np.random.lognormal(self.params[0], self.params[1], n_samples)
        elif self.distribution == 'normal':
            return np.random.normal(self.params[0], self.params[1], n_samples)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
    
    def log_prob(self, values: np.ndarray) -> np.ndarray:
        """Compute log probability of values."""
        if self.distribution == 'uniform':
            in_range = (values >= self.params[0]) & (values <= self.params[1])
            log_prob = np.where(in_range, -np.log(self.params[1] - self.params[0]), -np.inf)
            return log_prob
        elif self.distribution == 'lognormal':
            return stats.lognorm.logpdf(values, s=self.params[1], scale=np.exp(self.params[0]))
        elif self.distribution == 'normal':
            return stats.norm.logpdf(values, loc=self.params[0], scale=self.params[1])
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


class SummaryStatistics:
    """
    Compute summary statistics from FRAP/SPT data.
    
    These are used for distance-based inference methods.
    """
    
    @staticmethod
    def frap_statistics(recovery_curve: np.ndarray, time: np.ndarray) -> Dict[str, float]:
        """
        Extract summary statistics from FRAP recovery curve.
        
        Args:
            recovery_curve: Normalized recovery values
            time: Time points
            
        Returns:
            Dictionary of summary statistics
        """
        if len(recovery_curve) == 0:
            return {
                'half_time': np.nan,
                'mobile_fraction': np.nan,
                'initial_slope': np.nan,
                'final_value': np.nan
            }
        
        # Half-time (time to 50% recovery)
        half_recovery = 0.5
        idx_half = np.argmin(np.abs(recovery_curve - half_recovery))
        half_time = time[idx_half] if idx_half < len(time) else np.nan
        
        # Mobile fraction (final plateau)
        mobile_fraction = np.mean(recovery_curve[-5:]) if len(recovery_curve) >= 5 else recovery_curve[-1]
        
        # Initial slope
        if len(recovery_curve) >= 5:
            initial_slope = np.polyfit(time[:5], recovery_curve[:5], 1)[0]
        else:
            initial_slope = np.nan
        
        # Final value
        final_value = recovery_curve[-1]
        
        return {
            'half_time': half_time,
            'mobile_fraction': mobile_fraction,
            'initial_slope': initial_slope,
            'final_value': final_value
        }
    
    @staticmethod
    def spt_statistics(msd: np.ndarray, time_lags: np.ndarray) -> Dict[str, float]:
        """
        Extract summary statistics from SPT MSD curve.
        
        Args:
            msd: Mean squared displacement values
            time_lags: Time lag values
            
        Returns:
            Dictionary of summary statistics
        """
        if len(msd) < 3:
            return {
                'msd_slope': np.nan,
                'msd_intercept': np.nan,
                'alpha': np.nan,
                'D_apparent': np.nan
            }
        
        # Linear fit (first 20% of data)
        n_fit = max(3, min(len(msd) // 5, 20))
        
        # Log-log fit for alpha
        log_msd = np.log(msd[1:n_fit] + 1e-10)
        log_t = np.log(time_lags[1:n_fit] + 1e-10)
        
        alpha = np.polyfit(log_t, log_msd, 1)[0]
        
        # Linear fit for D
        slope, intercept = np.polyfit(time_lags[:n_fit], msd[:n_fit], 1)
        D_apparent = slope / 6.0  # 3D diffusion
        
        return {
            'msd_slope': slope,
            'msd_intercept': intercept,
            'alpha': alpha,
            'D_apparent': D_apparent
        }


class BayesianCalibration:
    """
    Bayesian parameter calibration using simulation-based inference.
    
    Implements ABC (Approximate Bayesian Computation) for parameter estimation.
    """
    
    def __init__(
        self,
        simulator: Callable,
        priors: List[PriorDistribution],
        summary_stats_func: Callable
    ):
        """
        Initialize Bayesian calibration.
        
        Args:
            simulator: Function that runs simulation given parameters
            priors: List of prior distributions for each parameter
            summary_stats_func: Function to compute summary statistics
        """
        self.simulator = simulator
        self.priors = priors
        self.summary_stats_func = summary_stats_func
    
    def abc_inference(
        self,
        observed_data: Dict,
        n_samples: int = 10000,
        epsilon: float = 0.1,
        distance_metric: str = 'euclidean'
    ) -> Dict:
        """
        ABC inference to estimate posterior distribution.
        
        Args:
            observed_data: Experimental data
            n_samples: Number of prior samples
            epsilon: Acceptance threshold
            distance_metric: Distance metric for comparing statistics
            
        Returns:
            Dictionary with posterior samples and acceptance rate
        """
        # Compute summary statistics from observed data
        obs_stats = self.summary_stats_func(observed_data)
        obs_stats_vector = np.array(list(obs_stats.values()))
        
        # Sample from prior and simulate
        accepted_params = []
        distances = []
        
        for _ in range(n_samples):
            # Sample parameters from prior
            params = {
                prior.name: prior.sample(1)[0]
                for prior in self.priors
            }
            
            # Run simulation
            sim_data = self.simulator(**params)
            
            # Compute summary statistics
            sim_stats = self.summary_stats_func(sim_data)
            sim_stats_vector = np.array(list(sim_stats.values()))
            
            # Compute distance
            distance = self._compute_distance(
                obs_stats_vector,
                sim_stats_vector,
                distance_metric
            )
            
            distances.append(distance)
            
            # Accept if distance < epsilon
            if distance < epsilon:
                accepted_params.append(params)
        
        # Convert to arrays
        posterior_samples = {}
        for prior in self.priors:
            posterior_samples[prior.name] = np.array([
                p[prior.name] for p in accepted_params
            ])
        
        acceptance_rate = len(accepted_params) / n_samples
        
        return {
            'posterior_samples': posterior_samples,
            'acceptance_rate': acceptance_rate,
            'distances': np.array(distances),
            'epsilon': epsilon,
            'n_accepted': len(accepted_params)
        }
    
    def maximum_likelihood_estimation(
        self,
        observed_data: Dict,
        bounds: Optional[Dict] = None
    ) -> Dict:
        """
        Maximum likelihood estimation using optimization.
        
        Args:
            observed_data: Experimental data
            bounds: Parameter bounds (optional, uses priors if not provided)
            
        Returns:
            Dictionary with MLE estimates
        """
        # Compute observed statistics
        obs_stats = self.summary_stats_func(observed_data)
        obs_stats_vector = np.array(list(obs_stats.values()))
        
        # Define objective function (negative log likelihood)
        def objective(params_array):
            params = {
                prior.name: params_array[i]
                for i, prior in enumerate(self.priors)
            }
            
            # Run simulation
            sim_data = self.simulator(**params)
            sim_stats = self.summary_stats_func(sim_data)
            sim_stats_vector = np.array(list(sim_stats.values()))
            
            # Compute squared error (assumes Gaussian likelihood)
            error = np.sum((obs_stats_vector - sim_stats_vector)**2)
            return error
        
        # Set bounds from priors if not provided
        if bounds is None:
            bounds = []
            for prior in self.priors:
                if prior.distribution == 'uniform':
                    bounds.append((prior.params[0], prior.params[1]))
                else:
                    # Use wide bounds for non-uniform priors
                    bounds.append((0.01, 100.0))
        
        # Optimize
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            seed=42
        )
        
        # Extract MLE estimates
        mle_estimates = {
            prior.name: result.x[i]
            for i, prior in enumerate(self.priors)
        }
        
        return {
            'mle_estimates': mle_estimates,
            'objective_value': result.fun,
            'success': result.success
        }
    
    def _compute_distance(
        self,
        stats1: np.ndarray,
        stats2: np.ndarray,
        metric: str
    ) -> float:
        """Compute distance between summary statistics."""
        # Handle NaN values
        valid = ~(np.isnan(stats1) | np.isnan(stats2))
        if not np.any(valid):
            return np.inf
        
        stats1 = stats1[valid]
        stats2 = stats2[valid]
        
        if metric == 'euclidean':
            return np.sqrt(np.sum((stats1 - stats2)**2))
        elif metric == 'manhattan':
            return np.sum(np.abs(stats1 - stats2))
        elif metric == 'relative':
            # Relative error (scale-invariant)
            return np.mean(np.abs((stats1 - stats2) / (stats1 + 1e-10)))
        else:
            raise ValueError(f"Unknown metric: {metric}")


class ParameterRecovery:
    """
    Validate parameter recovery capability.
    
    Tests whether inference methods can recover known ground truth parameters.
    """
    
    def __init__(
        self,
        simulator: Callable,
        inference_method: BayesianCalibration
    ):
        """
        Initialize parameter recovery validation.
        
        Args:
            simulator: Simulation function
            inference_method: Inference method to test
        """
        self.simulator = simulator
        self.inference_method = inference_method
    
    def run_recovery_test(
        self,
        true_params: Dict,
        n_trials: int = 10,
        noise_level: float = 0.0
    ) -> Dict:
        """
        Run parameter recovery test.
        
        Args:
            true_params: Ground truth parameters
            n_trials: Number of recovery attempts
            noise_level: Observational noise level
            
        Returns:
            Dictionary with recovery statistics
        """
        recovered_params = {name: [] for name in true_params.keys()}
        biases = {name: [] for name in true_params.keys()}
        
        for trial in range(n_trials):
            # Generate synthetic data with true parameters
            synthetic_data = self.simulator(**true_params)
            
            # Add noise if specified
            if noise_level > 0:
                synthetic_data = self._add_noise(synthetic_data, noise_level)
            
            # Attempt to recover parameters
            result = self.inference_method.maximum_likelihood_estimation(synthetic_data)
            
            if result['success']:
                for name in true_params.keys():
                    recovered = result['mle_estimates'][name]
                    true_val = true_params[name]
                    
                    recovered_params[name].append(recovered)
                    biases[name].append((recovered - true_val) / true_val)
        
        # Compute statistics
        recovery_stats = {}
        for name in true_params.keys():
            if len(recovered_params[name]) > 0:
                recovery_stats[name] = {
                    'true_value': true_params[name],
                    'mean_recovered': np.mean(recovered_params[name]),
                    'std_recovered': np.std(recovered_params[name]),
                    'mean_bias': np.mean(biases[name]),
                    'rmse': np.sqrt(np.mean(np.array(biases[name])**2))
                }
            else:
                recovery_stats[name] = {
                    'true_value': true_params[name],
                    'mean_recovered': np.nan,
                    'std_recovered': np.nan,
                    'mean_bias': np.nan,
                    'rmse': np.nan
                }
        
        return {
            'recovery_stats': recovery_stats,
            'success_rate': len(recovered_params[list(true_params.keys())[0]]) / n_trials,
            'n_trials': n_trials
        }
    
    def _add_noise(self, data: Dict, noise_level: float) -> Dict:
        """Add observational noise to data."""
        noisy_data = data.copy()
        
        # Add Gaussian noise to numerical arrays
        for key, value in noisy_data.items():
            if isinstance(value, np.ndarray):
                noise = np.random.randn(*value.shape) * noise_level * np.std(value)
                noisy_data[key] = value + noise
        
        return noisy_data
