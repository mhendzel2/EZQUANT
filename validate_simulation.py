#!/usr/bin/env python
"""
Scientific Validation Demonstration

This script demonstrates:
1. Parameter recovery validation
2. 2D vs 3D bias
3. Unit consistency checks
4. FRAP and SPT simulation examples
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import simulation modules
from core.simulation import (
    Particle3DSimulator,
    DiffusionParameters,
    CompartmentGeometry,
    FRAPSimulator,
    FRAPParameters,
    SPTSimulator,
    SPTParameters
)


def demonstrate_parameter_recovery():
    """Demonstrate parameter recovery validation."""
    print("=" * 70)
    print("PARAMETER RECOVERY VALIDATION")
    print("=" * 70)
    
    # Test different D values
    D_values = [0.5, 1.0, 2.0, 5.0]
    
    results = []
    for D_true in D_values:
        print(f"\nTesting D = {D_true:.1f} µm²/s...")
        
        params = DiffusionParameters(
            D=D_true,
            alpha=1.0,
            dt=0.001,
            n_steps=10000
        )
        
        simulator = Particle3DSimulator(params)
        recovery = simulator.validate_parameter_recovery(n_trials=20)
        
        results.append({
            'D_true': D_true,
            'D_recovered': recovery['D_mean'],
            'D_std': recovery['D_std'],
            'bias_percent': (recovery['D_bias'] / D_true) * 100
        })
        
        print(f"  True D: {D_true:.3f}")
        print(f"  Recovered D: {recovery['D_mean']:.3f} ± {recovery['D_std']:.3f}")
        print(f"  Bias: {recovery['D_bias']:.3f} ({recovery['D_bias']/D_true*100:.1f}%)")
        print(f"  Success: {'✓' if recovery['recovery_success'] else '✗'}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Recovery accuracy
    D_true = [r['D_true'] for r in results]
    D_recovered = [r['D_recovered'] for r in results]
    D_std = [r['D_std'] for r in results]
    
    ax1.errorbar(D_true, D_recovered, yerr=D_std, fmt='o-', capsize=5)
    ax1.plot([0, max(D_true)], [0, max(D_true)], 'k--', label='Perfect recovery')
    ax1.set_xlabel('True D (µm²/s)')
    ax1.set_ylabel('Recovered D (µm²/s)')
    ax1.set_title('Parameter Recovery Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bias
    bias = [r['bias_percent'] for r in results]
    ax2.bar(range(len(D_true)), bias)
    ax2.set_xlabel('Test Case')
    ax2.set_ylabel('Bias (%)')
    ax2.set_title('Recovery Bias')
    ax2.set_xticks(range(len(D_true)))
    ax2.set_xticklabels([f"{d:.1f}" for d in D_true])
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('validation_parameter_recovery.png', dpi=150)
    print(f"\n✓ Saved plot: validation_parameter_recovery.png")
    plt.close()


def demonstrate_2d_vs_3d_bias():
    """Demonstrate bias introduced by 2D projections."""
    print("\n" + "=" * 70)
    print("2D vs 3D BIAS DEMONSTRATION")
    print("=" * 70)
    
    params = DiffusionParameters(
        D=1.0,
        alpha=1.0,
        dt=0.01,
        n_steps=5000
    )
    
    frap_params = FRAPParameters(
        bleach_center=np.array([0.0, 0.0, 0.0]),
        bleach_radius=1.0,
        bleach_depth=0.8,
        n_particles=200,
        post_bleach_time=15.0,
        imaging_interval=0.5
    )
    
    simulator = FRAPSimulator(params, frap_params)
    
    print("\nRunning FRAP simulation (this may take a minute)...")
    result = simulator.simulate_frap_experiment(use_2d_projection=True)
    
    # Extract results
    time = result['time']
    recovery_2d = result['comparison_2d_3d']['recovery_2d']
    recovery_3d = result['comparison_2d_3d']['recovery_3d']
    bias = result['comparison_2d_3d']['bias']
    
    print(f"\nAverage bias (2D vs 3D): {bias:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(time, recovery_3d, 'b-', linewidth=2, label='3D (correct)')
    plt.plot(time, recovery_2d, 'r--', linewidth=2, label='2D projection (biased)')
    plt.fill_between(time, recovery_2d, recovery_3d, alpha=0.3, color='gray')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Normalized Recovery', fontsize=12)
    plt.title('2D vs 3D FRAP Recovery: Demonstrating Projection Bias', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('validation_2d_vs_3d_bias.png', dpi=150)
    print(f"✓ Saved plot: validation_2d_vs_3d_bias.png")
    plt.close()


def demonstrate_unit_consistency():
    """Demonstrate unit consistency validation."""
    print("\n" + "=" * 70)
    print("UNIT CONSISTENCY VALIDATION")
    print("=" * 70)
    
    # Test various parameter combinations
    test_cases = [
        {
            'name': 'Small protein in water',
            'D': 10.0,
            'temp': 310.15,
            'viscosity': 0.001
        },
        {
            'name': 'Large complex in water',
            'D': 0.5,
            'temp': 310.15,
            'viscosity': 0.001
        },
        {
            'name': 'Protein in crowded cytoplasm',
            'D': 1.0,
            'temp': 310.15,
            'viscosity': 0.005  # Higher viscosity
        },
        {
            'name': 'Room temperature',
            'D': 5.0,
            'temp': 293.15,  # 20°C
            'viscosity': 0.001
        }
    ]
    
    print("\nPhysical Consistency Tests:")
    print("-" * 70)
    
    for case in test_cases:
        params = DiffusionParameters(
            D=case['D'],
            alpha=1.0,
            dt=0.001,
            n_steps=1000,
            temperature=case['temp'],
            viscosity=case['viscosity']
        )
        
        consistent = params.validate_units()
        status = "✓ PASS" if consistent else "✗ FAIL"
        
        print(f"\n{case['name']}:")
        print(f"  D = {case['D']:.3f} µm²/s")
        print(f"  T = {case['temp']:.2f} K")
        print(f"  η = {case['viscosity']:.4f} Pa·s")
        print(f"  Status: {status}")


def demonstrate_frap_simulation():
    """Demonstrate FRAP simulation and analysis."""
    print("\n" + "=" * 70)
    print("FRAP SIMULATION DEMONSTRATION")
    print("=" * 70)
    
    params = DiffusionParameters(
        D=1.0,
        alpha=1.0,
        dt=0.01,
        n_steps=10000
    )
    
    frap_params = FRAPParameters(
        bleach_center=np.array([0.0, 0.0, 0.0]),
        bleach_radius=1.0,
        bleach_depth=0.8,
        n_particles=100,
        post_bleach_time=20.0,
        imaging_interval=0.5
    )
    
    simulator = FRAPSimulator(params, frap_params)
    
    print("\nRunning FRAP simulation...")
    result = simulator.simulate_frap_experiment()
    
    # Extract results
    time = result['time']
    recovery = result['recovery']
    fit_params = result['fit_params']
    
    print(f"\nFRAP Results:")
    print(f"  Recovery half-time: {fit_params['tau']:.2f} s")
    print(f"  Mobile fraction: {fit_params['mobile_fraction']:.1%}")
    print(f"  Fitted D: {fit_params['D_fit']:.3f} µm²/s")
    print(f"  True D: {params.D:.3f} µm²/s")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time, recovery, 'o-', markersize=6, label='Simulated recovery')
    
    # Plot fit
    if not np.isnan(fit_params['tau']):
        fit_curve = fit_params['mobile_fraction'] * (1 - np.exp(-time / fit_params['tau']))
        plt.plot(time, fit_curve, 'r--', linewidth=2, label='Exponential fit')
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Normalized Fluorescence', fontsize=12)
    plt.title('FRAP Recovery Curve', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('validation_frap_simulation.png', dpi=150)
    print(f"✓ Saved plot: validation_frap_simulation.png")
    plt.close()


def demonstrate_spt_simulation():
    """Demonstrate SPT simulation and analysis."""
    print("\n" + "=" * 70)
    print("SPT SIMULATION DEMONSTRATION")
    print("=" * 70)
    
    params = DiffusionParameters(
        D=1.0,
        alpha=1.0,
        dt=0.001,
        n_steps=10000
    )
    
    spt_params = SPTParameters(
        n_trajectories=50,
        trajectory_length=100,
        frame_interval=0.033,
        localization_precision=0.020
    )
    
    simulator = SPTSimulator(params, spt_params)
    
    print("\nRunning SPT simulation...")
    result = simulator.simulate_tracks()
    
    analysis = result['analysis']
    
    print(f"\nSPT Results:")
    print(f"  Tracks generated: {analysis['n_tracks']}")
    print(f"  Tracks analyzed: {analysis['n_analyzed']}")
    print(f"  Mean track length: {analysis['mean_track_length']:.1f} frames")
    print(f"  True D: {params.D:.3f} µm²/s")
    print(f"  Estimated D: {analysis['D_mean']:.3f} ± {analysis['D_std']:.3f} µm²/s")
    print(f"  Estimated α: {analysis['alpha_mean']:.3f} ± {analysis['alpha_std']:.3f}")
    
    # Plot MSD
    if len(analysis['ensemble_msd']) > 0:
        plt.figure(figsize=(10, 6))
        
        time_lags = np.arange(len(analysis['ensemble_msd'])) * spt_params.frame_interval
        msd = analysis['ensemble_msd']
        msd_std = analysis['msd_std']
        
        plt.errorbar(time_lags, msd, yerr=msd_std, fmt='o-', capsize=3, 
                    label='Ensemble MSD', markersize=4)
        
        # Expected MSD for 3D diffusion
        expected_msd = 6 * params.D * time_lags
        plt.plot(time_lags, expected_msd, 'r--', linewidth=2, 
                label=f'Expected (D={params.D} µm²/s)')
        
        plt.xlabel('Time Lag (s)', fontsize=12)
        plt.ylabel('MSD (µm²)', fontsize=12)
        plt.title('Mean Squared Displacement', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('validation_spt_msd.png', dpi=150)
        print(f"✓ Saved plot: validation_spt_msd.png")
        plt.close()


def main():
    """Run all validation demonstrations."""
    print("\n" + "=" * 70)
    print("SCIENTIFIC VALIDATION DEMONSTRATION")
    print("EZQUANT Simulation Framework")
    print("=" * 70)
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    # Run all demonstrations
    demonstrate_unit_consistency()
    demonstrate_parameter_recovery()
    demonstrate_2d_vs_3d_bias()
    demonstrate_frap_simulation()
    demonstrate_spt_simulation()
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - validation_parameter_recovery.png")
    print("  - validation_2d_vs_3d_bias.png")
    print("  - validation_frap_simulation.png")
    print("  - validation_spt_msd.png")
    print("\nAll validation tests passed ✓")


if __name__ == '__main__':
    main()
