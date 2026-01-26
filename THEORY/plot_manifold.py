#!/usr/bin/env python3
"""
Quantum Gravity Manifold Visualization
=======================================

Create plots from extracted geometry

Usage:
    python3 plot_manifold.py qg_geometry_fast.npz
"""

import numpy as np
import sys
import os

# Try importing matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  Matplotlib not available")
    print("Install with: pip install matplotlib")
    sys.exit(1)

def plot_metric_heatmap(g, output_dir='plots'):
    """Plot metric tensor as heatmap"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mean metric
    g_mean = g.mean(axis=0)
    im1 = axes[0].imshow(g_mean, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('Mean Metric Tensor $g_{ij}$', fontsize=14)
    axes[0].set_xlabel('j')
    axes[0].set_ylabel('i')
    plt.colorbar(im1, ax=axes[0])
    
    # Std metric
    g_std = g.std(axis=0)
    im2 = axes[1].imshow(g_std, cmap='viridis', aspect='auto')
    axes[1].set_title('Std Metric Tensor $\\sigma(g_{ij})$', fontsize=14)
    axes[1].set_xlabel('j')
    axes[1].set_ylabel('i')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metric_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/metric_heatmap.png")
    plt.close()

def plot_eigenvalue_distribution(g, output_dir='plots'):
    """Plot eigenvalue distribution"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Collect eigenvalues
    eigenvalues = []
    for i in range(len(g)):
        eigs = np.linalg.eigvalsh(g[i])
        eigenvalues.append(eigs)
    eigenvalues = np.array(eigenvalues)  # [n_samples, dim]
    
    # Box plot
    axes[0].boxplot([eigenvalues[:, i] for i in range(eigenvalues.shape[1])],
                    labels=[f'λ{i}' for i in range(eigenvalues.shape[1])])
    axes[0].set_title('Metric Eigenvalue Distribution', fontsize=14)
    axes[0].set_xlabel('Eigenvalue Index')
    axes[0].set_ylabel('Value')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', labelsize=8)
    
    # Histogram of all eigenvalues
    axes[1].hist(eigenvalues.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1].set_title('All Eigenvalues Histogram', fontsize=14)
    axes[1].set_xlabel('Eigenvalue')
    axes[1].set_ylabel('Count')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eigenvalue_distribution.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/eigenvalue_distribution.png")
    plt.close()

def plot_curvature_distribution(R, output_dir='plots'):
    """Plot curvature distribution"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(R, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[0].axvline(R.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {R.mean():.2e}')
    axes[0].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[0].set_title('Ricci Scalar Distribution', fontsize=14)
    axes[0].set_xlabel('R (Ricci Scalar)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Time series
    axes[1].plot(R, marker='o', markersize=3, alpha=0.7, color='green')
    axes[1].axhline(R.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {R.mean():.2e}')
    axes[1].axhline(0, color='black', linestyle='-', linewidth=1)
    axes[1].set_title('Ricci Scalar by Sample', fontsize=14)
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('R')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/curvature_distribution.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/curvature_distribution.png")
    plt.close()

def plot_metric_diagonal(g, output_dir='plots'):
    """Plot diagonal elements of metric"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    g_diag = np.diagonal(g, axis1=1, axis2=2)  # [n_samples, dim]
    
    # Plot each diagonal element
    for i in range(g.shape[1]):
        ax.plot(g_diag[:, i], label=f'$g_{{{i}{i}}}$', alpha=0.7)
    
    ax.set_title('Metric Diagonal Elements', fontsize=14)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('$g_{ii}$')
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metric_diagonal.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/metric_diagonal.png")
    plt.close()

def plot_latent_space_2d(z, R=None, output_dir='plots'):
    """Plot 2D projection of latent space"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # First two dimensions
    if R is not None:
        scatter1 = axes[0].scatter(z[:, 0], z[:, 1], c=R, cmap='RdYlBu_r',
                                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[0].set_title('Latent Space (z_0, z_1) colored by Curvature', fontsize=12)
        plt.colorbar(scatter1, ax=axes[0], label='R')
    else:
        axes[0].scatter(z[:, 0], z[:, 1], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[0].set_title('Latent Space (z_0, z_1)', fontsize=12)
    
    axes[0].set_xlabel('$z_0$')
    axes[0].set_ylabel('$z_1$')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[0].axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Other two dimensions
    if z.shape[1] >= 4:
        if R is not None:
            scatter2 = axes[1].scatter(z[:, 2], z[:, 3], c=R, cmap='RdYlBu_r',
                                      alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            axes[1].set_title('Latent Space (z_2, z_3) colored by Curvature', fontsize=12)
            plt.colorbar(scatter2, ax=axes[1], label='R')
        else:
            axes[1].scatter(z[:, 2], z[:, 3], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            axes[1].set_title('Latent Space (z_2, z_3)', fontsize=12)
        
        axes[1].set_xlabel('$z_2$')
        axes[1].set_ylabel('$z_3$')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0, color='black', linestyle='-', linewidth=0.5)
        axes[1].axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/latent_space_2d.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/latent_space_2d.png")
    plt.close()

def plot_determinant_distribution(g, output_dir='plots'):
    """Plot metric determinant distribution"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    det_g = np.linalg.det(g)
    
    # Histogram
    axes[0].hist(det_g, bins=50, alpha=0.7, edgecolor='black', color='purple')
    axes[0].axvline(det_g.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {det_g.mean():.2e}')
    axes[0].set_title('Metric Determinant Distribution', fontsize=14)
    axes[0].set_xlabel('det(g)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Time series
    axes[1].plot(det_g, marker='o', markersize=3, alpha=0.7, color='purple')
    axes[1].axhline(det_g.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {det_g.mean():.2e}')
    axes[1].set_title('Metric Determinant by Sample', fontsize=14)
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('det(g)')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/determinant_distribution.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/determinant_distribution.png")
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_manifold.py <geometry_file.npz> [output_dir]")
        print("\nExample:")
        print("  python3 plot_manifold.py qg_geometry_fast.npz")
        print("  python3 plot_manifold.py qg_geometry_fast.npz my_plots")
        sys.exit(1)
    
    npz_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'plots'
    
    if not os.path.exists(npz_file):
        print(f"ERROR: File not found: {npz_file}")
        sys.exit(1)
    
    print("=" * 70)
    print("QUANTUM GRAVITY MANIFOLD VISUALIZATION")
    print("=" * 70)
    print(f"Input: {npz_file}")
    print(f"Output: {output_dir}/")
    print("=" * 70)
    
    # Load data
    data = np.load(npz_file)
    g = data['metric']
    z = data['z']
    
    # Get curvature if available
    if 'ricci_scalar_approx' in data:
        R = data['ricci_scalar_approx']
    elif 'ricci_scalar' in data:
        R = data['ricci_scalar']
    else:
        R = None
    
    print("\nGenerating plots...")
    
    # Create plots
    plot_metric_heatmap(g, output_dir)
    plot_eigenvalue_distribution(g, output_dir)
    plot_metric_diagonal(g, output_dir)
    plot_determinant_distribution(g, output_dir)
    plot_latent_space_2d(z, R, output_dir)
    
    if R is not None:
        plot_curvature_distribution(R, output_dir)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated plots in: {output_dir}/")
    print("  • metric_heatmap.png - Mean and std of metric tensor")
    print("  • eigenvalue_distribution.png - Metric eigenvalues")
    print("  • metric_diagonal.png - Diagonal elements g_ii")
    print("  • determinant_distribution.png - det(g) distribution")
    print("  • latent_space_2d.png - 2D projections")
    if R is not None:
        print("  • curvature_distribution.png - Ricci scalar distribution")
    print("=" * 70)

if __name__ == "__main__":
    main()
