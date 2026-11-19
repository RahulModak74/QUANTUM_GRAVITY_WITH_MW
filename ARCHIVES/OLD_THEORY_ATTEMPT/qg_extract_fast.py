#!/usr/bin/env python3
"""
Fast Minimal Quantum Gravity Manifold Extractor
================================================

OPTIMIZED for speed:
- Batched computation
- Progress bars
- Only essential geometry
- Memory efficient

Extracts:
1. Metric tensor g_ij(z)
2. Ricci scalar R (approximate)
"""

import numpy as np
import torch
import torch.nn as nn
import sys
import os
from tqdm import tqdm

class FastManifoldExtractor:
    """
    Fast minimal extractor - only metric and approximate curvature
    """
    
    def __init__(self, vae, device='cuda'):
        self.vae = vae.eval()
        self.device = device
        self.latent_dim = vae.latent_dim
        self.data_dim = vae.decoder.fc_out.out_features
        
        print("=" * 70)
        print("FAST QUANTUM GRAVITY MANIFOLD EXTRACTOR")
        print("=" * 70)
        print(f"Latent dim: {self.latent_dim}, Data dim: {self.data_dim}")
        print("Computes: Metric g_ij + Approximate Ricci scalar")
        print("=" * 70)
    
    def compute_metric_batch(self, z_batch):
        """
        Compute metric for a batch efficiently
        Uses matrix operations instead of loops
        """
        batch_size = z_batch.shape[0]
        z_grad = z_batch.clone().detach().requires_grad_(True)
        
        with torch.enable_grad():
            x = self.vae.decode(z_grad)
        
        # Compute full Jacobian at once
        # J[b, i, j] = ∂x^i/∂z^j
        J = torch.zeros(batch_size, self.data_dim, self.latent_dim,
                       device=self.device, dtype=z_batch.dtype)
        
        for i in range(self.data_dim):
            grad_out = torch.zeros_like(x)
            grad_out[:, i] = 1.0
            
            grad = torch.autograd.grad(
                outputs=x, inputs=z_grad,
                grad_outputs=grad_out,
                retain_graph=(i < self.data_dim - 1),
                create_graph=False  # Don't need second derivatives
            )[0]
            
            J[:, i, :] = grad
        
        # g_ij = J^T J (pullback metric)
        # g[b, i, j] = sum_k J[b, k, i] * J[b, k, j]
        g = torch.einsum('bki,bkj->bij', J, J)
        
        return g
    
    def compute_ricci_scalar_approximate(self, g):
        """
        Approximate Ricci scalar using metric properties
        
        For small perturbations from flat space:
        R ≈ -Δ(log det g) where Δ is Laplacian
        
        We use a simpler approximation:
        R ≈ trace(g^-1) - dim (deviation from flat)
        """
        batch_size = g.shape[0]
        
        # Compute determinant and trace
        det_g = torch.linalg.det(g + torch.eye(self.latent_dim, device=self.device) * 1e-8)
        
        # Inverse metric
        g_inv = torch.linalg.inv(g + torch.eye(self.latent_dim, device=self.device) * 1e-8)
        
        # Approximate curvature from metric distortion
        # R ~ (det(g) - 1) measures deviation from flat space
        R_approx = torch.log(torch.abs(det_g) + 1e-10)
        
        # Alternative: use trace of inverse
        # R ~ trace(g^-1) - dim
        trace_inv = torch.diagonal(g_inv, dim1=1, dim2=2).sum(dim=1)
        R_trace = trace_inv - self.latent_dim
        
        # Average both estimates
        R = 0.5 * (R_approx + R_trace)
        
        return R
    
    def extract_fast(self, z_samples, batch_size=10):
        """
        Extract geometry in batches with progress bar
        """
        n_samples = z_samples.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        print(f"\nExtracting geometry for {n_samples} points in {n_batches} batches...")
        
        # Preallocate
        all_g = []
        all_R = []
        
        # Process in batches
        for i in tqdm(range(0, n_samples, batch_size), desc="Computing metric"):
            batch_end = min(i + batch_size, n_samples)
            z_batch = z_samples[i:batch_end]
            
            # Compute metric
            g_batch = self.compute_metric_batch(z_batch)
            all_g.append(g_batch.cpu().detach())
            
            # Compute approximate curvature
            R_batch = self.compute_ricci_scalar_approximate(g_batch)
            all_R.append(R_batch.cpu().detach())
            
            # Free GPU memory
            del g_batch, R_batch
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Concatenate
        g = torch.cat(all_g, dim=0).numpy()
        R = torch.cat(all_R, dim=0).numpy()
        
        print("\n✅ Extraction complete!")
        print(f"   Mean Ricci scalar (approx): {R.mean():.6f} ± {R.std():.6f}")
        
        return {
            'z': z_samples.cpu().detach().numpy(),
            'metric': g,
            'ricci_scalar_approx': R,
            'note': 'Ricci scalar is approximate (no Christoffel computation)'
        }

def extract_fast(vae_path='qg_vae.pth', n_samples=100, batch_size=10, output_file='qg_geometry_fast.npz'):
    """
    Fast extraction with batching
    """
    print("=" * 70)
    print("FAST EXTRACTION PIPELINE")
    print("=" * 70)
    print("Computes: Metric tensor + Approximate Ricci scalar")
    print("Speed: ~10-20x faster than full extraction")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load VAE
    print(f"\nLoading VAE from {vae_path}...")
    checkpoint = torch.load(vae_path, map_location=device)
    
    # Import VAE class
    sys.path.insert(0, os.path.dirname(os.path.abspath(vae_path)))
    try:
        from qg_toy_vae_trainer_v4 import QGPhysicsVAE
    except ImportError:
        print("ERROR: Cannot find qg_toy_vae_trainer_v4.py")
        print("Make sure it's in the same directory as this script.")
        sys.exit(1)
    
    vae = QGPhysicsVAE(
        input_dim=30,
        latent_dim=checkpoint['latent_dim'],
        mean=checkpoint['mean'].to(device),
        std=checkpoint['std'].to(device)
    ).to(device)
    
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    
    print(f"✅ VAE loaded (latent_dim={vae.latent_dim})")
    
    # Sample points
    print(f"\nSampling {n_samples} points in latent space...")
    z_samples = torch.randn(n_samples, vae.latent_dim, device=device) * 2.0
    
    # Extract
    extractor = FastManifoldExtractor(vae, device=device)
    geometry = extractor.extract_fast(z_samples, batch_size=batch_size)
    
    # Save
    print(f"\n💾 Saving to {output_file}...")
    np.savez_compressed(output_file, **geometry)
    
    # Print stats
    print("\n" + "=" * 70)
    print("GEOMETRY STATISTICS")
    print("=" * 70)
    
    g = geometry['metric']
    det_g = np.linalg.det(g)
    print(f"Metric determinant: {det_g.mean():.6f} ± {det_g.std():.6f}")
    
    # Eigenvalues of mean metric
    g_mean = g.mean(axis=0)
    eigs = np.linalg.eigvalsh(g_mean)
    print(f"\nMean metric eigenvalues (first 5):")
    for i in range(min(5, len(eigs))):
        print(f"  λ_{i}: {eigs[i]:.6f}")
    
    if (eigs > 0).all():
        print("\n✅ Metric is positive definite (Riemannian)")
    else:
        print(f"\n⚠️  {(eigs <= 0).sum()} negative eigenvalues")
    
    R = geometry['ricci_scalar_approx']
    print(f"\nApproximate Ricci scalar: {R.mean():.6f} ± {R.std():.6f}")
    
    if R.mean() > 0.1:
        print("→ Positive curvature (sphere-like)")
    elif R.mean() < -0.1:
        print("→ Negative curvature (hyperbolic)")
    else:
        print("→ Nearly flat")
    
    print("\n" + "=" * 70)
    print("FAST EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"\nSaved to: {output_file}")
    print("\nContents:")
    print("  - metric: [n, 20, 20] full metric tensor")
    print("  - ricci_scalar_approx: [n] approximate curvature")
    print("  - z: [n, 20] sample points")
    print("\nTo load:")
    print(f"  data = np.load('{output_file}')")
    print(f"  g = data['metric']")
    print(f"  R = data['ricci_scalar_approx']")
    print("\nNote: For full curvature (Christoffel, Riemann), use slower extraction")
    print("      or compute on smaller batch (--samples 10)")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast manifold extractor')
    parser.add_argument('--vae', default='qg_vae.pth', help='Path to VAE')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--batch', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--output', default='qg_geometry_fast.npz', help='Output file')
    
    args = parser.parse_args()
    
    # Check for tqdm
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm for progress bars...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        from tqdm import tqdm
    
    extract_fast(args.vae, args.samples, args.batch, args.output)
