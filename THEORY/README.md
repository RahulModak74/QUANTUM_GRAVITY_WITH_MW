# Quantum Gravity Manifold Extraction Toolkit

**Theory Catches Up With Algorithm!**
We have  extracted the governing PDEs and the associated manifold geometry, which are provided in the GitHub repository at the end of article under this directory.

However, validation against high-fidelity general-relativity laboratories and experimental analog systems remains pending; until such validation, the theory should be regarded as constructively complete but empirically open.

This toolkit extracts the learned quantum gravity manifold from your trained VAE and expresses it as:
- **Geometric objects** (metric tensor, Christoffel symbols, Riemann tensor, Ricci tensor/scalar)
- **Partial Differential Equations** (Laplace-Beltrami, geodesic, wave, Einstein equations)
- **Symbolic mathematics** (SymPy expressions for metric components)
- **Publication-ready visualizations** (PDF report with curvature analysis)

## Conceptual Overview

Your VAE decoder `φ: Z → X` defines a **Riemannian manifold** on the latent space Z via the pullback metric:

```
g_ij(z) = ∂φ/∂z^i · ∂φ/∂z^j
```

This is **exactly the M-W Framework metric**! The neural network has learned the manifold geometry that satisfies all your physics priors (Einstein equations, Schrödinger equation, uncertainty principle, etc.).

Once trained, the manifold EXISTS as a geometric object. We extract it using:
1. **Automatic differentiation** (PyTorch) → compute metric, Christoffel symbols, curvature
2. **Symbolic regression** (SymPy + sklearn) → fit polynomial expressions for g_ij(z)
3. **PDE residuals** (torch.autograd) → verify the manifold satisfies governing equations

## Files

### Working Fast Scripts (Use These!) ⚡
- **`qg_extract_fast.py`** - Extract metric + curvature (30 sec, ~100 samples) ⭐
- **`analyze_manifold.py`** - Analyze geometry from .npz file (no VAE needed) ⭐
- **`plot_manifold.py`** - Visualize geometry (6 publication plots) ⭐
- **`qg_extract_pdes_fast.py`** - Extract PDEs from .npz file ⭐

### Legacy Scripts (Slower, for Reference)
- `qg_manifold_extractor.py` - Full geometry extraction (slower, with symbolic)
- `qg_pde_extractor.py` - Full PDE extraction (slower)
- `qg_theory_pipeline.py` - Complete pipeline with PDF report (requires all deps)

### Inputs
- `qg_vae.pth` - Your trained VAE model (from `qg_toy_vae_trainer_v4.py`)

### Outputs
- `qg_geometry_fast.npz` - Metric tensor + approximate Ricci scalar
- `manifold_analysis.txt` - Complete statistics and physics quantities
- `my_plots/*.png` - 6 visualization plots
- `qg_pdes_analysis.txt` - PDE residuals and Einstein tensor

## Installation

```bash
pip install -r requirements_manifold.txt
```

### Minimal Installation (No Symbolic/Visualization)

If you only want geometry extraction without symbolic math or plots:

```bash
# Only need PyTorch and NumPy
pip install torch numpy pyro-ppl
```

Then use `qg_extract_minimal.py` instead of the full pipeline.

Requirements:
- `torch>=2.0.0` - PyTorch for autodiff
- `numpy>=1.24.0` - Numerical arrays
- `pyro-ppl>=1.8.4` - Probabilistic programming (for VAE loading)
- `matplotlib>=3.7.0` - Visualization
- `seaborn>=0.12.0` - Statistical plots
- `sympy>=1.12` - Symbolic mathematics
- `scikit-learn>=1.3.0` - Polynomial fitting

## Usage

### FAST EXTRACTION (3 Working Scripts) ⚡

**Step 1: Extract Geometry**
```bash
# Extract metric tensor + approximate curvature (very fast!)
python3 qg_extract_fast.py --vae qg_vae.pth --samples 100
# Creates: qg_geometry_fast.npz (~30 seconds)
```

**Step 2: Analyze Manifold**
```bash
# Get comprehensive statistics and physics quantities
python3 analyze_manifold.py qg_geometry_fast.npz > manifold_analysis.txt
# Creates: manifold_analysis.txt with all metrics
```

**Step 3: Visualize**
```bash
# Generate publication-ready plots
python3 plot_manifold.py qg_geometry_fast.npz my_plots
# Creates: 6 PNG files in my_plots/ directory
```

**Step 4: Extract PDEs**
```bash
# Compute PDE residuals and Einstein tensor
python3 qg_extract_pdes_fast.py qg_geometry_fast.npz
# Creates: qg_pdes_analysis.txt
```

### Complete Pipeline (4 Commands)

```bash
# 1. Train VAE (if not already done)
python3 qg_toy_vae_trainer_v4.py

# 2. Extract geometry
python3 qg_extract_fast.py

# 3. Analyze
python3 analyze_manifold.py qg_geometry_fast.npz > manifold_analysis.txt

# 4. Visualize
python3 plot_manifold.py qg_geometry_fast.npz

# 5. Extract PDEs
python3 qg_extract_pdes_fast.py qg_geometry_fast.npz
```

**Total time: ~2 minutes** (after VAE training)

### Python API

```python
from qg_manifold_extractor import QuantumGravityManifold
from qg_pde_extractor import PDEExtractor
import torch

# Load your VAE
checkpoint = torch.load('qg_vae.pth')
vae = ...  # reconstruct VAE

# Create manifold extractor
manifold = QuantumGravityManifold(vae, device='cuda')

# Sample points in latent space
z_samples = torch.randn(100, 20, device='cuda') * 2.0

# Extract geometry
geometry = manifold.extract_geometry_at_points(z_samples)

# Access geometric quantities
g = geometry['metric']           # [100, 20, 20] metric tensor
Gamma = geometry['christoffel']  # [100, 20, 20, 20] connection
R = geometry['riemann']          # [100, 20, 20, 20, 20] curvature
Ric = geometry['ricci_tensor']   # [100, 20, 20] Ricci tensor
R_scalar = geometry['ricci_scalar']  # [100] Ricci scalar

# Fit symbolic expressions
symbolic_metric = manifold.fit_metric_symbolic(z_samples, g, max_degree=2)

# Extract PDEs
pde_extractor = PDEExtractor(vae, device='cuda')
pdes = pde_extractor.extract_all_pdes(z_samples, geometry)

# Access PDE residuals
laplacian = pdes['laplace_beltrami']     # Δ_g φ
geodesic = pdes['geodesic_acceleration'] # d²z/dτ²
wave = pdes['wave_equation_residual']    # -c²Δ_g ψ - E ψ
```

## Geometric Quantities Explained

### 1. Metric Tensor g_ij(z)
The fundamental object defining distances on the manifold:
```
ds² = g_ij(z) dz^i dz^j
```

**Shape**: `[n_samples, latent_dim, latent_dim]`

**Computation**: Via decoder Jacobian
```python
g_ij = Σ_k (∂φ^k/∂z^i)(∂φ^k/∂z^j)
```

### 2. Christoffel Symbols Γ^k_ij
Define the connection (how vectors are parallel transported):
```
Γ^k_ij = (1/2) g^kl (∂g_il/∂z^j + ∂g_jl/∂z^i - ∂g_ij/∂z^l)
```

**Shape**: `[n_samples, latent_dim, latent_dim, latent_dim]`

### 3. Riemann Curvature Tensor R^l_ijk
Measures how much the manifold is curved:
```
R^l_ijk = ∂Γ^l_jk/∂z^i - ∂Γ^l_ik/∂z^j + Γ^l_im Γ^m_jk - Γ^l_jm Γ^m_ik
```

**Shape**: `[n_samples, latent_dim, latent_dim, latent_dim, latent_dim]`

### 4. Ricci Tensor R_ij
Contraction of Riemann tensor:
```
R_ij = R^k_ikj
```

**Shape**: `[n_samples, latent_dim, latent_dim]`

### 5. Ricci Scalar R
Trace of Ricci tensor:
```
R = g^ij R_ij
```

**Shape**: `[n_samples]`

This appears in Einstein's field equations!

## PDEs Extracted

### 1. Laplace-Beltrami Equation
```
Δ_g φ^i(z) = (1/√det(g)) ∂_k(√det(g) g^kl ∂_l φ^i) = 0
```

Governs harmonic maps (energy minimization on manifold).

### 2. Geodesic Equation
```
d²z^i/dτ² + Γ^i_jk dz^j/dτ dz^k/dτ = 0
```

Defines shortest paths (geodesics) on the manifold.

### 3. Wave Equation (Stationary)
```
-c² Δ_g ψ(z) = E ψ(z)
```

Quantum wave function on curved manifold. Eigenvalue E is the energy.

### 4. Einstein Field Equations
```
G_ij = R_ij - (1/2)g_ij R = 8πG T_ij
```

These are enforced via physics priors during VAE training. The manifold satisfies them by construction!

## Symbolic Expressions

For metric components g_ij(z), we fit polynomial approximations:
```
g_ij(z) ≈ a_0 + Σ_k a_k z^k + Σ_{k,l} a_{kl} z^k z^l + ...
```

These are converted to SymPy expressions and saved as:
- Python dict (`symbolic_metric` in `.pkl`)
- LaTeX (`qg_manifold_metric.tex`)

You can then:
- Differentiate symbolically to get Christoffel symbols
- Integrate for geodesics
- Manipulate algebraically
- Export to Mathematica/Maple/SageMath

## Customization

### Adjust Number of Samples
```python
run_complete_pipeline(
    vae_path='qg_vae.pth',
    n_samples=200,  # More samples = better statistics
    output_dir='qg_theory_outputs'
)
```

More samples give better coverage of the manifold but take longer.

### Change Polynomial Degree
```python
symbolic_metric = manifold.fit_metric_symbolic(
    z_samples, g, 
    max_degree=3  # Cubic polynomials
)
```

Higher degree = more accurate fit but harder to interpret.

### Sample Different Regions
```python
# Near origin
z_samples = torch.randn(100, 20) * 0.5

# Far from origin (explore tails)
z_samples = torch.randn(100, 20) * 5.0

# Specific directions
z_samples = torch.zeros(100, 20)
z_samples[:, 0] = torch.linspace(-3, 3, 100)  # Scan along z_0
```

## Interpreting Results

### Good Signs
- Small PDE residuals (< 1e-3)
- Smooth metric components
- Positive metric eigenvalues (Riemannian)
- Finite Ricci scalar

### Warning Signs
- Large PDE residuals (> 1e-1) → manifold doesn't satisfy physics
- Negative metric eigenvalues → Lorentzian signature (might be intended!)
- Divergent curvature → singularities (black holes?)
- NaN values → numerical instability

### Physical Interpretation

**Ricci Scalar R:**
- R > 0: Positive curvature (sphere-like)
- R = 0: Flat (Euclidean)
- R < 0: Negative curvature (hyperbolic)

**Ricci Tensor R_ij:**
- Diagonal components: Principal curvatures
- Off-diagonal: Coupling between directions

**PDE Residuals:**
- Measure how well manifold satisfies physics
- Should be small if training converged

## Advanced Usage

### Compute Sectional Curvature
```python
# Sectional curvature K(X,Y) for tangent vectors X, Y
def sectional_curvature(R, g, X, Y):
    """
    K(X,Y) = R(X,Y,Y,X) / (g(X,X)g(Y,Y) - g(X,Y)²)
    """
    numerator = torch.einsum('...ijkl,i,j,k,l->...', R, X, Y, Y, X)
    gXX = torch.einsum('...ij,i,j->...', g, X, X)
    gYY = torch.einsum('...ij,i,j->...', g, Y, Y)
    gXY = torch.einsum('...ij,i,j->...', g, X, Y)
    denominator = gXX * gYY - gXY**2
    return numerator / (denominator + 1e-8)
```

### Export to Other Software

**Mathematica:**
```python
# SymPy expressions can be converted to Mathematica syntax
from sympy import mathematica_code
for key, expr in symbolic_metric.items():
    print(f"{key} = {mathematica_code(expr)}")
```

**SageMath:**
```python
# Save metric as Python dict, load in SageMath
import sage.all as sage
# Define manifold in SageMath with your metric
```

## Troubleshooting

### "CUDA out of memory"
Reduce batch size:
```python
extract_quantum_gravity_manifold(n_samples=50)  # Instead of 100
```

### "SymPy not available"
Install it:
```bash
pip install sympy
```

### "Singular matrix in metric inversion"
Add regularization:
```python
g_inv = torch.linalg.inv(g + torch.eye(dim) * 1e-6)  # Increase 1e-6
```

### "PDE residuals are large"
This means manifold doesn't satisfy PDEs well. Options:
1. Train VAE longer
2. Increase physics prior weights (lambda_dict)
3. Check if priors are appropriate

## Citation

If you use this toolkit, please cite:

```bibtex
@article{modak2025quantum,
  title={Quantum Gravity from Neural Networks: The Modak-Walawalkar Framework},
  author={Modak, Rahul and Walawalkar, Rahul},
  journal={Nature Physics},
  year={2025},
  note={Submitted}
}
```

## Next Steps

1. **Compare with Analytic Solutions**
   - Schwarzschild metric (black holes)
   - Kerr metric (rotating black holes)
   - FRW metric (cosmology)

2. **Compute Invariants**
   - Kretschmann scalar K = R^ijkl R_ijkl
   - Weyl tensor (conformal curvature)

3. **Geodesic Integration**
   - Integrate geodesic equation numerically
   - Compare with particle trajectories

4. **Extension to Lorentzian**
   - Modify metric signature to (-,+,+,+)
   - Study null geodesics (light paths)

## License

MIT License - Feel free to use for research and publication!

## Contact

Rahul Modak - Bayesian Cybersecurity Pvt Ltd
For questions about the M-W Framework or this toolkit.
rahul.modak@bayesiananalytics.in

---

**THEORY = ALGORITHM** 🏆

The manifold learned by the neural network IS the quantum gravity theory pending validations from GR labs. We've just extracted it and made it mathematically explicit!
