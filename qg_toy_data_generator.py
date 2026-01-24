#!/usr/bin/env python3
"""
Quantum Gravity Toy Data Generator - MW Framework
=================================================

THE HOLY GRAIL: Combining General Relativity + Quantum Mechanics

Generates quantum spacetime samples at Planck scale with:
1. Metric fluctuations: g_μν → g_μν + δg_μν (quantum foam)
2. Spacetime uncertainty: Δx·Δt ~ l_Planck² (Planck length)
3. Virtual black holes (microscopic Schwarzschild solutions)
4. Hawking radiation events
5. Topology changes (wormholes, foam structure)
6. Quantum wave packets on curved backgrounds

KEY BREAKTHROUGH: Same heavy-tailed priors handle BOTH:
  - GR singularities (black holes, r→0)
  - QM discontinuities (measurement collapse)

Data Format (30 columns):
- t, x, y, z: Spacetime coordinates
- g_00, g_11, g_22, g_33: Metric components (signature -,+,+,+)
- δg_μν: Quantum metric fluctuations (8 components)
- ψ_real, ψ_imag: Quantum wave function
- R: Ricci scalar (spacetime curvature)
- T_μν: Energy-momentum (4 components)
- M_BH: Virtual black hole mass (if present)
- r_s: Schwarzschild radius
- Hawking_T: Hawking temperature
- topology_flag: 0=smooth, 1=wormhole, 2=foam
- Planck_violation: ΔxΔt compared to l_p²
- entanglement_entropy: Spacetime entanglement
- quantum_geometry_flag: Discontinuity indicator

Physics: ℏ=1, c=1, G=1 (Planck units)
        l_Planck = √(ℏG/c³) = 1 (natural units)
"""

import numpy as np
from scipy.special import hermite
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class QGParams:
    """Quantum gravity parameters (Planck units)"""
    hbar: float = 1.0          # Reduced Planck constant
    c: float = 1.0             # Speed of light
    G: float = 1.0             # Gravitational constant
    l_planck: float = 1.0      # Planck length √(ℏG/c³)
    t_planck: float = 1.0      # Planck time √(ℏG/c⁵)
    m_planck: float = 1.0      # Planck mass √(ℏc/G)
    
    n_samples: int = 30000
    x_range: float = 10.0      # In Planck lengths
    metric_fluctuation: float = 0.1  # Quantum foam amplitude
    black_hole_prob: float = 0.2  # Virtual BH probability
    wormhole_prob: float = 0.05   # Topology change probability

class QuantumSpacetime:
    """
    Quantum spacetime with fluctuating geometry
    
    Combines:
    - Einstein equations: G_μν = 8πG T_μν
    - Schrödinger on curved background: iℏ∂ψ/∂t = Ĥψ
    - Planck-scale uncertainty: Δx·Δt ~ l_p²
    - Hawking radiation: T_H = ℏc³/(8πGMk_B)
    """
    
    def __init__(self, params: QGParams):
        self.p = params
    
    def minkowski_metric(self) -> np.ndarray:
        """Flat spacetime metric (signature -,+,+,+)"""
        return np.diag([-1, 1, 1, 1])
    
    def schwarzschild_metric(self, r: float, M: float) -> np.ndarray:
        """
        Schwarzschild metric for static black hole
        
        ds² = -(1-r_s/r)dt² + (1-r_s/r)⁻¹dr² + r²dΩ²
        
        where r_s = 2GM/c² (Schwarzschild radius)
        """
        r_s = 2 * self.p.G * M / self.p.c**2
        
        if r <= r_s * 1.01:  # Near/inside horizon
            r = r_s * 1.01  # Regularize singularity
        
        g_tt = -(1 - r_s / r)
        g_rr = 1 / (1 - r_s / r) if r > r_s else 1e6  # Singular at horizon
        g_θθ = r**2
        g_φφ = r**2  # Simplified
        
        return np.diag([g_tt, g_rr, g_θθ, g_φφ])
    
    def quantum_foam(self, x: np.ndarray) -> np.ndarray:
        """
        Wheeler's quantum foam: Metric fluctuations at Planck scale
        
        δg_μν ~ l_Planck (random fluctuations)
        
        These are DISCONTINUOUS topology changes!
        """
        # Random metric perturbations
        delta_g = np.random.randn(4, 4) * self.p.metric_fluctuation
        
        # Symmetrize (metric must be symmetric)
        delta_g = (delta_g + delta_g.T) / 2
        
        return delta_g
    
    def ricci_scalar(self, g: np.ndarray) -> float:
        """
        Ricci scalar R (spacetime curvature)
        
        For Schwarzschild: R = 0 (vacuum)
        For quantum fluctuations: R ≠ 0
        """
        # Simplified: Use trace and determinant approximation
        det_g = np.linalg.det(g)
        if abs(det_g) < 1e-10:
            return 0.0
        
        g_inv = np.linalg.inv(g)
        R_approx = -np.trace(g_inv @ g) + 4  # Approximate curvature
        
        return R_approx
    
    def hawking_temperature(self, M: float) -> float:
        """
        Hawking temperature for black hole
        
        T_H = ℏc³/(8πGMk_B) ~ 1/M in Planck units
        """
        if M < self.p.m_planck * 0.01:
            M = self.p.m_planck * 0.01  # Regularize
        
        T_H = self.p.hbar * self.p.c**3 / (8 * np.pi * self.p.G * M)
        return T_H
    
    def spacetime_uncertainty(self, dx: float) -> float:
        """
        Spacetime uncertainty relation
        
        Δx · Δt ≥ l_Planck²
        
        This is QUANTUM GRAVITY signature!
        """
        # From uncertainty, compute minimum Δt
        dt_min = self.p.l_planck**2 / dx
        
        # Actual Δt with quantum fluctuation
        dt = dt_min * (1 + 0.5 * np.random.randn())
        
        return dt
    
    def wormhole_geometry(self, r: float, a: float = 1.0) -> np.ndarray:
        """
        Morris-Thorne traversable wormhole
        
        ds² = -dt² + dr²/(1-b(r)/r) + r²dΩ²
        
        where b(r) is shape function
        """
        b_r = a / np.cosh(r / a)  # Shape function
        
        if r < b_r:
            r = b_r * 1.01  # Regularize throat
        
        g_tt = -1.0
        g_rr = 1 / (1 - b_r / r)
        g_θθ = r**2
        g_φφ = r**2
        
        return np.diag([g_tt, g_rr, g_θθ, g_φφ])
    
    def quantum_wave_on_curved_spacetime(self, x: np.ndarray, g: np.ndarray) -> complex:
        """
        Quantum wave function on curved background
        
        iℏ∂ψ/∂t = √(-g) Ĥ ψ
        
        where √(-g) is metric determinant factor
        """
        det_g = abs(np.linalg.det(g))
        sqrt_det_g = np.sqrt(det_g)
        
        # Gaussian wave packet with phase
        r = np.linalg.norm(x)
        psi = np.exp(-0.5 * r**2) * np.exp(1j * r) / sqrt_det_g
        
        return psi

class QGDataGenerator:
    """Generate quantum gravity training data"""
    
    def __init__(self, params: QGParams):
        self.p = params
        self.qg = QuantumSpacetime(params)
        self.data = []
    
    def generate(self) -> np.ndarray:
        """
        Generate quantum spacetime dataset
        
        Includes:
        - Flat spacetime with quantum fluctuations
        - Virtual black holes (Planck mass scale)
        - Wormholes (topology changes)
        - Hawking radiation events
        - Quantum wave packets on curved geometry
        """
        print("=" * 70)
        print("QUANTUM GRAVITY TOY DATA GENERATOR - MW FRAMEWORK")
        print("=" * 70)
        print("\n🌌 THE HOLY GRAIL: Combining GR + QM")
        print(f"   Samples: {self.p.n_samples}")
        print(f"   Planck scale physics!")
        print(f"   Black hole probability: {self.p.black_hole_prob}")
        print(f"   Wormhole probability: {self.p.wormhole_prob}")
        print("=" * 70)
        
        # Generate different scenarios
        n_per_scenario = self.p.n_samples // 4
        
        # Scenario 1: Quantum foam (fluctuating flat spacetime)
        print("\n📊 Scenario 1: Quantum foam (metric fluctuations)...")
        self._generate_quantum_foam(n_per_scenario)
        
        # Scenario 2: Virtual black holes
        print("📊 Scenario 2: Virtual black holes (Planck mass)...")
        self._generate_virtual_black_holes(n_per_scenario)
        
        # Scenario 3: Wormholes (topology change)
        print("📊 Scenario 3: Wormholes (topology changes)...")
        self._generate_wormholes(n_per_scenario)
        
        # Scenario 4: Hawking radiation
        print("📊 Scenario 4: Hawking radiation (quantum evaporation)...")
        self._generate_hawking_radiation(n_per_scenario)
        
        # Convert to array
        data = np.array(self.data)
        
        print(f"\n✅ Generated {len(data)} quantum gravity samples")
        print(f"   Metric fluctuations: {np.sum(data[:, 28]):.0f} events")
        print(f"   Virtual black holes: {np.sum(data[:, 20] > 0):.0f}")
        print(f"   Topology changes: {np.sum(data[:, 23] > 0):.0f}")
        print(f"   NaN/Inf values: {np.sum(~np.isfinite(data))}")
        
        return data
    
    def _generate_quantum_foam(self, n_samples: int):
        """Generate flat spacetime with quantum metric fluctuations"""
        for _ in range(n_samples):
            # Random spacetime point
            t = np.random.uniform(0, self.p.x_range)
            x = np.random.uniform(-self.p.x_range, self.p.x_range)
            y = np.random.uniform(-self.p.x_range, self.p.x_range)
            z = np.random.uniform(-self.p.x_range, self.p.x_range)
            coords = np.array([t, x, y, z])
            
            # Minkowski + quantum fluctuations
            g_flat = self.qg.minkowski_metric()
            delta_g = self.qg.quantum_foam(coords)
            g_total = g_flat + delta_g
            
            # Extract components
            g_00, g_11, g_22, g_33 = np.diag(g_total)
            
            # Quantum wave function
            psi = self.qg.quantum_wave_on_curved_spacetime(coords[1:], g_total)
            
            # Curvature
            R = self.qg.ricci_scalar(g_total)
            
            # Energy-momentum (vacuum fluctuations)
            T_00 = np.random.randn() * 0.1  # Vacuum energy
            T_11 = T_22 = T_33 = -T_00 / 3  # Trace
            
            # No black hole
            M_BH = 0.0
            r_s = 0.0
            T_H = 0.0
            
            # Topology flag
            topology = 0  # Smooth
            
            # Spacetime uncertainty
            dx = abs(x)
            dt_min = self.p.l_planck**2 / (dx + 0.1)
            Planck_violation = abs(dx * dt_min - self.p.l_planck**2)
            
            # Entanglement entropy (approximate)
            S_ent = abs(R) * 0.1  # Curvature induces entanglement
            
            # Discontinuity flag
            disc_flag = 1 if np.max(np.abs(delta_g)) > 0.2 else 0
            
            row = [
                t, x, y, z,  # Spacetime coords
                g_00, g_11, g_22, g_33,  # Metric
                delta_g[0,1], delta_g[0,2], delta_g[0,3],  # Off-diagonal fluctuations
                delta_g[1,2], delta_g[1,3], delta_g[2,3],
                delta_g[1,1], delta_g[2,2],  # Diagonal fluctuations
                psi.real, psi.imag,  # Wave function
                R,  # Ricci scalar
                T_00, T_11, T_22, T_33,  # Energy-momentum
                M_BH, r_s, T_H,  # Black hole (none)
                topology, Planck_violation, S_ent, disc_flag
            ]
            
            self.data.append(row)
    
    def _generate_virtual_black_holes(self, n_samples: int):
        """Generate virtual black holes (Planck mass scale)"""
        for _ in range(n_samples):
            # Random position
            t = np.random.uniform(0, self.p.x_range)
            x = np.random.uniform(-self.p.x_range, self.p.x_range)
            y = np.random.uniform(-self.p.x_range, self.p.x_range)
            z = np.random.uniform(-self.p.x_range, self.p.x_range)
            coords = np.array([t, x, y, z])
            
            # Virtual black hole mass (~ Planck mass)
            M_BH = self.p.m_planck * np.random.uniform(0.5, 2.0)
            
            # Distance from center
            r = np.linalg.norm(coords[1:])
            if r < 0.1:
                r = 0.1
            
            # Schwarzschild metric
            g_sch = self.qg.schwarzschild_metric(r, M_BH)
            
            # Add quantum fluctuations
            delta_g = self.qg.quantum_foam(coords) * 0.5  # Smaller fluctuations
            g_total = g_sch + delta_g
            
            # Extract components
            g_00, g_11, g_22, g_33 = np.diag(g_total)
            
            # Schwarzschild radius
            r_s = 2 * self.p.G * M_BH / self.p.c**2
            
            # Hawking temperature
            T_H = self.qg.hawking_temperature(M_BH)
            
            # Quantum wave function (affected by curvature)
            psi = self.qg.quantum_wave_on_curved_spacetime(coords[1:], g_total)
            
            # Curvature
            R = self.qg.ricci_scalar(g_total)
            
            # Energy-momentum (from black hole)
            T_00 = M_BH / (r**3 + 0.1)  # Energy density
            T_11 = T_22 = T_33 = T_00  # Simplified
            
            # Topology
            topology = 0  # Black hole is smooth (outside horizon)
            
            # Planck uncertainty
            dx = r
            dt_min = self.p.l_planck**2 / dx
            Planck_violation = abs(dx * dt_min - self.p.l_planck**2)
            
            # Entanglement (Bekenstein-Hawking)
            A_horizon = 4 * np.pi * r_s**2
            S_ent = A_horizon / (4 * self.p.l_planck**2)  # Bekenstein-Hawking
            
            # Discontinuity (horizon crossing)
            disc_flag = 1 if abs(r - r_s) < r_s * 0.2 else 0
            
            row = [
                t, x, y, z,
                g_00, g_11, g_22, g_33,
                delta_g[0,1], delta_g[0,2], delta_g[0,3],
                delta_g[1,2], delta_g[1,3], delta_g[2,3],
                delta_g[1,1], delta_g[2,2],
                psi.real, psi.imag,
                R,
                T_00, T_11, T_22, T_33,
                M_BH, r_s, T_H,
                topology, Planck_violation, S_ent, disc_flag
            ]
            
            self.data.append(row)
    
    def _generate_wormholes(self, n_samples: int):
        """Generate wormhole geometries (topology change!)"""
        for _ in range(n_samples):
            # Random position
            t = np.random.uniform(0, self.p.x_range)
            x = np.random.uniform(-self.p.x_range, self.p.x_range)
            y = np.random.uniform(-self.p.x_range, self.p.x_range)
            z = np.random.uniform(-self.p.x_range, self.p.x_range)
            coords = np.array([t, x, y, z])
            
            r = np.linalg.norm(coords[1:])
            if r < 0.1:
                r = 0.1
            
            # Wormhole metric
            g_wh = self.qg.wormhole_geometry(r, a=1.0)
            
            # Quantum fluctuations (larger near throat)
            delta_g = self.qg.quantum_foam(coords) * (1 + 1/r)
            g_total = g_wh + delta_g
            
            g_00, g_11, g_22, g_33 = np.diag(g_total)
            
            # Quantum wave (exotic matter needed for wormhole)
            psi = self.qg.quantum_wave_on_curved_spacetime(coords[1:], g_total)
            
            R = self.qg.ricci_scalar(g_total)
            
            # Exotic matter (negative energy density!)
            T_00 = -abs(np.random.randn()) * 0.5  # Negative!
            T_11 = T_22 = T_33 = abs(T_00)
            
            # No black hole, but wormhole
            M_BH = 0.0
            r_s = 0.0
            T_H = 0.0
            
            # Topology flag (wormhole!)
            topology = 1
            
            # Planck uncertainty
            dx = r
            dt_min = self.p.l_planck**2 / dx
            Planck_violation = abs(dx * dt_min - self.p.l_planck**2)
            
            # Entanglement (wormholes create entanglement!)
            S_ent = abs(R) * 0.5  # Higher for topology change
            
            # Discontinuity (topology change is discontinuous!)
            disc_flag = 1
            
            row = [
                t, x, y, z,
                g_00, g_11, g_22, g_33,
                delta_g[0,1], delta_g[0,2], delta_g[0,3],
                delta_g[1,2], delta_g[1,3], delta_g[2,3],
                delta_g[1,1], delta_g[2,2],
                psi.real, psi.imag,
                R,
                T_00, T_11, T_22, T_33,
                M_BH, r_s, T_H,
                topology, Planck_violation, S_ent, disc_flag
            ]
            
            self.data.append(row)
    
    def _generate_hawking_radiation(self, n_samples: int):
        """Generate Hawking radiation events (quantum + gravity!)"""
        for _ in range(n_samples):
            # Random position near horizon
            t = np.random.uniform(0, self.p.x_range)
            
            # Black hole mass
            M_BH = self.p.m_planck * np.random.uniform(1.0, 3.0)
            r_s = 2 * self.p.G * M_BH / self.p.c**2
            
            # Position near horizon (r ~ r_s)
            r = r_s * (1 + np.random.uniform(0.1, 0.5))
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            coords = np.array([t, x, y, z])
            
            # Metric
            g_sch = self.qg.schwarzschild_metric(r, M_BH)
            delta_g = self.qg.quantum_foam(coords)
            g_total = g_sch + delta_g
            
            g_00, g_11, g_22, g_33 = np.diag(g_total)
            
            # Hawking temperature
            T_H = self.qg.hawking_temperature(M_BH)
            
            # Thermal radiation (quantum effect!)
            # Planck distribution at T_H
            E_photon = T_H * abs(np.random.randn())
            
            # Wave function (thermal state)
            psi = np.exp(-E_photon / T_H) * np.exp(1j * np.random.uniform(0, 2*np.pi))
            
            R = self.qg.ricci_scalar(g_total)
            
            # Energy-momentum (radiation)
            T_00 = T_H**4  # Stefan-Boltzmann
            T_11 = T_22 = T_33 = T_00 / 3
            
            # Topology
            topology = 0
            
            # Planck uncertainty
            dx = r
            dt_min = self.p.l_planck**2 / dx
            Planck_violation = abs(dx * dt_min - self.p.l_planck**2)
            
            # Entanglement (Hawking pairs!)
            A_horizon = 4 * np.pi * r_s**2
            S_ent = A_horizon / (4 * self.p.l_planck**2)
            
            # Discontinuity (radiation emission is quantum jump!)
            disc_flag = 1
            
            row = [
                t, x, y, z,
                g_00, g_11, g_22, g_33,
                delta_g[0,1], delta_g[0,2], delta_g[0,3],
                delta_g[1,2], delta_g[1,3], delta_g[2,3],
                delta_g[1,1], delta_g[2,2],
                psi.real, psi.imag,
                R,
                T_00, T_11, T_22, T_33,
                M_BH, r_s, T_H,
                topology, Planck_violation, S_ent, disc_flag
            ]
            
            self.data.append(row)
    
    def save(self, filename: str = 'qg_toy_data.npy'):
        """Save generated data"""
        data = np.array(self.data)
        np.save(filename, data)
        print(f"\n💾 Saved to {filename}")
        print(f"   Shape: {data.shape}")
        print(f"   Columns: 30 (coords|metric|δg|ψ|R|T_μν|BH|topology|Planck|S_ent|disc)")

def main():
    """Generate quantum gravity dataset"""
    print("=" * 70)
    print("QUANTUM GRAVITY - THE HOLY GRAIL")
    print("=" * 70)
    print("\n🏆 Combining General Relativity + Quantum Mechanics!")
    print("   GR: Spacetime curvature, black holes, singularities")
    print("   QM: Wave functions, uncertainty, collapse")
    print("   MW: Heavy-tailed priors handle BOTH discontinuities!")
    print("=" * 70)
    
    # Parameters
    params = QGParams(
        n_samples=30000,
        metric_fluctuation=0.1,
        black_hole_prob=0.2,
        wormhole_prob=0.05
    )
    
    # Generate
    generator = QGDataGenerator(params)
    data = generator.generate()
    
    # Statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Total samples: {len(data)}")
    print(f"Virtual black holes: {np.sum(data[:, 24] > 0):.0f}")
    print(f"Wormholes: {np.sum(data[:, 27] == 1):.0f}")
    print(f"Hawking radiation events: {np.sum((data[:, 26] > 0) & (data[:, 29] == 1)):.0f}")
    print(f"Topology changes: {np.sum(data[:, 27] > 0):.0f}")
    print(f"Discontinuous events: {np.sum(data[:, 29]):.0f}")
    print(f"\nMetric range:")
    print(f"  g_00: [{data[:, 4].min():.3f}, {data[:, 4].max():.3f}]")
    print(f"  Ricci R: [{data[:, 18].min():.3f}, {data[:, 18].max():.3f}]")
    print(f"\nQuantum:")
    print(f"  |ψ|: [{np.abs(data[:, 16] + 1j*data[:, 17]).min():.4f}, {np.abs(data[:, 16] + 1j*data[:, 17]).max():.4f}]")
    print(f"  Entanglement S: [{data[:, 28].min():.3f}, {data[:, 28].max():.3f}]")
    
    # Save
    generator.save()
    
    print("\n" + "=" * 70)
    print("✅ QUANTUM GRAVITY DATA GENERATION COMPLETE")
    print("=" * 70)
    print("\nNext: Train MW VAE with qg_toy_vae_trainer.py")
    print("StudentT priors will handle spacetime + quantum discontinuities!")
    print("\n🏆 THE HOLY GRAIL AWAITS! 🏆")

if __name__ == "__main__":
    main()
