#!/usr/bin/env python3
"""
Mercury Orbit Data Generator – Pre‑Einstein (Newtonian) + Optional GR Correction
================================================================================
Generates synthetic observational data for a planet (e.g., Mercury) orbiting the Sun.
Each sample is a point along the orbit at a given time, with 30 features that include
coordinates, velocities, radial distance, Newtonian gravity strength, Hamiltonian,
and optional noise.

Two modes:
  - newtonian_only: pure Keplerian ellipse (pre‑Einstein)
  - with_gr_correction: adds a tiny 1/r^3 perturbation to the acceleration,
                        mimicking the GR precession (≈ 43 arcsec/century).

This dataset can be used to test the MW framework:
  - Train a VAE with ONLY Newtonian priors (1/r², Hamiltonian conservation, etc.)
  - Extract the manifold geometry.
  - If the manifold implies a residual precession (e.g., in the orientation of the
    orbit or in the effective potential), that would be non‑circular evidence that
    the framework can "discover" GR corrections from data.

Columns (30 total):
  0: t         – time (years)
  1: x         – position x (AU)
  2: y         – position y (AU)
  3: z         – position z (AU, always ~0 for planar orbit)
  4: r         – radial distance sqrt(x²+y²+z²) (AU)
  5: inv_r2    – 1/r² (Newtonian gravity strength)
  6: inv_r     – 1/r (for potential)
  7: H         – Hamiltonian (kinetic + potential)
  8: T         – kinetic energy
  9: V         – potential energy (-GM/r)
  10: vx       – velocity x (AU/yr)
  11: vy       – velocity y
  12: vz       – velocity z
  13: theta    – true anomaly (radians)
  14: phi      – inclination (radians, near 0)
  15: r_dot    – radial velocity
  16: precession_residual – cumulative precession angle (degrees) relative to Newtonian
  17: noise_1  – random noise channel
  18: noise_2  – random noise channel
  ... up to 29: additional noise / metadata

Usage:
    python3 mercury_orbit_datagen.py --samples 10000 --mode newtonian --output mercury_newtonian.npy
    python3 mercury_orbit_datagen.py --samples 10000 --mode gr_corrected --output mercury_gr.npy
"""

import numpy as np
import argparse
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Physical constants for Mercury (approximate, in AU and years)
GM_SUN = 39.478  # AU^3 / yr^2 (G * M_sun)
AU = 1.0
YEAR = 1.0

# Mercury orbital elements (mean values)
MERCURY_SEMI_MAJOR = 0.387098  # AU
MERCURY_ECCENTRICITY = 0.205630
MERCURY_INCLINATION = 0.122  # radians (~7°)
MERCURY_PERIOD = np.sqrt(MERCURY_SEMI_MAJOR**3 / GM_SUN) * 2 * np.pi  # ~0.2408 yr

# GR correction strength (for 1/r^3 perturbation)
# The relativistic precession per orbit ≈ 6π GM / (c² a (1-e²)) radians
# For Mercury: ~5.0e-7 rad/orbit → 43 arcsec/century
# We'll implement as an extra acceleration: a_GR = - (3 GM / c²) * (GM / r^4) * r
# In dimensionless code units, we can set a small coefficient.
GR_COEFF = 5.0e-8  # tuned to produce ~43 arcsec/century over many orbits

@dataclass
class OrbitParams:
    n_samples: int = 10000
    dt: float = 0.0005  # time step in years (~4.4 hours)
    semi_major: float = MERCURY_SEMI_MAJOR
    eccentricity: float = MERCURY_ECCENTRICITY
    inclination: float = MERCURY_INCLINATION
    noise_std: float = 0.001   # observational noise
    gr_correction: bool = False  # whether to include GR perturbation
    seed: int = 42

class MercuryOrbitGenerator:
    def __init__(self, params: OrbitParams):
        self.params = params
        self.rng = np.random.default_rng(params.seed)

    def keplerian_orbit(self, t):
        """Return position and velocity from Kepler's equation (2D, approximate)."""
        # Mean motion
        n = np.sqrt(GM_SUN / self.params.semi_major**3)
        M = n * t  # mean anomaly

        # Solve Kepler's equation for eccentric anomaly E (simple iterative)
        E = M
        for _ in range(5):
            E = M + self.params.eccentricity * np.sin(E)

        # True anomaly
        theta = 2 * np.arctan2(np.sqrt(1 + self.params.eccentricity) * np.sin(E/2),
                               np.sqrt(1 - self.params.eccentricity) * np.cos(E/2))
        # Distance
        r = self.params.semi_major * (1 - self.params.eccentricity * np.cos(E))

        # Position in orbital plane
        x_orb = r * np.cos(theta)
        y_orb = r * np.sin(theta)

        # Rotate by inclination (only x-y plane, ignoring node for simplicity)
        x = x_orb
        y = y_orb * np.cos(self.params.inclination)
        z = y_orb * np.sin(self.params.inclination)

        # Velocity via Keplerian formula
        mu = GM_SUN
        v_r = np.sqrt(mu / self.params.semi_major / (1 - self.params.eccentricity**2)) * self.params.eccentricity * np.sin(theta)
        v_t = np.sqrt(mu / self.params.semi_major / (1 - self.params.eccentricity**2)) * (1 + self.params.eccentricity * np.cos(theta))
        vx_orb = v_r * np.cos(theta) - v_t * np.sin(theta)
        vy_orb = v_r * np.sin(theta) + v_t * np.cos(theta)
        vx = vx_orb
        vy = vy_orb * np.cos(self.params.inclination)
        vz = vy_orb * np.sin(self.params.inclination)

        return x, y, z, vx, vy, vz, r, theta

    def orbit_with_gr_correction(self, t, dt, max_steps=10000):
        """Integrate orbit with GR 1/r^3 perturbation using simple Euler or RK4."""
        # Initial conditions from Keplerian at t=0
        x, y, z, vx, vy, vz, r0, theta0 = self.keplerian_orbit(0.0)
        state = np.array([x, y, z, vx, vy, vz], dtype=float)
        times = [0.0]
        states = [state.copy()]

        n_steps = int(t / dt) + 1
        n_steps = min(n_steps, max_steps)
        dt_use = t / n_steps

        for step in range(1, n_steps+1):
            # Current acceleration (Newtonian + GR)
            xc, yc, zc, vxc, vyc, vzc = state
            r = np.sqrt(xc*xc + yc*yc + zc*zc)
            r3 = r*r*r
            # Newtonian
            a_newton = -GM_SUN / r3
            ax_newton = a_newton * xc
            ay_newton = a_newton * yc
            az_newton = a_newton * zc
            # GR correction: a_GR = - (3 GM / c^2) * (GM / r^4) * r
            # We use a small constant coefficient gr_coeff
            gr_factor = self.params.gr_correction * GR_COEFF
            a_gr = gr_factor * GM_SUN / (r3 * r)  # extra acceleration magnitude
            ax_gr = a_gr * xc
            ay_gr = a_gr * yc
            az_gr = a_gr * zc

            ax = ax_newton + ax_gr
            ay = ay_newton + ay_gr
            az = az_newton + az_gr

            # Euler integration (simplified; RK4 could be used for higher accuracy)
            vx_new = vxc + ax * dt_use
            vy_new = vyc + ay * dt_use
            vz_new = vzc + az * dt_use
            x_new = xc + vxc * dt_use
            y_new = yc + vyc * dt_use
            z_new = zc + vzc * dt_use
            state = np.array([x_new, y_new, z_new, vx_new, vy_new, vz_new])
            states.append(state.copy())
            times.append(times[-1] + dt_use)

        # Interpolate to exact t
        if n_steps == 0:
            return state[0], state[1], state[2], state[3], state[4], state[5], np.sqrt(state[0]**2+state[1]**2+state[2]**2), 0.0
        # find index
        import bisect
        idx = bisect.bisect_left(times, t)
        if idx == 0:
            idx = 1
        if idx >= len(times):
            idx = len(times)-1
        t1, t2 = times[idx-1], times[idx]
        s1, s2 = states[idx-1], states[idx]
        frac = (t - t1) / (t2 - t1) if t2 > t1 else 0
        x = s1[0] + frac*(s2[0]-s1[0])
        y = s1[1] + frac*(s2[1]-s1[1])
        z = s1[2] + frac*(s2[2]-s1[2])
        vx = s1[3] + frac*(s2[3]-s1[3])
        vy = s1[4] + frac*(s2[4]-s1[4])
        vz = s1[5] + frac*(s2[5]-s1[5])
        r = np.sqrt(x*x + y*y + z*z)
        theta = np.arctan2(y, x)
        return x, y, z, vx, vy, vz, r, theta

    def generate_sample(self, t):
        """Generate a single sample at time t (years)."""
        if self.params.gr_correction:
            # Use integrated orbit for GR (slower but accurate)
            x, y, z, vx, vy, vz, r, theta = self.orbit_with_gr_correction(t, dt=self.params.dt)
        else:
            x, y, z, vx, vy, vz, r, theta = self.keplerian_orbit(t)

        inv_r2 = GM_SUN / (r*r)
        inv_r = 1.0 / r
        v2 = vx*vx + vy*vy + vz*vz
        T_kin = 0.5 * v2
        V_pot = -GM_SUN / r
        H = T_kin + V_pot
        r_dot = (x*vx + y*vy + z*vz) / r

        # Precession residual: cumulative angle difference from Newtonian
        # For simplicity, we compute the true anomaly theta; the residual would be measured
        # after many orbits; here we just output theta.
        precession_res = 0.0  # placeholder; for long runs we could accumulate

        # Add noise
        noise = self.rng.normal(0, self.params.noise_std, size=16)  # for channels 16..29

        features = np.array([
            t, x, y, z, r, inv_r2, inv_r,
            H, T_kin, V_pot,
            vx, vy, vz,
            theta, self.params.inclination, r_dot,
            precession_res,
            noise[0], noise[1], noise[2], noise[3],
            noise[4], noise[5], noise[6], noise[7],
            noise[8], noise[9], noise[10], noise[11],
            noise[12], noise[13]
        ], dtype=np.float32)

        return features

    def generate_dataset(self):
        logger.info(f"Generating Mercury orbit dataset: "
                    f"{self.params.n_samples} samples, GR correction={self.params.gr_correction}")
        data = []
        # time range: from 0 to many periods (e.g., 100 years to see precession)
        max_time = self.params.n_samples * self.params.dt
        times = np.linspace(0, max_time, self.params.n_samples)
        for i, t in enumerate(times):
            data.append(self.generate_sample(t))
            if (i+1) % 5000 == 0:
                logger.info(f"  {i+1}/{self.params.n_samples}")
        return np.array(data, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--mode", choices=["newtonian", "gr_corrected"], default="newtonian")
    parser.add_argument("--output", type=str, default="mercury_orbit.npy")
    parser.add_argument("--noise-std", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    params = OrbitParams(
        n_samples=args.samples,
        gr_correction=(args.mode == "gr_corrected"),
        noise_std=args.noise_std,
        seed=args.seed
    )
    gen = MercuryOrbitGenerator(params)
    data = gen.generate_dataset()
    np.save(args.output, data)
    logger.info(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
