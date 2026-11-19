"""
Kerr v2 (Nov 2026): shared physics and model.

Units G = c = M = 1. Boyer-Lindquist coordinates (t, r, theta, phi).
"""
import numpy as np
import torch
import torch.nn as nn

A_SPIN = 0.9
R_PLUS = 1 + np.sqrt(1 - A_SPIN ** 2)


# ---------------------------------------------------------------- exact Kerr
def metric(r, th, a=A_SPIN):
    """Covariant Kerr metric components (works on numpy or torch)."""
    lib = torch if isinstance(r, torch.Tensor) else np
    s2, c2 = lib.sin(th) ** 2, lib.cos(th) ** 2
    Sig = r ** 2 + a ** 2 * c2
    Del = r ** 2 - 2 * r + a ** 2
    gtt = -(1 - 2 * r / Sig)
    gtp = -2 * a * r * s2 / Sig
    gpp = (r ** 2 + a ** 2 + 2 * a ** 2 * r * s2 / Sig) * s2
    grr = Sig / Del
    gthth = Sig
    return gtt, gtp, gpp, grr, gthth


def inverse_metric(r, th, a=A_SPIN):
    s2, c2 = torch.sin(th) ** 2, torch.cos(th) ** 2
    Sig = r ** 2 + a ** 2 * c2
    Del = r ** 2 - 2 * r + a ** 2
    A = (r ** 2 + a ** 2) ** 2 - a ** 2 * Del * s2
    return (-A / (Sig * Del), -2 * a * r / (Sig * Del), (Del - a ** 2 * s2) / (Sig * Del * s2), Del / Sig, 1 / Sig)


def omega_exact(r, a=A_SPIN):        # prograde circular equatorial (Bardeen, Press & Teukolsky 1972)
    return 1 / (r ** 1.5 + a)


def energy_exact(r, a=A_SPIN):
    return (1 - 2 / r + a / r ** 1.5) / np.sqrt(1 - 3 / r + 2 * a / r ** 1.5)


def isco_exact(a=A_SPIN):
    Z1 = 1 + (1 - a ** 2) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
    Z2 = np.sqrt(3 * a ** 2 + Z1 ** 2)
    return 3 + Z2 - np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))


def circular_from_metric(gtt, gtp, gpp, dgtt, dgtp, dgpp):
    """Prograde circular equatorial orbit from ANY metric and its r-derivatives:
    geodesic condition  d_r g_tt + 2 Omega d_r g_tphi + Omega^2 d_r g_phiphi = 0,
    then E = -(g_tt + g_tphi Omega) u^t,  u^t = 1/sqrt(-(g_tt + 2 g_tphi Omega + g_phiphi Omega^2))."""
    lib = torch if isinstance(gtt, torch.Tensor) else np
    Om = (-dgtp + lib.sqrt(dgtp ** 2 - dgtt * dgpp)) / dgpp
    ut = 1 / lib.sqrt(-(gtt + 2 * gtp * Om + gpp * Om ** 2))
    E = -(gtt + gtp * Om) * ut
    L = (gtp + gpp * Om) * ut
    return Om, E, L


# ---------------------------------------------------------------- MW VAE
class KerrMWVAE(nn.Module):
    """Encoder/decoder over spacetime points x = (r, theta, cos phi, sin phi).
    t is dropped: the Kerr metric is stationary, so t carries no geometry."""

    def __init__(self, latent_dim=8, mean=None, std=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer('mean', torch.zeros(4) if mean is None else mean)
        self.register_buffer('std', torch.ones(4) if std is None else std)
        act = nn.Tanh
        self.enc = nn.Sequential(nn.Linear(4, 128), act(), nn.Linear(128, 128), act(), nn.Linear(128, 64), act())
        self.fc_mu, self.fc_lv = nn.Linear(64, latent_dim), nn.Linear(64, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, 64), act(), nn.Linear(64, 128), act(), nn.Linear(128, 128), act(),
                                 nn.Linear(128, 4))

    @staticmethod
    def feat(x):
        """Internal coordinates: log r (uniform relative precision in r), theta, cos phi, sin phi."""
        return torch.cat([torch.log(x[:, :1]), x[:, 1:]], 1)

    def encode(self, x):
        h = self.enc((self.feat(x) - self.mean) / self.std)
        return self.fc_mu(h), self.fc_lv(h).clamp(-14, 6)

    def decode(self, z):
        o = self.dec(z)
        r = torch.clamp(torch.exp(o[:, 0] * self.std[0] + self.mean[0]), min=1.05 * R_PLUS)
        th = torch.pi * torch.sigmoid(o[:, 1])
        cs = o[:, 2:4] / o[:, 2:4].norm(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.stack([r, th, cs[:, 0], cs[:, 1]], 1)

    def roundtrip(self, x):
        return self.decode(self.encode(x)[0])


def load_model(path, device='cpu'):
    ck = torch.load(path, map_location=device)
    m = KerrMWVAE(ck['latent_dim'], ck['mean'], ck['std']).double()
    m.load_state_dict(ck['state_dict'])
    return m.eval()
