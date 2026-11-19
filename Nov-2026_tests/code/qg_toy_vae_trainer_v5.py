#!/usr/bin/env python3
"""
Quantum Gravity VAE trainer - V5
================================

Same architecture, ELBO and prior set (C1-C6) as v4, with the column-index bug
fixed: every physics prior now reads its column BY NAME from the generator's
column list. In v4, M_BH/T_H/topology/Planck/S_ent were read one index too high,
so C3, C4 and C6 constrained the wrong columns.

Priors that need a column the run has dropped are skipped automatically.
use_priors=False gives a plain VAE (reconstruction + KL only) for comparison.
nu sets the Student-t degrees of freedom of the measurement prior (C6); default 0.8.
"""

import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam


L_PLANCK, G = 1.0, 1.0


# ---- architecture copied verbatim from qg_toy_vae_trainer_v4.py (unchanged) ----
class QGEncoder(nn.Module):
    """Encoder: Quantum spacetime → Latent manifold"""
    def __init__(self, input_dim=30, latent_dim=20, hidden_dims=[384, 256, 192]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.Tanh(),
                nn.Dropout(0.15)  # Higher dropout for stability
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10, 10)
        return mu, logvar

class QGDecoder(nn.Module):
    """Decoder: Latent manifold → Quantum spacetime"""
    def __init__(self, latent_dim=20, output_dim=30, hidden_dims=[192, 256, 384]):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.Tanh(),
                nn.Dropout(0.15)
            ])
            prev_dim = h_dim
        
        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, z):
        h = self.decoder(z)
        return self.fc_out(h)

class QGPhysicsVAE(nn.Module):
    """Physics-Informed VAE for Quantum Gravity"""
    def __init__(self, input_dim=30, latent_dim=20, mean=None, std=None):
        super().__init__()
        self.encoder = QGEncoder(input_dim, latent_dim)
        self.decoder = QGDecoder(latent_dim, input_dim)
        self.latent_dim = latent_dim
        
        self.register_buffer('mean', mean if mean is not None else torch.zeros(input_dim))
        self.register_buffer('std', std if std is not None else torch.ones(input_dim))
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def denormalize(self, x_norm):
        return x_norm * self.std + self.mean
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z



def physics_terms(x, ci):
    """x: denormalised batch; ci: {column name: index}. Returns {prior: per-batch scalar}."""
    out = {}
    col = lambda n: x[:, ci[n]]
    T_tr = -col('T_00') + col('T_11') + col('T_22') + col('T_33')
    out['einstein'] = torch.abs(col('R') - 8 * np.pi * G * T_tr).mean()                    # C1
    out['wavefunction'] = torch.abs(col('psi_re') ** 2 + col('psi_im') ** 2 - 0.1).mean()  # C2 (as v4)
    out['uncertainty'] = torch.relu(L_PLANCK ** 2 - col('dxdt')).mean()                     # C3
    normal = (col('topology') < 0.5).float()
    out['energy'] = (torch.relu(-col('T_00')) * normal).mean()                              # C4
    out['signature'] = (torch.relu(col('g_00')) + torch.relu(-col('g_11'))
                        + torch.relu(-col('g_22')) + torch.relu(-col('g_33'))).mean()       # C5
    out['planck_vals'] = col('topology') + col('disc_flag')                                 # C6
    return out


PRIOR_SCALE = {'einstein': 2.0, 'wavefunction': 1.5, 'uncertainty': 1.0,
               'energy': 1.0, 'signature': 1.0}


def make_model(ci, use_priors, nu=0.8):
    def model(x_norm, vae):
        pyro.module('qg_vae', vae)
        B, dev = x_norm.shape[0], x_norm.device
        with pyro.plate('data', B):
            z = pyro.sample('latent', dist.Normal(torch.zeros(B, vae.latent_dim, device=dev),
                                                  torch.ones(B, vae.latent_dim, device=dev)).to_event(1))
            x_rec = vae.decode(z)
            pyro.sample('obs', dist.Normal(x_rec, 0.1 * torch.ones_like(x_norm)).to_event(1), obs=x_norm)
        if not use_priors:
            return
        ph = physics_terms(vae.denormalize(x_rec), ci)
        zero = torch.tensor(0.0, device=dev)
        for k, s in PRIOR_SCALE.items():
            pyro.sample(f'phys_{k}', dist.Normal(zero, torch.tensor(s, device=dev)), obs=ph[k])
        pyro.sample('phys_planck', dist.StudentT(torch.tensor(float(nu), device=dev), zero,
                                                 torch.tensor(3.0, device=dev)).expand([B]).to_event(1),
                    obs=ph['planck_vals'])
    return model


def guide(x_norm, vae):
    pyro.module('qg_vae', vae)
    with pyro.plate('data', x_norm.shape[0]):
        mu, logvar = vae.encode(x_norm)
        pyro.sample('latent', dist.Normal(mu, torch.exp(0.5 * logvar)).to_event(1))


def train(data, cols, use_priors=True, epochs=150, batch=128, lr=5e-4, latent_dim=20, seed=0, log=print, nu=0.8):
    torch.manual_seed(seed); np.random.seed(seed); pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(np.nan_to_num(data), dtype=torch.float32)
    mean, std = X.mean(0), X.std(0) + 1e-8
    Xn = ((X - mean) / std).to(dev)
    vae = QGPhysicsVAE(len(cols), latent_dim, mean.to(dev), std.to(dev)).to(dev)
    ci = {c: i for i, c in enumerate(cols)}
    svi = SVI(make_model(ci, use_priors, nu), guide,
              ClippedAdam({'lr': lr, 'clip_norm': 10.0, 'weight_decay': 1e-5}), loss=Trace_ELBO())
    n = len(Xn)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = sum(svi.step(Xn[perm[i:i + batch]], vae) for i in range(0, n, batch))
        if (ep + 1) % 25 == 0 or ep == 0:
            log(f'    epoch {ep + 1:4d}  ELBO loss/sample {tot / n:10.2f}')
    vae.eval()
    return vae
