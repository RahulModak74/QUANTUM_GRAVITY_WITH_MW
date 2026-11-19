"""
Exact first and second derivatives of the M-W decoder by analytic Taylor propagation.

The v5 decoder is  [Linear -> LayerNorm -> Tanh -> Dropout] x 3 -> Linear.  For a batch of
latent points we carry (value, Jacobian, Hessian) through every layer with the closed-form
chain rule, so f, df/du and d2f/du2 are exact to float64 round-off and cost a few einsums.
(Nested forward-mode autodiff through LayerNorm returned a wrong Hessian in torch 2.13;
nested reverse mode is correct but ~40 ms per pass. This module is checked against
reverse-over-reverse autodiff in mw_exact_geometry.py --selftest.)
"""
import torch
import torch.nn as nn


def _linear(layer, v, J, H):
    W, b = layer.weight, layer.bias
    return v @ W.T + b, torch.einsum('oh,bhn->bon', W, J), torch.einsum('oh,bhnm->bonm', W, H)


def _tanh(v, J, H):
    t = torch.tanh(v); d1 = 1 - t * t; d2 = -2 * t * d1
    return t, d1[..., None] * J, d1[..., None, None] * H + d2[..., None, None] * torch.einsum('bhn,bhm->bhnm', J, J)


def _layernorm(ln, v, J, H):
    c = v - v.mean(-1, keepdim=True)
    Jc = J - J.mean(1, keepdim=True)
    Hc = H - H.mean(1, keepdim=True)
    s = (c * c).mean(-1) + ln.eps                                   # [B]
    Js = 2 * torch.einsum('bh,bhn->bn', c, Jc) / c.shape[1]
    Hs = 2 * (torch.einsum('bhn,bhm->bnm', Jc, Jc) + torch.einsum('bh,bhnm->bnm', c, Hc)) / c.shape[1]
    r = s ** -0.5
    Jr = -0.5 * s[:, None] ** -1.5 * Js
    Hr = 0.75 * s[:, None, None] ** -2.5 * torch.einsum('bn,bm->bnm', Js, Js) - 0.5 * s[:, None, None] ** -1.5 * Hs
    y = c * r[:, None]
    Jy = Jc * r[:, None, None] + c[..., None] * Jr[:, None, :]
    Hy = (Hc * r[:, None, None, None] + torch.einsum('bhn,bm->bhnm', Jc, Jr) + torch.einsum('bn,bhm->bhnm', Jr, Jc)
          + c[..., None, None] * Hr[:, None])
    w, bb = ln.weight, ln.bias
    return y * w + bb, Jy * w[None, :, None], Hy * w[None, :, None, None]


def decoder_jet(vae, Z, active):
    """Z: [B, 20] latent points; active: list of chart axes. Returns f [B,D], J [B,D,n], H [B,D,n,n]
    with derivatives taken w.r.t. the active axes only (others held fixed)."""
    B, n = Z.shape[0], len(active)
    v = Z
    J = torch.zeros(B, Z.shape[1], n, dtype=Z.dtype)
    J[:, active, torch.arange(n)] = 1.0
    H = torch.zeros(B, Z.shape[1], n, n, dtype=Z.dtype)
    for layer in vae.decoder.decoder:
        if isinstance(layer, nn.Linear):
            v, J, H = _linear(layer, v, J, H)
        elif isinstance(layer, nn.LayerNorm):
            v, J, H = _layernorm(layer, v, J, H)
        elif isinstance(layer, nn.Tanh):
            v, J, H = _tanh(v, J, H)
        elif isinstance(layer, nn.Dropout):
            pass                                                      # eval mode: identity
        else:
            raise TypeError(layer)
    return _linear(vae.decoder.fc_out, v, J, H)
