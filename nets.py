"""
nets.py -- backbones (MLP, GRU, TCN, Transformer encoder) and output heads, built on ag.py.
The GRU recurrence is a fused op with a hand-written BPTT backward (checked in test_ag.py).
"""
from __future__ import annotations
import numpy as np
from ag import (Tensor, Param, Module, Linear, concat, relu, softplus, softmax, layernorm, DT)


def _sig(a):
    return 0.5 * (1.0 + np.tanh(0.5 * a))


def gru_seq(gi, Wh, bh):
    """Fused GRU recurrence, PyTorch gate convention (r, z, n), h0 = 0.
    gi: Tensor (B, T, 3H) = x W_i + b_i ; Wh: Param (H, 3H) ; bh: Param (3H,). Returns (B, T, H)."""
    B, T, H3 = gi.data.shape
    H = H3 // 3
    W, b = Wh.data, bh.data
    h = np.zeros((B, H))
    hs = np.empty((B, T, H))
    cache = []
    for t in range(T):
        gh = h @ W + b
        g = gi.data[:, t]
        r = _sig(g[:, :H] + gh[:, :H])
        z = _sig(g[:, H:2 * H] + gh[:, H:2 * H])
        ghn = gh[:, 2 * H:]
        n = np.tanh(g[:, 2 * H:] + r * ghn)
        cache.append((h, r, z, n, ghn))
        h = (1.0 - z) * n + z * h
        hs[:, t] = h
    out = Tensor(hs, gi.req or Wh.req or bh.req, (gi, Wh, bh))

    def back(gout):
        dgi = np.zeros_like(gi.data)
        dW = np.zeros_like(W)
        db = np.zeros_like(b)
        dh_next = np.zeros((B, H))
        for t in range(T - 1, -1, -1):
            hp, r, z, n, ghn = cache[t]
            dh = gout[:, t] + dh_next
            dn = dh * (1.0 - z)
            dz = dh * (hp - n)
            dan = dn * (1.0 - n * n)
            dar = dan * ghn * r * (1.0 - r)
            daz = dz * z * (1.0 - z)
            dgh = np.concatenate([dar, daz, dan * r], axis=1)
            dgi[:, t] = np.concatenate([dar, daz, dan], axis=1)
            dW += hp.T @ dgh
            db += dgh.sum(0)
            dh_next = dh * z + dgh @ W.T
        if gi.req: gi._acc(dgi)
        if Wh.req: Wh._acc(dW)
        if bh.req: bh._acc(db)
    out._back = back
    return out


# ------------------------------------------------------------------------------ backbones
class MLPNet(Module):
    def __init__(self, T, F, S, hid, rng):
        self.l1 = Linear(T * F + S, hid, rng)
        self.l2 = Linear(hid, hid, rng)
        self.out_dim = hid

    def __call__(self, xs, xt):
        h = Tensor(np.concatenate([xs.reshape(xs.shape[0], -1), xt], 1))
        return relu(self.l2(relu(self.l1(h))))


class GRUNet(Module):
    def __init__(self, T, F, S, hid, rng):
        self.inp = Linear(F, 3 * hid, rng)
        q, _ = np.linalg.qr(rng.standard_normal((3 * hid, hid)))
        self.Wh = Param(q.T.copy())                              # (H, 3H), orthogonal rows
        self.bh = Param(np.zeros(3 * hid))
        self.out = Linear(hid + S, hid, rng)
        self.out_dim = hid

    def __call__(self, xs, xt):
        hs = gru_seq(self.inp(Tensor(xs)), self.Wh, self.bh)
        return relu(self.out(concat([hs[:, -1, :], Tensor(xt)], 1)))


class CausalConv(Module):
    def __init__(self, c, rng, k, d):
        self.k, self.d = k, d
        self.taps = [Linear(c, c, rng, bias=(j == 0)) for j in range(k)]

    def __call__(self, x):
        B, T, C = x.shape
        pad = (self.k - 1) * self.d
        xp = concat([Tensor(np.zeros((B, pad, C))), x], 1)
        y = None
        for j, tap in enumerate(self.taps):
            s = pad - j * self.d                                 # x[t - j d]
            yj = tap(xp[:, s:s + T, :])
            y = yj if y is None else y + yj
        return y


class TCNNet(Module):
    def __init__(self, T, F, S, hid, rng, k=3, dil=(1, 2, 4)):
        self.inp = Linear(F, hid, rng)
        self.blocks = [CausalConv(hid, rng, k, d) for d in dil]
        self.out = Linear(hid + S, hid, rng)
        self.out_dim = hid

    def __call__(self, xs, xt):
        h = self.inp(Tensor(xs))
        for blk in self.blocks:
            h = relu(blk(h)) + h
        return relu(self.out(concat([h[:, -1, :], Tensor(xt)], 1)))


class TransNet(Module):
    def __init__(self, T, F, S, d, rng, heads=2, ff=64):
        self.inp = Linear(F, d, rng)
        self.pos = Param(rng.normal(0, 0.1, (T, d)))
        self.q, self.k, self.v, self.o = (Linear(d, d, rng) for _ in range(4))
        self.g1, self.b1 = Param(np.ones(d)), Param(np.zeros(d))
        self.f1, self.f2 = Linear(d, ff, rng), Linear(ff, d, rng)
        self.g2, self.b2 = Param(np.ones(d)), Param(np.zeros(d))
        self.out = Linear(d + S, d, rng)
        self.h, self.d, self.out_dim = heads, d, d

    def __call__(self, xs, xt):
        B, T, _ = xs.shape
        h, d = self.h, self.d
        dh = d // h
        x = self.inp(Tensor(xs)) + self.pos
        Q = self.q(x).reshape(B, T, h, dh).transpose(0, 2, 1, 3)
        Kt = self.k(x).reshape(B, T, h, dh).transpose(0, 2, 3, 1)
        V = self.v(x).reshape(B, T, h, dh).transpose(0, 2, 1, 3)
        A = softmax((Q @ Kt) * (1.0 / np.sqrt(dh)), axis=-1)
        att = (A @ V).transpose(0, 2, 1, 3).reshape(B, T, d)
        x = layernorm(x + self.o(att), self.g1, self.b1)
        x = layernorm(x + self.f2(relu(self.f1(x))), self.g2, self.b2)
        return relu(self.out(concat([x[:, -1, :], Tensor(xt)], 1)))


BACKBONES = {"MLP": MLPNet, "GRU": GRUNet, "TCN": TCNNet, "TRF": TransNet}


# ------------------------------------------------------------------------------ heads
class ClsHead(Module):
    def __init__(self, hid, K, rng):
        self.l = Linear(hid, K, rng)

    def __call__(self, h):
        return self.l(h)                                         # logits (B, K)


class CumHead(Module):
    """Rank-consistent cumulative-logit head (CORAL-type, monotone thresholds by construction)."""
    def __init__(self, hid, K, rng):
        self.w = Linear(hid, 1, rng, bias=False)
        self.b0 = Param(np.array([0.5]))
        self.delta = Param(np.full((K - 2, 1), 0.5))
        L = np.zeros((K - 1, K - 2))
        for k in range(K - 1):
            L[k, :k] = 1.0
        self.L, self.K = L, K

    def __call__(self, h):
        b = (Tensor(self.L) @ softplus(self.delta)).reshape(1, self.K - 1)
        return self.w(h) + self.b0 - b                           # logits of P(y >= k), k = 1..K-1


class LatentHead(Module):
    """Index-decoding head: per-pollutant latent TRUE log-concentration N(mu, sigma^2) (standardised)."""
    def __init__(self, hid, P, rng, s_min=0.02):
        self.m = Linear(hid, P, rng)
        self.s = Linear(hid, P, rng)
        self.s.b.data[:] = np.log(np.expm1(0.6))
        self.s_min = s_min

    def __call__(self, h):
        return self.m(h), softplus(self.s(h)) + self.s_min
