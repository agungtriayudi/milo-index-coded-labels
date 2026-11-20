"""
ag.py -- a minimal reverse-mode automatic differentiation engine on NumPy arrays,
with the layers, losses and optimiser needed by the experiments (no PyTorch available).
All ops are verified by numerical gradient checks in test_ag.py.
"""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr, log_ndtr

DT = np.float64


def _unbroadcast(g, shape):
    """Sum a broadcast gradient back to `shape`."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Tensor:
    __slots__ = ("data", "grad", "_back", "_prev", "req")

    def __init__(self, data, req=False, _prev=(), _back=None):
        self.data = data if isinstance(data, np.ndarray) and data.dtype == DT else np.asarray(data, dtype=DT)
        self.grad = None
        self.req = req
        self._prev = _prev
        self._back = _back

    # ----------------------------------------------------------------- plumbing
    @property
    def shape(self):
        return self.data.shape

    def _acc(self, g):
        if self.grad is None:
            self.grad = np.array(g, dtype=DT, copy=True)
        else:
            self.grad += g

    def backward(self, g=None):
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:                                  # iterative topological sort
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._prev:
                if p.req and id(p) not in seen:
                    stack.append((p, False))
        self.grad = np.ones_like(self.data) if g is None else g
        for node in reversed(topo):
            if node._back is not None and node.grad is not None:
                node._back(node.grad)

    # ----------------------------------------------------------------- arithmetic
    def __add__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(o)
        out = Tensor(self.data + o.data, self.req or o.req, (self, o))
        def back(g):
            if self.req: self._acc(_unbroadcast(g, self.data.shape))
            if o.req: o._acc(_unbroadcast(g, o.data.shape))
        out._back = back
        return out
    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.data, self.req, (self,))
        out._back = lambda g: self._acc(-g)
        return out

    def __sub__(self, o):
        return self + (-(o if isinstance(o, Tensor) else Tensor(o)))

    def __rsub__(self, o):
        return Tensor(o) + (-self)

    def __mul__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(o)
        out = Tensor(self.data * o.data, self.req or o.req, (self, o))
        def back(g):
            if self.req: self._acc(_unbroadcast(g * o.data, self.data.shape))
            if o.req: o._acc(_unbroadcast(g * self.data, o.data.shape))
        out._back = back
        return out
    __rmul__ = __mul__

    def __truediv__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(o)
        out = Tensor(self.data / o.data, self.req or o.req, (self, o))
        def back(g):
            if self.req: self._acc(_unbroadcast(g / o.data, self.data.shape))
            if o.req: o._acc(_unbroadcast(-g * self.data / o.data ** 2, o.data.shape))
        out._back = back
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float))
        out = Tensor(self.data ** p, self.req, (self,))
        out._back = lambda g: self._acc(g * p * self.data ** (p - 1))
        return out

    def __matmul__(self, o):
        out = Tensor(self.data @ o.data, self.req or o.req, (self, o))
        def back(g):
            if self.req:
                self._acc(_unbroadcast(g @ np.swapaxes(o.data, -1, -2), self.data.shape))
            if o.req:
                o._acc(_unbroadcast(np.swapaxes(self.data, -1, -2) @ g, o.data.shape))
        out._back = back
        return out

    # ----------------------------------------------------------------- shape ops
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.req, (self,))
        def back(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.broadcast_to(g, self.data.shape))
        out._back = back
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod([self.data.shape[a] for a in np.atleast_1d(axis)])
        return self.sum(axis, keepdims) * (1.0 / n)

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), self.req, (self,))
        out._back = lambda g: self._acc(g.reshape(self.data.shape))
        return out

    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), self.req, (self,))
        inv = np.argsort(axes)
        out._back = lambda g: self._acc(g.transpose(*inv))
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], self.req, (self,))
        def back(g):
            full = np.zeros_like(self.data)
            np.add.at(full, idx, g)
            self._acc(full)
        out._back = back
        return out


def concat(ts, axis=-1):
    out = Tensor(np.concatenate([t.data for t in ts], axis=axis), any(t.req for t in ts), tuple(ts))
    sizes = [t.data.shape[axis] for t in ts]
    def back(g):
        parts = np.split(g, np.cumsum(sizes)[:-1], axis=axis)
        for t, p in zip(ts, parts):
            if t.req:
                t._acc(p)
    out._back = back
    return out


# --------------------------------------------------------------------- elementwise
def _unary(x, f, df):
    y = f(x.data)
    out = Tensor(y, x.req, (x,))
    out._back = lambda g: x._acc(g * df(x.data, y))
    return out

def exp(x): return _unary(x, np.exp, lambda a, y: y)
def log(x): return _unary(x, np.log, lambda a, y: 1.0 / a)
def tanh(x): return _unary(x, np.tanh, lambda a, y: 1.0 - y * y)
def sigmoid(x):
    return _unary(x, lambda a: 0.5 * (1 + np.tanh(0.5 * a)), lambda a, y: y * (1 - y))
def relu(x): return _unary(x, lambda a: np.maximum(a, 0), lambda a, y: (a > 0).astype(DT))
def softplus(x):
    return _unary(x, lambda a: np.logaddexp(0, a), lambda a, y: 0.5 * (1 + np.tanh(0.5 * a)))


def softmax(x, axis=-1):
    z = x.data - x.data.max(axis=axis, keepdims=True)
    e = np.exp(z)
    y = e / e.sum(axis=axis, keepdims=True)
    out = Tensor(y, x.req, (x,))
    out._back = lambda g: x._acc(y * (g - (g * y).sum(axis=axis, keepdims=True)))
    return out


def log_softmax(x, axis=-1):
    z = x.data - x.data.max(axis=axis, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=axis, keepdims=True))
    y = z - lse
    sm = np.exp(y)
    out = Tensor(y, x.req, (x,))
    out._back = lambda g: x._acc(g - sm * g.sum(axis=axis, keepdims=True))
    return out


def layernorm(x, gamma, beta, eps=1e-5):
    mu = x.data.mean(-1, keepdims=True)
    xc = x.data - mu
    var = (xc ** 2).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xh = xc * inv
    out = Tensor(xh * gamma.data + beta.data, x.req or gamma.req or beta.req, (x, gamma, beta))
    def back(g):
        if gamma.req: gamma._acc(_unbroadcast(g * xh, gamma.data.shape))
        if beta.req: beta._acc(_unbroadcast(g, beta.data.shape))
        if x.req:
            gh = g * gamma.data
            n = x.data.shape[-1]
            x._acc(inv / n * (n * gh - gh.sum(-1, keepdims=True) - xh * (gh * xh).sum(-1, keepdims=True)))
    out._back = back
    return out


# --------------------------------------------------------------------- probabilistic losses
_LOG2PI = np.log(2 * np.pi)

def _log_interval(a, b):
    """log(Phi(b) - Phi(a)) elementwise, stable; a < b, may be +-inf."""
    out = np.empty(np.broadcast(a, b).shape)
    a = np.broadcast_to(a, out.shape); b = np.broadcast_to(b, out.shape)
    m1 = b <= 0
    m2 = (a >= 0) & ~m1
    m3 = ~(m1 | m2)
    if m1.any():
        lb, la = log_ndtr(b[m1]), log_ndtr(a[m1])
        out[m1] = lb + np.log1p(-np.exp(np.minimum(la - lb, -1e-300)))
    if m2.any():
        lb, la = log_ndtr(-a[m2]), log_ndtr(-b[m2])
        out[m2] = lb + np.log1p(-np.exp(np.minimum(la - lb, -1e-300)))
    if m3.any():
        out[m3] = np.log(np.maximum(ndtr(b[m3]) - ndtr(a[m3]), 1e-300))
    return out


def censored_gauss_nll(mu, tau, lo, hi, w):
    """Negative log-likelihood that a Gaussian N(mu, tau^2) falls in [lo, hi] (lo/hi may be +-inf).
    mu, tau: Tensors (B, P); lo, hi, w: arrays (B, P); w is a 0/1 mask. Returns the mean over
    observed entries."""
    a = (lo - mu.data) / tau.data
    b = (hi - mu.data) / tau.data
    lp = _log_interval(a, b)
    nobs = max(w.sum(), 1.0)
    val = -(lp * w).sum() / nobs
    out = Tensor(val, mu.req or tau.req, (mu, tau))
    def back(g):
        lpa = np.where(np.isfinite(a), -0.5 * a * a - 0.5 * _LOG2PI, -np.inf)
        lpb = np.where(np.isfinite(b), -0.5 * b * b - 0.5 * _LOG2PI, -np.inf)
        ra = np.exp(lpa - lp); rb = np.exp(lpb - lp)                  # phi(a)/P, phi(b)/P
        dmu = (ra - rb) / tau.data                                    # d logP / d mu
        aa = np.where(np.isfinite(a), a, 0.0) * ra; bb = np.where(np.isfinite(b), b, 0.0) * rb
        dtau = (aa - bb) / tau.data                                   # d logP / d tau
        coef = -g * w / nobs
        if mu.req: mu._acc(coef * dmu)
        if tau.req: tau._acc(coef * dtau)
    out._back = back
    return out


# --------------------------------------------------------------------- parameters & optimiser
class Param(Tensor):
    def __init__(self, data):
        super().__init__(np.asarray(data, dtype=DT), req=True)


class Module:
    def params(self):
        ps = []
        for v in self.__dict__.values():
            if isinstance(v, Param):
                ps.append(v)
            elif isinstance(v, Module):
                ps.extend(v.params())
            elif isinstance(v, (list, tuple)):
                for u in v:
                    if isinstance(u, Module): ps.extend(u.params())
                    elif isinstance(u, Param): ps.append(u)
        return ps

    def state(self):
        return [p.data.copy() for p in self.params()]

    def load(self, st):
        for p, d in zip(self.params(), st):
            p.data[...] = d


class Linear(Module):
    def __init__(self, nin, nout, rng, bias=True):
        lim = np.sqrt(6.0 / (nin + nout))
        self.W = Param(rng.uniform(-lim, lim, (nin, nout)))
        self.b = Param(np.zeros(nout)) if bias else None

    def __call__(self, x):
        y = x @ self.W
        return y + self.b if self.b is not None else y


class Adam:
    def __init__(self, params, lr=1e-3, wd=1e-4, clip=5.0, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.wd, self.clip, self.b1, self.b2, self.eps = params, lr, wd, clip, b1, b2, eps
        self.m = [np.zeros_like(q.data) for q in params]
        self.v = [np.zeros_like(q.data) for q in params]
        self.t = 0

    def step(self):
        self.t += 1
        gs = [q.grad if q.grad is not None else np.zeros_like(q.data) for q in self.p]
        norm = np.sqrt(sum((g * g).sum() for g in gs))
        sc = min(1.0, self.clip / (norm + 1e-12))
        for i, (q, g) in enumerate(zip(self.p, gs)):
            g = g * sc
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            q.data -= self.lr * (mh / (np.sqrt(vh) + self.eps) + self.wd * q.data)   # AdamW-style decay

    def zero(self):
        for q in self.p:
            q.grad = None
