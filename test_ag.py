"""Numerical gradient checks for every op, layer, head and loss used in the experiments."""
import numpy as np
from ag import (Tensor, Param, concat, exp, log, tanh, sigmoid, relu, softplus, softmax, log_softmax,
                layernorm, censored_gauss_nll)
from nets import gru_seq, MLPNet, GRUNet, TCNNet, TransNet, CumHead, LatentHead

rng = np.random.default_rng(0)
worst = {}


def check(name, f, params, n=12, eps=1e-6):
    for p in params:
        p.grad = None
    out = f()
    out.backward()
    err = 0.0
    gscale = max(1.0, max(float(np.abs(p.grad).max()) for p in params if p.grad is not None))
    for p in params:
        g = p.grad if p.grad is not None else np.zeros_like(p.data)
        flat = p.data.reshape(-1)
        idx = rng.choice(flat.size, min(n, flat.size), replace=False)
        for i in idx:
            old = flat[i]
            flat[i] = old + eps; fp = f().data
            flat[i] = old - eps; fm = f().data
            flat[i] = old
            num = (fp - fm) / (2 * eps)
            ana = g.reshape(-1)[i]
            err = max(err, abs(num - ana) / max(gscale * 1e-3, abs(num) + abs(ana)))
    worst[name] = err
    print(f"{name:34s} max rel err {err:.2e}")


A = Param(rng.standard_normal((4, 3, 5)))
B = Param(rng.standard_normal((5, 2)))
C = Param(rng.standard_normal((3, 1)))
D = Param(rng.uniform(0.5, 2.0, (4, 3, 5)))
check("add/mul/sub/div bcast", lambda: (((A * D) - C.reshape(1, 3, 1) + 2.0) / (D + C.reshape(1, 3, 1) ** 2 + 1.0)).sum(), [A, C, D])
check("pow/neg/rsub", lambda: ((1.0 - D) ** 3 + (-(D ** 0.5))).sum(), [D])
check("batched matmul", lambda: ((A @ B) ** 2).mean(), [A, B])
check("transpose/reshape", lambda: ((A.transpose(0, 2, 1).reshape(4, 15) @ Tensor(rng.standard_normal((15, 2)) if False else np.ones((15, 2)))) ** 2).sum(), [A])
check("getitem/concat", lambda: (concat([A[:, 1:, :], A[:, :2, 2:4] * 3.0], 2) ** 2).sum(), [A])
check("sum/mean axes", lambda: (A.sum(1) * A.mean((0, 2), keepdims=True).reshape(1, 3)[:, :1]).sum() + (A.mean(2) ** 2).sum(), [A])
check("exp/log/tanh/sigmoid", lambda: (exp(tanh(A)) + log(D) * sigmoid(A)).sum(), [A, D])
check("relu/softplus", lambda: (relu(A) * softplus(A * 2.0)).sum(), [A])
check("softmax", lambda: (softmax(A, -1) * Tensor(rng.standard_normal((4, 3, 5)) * 0 + np.arange(5.))).sum(), [A])
check("log_softmax", lambda: (log_softmax(A, -1) * Tensor(np.linspace(0, 1, 60).reshape(4, 3, 5))).sum(), [A])
g, b = Param(rng.uniform(0.5, 1.5, 5)), Param(rng.standard_normal(5))
check("layernorm", lambda: (layernorm(A, g, b) * Tensor(np.linspace(-1, 1, 60).reshape(4, 3, 5))).sum(), [A, g, b])

# censored, noise-convolved Gaussian NLL: regular, one-sided, far-tail and masked cases
mu = Param(np.array([[0.0, 1.0, -2.0], [3.0, -6.0, 0.4], [0.1, 9.0, -1.0]]))
sg = Param(np.array([[0.5, 1.0, 0.3], [0.2, 0.7, 1.5], [0.05, 0.4, 0.9]]))
lo = np.array([[-0.2, -np.inf, 1.0], [-1.0, 0.0, 0.3], [0.0, -np.inf, -1.2]])
hi = np.array([[0.3, 0.5, 1.5], [-0.5, 0.2, 0.5], [0.01, 1.0, np.inf]])
w = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 1.]])
check("censored NLL (tau = sqrt(s2+.1))", lambda: censored_gauss_nll(mu, (sg ** 2 + 0.1) ** 0.5, lo, hi, w), [mu, sg], n=9)

gi = Param(rng.standard_normal((3, 5, 12)) * 0.8)
Wh, bh = Param(rng.standard_normal((4, 12)) * 0.5), Param(rng.standard_normal(12) * 0.1)
cw = Tensor(rng.standard_normal((3, 5, 4)))
check("fused GRU (all states)", lambda: (gru_seq(gi, Wh, bh) * cw).sum(), [gi, Wh, bh], n=20)

xs = rng.standard_normal((6, 14, 4)); xt = rng.standard_normal((6, 3))
for nm, cls in [("MLP", MLPNet), ("GRU", GRUNet), ("TCN", TCNNet), ("TRF", TransNet)]:
    net = cls(14, 4, 3, 8, rng)
    head = LatentHead(net.out_dim, 2, rng)
    tgt = rng.standard_normal((6, 2))
    def f(net=net, head=head, tgt=tgt):
        m, s = head(net(xs, xt))
        return (((m - tgt) ** 2) * 0.5 / (s ** 2) + log(s)).mean()
    check(f"{nm} backbone + latent head", f, net.params() + head.params(), n=6)
ch = CumHead(5, 4, rng)
hh = Param(rng.standard_normal((6, 5)))
check("cumulative (CORAL) head", lambda: (sigmoid(ch(hh)) * Tensor(np.arange(18.).reshape(6, 3))).sum(), ch.params() + [hh])
print("WORST", max(worst.values()))
assert max(worst.values()) < 1e-4
