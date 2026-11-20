"""train.py -- one training run (task x method x backbone x seed) with early stopping on the
method's own validation loss; returns metrics and saves test-set predictions."""
from __future__ import annotations
import os
import time, json, os
import numpy as np
from ag import Adam
from nets import BACKBONES
from methods import Method
from metrics import class_metrics, latent_scores

HID = {"MLP": 64, "GRU": 32, "TCN": 32, "TRF": 32}


def init_head(head, M, t):
    """Prior-informed output biases (identical policy for every method)."""
    y = t["y"][t["split"] == 0]
    pri = np.bincount(y, minlength=3) / len(y)
    if M.kind == "cls":
        head.l.b.data[:] = np.log(pri + 1e-3)
    elif M.kind == "cum":
        ge = np.clip(1.0 - np.cumsum(pri)[:-1], 1e-3, 1 - 1e-3)        # P(y >= 1), P(y >= 2)
        lg = np.log(ge / (1 - ge))
        head.b0.data[:] = lg[0]
        head.delta.data[:] = np.log(np.expm1(max(lg[0] - lg[1], 1e-3)))
    else:
        head.s.b.data[:] = np.log(np.expm1(0.9))


def forward(net, head, t, idx):
    return head(net(t["Xseq"][idx], t["Xstat"][idx]))


def eval_loss(M, net, head, t, idx, bs=1024):
    tot, n = 0.0, 0
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        tot += float(M.loss(forward(net, head, t, b), b).data) * len(b)
        n += len(b)
    return tot / n


def raw_outputs(M, net, head, t, idx, bs=1024):
    parts = [M.raw(forward(net, head, t, idx[s:s + bs])) for s in range(0, len(idx), bs)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def run(t, method, backbone, seed, max_epochs=60, patience=8, lr=1e-3, bs=128, wd=1e-4, outdir=None, tag=""):
    rng = np.random.default_rng(seed)
    M = Method(method, t)
    L, F = t["Xseq"].shape[1:]
    S = t["Xstat"].shape[1]
    net = BACKBONES[backbone](L, F, S, HID[backbone], rng)
    head = M.make_head(net.out_dim, rng)
    init_head(head, M, t)
    params = net.params() + head.params()
    opt = Adam(params, lr=lr, wd=wd, clip=5.0)
    tr, va, te = (np.where(t["split"] == i)[0] for i in range(3))
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(max_epochs):
        perm = rng.permutation(tr)
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            loss = M.loss(forward(net, head, t, b), b)
            opt.zero()
            loss.backward()
            opt.step()
        vl = eval_loss(M, net, head, t, va)
        hist.append(vl)
        if vl < best - 1e-5:
            best, best_state, bad = vl, [p.data.copy() for p in params], 0
        else:
            bad += 1
            if bad >= patience:
                break
    for p, d in zip(params, best_state):
        p.data[...] = d
    train_time = time.time() - t0
    fin = M.finalize(raw_outputs(M, net, head, t, va), raw_outputs(M, net, head, t, te), va, te)
    b = t["boundary"]
    res = {"task": t["task"], "scen": t["scen"], "method": method, "backbone": backbone, "seed": seed, "tag": tag,
           "epochs": len(hist), "best_epoch": int(np.argmin(hist)) + 1, "train_time_s": round(train_time, 2),
           "n_params": int(sum(p.data.size for p in params)), "val_loss": best}
    for nm, ix in (("val", va), ("test", te)):
        d = fin[nm]
        res[f"{nm}_obs"] = class_metrics(d["p_obs"], t["y"][ix], b)
        if "y_true" in t:
            res[f"{nm}_true"] = class_metrics(d["p_true"], t["y_true"][ix], b)
            if "mu" in d:
                res[f"{nm}_latent"] = latent_scores(d["mu"][:, 0], d["sig"][:, 0], t["lc_true"][ix, 0], t["sd_lat"][0])
        if "p_obs_ind" in d:
            res[f"{nm}_obs_ind"] = class_metrics(d["p_obs_ind"], t["y"][ix], b)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        stem = f"{t['task']}{tag}_{method}_{backbone}_s{seed}"
        save = {f"{nm}_{k}": v for nm in ("val", "test") for k, v in fin[nm].items() if v is not None}
        for k in ("R_obs", "R_lat"):
            if fin.get(k) is not None:
                save[k] = fin[k]
        np.savez_compressed(os.path.join(outdir, stem + ".npz"), **save)
        with open(os.path.join(outdir, stem + ".json"), "w") as f:
            json.dump(res, f)
    return res
