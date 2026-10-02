"""Auditable CPU revision experiments; original files are read-only inputs.

Main selection is final prescribed epoch. Oracle is diagnostic only.
Outputs are new experiments, not replacements for submitted numbers.
"""
import argparse
import csv
import hashlib
import json
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.special import roots_jacobi
import torch
from torch import nn

OUT = Path(__file__).resolve().parent / "results"
LAM = 0.1
SEEDS = [42, 123, 777, 2024, 9999]


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float64)


def g(u, kind):
    if kind == "square": return u * u
    if kind == "linear": return u
    if kind == "sin": return torch.sin(u) if torch.is_tensor(u) else np.sin(u)
    if kind == "tanh": return torch.tanh(u) if torch.is_tensor(u) else np.tanh(u)
    raise ValueError(kind)


class PINN(nn.Module):
    def __init__(self, width=256, depth=3):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1): layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def weights(x, nodes, alpha, method):
    x = np.asarray(x).reshape(-1, 1)
    a, b = nodes[:-1], nodes[1:]
    mids = (a + b) / 2
    if method in ("EP-product", "MP-product"):
        beta = 1 - alpha
        primitive = lambda t: np.sign(t) * np.abs(t)**beta / beta
        w = primitive(b - x) - primitive(a - x)
        samples = a if method == "EP-product" else mids
    elif method == "MP-raw":
        d = np.abs(x - mids)
        w = np.zeros_like(d)
        np.power(d, -alpha, out=w, where=d >= 1e-14)
        w *= b - a
        samples = mids if method == "MP-raw" else a
    elif method == "EP-zero":
        d = np.abs(x - nodes)
        w = np.zeros_like(d)
        np.power(d, -alpha, out=w, where=d >= 1e-14)
        q = np.r_[np.diff(nodes)[0]/2, (nodes[2:]-nodes[:-2])/2, np.diff(nodes)[-1]/2]
        w *= q
        samples = nodes
    else: raise ValueError(method)
    return LAM * w, samples


def independent_integral(x, fn, alpha, order=64):
    """Split at x; integrate t^-alpha via Jacobi (0,-alpha).

    s=x +/- L*t; factor L^(1-alpha). No singular samples or cutoffs.
    fn accepts a numpy array and returns its pointwise regular factor.
    """
    z, w = roots_jacobi(order, 0, -alpha)
    t, w = (z + 1)/2, w / 2**(1-alpha)
    x = np.asarray(x).reshape(-1)
    left = x[:, None] * (1-t)
    right = x[:, None] + (1-x[:, None])*t
    return x**(1-alpha)*(fn(left) @ w) + (1-x)**(1-alpha)*(fn(right) @ w)


def exact_source(x, alpha, kind, order=128):
    x = np.asarray(x)
    return np.sin(np.pi*x) - LAM*independent_integral(x, lambda s: g(np.sin(np.pi*s), kind), alpha, order)


def predict(model, x):
    shape = np.asarray(x).shape
    flat = np.asarray(x).reshape(-1)
    with torch.no_grad():
        y = [model(tensor(flat[i:i+2048, None])).numpy().ravel() for i in range(0, len(flat), 2048)]
    return np.concatenate(y).reshape(shape)


def evaluate(model, c, nodes, source):
    alpha, kind, method = c["alpha"], c["g"], c["method"]
    # Full-interval solution metrics include both endpoints.
    grid = np.linspace(0, 1, 401)
    p = predict(model, grid)
    truth = np.sin(np.pi*grid)
    err = p - truth
    interior = np.linspace(.001, .999, 400)
    ei = predict(model, interior) - np.sin(np.pi*interior)
    # Fixed independent residual grid: use noncoincident cell-independent points.
    xr = np.r_[0., (np.arange(100)+.38196601125)/100, 1.]
    factor = lambda s: g(predict(model, s), kind)
    integrals = {n: independent_integral(xr, factor, alpha, n) for n in (32, 64, 128)}
    f = exact_source(xr, alpha, kind)
    w, samples = weights(xr, nodes, alpha, method)
    discrete = w @ g(predict(model, samples), kind)
    continuous = LAM*integrals[128]
    pred = predict(model, xr)
    rc = pred - f - continuous
    rd = pred - f - discrete
    defect = continuous - discrete
    source_error = source(xr) - f
    wn, sn = weights(nodes, nodes, alpha, method)
    rn = predict(model, nodes) - source(nodes) - wn @ g(predict(model, sn), kind)
    # Distinguish unscaled operator convergence and residual-scaled convergence.
    dq = float(LAM*np.max(np.abs(integrals[128]-integrals[64])))
    metrics = dict(rel_l2_full=float(np.linalg.norm(err)/np.linalg.norm(truth)),
                   max_error_full=float(np.max(np.abs(err))), endpoint_0=float(abs(err[0])), endpoint_1=float(abs(err[-1])),
                   rel_l2_legacy_interior=float(np.linalg.norm(ei)/np.linalg.norm(np.sin(np.pi*interior))),
                   residual_continuous_sampled_max=float(np.max(np.abs(rc))),
                   residual_discrete_same_grid_sampled_max=float(np.max(np.abs(rd))),
                   quadrature_defect_sampled_max=float(np.max(np.abs(defect))),
                   training_residual_max=float(np.max(np.abs(rn))),
                   source_error_sampled_max=float(np.max(np.abs(source_error))),
                   independent_quad_delta32_64=float(LAM*np.max(np.abs(integrals[64]-integrals[32]))),
                   independent_quad_delta64_128=dq,
                   independent_quad_converged=bool(dq < 1e-8),
                   residual_identity_error=float(np.max(np.abs(rd-rc-defect))))
    arrays = dict(x=xr.tolist(), continuous_residual=rc.tolist(), discrete_residual=rd.tolist(),
                  quadrature_defect=defect.tolist(), source_error=source_error.tolist())
    return metrics, arrays


def manifest():
    items = []
    for seed in SEEDS:
        for kind in ("square", "sin"):
            for alpha in (.25, .5, .75):
                for method in ("EP-product", "MP-raw", "MP-product"):
                    items.append(dict(g=kind, alpha=alpha, method=method, seed=seed, source="direct-jacobi128"))
    for c in items:
        for k, v in dict(suite="core", mesh="uniform", loss="nodal", width=256, depth=3, epochs=2500).items(): c.setdefault(k, v)
        c["id"] = f'core_{c["g"]}_a{c["alpha"]}_{c["method"]}_s{c["seed"]}_{c["mesh"]}_{c["loss"]}'
    return items


def run(c, source):
    dest = OUT / c["id"]
    dest.mkdir(parents=True, exist_ok=True)
    nodes = np.linspace(0, 1, 50)
    t0 = time.perf_counter()
    w, samples = weights(nodes, nodes, c["alpha"], c["method"])
    assembly = time.perf_counter()-t0
    x, s, w = tensor(nodes[:, None]), tensor(samples[:, None]), tensor(w)
    f, truth = tensor(source(nodes)[:, None]), tensor(np.sin(np.pi*nodes)[:, None])
    spatial = np.r_[np.diff(nodes)[0]/2, (nodes[2:]-nodes[:-2])/2, np.diff(nodes)[-1]/2]
    lw = tensor(spatial[:, None]) if c["loss"] == "space-weighted" else torch.ones_like(x)/len(x)
    torch.manual_seed(c["seed"])
    np.random.seed(c["seed"])
    model = PINN(c["width"], c["depth"]).double()
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["epochs"], eta_min=1e-7)
    best, best_epoch, state = float("inf"), None, None
    history = []
    t0 = time.perf_counter()
    for epoch in range(c["epochs"]):
        opt.zero_grad(set_to_none=True)
        p = model(x)
        # Avoid a second network pass where factor samples are endpoints.
        ps = p[:-1] if c["method"] == "EP-product" else model(s)
        r = p - f - w @ g(ps, c["g"])
        loss = torch.sum(lw * r.square())
        error = float((p.detach()-truth).abs().max())
        if error < best:
            best, best_epoch = error, epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 100 == 0: history.append(dict(completed_updates=epoch, loss=float(loss.detach()), oracle_nodal_error=error))
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        sch.step()
    elapsed = time.perf_counter()-t0
    final_state = {k: v.detach().clone() for k,v in model.state_dict().items()}
    with torch.no_grad(): final_error = float((model(x)-truth).abs().max())
    if final_error < best: best, best_epoch, state = final_error, c["epochs"], final_state
    torch.save(dict(config=c, completed_updates=c["epochs"], state_dict=final_state), dest / "final.pt")
    torch.save(dict(config=c, completed_updates=best_epoch, state_dict=state), dest / "oracle.pt")
    results = dict(config=c, machine=socket.gethostname(), device="cpu", pytorch=torch.__version__,
                   nodes=nodes.tolist(), assembly_seconds=assembly, training_seconds=elapsed,
                   oracle_epoch=best_epoch, oracle_nodal_error=best, history=history)
    for label, st in (("final", final_state), ("oracle", state)):
        model.load_state_dict(st)
        metrics, arrays = evaluate(model, c, nodes, source)
        results[label] = metrics
        dump(dest / (label+"_residuals.json"), arrays)
    dump(dest / "result.json", results)
    return results


def self_test():
    xs = np.array([0., .123, .5, 1.])
    for alpha in (.25,.5,.75):
        want = (xs**(1-alpha)+(1-xs)**(1-alpha))/(1-alpha)
        got = independent_integral(xs, np.ones_like, alpha)
        assert np.max(abs(want-got)) < 2e-12
        w, _ = weights(xs, np.linspace(0,1,50), alpha, "EP-product")
        assert np.max(abs(w.sum(1)-LAM*want)) < 2e-12
        for kind in ("square", "sin"):
            f64 = exact_source(xs,alpha,kind,64)
            assert np.max(abs(f64-exact_source(xs,alpha,kind,128))) < 1e-9
    nodes = np.linspace(0, 1, 50)
    for alpha in (.25, .5, .75):
        for method in ("EP-product", "MP-product", "MP-raw"):
            w, samples = weights(xs, nodes, alpha, method)
            assert np.all(np.isfinite(w)) and np.all(np.isfinite(samples))
    return "PASS: analytic constants, source order convergence, finite core weights"


def summarize():
    rows=[]
    for p in sorted(OUT.glob("*/result.json")):
        d=json.loads(p.read_text(encoding="utf-8"))
        for selection in ("final","oracle"):
            rows.append(dict(**d["config"],selection=selection,training_seconds=d["training_seconds"],oracle_epoch=d["oracle_epoch"],**d[selection]))
    if rows:
        with (OUT / "metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    return len(rows)//2


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--threads",type=int,default=1)
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--self-test-only",action="store_true")
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=True)
    test=self_test()
    print(test,flush=True)
    info=dict(host=socket.gethostname(),platform=platform.platform(),processor=platform.processor(),
              executable=sys.executable,python=sys.version,torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,
              device="cpu",cuda_available=torch.cuda.is_available(),threads=torch.get_num_threads(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),self_test=test)
    dump(OUT/"environment_core.json",info)
    if args.self_test_only: return
    jobs=manifest()
    dump(OUT/"manifest_core.json",jobs)
    count=0
    for idx,c in enumerate(jobs):
        if (OUT/c["id"]/"result.json").exists(): continue
        if args.limit and count>=args.limit: break
        source=lambda x,a=c["alpha"],k=c["g"]: exact_source(x,a,k)
        print(f"START {idx+1}/{len(jobs)} {c['id']}",flush=True)
        try:
            d=run(c,source)
        except Exception as exc:
            dump(OUT/c["id"]/"failure.json",dict(config=c,error=repr(exc)))
            raise
        count+=1
        summarize()
        print(f"DONE seconds={d['training_seconds']:.1f} final={d['final']['rel_l2_full']:.6g} oracle={d['oracle']['rel_l2_full']:.6g}",flush=True)
    print(f"Completed result files: {summarize()}",flush=True)


if __name__ == "__main__": main()
