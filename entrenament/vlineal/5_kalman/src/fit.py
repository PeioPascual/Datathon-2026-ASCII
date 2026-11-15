"""Estimate the structural model on the fold's TRAINING hours only (fold days +-2 masked as missing).
Usage: fit.py <plec> <crit: ml|ms> [tag] [opts json]  -> data/kf_<plec>_<tag>.npz"""
import sys, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kf, kf_fast
Q3 = Path(__file__).resolve().parent.parent
plec, crit = sys.argv[1], sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else crit
OPT = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
HMAX = 21
d = np.load(Q3 / "data" / f"ser_{plec}.npz")
y = d["y"].T.copy()                     # (N, 2)
mask_tr = ~d["fora"]
extra = OPT.get("extra_mask")           # for inner CV: additional days masked (list of day indices)
day = np.arange(len(y)) // 24 + (int(d["k0"]) // 24)
if extra is not None:
    mask_tr &= ~np.isin(day, extra)
ytr = y.copy(); ytr[~mask_tr] = np.nan
mu = np.nanmean(ytr, 0)
# loadings: PCA of training-day residual curves (day mean removed), RMS-normalised
h0 = int(d["hour"][0]); assert h0 == 0
D = ytr[: (len(y) // 24) * 24].reshape(-1, 24, 2)
okd = ~np.isnan(D).any((1, 2))
lam = np.zeros((2, 2, 24))
for s in (0, 1):
    R = D[okd, :, s] - D[okd, :, s].mean(1, keepdims=True)
    U, S, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
    for k in range(2):
        v = Vt[k] * np.sqrt(24)
        lam[s, k] = v * np.sign(v[13] if k == 0 else -v[20])
if OPT.get("nf", 2) < 2:
    lam[:, OPT.get("nf", 2):] = 0.0
Z = kf.zmat(lam, d["hour"])
W = np.array(OPT.get("w", [0.7, 0.3]))


obs_tr = ~np.isnan(ytr).any(1)
orig = np.where(obs_tr)[0]
orig = orig[orig + HMAX < len(y)]
tgt = ytr[orig[:, None] + np.arange(1, HMAX + 1)[None, :]]          # (no, h, 2)
Zf = Z[orig[:, None] + np.arange(1, HMAX + 1)[None, :]]            # (no, h, 2, 8)
EPS = 1e-4


def unb(TH):
    U = [kf.unpack(t) for t in TH]
    return np.array([u[0] for u in U]), np.array([u[1] for u in U]), np.array([u[2] for u in U])


def batch_obj(TH):
    phi, Q, H = unb(TH)
    if crit == "ml":
        ll, n, _ = kf_fast.filt_batch(ytr, Z, phi, Q, H, mu)
        return -ll / n
    _, _, A = kf_fast.filt_batch(ytr, Z, phi, Q, H, mu, keep=True)
    out = []
    for b in range(len(TH)):
        pw = phi[b][None, :] ** np.arange(1, HMAX + 1)[:, None]
        e = mu + np.einsum("ohsk,hk,ok->ohs", Zf, pw, A[b][orig]) - tgt
        out.append(float(np.nansum(np.nanmean(e ** 2, (0, 1)) * W)))
    return np.array(out)


def fg(th):
    TH = np.vstack([th[None, :], th[None, :] + EPS * np.eye(len(th))])
    v = batch_obj(TH)
    if not np.isfinite(v[0]):
        return 1e6, np.zeros_like(th)
    g = (v[1:] - v[0]) / EPS
    g[~np.isfinite(g)] = 0.0
    return float(v[0]), g


th0 = kf.pack([0.99, 0.99, 0.97, 0.97, 0.97, 0.97, 0.8, 0.8], [[0.03, 0.02], [0.03, 0.03], [0.03, 0.03], [0.05, 0.05]],
              [0.5, 0.3, 0.3, 0.5], [0.02, 0.03])
if "th0" in OPT:
    th0 = np.array(OPT["th0"])
t0 = time.time()
if crit == "ms" and "th0" not in OPT:   # warm start from ML
    try:
        th0 = np.load(Q3 / "data" / f"kf_{plec}_ml.npz")["th"]
    except Exception:
        pass
r = minimize(fg, th0, jac=True, method="L-BFGS-B", options={"maxiter": OPT.get("maxiter", 150)})
phi, Q, H = kf.unpack(r.x)
sd = np.sqrt(np.diag(Q))
print(plec, tag, crit, "obj", round(r.fun, 6), "nit", r.nit, "t", round(time.time() - t0), "s", flush=True)
names = ["L_c", "L_p", "F1_c", "F1_p", "F2_c", "F2_p", "S_c", "S_p"]
print(" phi", {n: round(float(p), 4) for n, p in zip(names, phi)})
print(" half-life h", {n: round(float(np.log(0.5) / np.log(p)), 1) for n, p in zip(names, phi)})
print(" innov sd", {n: round(float(s), 4) for n, s in zip(names, sd)}, "rho", np.round(np.tanh(r.x[16:20]), 3), "eps sd", np.round(np.sqrt(H), 4))
st = kf.stationary(phi, Q)
print(" stationary sd", {n: round(float(np.sqrt(st[i, i])), 4) for i, n in enumerate(names)})
np.savez(Q3 / "data" / f"kf_{plec}_{tag}.npz", th=r.x, phi=phi, Q=Q, H=H, mu=mu, lam=lam, obj=r.fun, crit=crit)
