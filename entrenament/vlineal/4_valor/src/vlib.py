"""Shared pieces for the N2 value models: perfect-foresight targets, design, ridge / MLP / tree fits."""
import sys, json
sys.path.insert(0, "local_eval"); sys.path.insert(0, "entrenament")
from pathlib import Path
import importlib.util as iu
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import as_strided
import dades, modes
import entrenar_model_mercat as emm
N2 = Path(__file__).resolve().parent.parent
CAP, POT, PAS = 6000.0, 1500.0, 50.0
EDGE = 36
hor = dades.carregar_horari()
tot = modes.preparar_conjunt(hor, "dificil")


def agent_mod(plec):
    sp = iu.spec_from_file_location("ag_" + plec, str(N2 / f"ag_{plec}" / "agent.py")); m = iu.module_from_spec(sp)
    sp.loader.exec_module(m); return m


def cost_to_go(w, net, ETA):
    n_g = int(CAP / PAS)
    puja = int(np.floor(POT * ETA / PAS + 1e-9)); baixa = int(np.floor(POT / ETA / PAS + 1e-9))
    energia = np.arange(-baixa, puja + 1) * PAS
    potencia = np.where(energia >= 0, energia / ETA, energia * ETA)
    n = len(w); C = np.zeros((n + 1, n_g + 1))
    mem = np.full(baixa + n_g + 1 + puja, np.inf); sb = mem.strides[0]
    fin = as_strided(mem, shape=(n_g + 1, baixa + puja + 1), strides=(sb, sb), writeable=False)
    for t in range(n - 1, -1, -1):
        mem[baixa:baixa + n_g + 1] = C[t + 1]
        C[t] = (fin + w[t] * np.maximum(net[t] + potencia, 0.0)).min(axis=1)
    return C


def targets(m, excl_dies, Ks):
    """{K: {clau: marginal value at 12 SOC bins (absolute w units / kWh)}} from perfect-foresight DP over the
    train table with excl_dies removed. Offline only (never stored by date)."""
    d = tot[~tot.index.normalize().isin(excl_dies)]
    cl = emm.clau_hora(d.index, m.NS_PER_HORA).values.astype(np.int64)
    w = (0.7 * d.co2_kg_per_kwh / m.PARAMS["co2_mitja"] + 0.3 * d.precio_eur_per_kwh / m.PARAMS["preu_mitja"]).values
    net = (d.load_kw - d.pv_kw).values
    talls = np.where(np.diff(cl) > 6)[0] + 1
    n_g = int(CAP / PAS); NODES = (m.SOC_NODES * n_g).round().astype(int)
    out = {K: {} for K in Ks}
    for seg in np.split(np.arange(len(d)), talls):
        if len(seg) < 48: continue
        C = cost_to_go(w[seg], net[seg], m.ETA); n = len(seg)
        lam = -(np.diff(C[:, NODES], axis=1)) / (PAS * np.diff(NODES))
        for K in Ks:
            for i in range(n):
                if i + K <= n - EDGE:
                    out[K][int(cl[seg[i]])] = lam[i + K]
    return out


def design(m, F):
    return np.array([m.disseny_valor(f) for f in F])


def standard(X):
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-9] = 1.0; return mu, sd


def ridge_multi(Z, Y, alpha, wt=None):
    if wt is None: wt = np.ones(len(Z))
    zm = (Z * wt[:, None]).sum(0) / wt.sum(); ym = (Y * wt[:, None]).sum(0) / wt.sum()
    Zc, Yc = Z - zm, Y - ym
    A = (Zc * wt[:, None]).T @ Zc + alpha * np.eye(Z.shape[1])
    W = np.linalg.solve(A, (Zc * wt[:, None]).T @ Yc)
    return W, ym - zm @ W


def mono(y):
    return np.minimum.accumulate(np.maximum(y, 0.0), axis=1)


# ------------------------------------------------------------------ MLP with monotone output
def sp(x): return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))
def sig(x): return 1 / (1 + np.exp(-x))


def mlp_fwd(P, Z):
    h = np.tanh(Z @ P["W0"] + P["b0"]); o = h @ P["W1"] + P["b1"]
    return h, o, np.cumsum(sp(o)[:, ::-1], axis=1)[:, ::-1]


def mlp_fit(Z, Y, Zv, Yv, wt=None, wtv=None, H=32, lr=2e-3, wd=1e-4, epochs=300, batch=128, seed=0, huber=0.05, patience=30):
    rng = np.random.default_rng(seed); d = Z.shape[1]; nb = Y.shape[1]
    if wt is None: wt = np.ones(len(Z))
    if wtv is None: wtv = np.ones(len(Zv))
    ymu = Y.mean(0); dec = np.maximum(np.r_[ymu[:-1] - ymu[1:], ymu[-1]], 1e-3)
    P = {"W0": rng.normal(0, 1 / np.sqrt(d), (d, H)), "b0": np.zeros(H), "W1": rng.normal(0, 0.01, (H, nb)),
         "b1": np.log(np.expm1(dec))}
    Mo = {k: np.zeros_like(v) for k, v in P.items()}; Ve = {k: np.zeros_like(v) for k, v in P.items()}
    best, bestP, bad, t = np.inf, None, 0, 0
    for ep in range(epochs):
        perm = rng.permutation(len(Z))
        for s in range(0, len(Z), batch):
            ix = perm[s:s + batch]; z, y, ww = Z[ix], Y[ix], wt[ix]
            h, o, lam = mlp_fwd(P, z)
            e = lam - y
            g = np.clip(e / huber, -1, 1) * ww[:, None] / (ww.sum() * nb)
            dsp = np.cumsum(g, axis=1); do = dsp * sig(o)
            G = {"W1": h.T @ do, "b1": do.sum(0)}
            dh = (do @ P["W1"].T) * (1 - h ** 2)
            G["W0"] = z.T @ dh; G["b0"] = dh.sum(0)
            t += 1
            for k in P:
                gk = G[k] + (wd * P[k] if k[0] == "W" else 0)
                Mo[k] = 0.9 * Mo[k] + 0.1 * gk; Ve[k] = 0.999 * Ve[k] + 0.001 * gk ** 2
                P[k] -= lr * (Mo[k] / (1 - 0.9 ** t)) / (np.sqrt(Ve[k] / (1 - 0.999 ** t)) + 1e-8)
        lv = float((np.abs(mlp_fwd(P, Zv)[2] - Yv).mean(1) * wtv).sum() / wtv.sum())
        if lv < best - 1e-5: best, bestP, bad = lv, {k: v.copy() for k, v in P.items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    return bestP, best
