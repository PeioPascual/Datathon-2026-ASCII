"""Trains the no-tree market models for one fold on data/mkt_<plec>.npz (rows whose origin day is NOT in the fold
+-2 days), selects hyper-parameters by inner grouped CV on the training rows only, reports forecast metrics on the
fold's own days, and exports art/<plec>/nt_mercat_<family>.npz.
Usage: train_mkt.py <plec> [families csv: ridge,rrr,mlp]"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ntlib as nl
P11 = Path(__file__).resolve().parent.parent
plec = sys.argv[1]
FAM = (sys.argv[2] if len(sys.argv) > 2 else "ridge,rrr,mlp").split(",")
SUF = sys.argv[4] if len(sys.argv) > 4 else ""
d = np.load(P11 / "data" / f"mkt_{plec}{SUF}.npz")
X, Y, T, aug, fora, dins = d["X"], d["Y"], d["T"], d["aug"], d["fora"], d["dins"]
day = T // (86400 * 10**6)
tr = ~fora
comp = ~np.isnan(Y).any(1)
mu, sd = nl.standard(X[tr])
Z = (X - mu) / sd
grp = (day // 7) % 5
rng = np.random.default_rng(0)


def cv_ridge(alpha, r=None):
    err = []
    for g in range(5):
        a = tr & comp & (grp != g); b = tr & comp & (grp == g) & (aug == 0)
        W, b0 = nl.ridge_fit(Z[a], Y[a], alpha)
        if r is not None:
            W, b0 = nl.rrr(Z[a], W, b0, r)
        err.append(((Z[b] @ W + b0 - Y[b]) ** 2).mean())
    return float(np.mean(err))


res = {}
fit = tr & comp
if "ridge" in FAM or "rrr" in FAM:
    alphas = [30, 100, 300, 1000, 3000, 10000]
    cv = {a: cv_ridge(a) for a in alphas}
    a_best = min(cv, key=cv.get)
    print(plec, "ridge inner-CV MSE", {a: round(v, 5) for a, v in cv.items()}, "->", a_best, flush=True)
    W, b0 = nl.ridge_fit(Z[fit], Y[fit], a_best)
    s2 = ((Z[fit] @ W + b0 - Y[fit]) ** 2).mean(0)
    res["ridge"] = nl.lin_dict(mu, sd, W, b0, s2)
    if "rrr" in FAM:
        cvr = {r: cv_ridge(a_best, r) for r in (1, 2, 3, 5, 8, 12)}
        r_best = min(cvr, key=cvr.get)
        print(plec, "rrr inner-CV MSE", {r: round(v, 5) for r, v in cvr.items()}, "->", r_best, flush=True)
        Wr, br = nl.rrr(Z[fit], W, b0, r_best)
        s2 = ((Z[fit] @ Wr + br - Y[fit]) ** 2).mean(0)
        res["rrr"] = nl.lin_dict(mu, sd, Wr, br, s2)
if "mlp" in FAM:
    H = json.loads(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else {}
    hid = tuple(H.get("hidden", [32])); wd = H.get("wd", 1e-2); nm = H.get("members", 5); dr = H.get("drop_in", 0.1)
    Yf = np.nan_to_num(Y); M = (~np.isnan(Y)).astype(float)
    ymu = np.nanmean(Y[tr], 0); ysd = np.nanstd(Y[tr], 0)
    Ys = np.nan_to_num((Y - ymu) / ysd)
    days_tr = np.unique(day[tr])
    members = []
    for s in range(nm):
        r2 = np.random.default_rng(100 + s)
        vd = r2.choice(days_tr, size=len(days_tr) // 6, replace=False)
        v = tr & np.isin(day, vd); t_ = tr & ~np.isin(day, vd)
        lv, (Ws, bs), ep = nl.mlp_train(Z[t_], Ys[t_], Z[v], Ys[v], hidden=hid, wd=wd, seed=s, M=M[t_], Mv=M[v], drop_in=dr)
        print(plec, f"mlp member {s}: val MSE(std) {lv:.4f} at epoch {ep}", flush=True)
        members.append((Ws, bs))
    pr = ymu + ysd * np.mean([nl.mlp_predict(Ws, bs, Z[tr]) for Ws, bs in members], 0)
    s2 = np.nanmean((pr - Y[tr]) ** 2, 0)
    res["mlp"] = nl.mlp_dict(mu, sd, ymu, ysd, members, s2)
    tag = H.get("tag", "mlp")
    if tag != "mlp":
        res[tag] = res.pop("mlp")


# ---------------- forecast metrics on the fold's days (warm origins)
def absolute(P, rows):
    import importlib.util as iu
    sp = iu.spec_from_file_location("agm", str(P11 / f"ag_{plec}" / "agent.py")); m = iu.module_from_spec(sp); sp.loader.exec_module(m)
    y = np.array([m.prediu_nt(P, x) for x in X[rows]]).reshape(-1, 2, 21) + 0.5 * P["s2"].reshape(2, 21)
    out = d["prof"][rows] * np.exp(d["lev"][rows][:, :, None])
    out[:, :, 3:] = d["prof"][rows][:, :, 3:] * np.exp(d["lev"][rows][:, :, None] + np.clip(y, -2.5, 1.6))
    return out


ev = dins & (aug == 0)
if not ev.any():
    for k, Pd in res.items():
        np.savez_compressed(P11 / 'art' / plec / f'nt_mercat_{k}{SUF}.npz', **Pd)
    sys.exit(0)
real = d["real"][ev]
preds = {"recipe": d["recipe"][ev], "tree": d["tree"][ev]}
for k, Pd in res.items():
    preds[k] = absolute(Pd, ev)
    np.savez_compressed(P11 / "art" / plec / f"nt_mercat_{k}{SUF}.npz", **Pd)
for k in list(res):
    preds[f"avg(tree,{k})"] = 0.5 * (preds["tree"] + preds[k])
PAR = json.load(open(P11 / f"ag_{plec}" / "parametres.json"))
cm, pm = PAR["co2_mitja"], PAR["preu_mitja"]


def metr(p):
    o = {}
    for i, nm_ in ((0, "co2"), (1, "preu")):
        e = np.abs(p[:, i, 3:] - real[:, i, 3:]) / np.nanmean(real[:, i, 3:])
        o[nm_ + "_relMAE_3-23"] = np.nanmean(e)
        o[nm_ + "_relMAE_h3/h12/h23"] = [round(float(np.nanmean(e[:, j])), 3) for j in (0, 9, 20)]
        lp, lr = np.nanmean(p[:, i, 3:12], 1), np.nanmean(real[:, i, 3:12], 1)
        o[nm_ + "_lvl3-11"] = np.nanmean(np.abs(lp - lr) / lr)
        o[nm_ + "_bias"] = np.nanmean((p[:, i, 3:] - real[:, i, 3:])) / np.nanmean(real[:, i, 3:])
    wp = 0.7 * p[:, 0] / cm + 0.3 * p[:, 1] / pm; wr = 0.7 * real[:, 0] / cm + 0.3 * real[:, 1] / pm
    o["w_relMAE"] = np.nanmean(np.abs(wp[:, 3:] - wr[:, 3:])) / np.nanmean(wr[:, 3:])
    ok = ~np.isnan(wr).any(1)
    from scipy.stats import spearmanr
    o["w_rank24"] = float(np.mean([spearmanr(a, b)[0] for a, b in zip(wp[ok], wr[ok])]))
    return {k: (round(float(v), 4) if not isinstance(v, list) else v) for k, v in o.items()}


out = {k: metr(p) for k, p in preds.items()}
for k, v in out.items():
    print(plec, k, json.dumps(v), flush=True)
json.dump(out, open(P11 / "data" / f"metr_mkt_{plec}.json", "a"))
