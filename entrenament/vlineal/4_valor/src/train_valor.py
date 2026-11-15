"""Fit ridge / MLP / trees value models for one fold on the SAME features and targets.
Features: data/feat_<plec>.npz (vlineal1 agent run with fold-clean artefacts). Targets: perfect-foresight DP on
train with fold days +-2 removed. Training rows = hours not in fold days +-2. Eval (offline only) = fold days,
targets from DP on all train. Usage: train_valor.py <plec> [K list, default 3,24] [models, default lin,mlp,arb]"""
import sys, json, time, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import vlib as V
plec = sys.argv[1]
KS = [int(k) for k in (sys.argv[2] if len(sys.argv) > 2 else "3,24").split(",")]
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0; SUF = f"_s{SEED}" if SEED else ""
MODS = (sys.argv[3] if len(sys.argv) > 3 else "lin,mlp,arb").split(",")
m = V.agent_mod(plec)
z = np.load(V.N2 / "data" / f"feat_{plec}.npz"); CL, F, NIV = z["CL"], z["F"], z["NIV"]
DIA = pd.DatetimeIndex(pd.to_datetime(CL * m.NS_PER_HORA)).normalize()
if plec == "tot":   # all-data package: no exclusion; "eval" = in-sample (sanity only)
    excl, fold = [], list(pd.DatetimeIndex(DIA).unique())
else:
    excl = V.emm.dies_plec(V.hor, plec, marge_dies=2); fold = V.emm.dies_plec(V.hor, plec)
tg = V.targets(m, excl, KS); tge = V.targets(m, [], KS)
X = V.design(m, F); NB = 12; ALPHAS = [1, 3, 10, 30, 100, 300, 1000, 3000]
OUT = V.N2 / "art" / plec; res = []


def rows(t, mask):
    idx = np.array([i for i in np.where(mask)[0] if CL[i] in t])
    return idx, np.array([t[CL[i]] for i in idx]) / NIV[idx][:, None]


for K in KS:
    itr, ytr = rows(tg[K], ~DIA.isin(excl)); iev, yev = rows(tge[K], DIA.isin(fold))
    grp = (CL[itr] // 24 // 7) % 5
    mu, sd = V.standard(X[itr]); Ztr, Zev = (X[itr] - mu) / sd, (X[iev] - mu) / sd
    r = dict(plec=plec, K=K, n_tr=len(itr), n_ev=len(iev))
    base = V.mono(np.nan_to_num(F[iev][:, -NB:])) if K == 3 else np.zeros_like(yev)
    r["mae_plan"] = round(float(np.abs(base - yev).mean()), 4)
    hm = pd.DataFrame(ytr).groupby(F[itr][:, 0].astype(int)).mean()
    r["mae_hourmean"] = round(float(np.abs(hm.reindex(F[iev][:, 0].astype(int)).values - yev).mean()), 4)
    if "lin" in MODS:
        t0 = time.time(); cv = np.zeros(len(ALPHAS))
        for g in range(5):
            a, b = grp != g, grp == g
            for j, al in enumerate(ALPHAS):
                W, b0 = V.ridge_multi(Ztr[a], ytr[a], al); cv[j] += np.abs(V.mono(Ztr[b] @ W + b0) - ytr[b]).sum()
        ia = int(cv.argmin()); W, b0 = V.ridge_multi(Ztr, ytr, ALPHAS[ia])
        p = V.mono(Zev @ W + b0)
        r.update(lin_alpha=ALPHAS[ia], lin_cv=round(cv[ia] / ytr.size, 4), mae_lin=round(float(np.abs(p - yev).mean()), 4))
        np.savez_compressed(OUT / f"nt_valor_lin_K{K}.npz", tipus="lin", mu=mu, sd=sd, W=W, b=b0)
    if "mlp" in MODS:
        t0 = time.time(); Ps, cvs = [], []
        for g in range(5):
            a, b = grp != g, grp == g
            P, lv = V.mlp_fit(Ztr[a], ytr[a], Ztr[b], ytr[b], H=32, seed=g + SEED); Ps.append(P); cvs.append(lv)
        p = np.mean([V.mlp_fwd(P, Zev)[2] for P in Ps], axis=0)
        r.update(mlp_cv=round(float(np.mean(cvs)), 4), mae_mlp=round(float(np.abs(p - yev).mean()), 4), mlp_t=round(time.time() - t0))
        D = dict(tipus="mlp", mu=mu, sd=sd, n_membres=len(Ps), n_capes=2, ysc=1.0)
        for i, P in enumerate(Ps):
            D[f"W{i}_0"], D[f"b{i}_0"], D[f"W{i}_1"], D[f"b{i}_1"] = P["W0"], P["b0"], P["W1"], P["b1"]
        np.savez_compressed(OUT / f"nt_valor_mlp_K{K}{SUF}.npz", **D)
        # check agent inference matches
        q = np.array([m.prediu_valor({k: v for k, v in D.items()}, F[i]) for i in iev[:50]])
        assert np.abs(q - p[:50]).max() < 1e-6, np.abs(q - p[:50]).max()
    if "arb" in MODS:
        import xgboost as xgb
        t0 = time.time(); DEPTH = 6
        def tx(ix): return np.concatenate([np.repeat(F[ix], NB, axis=0), np.tile(m.SOC_MIG, len(ix))[:, None]], axis=1)
        Xt = tx(itr); mono_c = "(" + ",".join(["0"] * (Xt.shape[1] - 1) + ["-1"]) + ")"
        reg = xgb.XGBRegressor(n_estimators=250, learning_rate=0.06, max_depth=DEPTH, subsample=0.8, colsample_bytree=0.7,
                               min_child_weight=20, tree_method="hist", n_jobs=4, random_state=0, monotone_constraints=mono_c)
        reg.fit(Xt, ytr.ravel())
        p = V.mono(reg.predict(tx(iev)).reshape(-1, NB))
        r.update(mae_arb=round(float(np.abs(p - yev).mean()), 4), arb_t=round(time.time() - t0))
        P, booster = V.emm.exporta_arbres(reg, Xt.shape[1]); P["prof"] = np.int32(DEPTH)
        pr = Xt[np.random.default_rng(1).integers(0, len(Xt), 500)].astype(np.float32)
        ref = booster.inplace_predict(pr); P["base"] = np.float32(np.mean(ref - m.prediu_arbres(P, pr)))
        assert np.abs(m.prediu_arbres(P, pr) - ref).max() < 1e-3
        P["tipus"] = "arb"; np.savez_compressed(OUT / f"nt_valor_arb_K{K}.npz", **P)
    print(json.dumps(r), flush=True); res.append(r)
json.dump(res, open(V.N2 / "data" / f"offline_{plec}{SUF}.json", "w"))
