"""Models de quantils de la claredat solar de l'hora actual (k=0).
Us: train_s0.py <dir_agent> <rec2.pkl> <out_base> [json opcions]
Desa solar0q<i>_* dins <out>/<plec>/model_campus.npz (afegint-los a consum0 si ja hi és)."""
import sys, json, pickle, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "local_eval"); sys.path.insert(0, "entrenament")
import importlib.util as iu, numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
import dades, entrenar_model_mercat as emm

A = Path(sys.argv[1]); REC = Path(sys.argv[2]); OUT = Path(sys.argv[3])
O = dict(Q=7, env_min=20.0, n_estimators=200, max_depth=3, lr=0.05, mcw=20, eval_only=False)
if len(sys.argv) > 4:
    O.update(json.loads(sys.argv[4]))
sp = iu.spec_from_file_location("ags", str(A / "agent.py")); m = iu.module_from_spec(sp); sp.loader.exec_module(m)
emm.HIPER["max_depth"] = O["max_depth"]
d = pickle.load(open(REC, "rb"))
R = d["reals"].set_index("clau")
PV, L, DIA = R.pv.to_dict(), R.load.to_dict(), R.dia.to_dict()
X, Y, W, T, PH, BC, BE = [], [], [], [], [], [], []
P = m.PARAMS
for r in d["rec"]:
    c = r["clau"]
    if c not in PV or L.get(c, 0) <= 1.0:
        continue
    # cel clar que va fer servir l agent a l hora actual
    e0 = r["env"][0]
    if e0 <= O["env_min"]:
        continue
    X.append(r["fs0"]); Y.append(min(max(PV[c] / e0, 0.0), 1.5)); W.append(e0); T.append(DIA[c]); PH.append(r["s"][0])
X = np.array(X, float); Y = np.array(Y); W = np.array(W); T = pd.DatetimeIndex(T); PH = np.array(PH)
alphas = (np.arange(O["Q"]) + 0.5) / O["Q"]
print("exemples", X.shape)
hor = dades.carregar_horari()


def entrena(mask, a):
    r = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=a, n_estimators=O["n_estimators"],
                         learning_rate=O["lr"], max_depth=O["max_depth"], subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=O["mcw"], tree_method="hist", n_jobs=2, random_state=0)
    return r.fit(X[mask], Y[mask], sample_weight=W[mask])


def pinball(y, q, a):
    e = y - q
    return np.maximum(a * e, (a - 1) * e)


def desa(regs, fitxer):
    out = {}
    if fitxer.exists():
        z = np.load(fitxer)
        out = {k: z[k] for k in z.files if not k.startswith("solar0")}
    rng = np.random.default_rng(1)
    for i, reg in enumerate(regs):
        Pq, booster = emm.exporta_arbres(reg, X.shape[1])
        prova = X[rng.integers(0, len(X), 500)].astype(np.float32)
        ref = booster.inplace_predict(prova)
        Pq["base"] = np.float32(np.mean(ref - m.prediu_arbres(Pq, prova)))
        assert np.abs(m.prediu_arbres(Pq, prova) - ref).max() < 1e-2
        for k, v in Pq.items():
            out[f"solar0q{i}_{k}"] = v
    fitxer.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fitxer, **out)


for plec in dades.PLECS:
    fora = T.isin(emm.dies_plec(hor, plec, 2)); dins = T.isin(emm.dies_plec(hor, plec))
    regs = [entrena(~fora, a) for a in alphas]
    Qm = np.sort(np.column_stack([r.predict(X[dins]) for r in regs]), axis=1)
    # taula: quantils dels residus de la recepta (sense el plec)
    e0 = W[dins]
    rp_tr = (Y[~fora] * W[~fora] - PH[~fora]) / W[~fora]
    qt = np.quantile(rp_tr, alphas)
    Qt = (PH[dins][:, None] + qt[None, :] * e0[:, None]) / e0[:, None]
    pl_m = np.mean([np.mean(pinball(Y[dins], Qm[:, i], a) * e0) for i, a in enumerate(alphas)])
    pl_t = np.mean([np.mean(pinball(Y[dins], Qt[:, i], a) * e0) for i, a in enumerate(alphas)])
    mae_med = np.mean(np.abs(Y[dins] - Qm[:, O["Q"] // 2]) * e0)
    mae_v5 = np.mean(np.abs(Y[dins] * e0 - PH[dins]))
    print(f"[{plec}] pinball kW: taula-global {pl_t:.1f} model {pl_m:.1f} | MAE v5 {mae_v5:.1f} mediana model {mae_med:.1f}", flush=True)
    if not O["eval_only"]:
        desa(regs, OUT / plec / "model_campus.npz")
if not O["eval_only"]:
    desa([entrena(np.ones(len(Y), bool), a) for a in alphas], OUT / "tot" / "model_campus.npz")
