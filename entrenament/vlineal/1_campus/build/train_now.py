"""Entrena el model de consum de l'hora actual (k=0): objectiu consum(t) - consum(t-1).
Les variables es calculen amb la mateixa funció de l'agent (feats_consum0).
Desa per plec <out>/<plec>/model_campus.npz (honest: sense els dies del plec +-2), un model amb tot
el train a <out>/tot/model_campus.npz, i els residus fora de mostra (OOF) a <out>/oof_consum0.pkl.
v7: les variables fan servir el perfil per tipus de dia de <out>/<plec>/perfils_consum.json (el de
cada plec per al seu model; el de tot el train per al model final i els OOF).
Us: train_now.py <dir_agent> <out_base>"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "local_eval"); sys.path.insert(0, "entrenament")
import importlib.util as iu, numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
import dades, modes, entrenar_model_mercat as emm

A = Path(sys.argv[1]); OUT = Path(sys.argv[2])
sp = iu.spec_from_file_location("agn", str(A / "agent.py")); m = iu.module_from_spec(sp); sp.loader.exec_module(m)
HIPER = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=20, tree_method="hist", n_jobs=2, random_state=0)
emm.HIPER["max_depth"] = HIPER["max_depth"]   # exporta_arbres llegeix la profunditat d'aquí

hor = dades.carregar_horari()
tot = modes.preparar_conjunt(hor, "dificil")
claus = (tot.index.astype("datetime64[ns]").astype("int64") // m.NS_PER_HORA).values
LOAD = {int(c): float(v) for c, v in zip(claus, tot.load_kw.values) if v > 1.0}
TA = tot.temp_ambient_c.values


class Limitat(dict):
    hi = lo = 0

    def get(self, k, d=None):
        return d if (k > self.hi or k < self.lo) else dict.get(self, k, d)


def exemples(ptip):
    ag = m.BatteryAgent(usa_model=False)
    if ptip is not None:
        ag.models = {"perfil_tipus": ptip}
    ag.consum = Limitat(LOAD)
    rng = np.random.default_rng(0)
    X, Y, T, H, CAP, BASE, AUG = [], [], [], [], [], [], []
    for i, (c, ts) in enumerate(zip(claus, tot.index)):
        c = int(c)
        if i == 0 or c not in LOAD or (c - 1) not in LOAD:
            continue
        ag.temp_prev = float(TA[i - 1])
        for aug in (0, 1):
            ag.consum.hi = c - 1
            ag.consum.lo = -10**18 if aug == 0 else c - int(rng.integers(1, 30))
            if aug == 1 and rng.random() > 0.3:
                continue
            X.append(ag.feats_consum0(c, ts.hour, ts.dayofweek, ts.month))
            Y.append(LOAD[c] - LOAD[c - 1]); T.append(ts.normalize()); H.append(ts.hour)
            CAP.append(int(m.tipus_consum(c) in (1, 2)) if ptip is not None else int(ts.dayofweek >= 5))
            BASE.append(LOAD[c - 1]); AUG.append(aug)
    return (np.array(X, float), np.array(Y), pd.DatetimeIndex(T), np.array(H), np.array(CAP), np.array(BASE),
            np.array(AUG))


def entrena(X, Y, mask):
    return xgb.XGBRegressor(**HIPER).fit(X[mask], Y[mask])


def desa(reg, X, fitxer):
    rng = np.random.default_rng(0)
    P, booster = emm.exporta_arbres(reg, X.shape[1])
    prova = X[rng.integers(0, len(X), 500)].astype(np.float32)
    ref = booster.inplace_predict(prova)
    P["base"] = np.float32(np.mean(ref - m.prediu_arbres(P, prova)))
    assert np.abs(m.prediu_arbres(P, prova) - ref).max() < 1e-2
    fitxer.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fitxer, **{f"consum0_{k}": v for k, v in P.items()})


# model final i OOF per blocs de 14 dies (per a la taula d'errors), amb el perfil de tot el train
X, Y, T, H, CAP, BASE, AUG = exemples(m.carrega_models(OUT / "tot").get("perfil_tipus"))
print("exemples", X.shape, "augmentats", AUG.sum(), flush=True)
grup = (T.map(pd.Timestamp.toordinal).values // 14) % 8
oof = np.zeros(len(Y))
for g in range(8):
    oof[grup == g] = entrena(X, Y, grup != g).predict(X[grup == g])
pd.DataFrame({"dia": T, "h": H, "cap": CAP, "aug": AUG, "rl": (Y - oof) / (BASE + oof)}).to_pickle(OUT / "oof_consum0.pkl")
desa(entrena(X, Y, np.ones(len(Y), bool)), X, OUT / "tot" / "model_campus.npz")   # paquet final: tot el train
for plec in dades.PLECS:
    X, Y, T, H, CAP, BASE, AUG = exemples(m.carrega_models(OUT / plec).get("perfil_tipus"))
    fora = T.isin(emm.dies_plec(hor, plec, 2)); dins = T.isin(emm.dies_plec(hor, plec)) & (AUG == 0)
    r = entrena(X, Y, ~fora)
    p = r.predict(X[dins])
    print(f"[{plec}] MAE persistencia {np.abs(Y[dins]).mean():.1f}  model {np.abs(Y[dins] - p).mean():.1f}", flush=True)
    desa(r, X, OUT / plec / "model_campus.npz")
