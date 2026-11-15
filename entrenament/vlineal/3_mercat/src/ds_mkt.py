"""Dataset of the no-tree market model for one fold (features computed with the agent's own feats_nt and the
fold's CLEAN parametres.json). Also records the tree model's and the recipe's predictions on the same origins.
Usage (repo root): ds_mkt.py <plec>   -> data/mkt_<plec>.npz"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "local_eval"); sys.path.insert(0, "entrenament")
import importlib.util as iu
from pathlib import Path
import numpy as np, pandas as pd
import dades
from entrenar_model_mercat import dies_plec as _dp
def dies_plec(h, p, marge_dies=0):
    return [] if p == 'tot' else _dp(h, p, marge_dies=marge_dies)
P11 = Path(__file__).resolve().parent.parent
plec = sys.argv[1]; EXT = len(sys.argv) > 2 and sys.argv[2] == "ext"
sp = iu.spec_from_file_location("agm", str(P11 / f"ag_{plec}" / "agent.py")); m = iu.module_from_spec(sp); sp.loader.exec_module(m)
preu = pd.read_csv(dades.DIR_DADES / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
co2 = pd.read_csv(dades.DIR_DADES / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")
mk = pd.DataFrame({"co2": co2["co2_kg_per_kwh"], "preu": preu["precio_eur_per_kwh"]}).dropna()
claus = (mk.index.astype("datetime64[ns]").astype("int64") // m.NS_PER_HORA).values
reals = {int(k): (float(a), float(b)) for k, a, b in zip(claus, mk.co2, mk.preu)}
hor = dades.carregar_horari(); hor = hor[hor.load_kw > 0]
claredat = {}
for c, ts, pv in zip(hor.index.astype("datetime64[ns]").astype("int64") // m.NS_PER_HORA, hor.index, hor.pv_kw * 12.0):
    env = m.PARAMS["envolupant_pv"][str(ts.month)][ts.hour]
    if env > 100:
        claredat[int(c)] = float(np.clip(pv / env, 0, 1.3))


class Lim(dict):
    lo, hi = -10**18, 10**18
    def get(self, k, d=None):
        return d if (k > self.hi or k < self.lo) else dict.get(self, k, d)


ag = m.BatteryAgent(usa_model=False)
ag.models = {k: v for k, v in m.carrega_models(P11 / "art" / plec).items() if k in (0, 1)}
ag.mercat = Lim(reals)
rng = np.random.default_rng(0)
X, Y, M, T, AUG, TR, RC, PR, LV, CL = [], [], [], [], [], [], [], [], [], []
for ts, c in zip(mk.index, claus):
    c = int(c)
    if c + 1 not in reals or c + 2 not in reals:
        continue
    for aug in (0, 1):
        if aug == 1 and rng.random() > 0.2:
            continue
        ag.mercat.hi = c + 2
        ag.mercat.lo = -10**18 if aug == 0 else c - int(rng.integers(0, 25))
        ag.claredat_mercat, ag.clau_claredat_mercat = None, None
        for j in range(1, 7):
            if c - j in claredat:
                ag.claredat_mercat, ag.clau_claredat_mercat = claredat[c - j], c - j
                break
        f = ag.feats_nt(c, ts.hour, ts.dayofweek, ts.month, ext=EXT)
        if f is None:
            continue
        x, prof, lev = f
        y = np.full((2, 21), np.nan)
        for i, k in enumerate(range(3, 24)):
            r = reals.get(c + k)
            if r is not None:
                y[0, i], y[1, i] = np.log(r[0] / prof[0, k + 0]) - lev[0], np.log(r[1] / prof[1, k]) - lev[1]
        if np.isnan(y).all():
            continue
        # tree and recipe predictions (absolute values, k=0..23)
        ag.models_backup = ag.models
        tr = np.array(ag.prediu_mercat(c, ts.hour, ts.dayofweek, ts.month, 24))
        ag.models = {}
        rc = np.array(ag.prediu_mercat(c, ts.hour, ts.dayofweek, ts.month, 24))
        ag.models = ag.models_backup
        X.append(x); Y.append(y.ravel()); T.append(ts.normalize()); AUG.append(aug); TR.append(tr); RC.append(rc)
        PR.append(prof); LV.append(lev); CL.append(c)
T = pd.DatetimeIndex(T)
fora = T.isin(dies_plec(dades.carregar_horari(), plec, marge_dies=2))
dins = T.isin(dies_plec(dades.carregar_horari(), plec, marge_dies=0))
REAL = np.array([[[reals.get(c + k, (np.nan, np.nan))[i] for k in range(24)] for i in (0, 1)] for c in CL])
np.savez_compressed(P11 / "data" / f"mkt_{plec}{'_ext' if EXT else ''}.npz", X=np.array(X), Y=np.array(Y), T=T.values.astype("int64"), aug=np.array(AUG),
                    tree=np.array(TR), recipe=np.array(RC), prof=np.array(PR), lev=np.array(LV), clau=np.array(CL),
                    fora=fora, dins=dins, real=REAL)
print(plec, len(X), "train", (~fora).sum(), "eval", (dins & (np.array(AUG) == 0)).sum())
