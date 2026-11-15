"""Hourly log-anomaly series of CO2 and price for one fold (profile = the fold's CLEAN parametres.json, read via the
agent's own perfils_mercat), plus calendar, clearness and masks. Usage (repo root): series.py <plec> -> data/ser_<plec>.npz"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "local_eval"); sys.path.insert(0, "entrenament")
import importlib.util as iu
from pathlib import Path
import numpy as np, pandas as pd
import dades
sys.path.insert(0, str(Path(__file__).resolve().parent)); import fw; dies_plec = fw.dies_plec
Q3 = Path(__file__).resolve().parent.parent
plec = sys.argv[1]
sp = iu.spec_from_file_location("agm", str(Q3 / f"ag_{plec}" / "agent.py")); m = iu.module_from_spec(sp); sp.loader.exec_module(m)
preu = pd.read_csv(dades.DIR_DADES / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
co2 = pd.read_csv(dades.DIR_DADES / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")
mk = pd.DataFrame({"co2": co2["co2_kg_per_kwh"], "preu": preu["precio_eur_per_kwh"]}).dropna()
claus = (mk.index.astype("datetime64[ns]").astype("int64") // m.NS_PER_HORA).values
k0, k1 = int(claus.min()), int(claus.max())
N = k1 - k0 + 1
idx = np.arange(k0, k1 + 1)
ts = pd.to_datetime(idx * m.NS_PER_HORA)
raw = np.full((2, N), np.nan); raw[0, claus - k0] = mk.co2.values; raw[1, claus - k0] = mk.preu.values
ag = m.BatteryAgent(usa_model=False, cal_co2=False)   # el filtre llegeix el CO2 amb el perfil feiner/cap de setmana
prof = np.zeros((2, N))
for i, (c, t) in enumerate(zip(idx, ts)):
    for s in (0, 1):
        prof[s, i] = ag.perfils_mercat(s, int(c), t.hour, t.dayofweek, t.month, j0=0)[0]
y = np.log(raw / prof)
hor = dades.carregar_horari(); hor = hor[hor.load_kw > 0]
clar = np.full(N, np.nan)
for c, t, pv in zip(hor.index.astype("datetime64[ns]").astype("int64") // m.NS_PER_HORA, hor.index, hor.pv_kw * 12.0):
    env = m.PARAMS["envolupant_pv"][str(t.month)][t.hour]
    if env > 100 and k0 <= c <= k1:
        clar[c - k0] = float(np.clip(pv / env, 0, 1.3))
day = ts.normalize()
fora = np.asarray(day.isin(dies_plec(dades.carregar_horari(), plec, marge_dies=2)))
dins = np.asarray(day.isin(dies_plec(dades.carregar_horari(), plec, marge_dies=0)))
blocs = [(int(b.index.min().value // m.NS_PER_HORA), int(b.index.max().value // m.NS_PER_HORA)) for b in (fw.setmanes(dades.carregar_horari(), plec) if plec != "tot" else [])]
cap = np.array([m.cap_setmana_preu(int(c)) for c in idx])
np.savez_compressed(Q3 / "data" / f"ser_{plec}.npz", k0=k0, y=y, raw=raw, prof=prof, hour=ts.hour.values, dow=ts.dayofweek.values,
                    month=ts.month.values, cap=cap, clar=clar, fora=fora, dins=dins, blocs=np.array(blocs))
print(plec, N, "obs", (~np.isnan(y[0])).sum(), "train", (~np.isnan(y[0]) & ~fora).sum(), "blocs", len(blocs))
