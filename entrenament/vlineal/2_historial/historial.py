"""historial.npz com v6c/build/historial.py, pero omplint les hores on hi ha comptador i meteo i NO hi ha
inversors (p.ex. 17-24 d'agost de 2026): consum del comptador cru, solar x12 estimada de la irradiancia
(regressio lineal pv ~ irr + irr*(Tpanel-25) ajustada amb totes les hores que tenen les dues coses).
Us (des de l'arrel del repo): historial_N.py <dir_sortida> [<plec_a_excloure>]"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "local_eval"); sys.path.insert(0, "entrenament")
from pathlib import Path
import numpy as np, pandas as pd
import dades
from entrenar_model_mercat import dies_plec

NS = 3_600_000_000_000
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
p = pd.read_csv(dades.DIR_DADES / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")["precio_eur_per_kwh"]
c = pd.read_csv(dades.DIR_DADES / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")["co2_kg_per_kwh"]
hor = dades.carregar_horari()
met = ["irradiancia_wm2", "temp_ambient_c", "temp_panel_c"]
df = pd.DataFrame({"co2": c, "preu": p}).join(hor[["load_kw", "pv_kw"] + met], how="outer")
# --- hores sense inversors: comptador i meteo crus
cc = pd.read_csv(dades.DIR_DADES / "contador.csv", usecols=["Fecha", dades.COL_CONTADOR], parse_dates=["Fecha"]).set_index("Fecha")
load = (-cc[dades.COL_CONTADOR]).clip(lower=0, upper=5000).resample("1h").mean()
cap = pd.read_csv(dades.DIR_DADES / "meteo.csv", nrows=0).columns
tria = {nom: next(x for x in cap if text in x) for nom, text in dades.COLS_METEO.items() if nom in met}
mt = pd.read_csv(dades.DIR_DADES / "meteo.csv", usecols=["Fecha", *tria.values()], parse_dates=["Fecha"]).set_index("Fecha")
mt = mt.apply(pd.to_numeric, errors="coerce").resample("1h").mean().rename(columns={v: k for k, v in tria.items()})
mt["irradiancia_wm2"] = mt["irradiancia_wm2"].clip(lower=0)
X = np.column_stack([hor.irradiancia_wm2, hor.irradiancia_wm2 * (hor.temp_panel_c - 25)])
coef, *_ = np.linalg.lstsq(X, hor.pv_kw.values, rcond=None)
falta = df.index[df.load_kw.isna()]
nou = load.reindex(falta)
ok = nou.notna() & (nou > 1.0) & mt.irradiancia_wm2.reindex(falta).notna()
idx = falta[ok.values]
df.loc[idx, "load_kw"] = load.reindex(idx).values
for k in met:
    df.loc[idx, k] = mt[k].reindex(idx).values
pv = coef[0] * df.loc[idx, "irradiancia_wm2"] + coef[1] * df.loc[idx, "irradiancia_wm2"] * (df.loc[idx, "temp_panel_c"] - 25)
df.loc[idx, "pv_kw"] = pv.clip(lower=0).values
print("hores omplertes", len(idx), "de", idx.min() if len(idx) else None, "a", idx.max() if len(idx) else None, "coef", coef)
if len(sys.argv) > 2:
    fora = pd.DatetimeIndex(dies_plec(hor, sys.argv[2], marge_dies=0))
    df = df[~df.index.normalize().isin(fora)]
claus = df.index.astype("datetime64[ns]").astype("int64") // NS
assert np.all(np.diff(claus.values) > 0)
f32 = lambda s: s.values.astype(np.float32)  # noqa: E731
np.savez_compressed(out / "historial.npz", clau=claus.values.astype(np.int64), co2=f32(df.co2), preu=f32(df.preu),
                    consum=f32(df.load_kw), solar=f32(df.pv_kw * 12.0), **{m: f32(df[m]) for m in met})
print(out / "historial.npz", len(df))
