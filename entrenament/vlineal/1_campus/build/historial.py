"""Construeix historial.npz: taula horària del train (co2, preu, consum, solar x12 i meteo) que
l'agent fa servir per recuperar les hores anteriors a un salt de temps.
Us (des de l'arrel del repo): historial.py <dir_sortida> [<plec_a_excloure>]
Amb un plec, se'n treuen els dies (avaluacio honesta: l'agent nomes troba els dies anteriors a
cada setmana de validacio, com al test ocult)."""
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
if len(sys.argv) > 2:
    fora = pd.DatetimeIndex(dies_plec(hor, sys.argv[2], marge_dies=0))
    df = df[~df.index.normalize().isin(fora)]
claus = df.index.astype("datetime64[ns]").astype("int64") // NS
assert np.all(np.diff(claus.values) > 0)
f32 = lambda s: s.values.astype(np.float32)  # noqa: E731
np.savez_compressed(out / "historial.npz", clau=claus.values.astype(np.int64), co2=f32(df.co2), preu=f32(df.preu),
                    consum=f32(df.load_kw), solar=f32(df.pv_kw * 12.0), **{m: f32(df[m]) for m in met})
print(out / "historial.npz", len(df))
