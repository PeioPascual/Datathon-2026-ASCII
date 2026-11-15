"""Registra les previsions de preu i CO2 (24 h) que fa l'agent a cada hora d'un plec, amb els
artefactes honestos d'aquell plec (model de mercat entrenat sense el plec), i els valors reals.
Us: log_mercat.py <dir_agent> <plec> <dir_artefactes_base> <sortida.npz>"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "local_eval")
import importlib.util as iu
from pathlib import Path
import numpy as np
import avaluar, dades, modes

D = Path(sys.argv[1]); plec = sys.argv[2]; base = Path(sys.argv[3]); out = sys.argv[4]
sp = iu.spec_from_file_location("ag", str(D / "agent.py")); m = iu.module_from_spec(sp); sp.loader.exec_module(m)
df = dades.carregar_horari(); conj = modes.preparar_conjunt(dades.conjunt_validacio(df, plec), "dificil")
mods = m.carrega_models(base / plec)
claus = (conj.index.astype("datetime64[ns]").astype("int64") // m.NS_PER_HORA).values
REAL = {int(k): (float(a), float(b)) for k, a, b in zip(claus, conj.co2_kg_per_kwh, conj.precio_eur_per_kwh)}
REC = []


class A(m.BatteryAgent):
    def prediu_mercat(self, clau, hora, dow, mes, n):
        c, p = super().prediu_mercat(clau, hora, dow, mes, n)
        if n == 24:
            REC.append((clau, hora, c.copy(), p.copy()))
        return c, p


def f():
    a = A(usa_model=False, escenaris=0, n_esc=0); a.models = mods; return a


avaluar.simular(f, conj, 0.0, "dificil")
N = len(REC)
arr = {k: np.full((N, 24), np.nan) for k in ("co2", "preu", "rco2", "rpreu")}
meta = np.zeros((N, 2), np.int64)
for i, (clau, hora, c, p) in enumerate(REC):
    meta[i] = (clau, hora); arr["co2"][i] = c; arr["preu"][i] = p
    for j in range(24):
        v = REAL.get(clau + j)
        if v is not None:
            arr["rco2"][i, j], arr["rpreu"][i, j] = v
np.savez(out, meta=meta, **arr)
print(plec, N)
