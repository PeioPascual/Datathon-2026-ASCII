"""Construeix escenaris_mercat.npz: trajectories d'error relatiu log(real/previst) del CO2 i del
preu (horitzons 0..23), a partir dels registres fora de mostra de log_mercat.py dels 4 plecs.
Us: esc_mercat.py <dir_logs> <plec_exclos|tot> <sortida.npz>
Amb un plec, s'exclouen les decisions a +-2 dies de qualsevol dia del plec."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "local_eval"); sys.path.insert(0, "entrenament")
import numpy as np, pandas as pd
import dades
from entrenar_model_mercat import dies_plec

L, excl, out = sys.argv[1], sys.argv[2], sys.argv[3]
NS = 3_600_000_000_000
z = [np.load(f"{L}/prev_{p}.npz") for p in dades.PLECS]
meta = np.concatenate([a["meta"] for a in z])
lc = np.log(np.concatenate([a["rco2"] for a in z]) / np.concatenate([a["co2"] for a in z]))
lp = np.log(np.concatenate([a["rpreu"] for a in z]) / np.concatenate([a["preu"] for a in z]))
dia = pd.DatetimeIndex((meta[:, 0] * NS).astype("datetime64[ns]")).normalize()
keep = np.isfinite(lc).all(1) & np.isfinite(lp).all(1)
if excl != "tot":
    keep &= ~dia.isin(pd.DatetimeIndex(dies_plec(dades.carregar_horari(), excl, marge_dies=2)))
# els plecs final i d15 es solapen: una sola fila per hora
_, u = np.unique(meta[:, 0], return_index=True)
unic = np.zeros(len(meta), bool); unic[u] = True; keep &= unic
o = np.argsort(meta[keep, 0], kind="stable")           # ordre cronologic
np.savez_compressed(out, hora=meta[keep, 1][o].astype(np.int8), co2=lc[keep][o].astype(np.float32),
                    preu=lp[keep][o].astype(np.float32))
print(excl, "escenaris", int(keep.sum()))
