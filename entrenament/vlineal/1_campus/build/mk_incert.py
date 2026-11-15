"""Construeix incert.json (quantils dels errors de consum i solar) honest per plec.
Us: mk_incert.py <llarg.pkl> <dir_sortida_base> [json opcions]"""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "local_eval"); sys.path.insert(0, "entrenament")
import numpy as np, pandas as pd
from pathlib import Path
import dades, entrenar_model_mercat as emm

df = pd.read_pickle(sys.argv[1]); OUT = Path(sys.argv[2])
O = dict(Q=7, QP=7, llindars_clar=[0.35, 0.7], llindars_alt=[0.3, 0.65], env_min=20.0, k_max=23, fitxer="incert.json")
if len(sys.argv) > 3:
    O.update(json.loads(sys.argv[3]))
hor = dades.carregar_horari()
df = df[df.k <= O["k_max"]]
df["rl"] = (df.L - df.Lh) / df.Lh
df["rp"] = (df.P - df.Ph) / df.env.clip(lower=1)
df["be"] = np.digitize(df.env / df.envmax, O["llindars_alt"])
df["bc"] = np.digitize(df.kav, O["llindars_clar"])


def quantils(v, Q):
    v = np.asarray(v)
    if len(v) < 5:
        return [0.0] * Q
    return [float(x) for x in np.quantile(v, (np.arange(Q) + 0.5) / Q)]


OOF = pd.read_pickle(O["oof"]) if O.get("oof") else None
if OOF is not None:
    OOF = OOF[OOF.aug == 0]


def construeix(d, oof=None):
    t = dict(Q=O["Q"], nq_p=O["QP"], llindars_clar=O["llindars_clar"], llindars_alt=O["llindars_alt"],
             env_min=O["env_min"], k_max=O["k_max"], L={}, P={})
    gl = {key: g.rl.values for key, g in d.groupby(["k", "cap", "h"])}
    dp = d[d.env > O["env_min"]]
    gp = {key: g.rp.values for key, g in dp.groupby(["k", "bc", "be"])}
    for k in range(O["k_max"] + 1):
        t["L"][str(k)] = [[quantils(gl.get((k, c, h), []), O["Q"]) for h in range(24)] for c in (0, 1)]
        t["P"][str(k)] = [[quantils(gp.get((k, bc, be), []), O["QP"]) for be in range(len(O["llindars_alt"]) + 1)]
                          for bc in range(len(O["llindars_clar"]) + 1)]
    if oof is not None:
        g0 = {key: g.rl.values for key, g in oof.groupby(["cap", "h"])}
        t["L"]["0"] = [[quantils(g0.get((c, h), []), O["Q"]) for h in range(24)] for c in (0, 1)]
        # hores de nit amb el consum congelat: l'agent no fa servir el model consum0 sino la
        # lectura repetida, i l'error es el de les prediccions recollides
        for h in O.get("k0_nit", []):
            for c in (0, 1):
                t["L"]["0"][c][h] = quantils(gl.get((0, c, h), []), O["Q"])
    return t


for plec in list(dades.PLECS) + ["tot"]:
    oof = OOF
    if plec == "tot":
        d = df
    else:
        d = df[~df.dia.isin(emm.dies_plec(hor, plec, 2))]
        if oof is not None:
            oof = oof[~oof.dia.isin(emm.dies_plec(hor, plec, 2))]
    (OUT / plec).mkdir(parents=True, exist_ok=True)
    json.dump(construeix(d, oof), open(OUT / plec / O["fitxer"], "w"))
print("ok", O)
