"""perfils_consum.json: consum mitjà per tipus de dia (feiner, dissabte, diumenge/festiu, feiner de
vacances) i hora, amb el calendari de l'agent (agent_F). Honest per plec (sense els dies del plec +-2)
i amb tot el train. Us (des de l'arrel del repo): perfils.py <dir_agent> <out_base>"""
import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "local_eval"); sys.path.insert(0, "entrenament")
import importlib.util as iu
from pathlib import Path
import numpy as np
import dades
from entrenar_model_mercat import dies_plec

A, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sp = iu.spec_from_file_location("agp", str(A / "agent.py")); m = iu.module_from_spec(sp); sp.loader.exec_module(m)
hor = dades.carregar_horari()
df = hor[hor.load_kw > 0]
claus = (df.index.astype("datetime64[ns]").astype("int64") // m.NS_PER_HORA).values
tip = np.array([m.tipus_consum(int(c)) for c in claus])
hora = df.index.hour.values


def perfil(sel):
    t = [[None] * 24 for _ in range(4)]
    for c in range(4):
        for h in range(24):
            v = df.load_kw.values[sel & (tip == c) & (hora == h)]
            t[c][h] = float(v.mean()) if len(v) >= 3 else None
    for c in range(4):          # tipus sense dades -> dissabte (per al 2) o feiner
        for h in range(24):
            if t[c][h] is None:
                t[c][h] = t[1][h] if (c == 2 and t[1][h] is not None) else t[0][h]
    return t


for plec in list(dades.PLECS) + ["tot"]:
    sel = np.ones(len(df), bool) if plec == "tot" else ~df.index.normalize().isin(dies_plec(hor, plec, marge_dies=2))
    (OUT / plec).mkdir(parents=True, exist_ok=True)
    json.dump({"perfil_consum_tipus": perfil(sel)}, open(OUT / plec / "perfils_consum.json", "w"))
    print(plec, int(sel.sum()), flush=True)
