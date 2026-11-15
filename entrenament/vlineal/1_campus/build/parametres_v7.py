"""parametres.json de v7 = parametres.json de v6 + perfil del preu amb festius estatals (agent_F) +
perfils de mercat de novembre refets amb les dades reals (agent_A; abans eren la mitjana d'oct. i des.).
Us (des de l'arrel del repo): parametres_v7.py <parametres v6> <parametres F> <sortida>"""
import sys, json, importlib.util as iu
from pathlib import Path
import numpy as np, pandas as pd

P = json.load(open(sys.argv[1], encoding="utf-8"))
F = json.load(open(sys.argv[2], encoding="utf-8"))
sp = iu.spec_from_file_location("ag7", str(Path(__file__).resolve().parent.parent / "agent" / "agent.py"))
m = iu.module_from_spec(sp); sp.loader.exec_module(m)
P["perfil_preu_festius"] = F["mercat_cal"]["1"]["perfil_preu"]
preu = pd.read_csv("data/train/precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")["precio_eur_per_kwh"]
co2 = pd.read_csv("data/train/co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")["co2_kg_per_kwh"]
for nom, s, festius in (("perfil_co2", co2, False), ("perfil_preu", preu, False), ("perfil_preu_festius", preu, True)):
    s = s[s.index.month == 11]
    claus = s.index.astype("datetime64[ns]").astype("int64") // m.NS_PER_HORA
    cap = np.array([m.cap_setmana_preu(int(c)) for c in claus]) if festius else (s.index.dayofweek >= 5).astype(int)
    vell = np.array(P[nom]["11"])
    P[nom]["11"] = [[float(s[(cap == c) & (s.index.hour == h)].mean()) for h in range(24)] for c in (0, 1)]
    print(nom, "nov: dies", s.index.normalize().nunique(), "mitjana vella", vell.mean(axis=1).round(4),
          "nova", np.array(P[nom]["11"]).mean(axis=1).round(4))
open(sys.argv[3], "w", encoding="utf-8").write(json.dumps(P))
