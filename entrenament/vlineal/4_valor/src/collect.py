"""Run the vlineal1 agent (fold-clean artefacts ag_<plec>, art/<plec>) over the whole train table and record
the SOC-independent value-model features at every hour (+ the SOC the agent holds). Usage: collect.py <plec>
Writes <N2>/data/feat_<plec>.npz (CL, F, NIV, SOC). Features only; no targets stored here."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "local_eval")
import importlib.util as iu
from pathlib import Path
import numpy as np
N2 = Path(__file__).resolve().parent.parent
plec = sys.argv[1]
sp = iu.spec_from_file_location("ag", str(N2 / f"ag_{plec}" / "agent.py")); m = iu.module_from_spec(sp); sp.loader.exec_module(m)
import avaluar, dades, modes
tot = modes.preparar_conjunt(dades.carregar_horari(), "dificil")
mods = m.carrega_models(N2 / "art" / plec)
z = np.load(N2 / "art" / plec / "nt_mercat_ridge.npz"); mods["nt_mercat"] = {k: z[k] for k in z.files}
mods.pop("valor", None)
REC, SOC = [], {}
class A(m.BatteryAgent):
    def act(self, obs):
        c = int(__import__("pandas").Timestamp(obs["timestamp"]).value // m.NS_PER_HORA); SOC[c] = float(obs["soc_kwh"])
        return super().act(obs)
def f():
    a = A(usa_model=False, mercat_nt=1.0); a.models = mods; a.registre = REC; return a
env, _, tt = avaluar.simular(f, tot, 0.0, "dificil"); r = avaluar.metriques(env)
CL = np.array([x[0] for x in REC], np.int64)
np.savez_compressed(N2 / "data" / f"feat_{plec}.npz", CL=CL, F=np.array([x[1] for x in REC]), NIV=np.array([x[2] for x in REC]),
                    SOC=np.array([SOC.get(int(c), np.nan) for c in CL]), SOCC=np.array(sorted(SOC.items())))
print(plec, len(REC), "score over train", round(r["score_brut"], 3), "t", round(tt, 1), flush=True)
