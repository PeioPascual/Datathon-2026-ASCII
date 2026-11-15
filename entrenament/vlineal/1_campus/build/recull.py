"""Recull les prediccions puntuals de consum i solar (k = 0..23) que fa l'agent sobre tot el train,
amb l'estat intern de l'agent (incloses les variables dels models de solar de l'hora actual), per
estimar-ne els errors. L'agent corre sense models (consum0 no hi és: k=0 es la recepta) i amb la
taula historial.npz i el perfil per tipus de dia (perfils_consum.json) de <dir_hist>.
Tres passades: tot el train, i setmanes alternes (mes arrencades despres d'un salt).
Us: recull.py <dir_agent> <dir_hist> <sortida.pkl> ['<json kwargs agent>']"""
import sys, json, pickle, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "local_eval")
import importlib.util as iu, numpy as np, pandas as pd
from pathlib import Path
import avaluar, dades, modes

A = Path(sys.argv[1]); HD = Path(sys.argv[2]); OUT = Path(sys.argv[3])
KW = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
sp = iu.spec_from_file_location("agc", str(A / "agent.py")); m = iu.module_from_spec(sp); sp.loader.exec_module(m)
hor = dades.carregar_horari()
tot = modes.preparar_conjunt(hor, "dificil")
MH = m.carrega_models(HD)
hist, ptip = MH.get("historial"), MH.get("perfil_tipus")
print("perfil per tipus de dia", ptip is not None)
print("historial", None if hist is None else len(hist["clau"]))
REC = []


class Rec(m.BatteryAgent):
    def prediu_consum(self, clau, hora, dow, n):
        self._c = super().prediu_consum(clau, hora, dow, n)
        return self._c

    def prediu_solar(self, clau, hora, mes, n):
        s = super().prediu_solar(clau, hora, mes, n)
        if n == 24:
            REC.append(dict(clau=clau, hora=hora, mes=mes, dow=self._dow, pas=self._pas, c=self._c.copy(), s=s.copy(),
                            env=np.array([self.env_hora(clau + k) for k in range(n)]), envmax=self.env_max(mes),
                            fs0=self.feats_solar0(clau, hora, mes, self.k_avui), k_avui=self.k_avui,
                            lp=self.consum_prev,
                            cap=[int(self.tipus(clau, hora, self._dow, k) in (1, 2)) if ptip is not None
                                 else self._hora_futura(hora, self._dow, k)[1] for k in range(n)]))
        return s

    def _decidir(self, obs, soc, cap, pot):
        self._dow = int(obs["day_of_week"])
        return super()._decidir(obs, soc, cap, pot)


def nou():
    a = Rec(usa_model=False, escenaris=0, **KW)
    a.models = {k: v for k, v in (("historial", hist), ("perfil_tipus", ptip)) if v is not None}
    return a


bloc = (tot.index.normalize().map(pd.Timestamp.toordinal) // 7) % 2
for nom, sub in (("A", tot), ("B", tot[bloc == 0]), ("C", tot[bloc == 1])):
    Rec._pas = nom
    avaluar.simular(nou, sub, 0.0, "dificil")
    print("passada", nom, len(sub), len(REC), flush=True)
claus = (tot.index.astype("datetime64[ns]").astype("int64") // m.NS_PER_HORA).values
reals = pd.DataFrame({"clau": claus, "load": tot.load_kw.values, "pv": tot.pv_kw.values, "dia": tot.index.normalize()})
pickle.dump({"rec": REC, "reals": reals}, open(OUT, "wb"))
