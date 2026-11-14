"""
agent.py — Agent per al MODE DIFÍCIL (Repte UPC Sostenible, Datathon 2026).

Versió 3: "predir i després planificar".
  1. Prediu les properes 24 h de preu, CO2, consum del campus i producció solar
     (perfils apresos del train + ajust amb el que s'ha observat les darreres hores).
  2. Busca el pla de càrrega/descàrrega que minimitza el que mesura el Score
     (0.7*CO2 + 0.3*cost de l'energia importada), sabent que l'excedent solar és gratuït.
     Es fa amb programació dinàmica sobre l'energia guardada (ràpid: ~1 ms per pla).
  3. Aplica només la consigna de l'hora actual i replanifica a l'hora següent.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided

ETA = 0.9 ** 0.5
EPS = 1e-4
NS_PER_HORA = 3_600_000_000_000

PARAMS = json.loads((Path(__file__).parent / "parametres.json").read_text(encoding="utf-8"))


class BatteryAgent:
    def __init__(self, horitzo=24, pes_ahir=0.5, marge_descarrega=0.0, valor_final=0.0, pas_kwh=50.0):
        self.H = horitzo
        self.pes_ahir = pes_ahir                  # pes de "ahir a la mateixa hora" en la predicció
        self.marge_descarrega = marge_descarrega  # fracció de prudència en descarregar l'hora actual
        self.valor_final = valor_final            # valor de l'energia que queda al final del pla
        self.pas_kwh = pas_kwh                    # resolució de l'energia guardada al planificador
        self.mercat = {}        # hora absoluta -> (co2, preu) reals ja vistos
        self.consum = {}        # hora absoluta -> consum real mesurat (kW)
        self.nivell_consum = 1.0
        self.claredat = None    # última claredat mesurada (solar real / solar de dia clar)
        self.clau_claredat = None
        self._dp = None

    # ------------------------------------------------------------------ utilitats de temps
    @staticmethod
    def _temps(obs):
        ts = pd.Timestamp(obs["timestamp"])
        return int(ts.value // NS_PER_HORA), int(obs["hour"]), int(obs["day_of_week"]), int(ts.month)

    @staticmethod
    def _hora_futura(hora, dow, k):
        h = hora + k
        return h % 24, int(((dow + h // 24) % 7) >= 5)

    # ------------------------------------------------------------------ prediccions
    def prediu_mercat(self, clau, hora, dow, mes, n):
        """Prediu CO2 i preu de les properes n hores (les 3 primeres ja són reals)."""
        co2, preu = np.zeros(n), np.zeros(n)
        for idx, nom, out in ((0, "perfil_co2", co2), (1, "perfil_preu", preu)):
            perfil = PARAMS[nom][str(mes)]
            # Nivell d'avui: valors reals recents dividits pel perfil típic de les mateixes hores
            num = den = 0.0
            for j in range(-21, 3):
                v = self.mercat.get(clau + j)
                if v is not None:
                    hj, cj = self._hora_futura(hora, dow, j)
                    num += v[idx]
                    den += perfil[cj][hj]
            nivell = num / den if den > 0 else 1.0
            for k in range(n):
                v = self.mercat.get(clau + k)
                if v is not None:
                    out[k] = v[idx]
                    continue
                hk, ck = self._hora_futura(hora, dow, k)
                pred = perfil[ck][hk] * nivell
                ahir = self.mercat.get(clau + k - 24)
                if ahir is not None:
                    pred = self.pes_ahir * ahir[idx] + (1 - self.pes_ahir) * pred
                out[k] = pred
        return co2, preu

    def prediu_consum(self, clau, hora, dow, n):
        perfil = PARAMS["perfil_consum"]
        out = np.zeros(n)
        for k in range(n):
            hk, ck = self._hora_futura(hora, dow, k)
            pred = perfil[ck][hk] * self.nivell_consum
            ahir = self.consum.get(clau + k - 24)
            _, cahir = self._hora_futura(hora, dow, k - 24)
            if k > 0 and ahir is not None and ahir > 1.0 and cahir == ck:
                pred = 0.5 * ahir + 0.5 * pred
            out[k] = pred
        return out

    def prediu_solar(self, clau, hora, mes, n):
        env = PARAMS["envolupant_pv"][str(mes)]
        mitja = PARAMS["claredat_mitja"]
        if self.claredat is None or clau - self.clau_claredat > 30:
            k_avui = mitja
        elif clau - self.clau_claredat <= 2:
            k_avui = self.claredat
        else:
            k_avui = 0.5 * self.claredat + 0.5 * mitja
        k_dema = 0.5 * k_avui + 0.5 * mitja
        out = np.zeros(n)
        for k in range(n):
            h = hora + k
            out[k] = env[h % 24] * (k_avui if h < 24 else k_dema)
        return out

    # ------------------------------------------------------------------ planificació
    def _prepara_dp(self, cap, pot):
        """Graella d'energia guardada i moviments possibles en una hora (només depenen de
        quants graons es puja o es baixa, per això es poden precalcular)."""
        n_graons = int(round(cap / self.pas_kwh))
        graella = np.linspace(0.0, cap, n_graons + 1)
        pas = cap / n_graons
        puja = int(np.floor(pot * ETA / pas + 1e-9))       # graons màxims carregant
        baixa = int(np.floor(pot / ETA / pas + 1e-9))      # graons màxims descarregant
        energia = np.arange(-baixa, puja + 1) * pas
        potencia = np.where(energia >= 0, energia / ETA, energia * ETA)   # kW vistos des de la xarxa
        # Memòria de treball: valor de cada nivell d'energia, envoltat d'infinits, i una "vista"
        # on la fila i conté els valors dels nivells als quals es pot arribar des del nivell i.
        mem = np.full(baixa + n_graons + 1 + puja, np.inf)
        pas_bytes = mem.strides[0]
        finestres = as_strided(mem, shape=(n_graons + 1, baixa + puja + 1),
                               strides=(pas_bytes, pas_bytes), writeable=False)
        self._dp = {"cap": cap, "pot": pot, "graella": graella, "baixa": baixa, "potencia": potencia,
                    "mem": mem, "finestres": finestres, "tmp": np.empty((n_graons + 1, baixa + puja + 1))}

    def planificar(self, w, net, soc, cap, pot):
        if self._dp is None or self._dp["cap"] != cap or self._dp["pot"] != pot:
            self._prepara_dp(cap, pot)
        d = self._dp
        graella, potencia, mem, tmp = d["graella"], d["potencia"], d["mem"], d["tmp"]
        centre = slice(d["baixa"], d["baixa"] + len(graella))
        n = len(w)
        V = -self.valor_final * float(np.mean(w)) * graella       # valor de l'energia al final
        for k in range(n - 1, 0, -1):
            mem[centre] = V
            cost_salt = w[k] * np.maximum(net[k] + potencia, 0.0)  # cost de cada moviment a l'hora k
            np.add(d["finestres"], cost_salt, out=tmp)
            V = tmp.min(axis=1)

        # Hora actual: accions candidates contínues (el SOC real no cau sobre la graella)
        max_c = max(0.0, min(pot, (cap - soc) / ETA - EPS))
        max_d = max(0.0, min(pot, soc * ETA - EPS, max(0.0, net[0] * (1.0 - self.marge_descarrega))))
        cand = np.concatenate([np.linspace(-max_d, max_c, 241), [0.0, min(max_c, max(0.0, -net[0]))]])
        soc1 = soc + np.where(cand >= 0, cand * ETA, cand / ETA)
        cost = w[0] * np.maximum(net[0] + cand, 0.0) + np.interp(soc1, graella, V) + 1e-6 * np.abs(cand)
        return float(cand[int(np.argmin(cost))])

    # ------------------------------------------------------------------ decisió
    def act(self, obs: dict) -> float:
        cap = float(obs.get("cap_max_kwh", 6000.0))
        pot = float(obs.get("power_max_kw", 1500.0))
        soc = float(obs["soc_kwh"])
        try:
            consigna = self._decidir(obs, soc, cap, pot)
        except Exception:
            consigna = 0.0
        max_c = max(0.0, min(pot, (cap - soc) / ETA - EPS))
        max_d = max(0.0, min(pot, soc * ETA - EPS))
        return float(max(-max_d, min(max_c, consigna)))

    def _decidir(self, obs, soc, cap, pot):
        clau, hora, dow, mes = self._temps(obs)
        preus, co2s = list(obs["preus_properes_24h"]), list(obs["co2_properes_24h"])
        for j, (c, p) in enumerate(zip(co2s, preus)):
            self.mercat[clau + j] = (float(c), float(p))

        # --- telemetria de l'hora anterior
        dades = obs.get("noves_dades_5min") or {}
        consum_prev = float(dades.get("ultim_consum_kw", 0.0))
        solar_prev = float(dades.get("ultima_solar_kw", 0.0))
        avaria = consum_prev <= 1.0          # el comptador no envia dades
        h_prev, c_prev = self._hora_futura(hora, dow, -1)
        if not avaria:
            self.consum[clau - 1] = consum_prev
            self.nivell_consum = consum_prev / PARAMS["perfil_consum"][c_prev][h_prev]
            env_prev = PARAMS["envolupant_pv"][str(mes)][h_prev]
            if env_prev > 100.0:
                self.claredat = float(np.clip(solar_prev / env_prev, 0.0, 1.3))
                self.clau_claredat = clau - 1

        # --- horitzó: si el vector de mercat s'escurça, s'acaba la simulació
        n = self.H if len(preus) >= 3 else len(preus)
        co2_f, preu_f = self.prediu_mercat(clau, hora, dow, mes, n)
        w = 0.7 * co2_f / PARAMS["co2_mitja"] + 0.3 * preu_f / PARAMS["preu_mitja"]
        net = self.prediu_consum(clau, hora, dow, n) - self.prediu_solar(clau, hora, mes, n)
        if avaria:
            net[0] = 0.0                     # el simulador comptarà consum zero aquesta hora
        return self.planificar(w, net, soc, cap, pot)
