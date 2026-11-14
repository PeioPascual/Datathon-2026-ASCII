"""
agent.py — Agent de control de la bateria (Repte UPC Sostenible, Datathon 2026).

Versió 2: planificació òptima amb programació lineal (MPC) sobre les properes 24 hores.

Idea: mentre la bateria no faci exportar el campus, el Score és lineal en la consigna:
    Score = -sum_t P_t * w_t,   amb  w_t = 0.7*co2_t/CO2_mitja + 0.3*preu_t/preu_mitja
Per tant, minimitzar sum w_t*(carrega_t - descarrega_t) amb les restriccions físiques
de la bateria dona exactament l'estratègia òptima per al Score. Com que els vectors
de 24 h de preu i CO2 són coneguts, el pla és vàlid i només cal replanificar cada
poques hores (més ràpid) o si l'estat se'n desvia.
"""

import numpy as np
from scipy.optimize import linprog

ETA = 0.9 ** 0.5          # rendiment de càrrega i de descàrrega
EPS = 1e-4                # marge contra errors de coma flotant
CAPACITAT_KWH = 2000.0
POTENCIA_MAX_KW = 500.0
MARGE_SOC_KWH = 0.01      # el pla no toca exactament 0 ni 2000 kWh

# Mitjanes del train ponderades per la importació del campus (normalitzen CO2 i preu
# a la mateixa escala, tal com ho fa el Score respecte al baseline).
CO2_MITJA = 0.0840058      # kg CO2/kWh
PREU_MITJA = 0.1430437     # €/kWh
PES_CO2, PES_COST = 0.70, 0.30


class BatteryAgent:
    def __init__(self, replanificar_cada: int = 6):
        self.replanificar_cada = replanificar_cada
        self.pla = []             # consignes planificades (kW)
        self.soc_esperat = []     # SOC esperat a l'inici de cada hora del pla
        self.idx = 0
        self._L = np.tril(np.ones((24, 24)))

    # ------------------------------------------------------------------ planificació
    def planificar(self, preus, co2s, soc_kwh, max_desc_ara):
        w = PES_CO2 * np.asarray(co2s) / CO2_MITJA + PES_COST * np.asarray(preus) / PREU_MITJA
        h = len(w)
        L = self._L[:h, :h]
        # SOC al final de cada hora k: soc + ETA*cumsum(c) - cumsum(d)/ETA
        A = np.hstack([ETA * L, -L / ETA])
        A_ub = np.vstack([A, -A])
        b_ub = np.concatenate([np.full(h, CAPACITAT_KWH - MARGE_SOC_KWH - soc_kwh),
                               np.full(h, soc_kwh - MARGE_SOC_KWH)])
        cost = np.concatenate([w, -w])
        limits = [(0.0, POTENCIA_MAX_KW)] * h + [(0.0, POTENCIA_MAX_KW)] * h
        limits[h] = (0.0, max(0.0, min(POTENCIA_MAX_KW, max_desc_ara)))
        r = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=limits, method="highs")
        if r.status != 0:
            return [0.0]
        return list(r.x[:h] - r.x[h:])

    # ------------------------------------------------------------------ decisió
    def act(self, obs: dict) -> float:
        soc = float(obs["soc_kwh"])
        dades = obs.get("noves_dades_5min") or {}
        # No descarregar més del consum net de l'hora anterior (evita exportar a 0 €).
        net_prev = float(dades.get("ultim_consum_kw", POTENCIA_MAX_KW)) - float(dades.get("ultima_solar_kw", 0.0))

        cal_replanificar = (
            self.idx >= len(self.pla)
            or self.idx >= self.replanificar_cada
            or abs(self.soc_esperat[self.idx] - soc) > 1.0
        )
        if cal_replanificar:
            try:
                self.pla = self.planificar(obs["preus_properes_24h"], obs["co2_properes_24h"], soc, net_prev)
            except Exception:
                self.pla = [0.0]
            self.soc_esperat = [soc]
            for p in self.pla:
                s = self.soc_esperat[-1]
                self.soc_esperat.append(s + p * ETA if p > 0 else s + p / ETA)
            self.idx = 0

        consigna = self.pla[self.idx]
        self.idx += 1
        if consigna < 0:
            consigna = max(consigna, -max(0.0, net_prev))
        return self.clamp(consigna, soc)

    @staticmethod
    def clamp(consigna: float, soc_kwh: float) -> float:
        """Limita la consigna als marges físics exactes del simulador (0 infraccions)."""
        max_carrega = max(0.0, min(POTENCIA_MAX_KW, (CAPACITAT_KWH - soc_kwh) / ETA - EPS))
        max_descarrega = max(0.0, min(POTENCIA_MAX_KW, soc_kwh * ETA - EPS))
        return float(max(-max_descarrega, min(max_carrega, consigna)))
