"""
agent.py — Agent de control de la bateria (Repte UPC Sostenible, Datathon 2026).

Versió 1: mateixes regles que la plantilla oficial, amb el clamp de seguretat corregit
(rendiment exacte sqrt(0.9) i marge eps) per garantir 0 infraccions.
"""

ETA = 0.9 ** 0.5          # rendiment de càrrega i de descàrrega (exacte, no 0.9487)
EPS = 1e-4                # marge contra errors de coma flotant al límit del SOC
CAPACITAT_KWH = 2000.0
POTENCIA_MAX_KW = 500.0


class BatteryAgent:
    def __init__(self):
        self.llindar_preu_carrega = 0.08      # €/kWh
        self.llindar_preu_descarrega = 0.16   # €/kWh

    def act(self, obs: dict) -> float:
        hora = obs["hour"]
        preu = obs["preu_actual"]
        soc = obs["soc_pct"]
        co2 = obs["co2_actual"]

        consigna = 0.0
        # 1. Vall nocturna (01:00-06:00): càrrega amb preu baix
        if 1 <= hora <= 6 and preu < self.llindar_preu_carrega and soc < 0.85:
            consigna = 350.0
        # 2. Migdia solar (11:00-15:00): càrrega amb preus moderats
        elif 11 <= hora <= 15 and preu < 0.10 and soc < 0.95:
            consigna = 250.0
        # 3. Punta vespertina (18:00-22:00): descàrrega
        elif 18 <= hora <= 22 and (preu > self.llindar_preu_descarrega or co2 > 0.10) and soc > 0.15:
            consigna = -450.0

        return self.clamp(consigna, float(obs["soc_kwh"]))

    @staticmethod
    def clamp(consigna: float, soc_kwh: float) -> float:
        """Limita la consigna als marges físics exactes del simulador (0 infraccions)."""
        max_carrega = max(0.0, min(POTENCIA_MAX_KW, (CAPACITAT_KWH - soc_kwh) / ETA - EPS))
        max_descarrega = max(0.0, min(POTENCIA_MAX_KW, soc_kwh * ETA - EPS))
        return float(max(-max_descarrega, min(max_carrega, consigna)))
