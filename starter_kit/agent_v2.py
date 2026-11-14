"""
agent.py
Plantilla oficial de l'agent de control per al repte de la bateria de UPC Sostenible.
Datathon 2026.
"""

# import joblib  # Descomenteu si utilitzeu models serialitzats (.pkl)
import numpy as np

class BatteryAgent:
    def __init__(self):
        """Inicialització de l'agent.
        S'executa una sola vegada a l'inici del període d'avaluació.
        Permet carregar models predictius o definir llindars d'operació.
        """
        # Paràmetres d'operació de la política de referència
        self.llindar_preu_carrega = 0.08      # €/kWh
        self.llindar_preu_descarrega = 0.16   # €/kWh
        self.potencia_nominal_kw = 1500.0      # Potència màxima de l'inversor (kW)

        # Memòria històrica d'hores passades (permet consultar el passat dins de la simulació):
        self.historial = []

    def act(self, obs: dict) -> float:
        """Determina la consigna de potència activa de la bateria per a l'hora actual.

        Paràmetres:
        -----------
        obs : dict amb les variables d'estat disponibles a l'hora actual:
            - "timestamp": pd.Timestamp amb la data i hora d'avaluació.
            - "hour": int de 0 a 23 (hora del dia).
            - "day_of_week": int de 0 (dilluns) a 6 (diumenge).
            - "soc_kwh": float amb l'energia emmagatzemada actual (0.0 a 2.000.0 kWh).
            - "soc_pct": float amb l'estat de càrrega relatiu [0.0, 1.0].
            - "preu_actual": float amb el preu del PVPC d'aquesta hora (€/kWh).
            - "co2_actual": float amb la intensitat de CO2 peninsular (kg CO2/kWh).
            - "preus_properes_24h": llista de preus horaris de les properes 24 hores (€/kWh).
            - "co2_properes_24h": llista de factors d'emissió de CO2 de les properes 24 hores (kg/kWh).
            - "noves_dades_5min": dades mesurades pels sensors en la darrera hora transcorreguda.

        Retorna:
        --------
        float : Consigna de potència activa en kW:
            > 0  -> Càrrega d'energia des de la xarxa interna (fins a +500.0 kW).
            < 0  -> Descàrrega d'energia cap al campus (fins a -500.0 kW).
            = 0  -> Bateria en mode d'espera (0.0 kW).
        """

        if obs["day_of_week"] > 4 or obs["hour"] < 8 or obs["hour"] > 20:
            return 1 - obs["soc_pct"]

        self.historial.append(obs)
        len_hist = len(self.historial)

        metode_medianes = 0
        if len_hist - 130 > 0:
            ultims_10_dies_preu = []
            ultimes_10_hores_preu = []
            ultims_10_dies_co2 = []
            ultimes_10_hores_co2 = []
            j = len_hist - 13
            for i in range(len_hist - 10, len_hist - 1):
                j -= 13
                ultims_10_dies_preu.append(self.historial[j]["preu_actual"])
                ultimes_10_hores_preu.append(self.historial[i]["preu_actual"])
                ultims_10_dies_co2.append(self.historial[j]["co2_actual"])
                ultimes_10_hores_co2.append(self.historial[i]["co2_actual"])
            preu = np.median(ultims_10_dies_preu + ultimes_10_hores_preu)
            co2 = np.median(ultims_10_dies_co2 + ultimes_10_hores_co2)
            metode_medianes = 7 * (obs["preu_actual"] - preu) / preu \
                + 3 * (obs["co2_actual"] - co2 / co2)

        llindar = 7 * (obs["preu_actual"] - np.mean(obs["preus_properes_24h"])) / np.mean(obs["preus_properes_24h"]) \
            + 3 * (obs["co2_actual"] - np.mean(obs["co2_properes_24h"])) / np.mean(obs["co2_properes_24h"])

        llindar += metode_medianes

        return llindar
