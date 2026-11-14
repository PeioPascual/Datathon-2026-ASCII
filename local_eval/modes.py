"""
local_eval/modes.py
Definició dels dos modes del repte (normal i difícil) i entorn de simulació que els imita.

La física i la puntuació són sempre les del CampusBatteryEnv oficial (starter_kit/environment.py);
aquí només s'afegeix el que la pestanya "Mode Difícil" de la web descriu i el simulador
descarregat no implementa:
  - vectors de preu i CO2 retallats a 3 hores (l'actual i les 2 següents)
  - meteorologia de la darrera hora dins obs["noves_dades_5min"]
  - obs["cap_max_kwh"] i obs["power_max_kw"]
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "starter_kit"))
from environment import CampusBatteryEnv  # noqa: E402

CAMPS_METEO = ("irradiancia_wm2", "temp_ambient_c", "temp_panel_c", "vent_ms")

MODES = {
    "normal": {
        "spec": None,            # el battery_spec.json oficial (2.000 kWh, 500 kW)
        "factor_solar": 1.0,
        "horitzo": 24,
        "extres_obs": False,     # el simulador oficial no dona meteo ni cap_max/power_max
    },
    "dificil": {
        "spec": {"capacity_kwh": 6000.0, "power_kw": 1500.0, "efficiency": 0.90, "soc_init": 0.50,
                 "soc_min": 0.0, "soc_max": 1.0, "dt_hours": 1.0, "forecast_horizon": 3},
        "factor_solar": 12.0,
        "horitzo": 3,
        "extres_obs": True,
    },
}


def preparar_conjunt(conjunt: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Aplica al conjunt de validació els canvis de dades del mode (solar x12 al Mode Difícil)."""
    out = conjunt.copy()
    out["pv_kw"] = out["pv_kw"] * MODES[mode]["factor_solar"]
    return out


class EnvMode(CampusBatteryEnv):
    """CampusBatteryEnv oficial + les diferències d'observació de cada mode."""

    def __init__(self, hourly_data: pd.DataFrame, mode: str = "normal", export_price: float = 0.0):
        cfg = MODES[mode]
        self.mode = mode
        self.horitzo = cfg["horitzo"]
        self.extres_obs = cfg["extres_obs"]
        super().__init__(hourly_data, spec=cfg["spec"], export_price=export_price)

    def _get_obs(self):
        obs = super()._get_obs()
        if obs is None:
            return None
        if self.horitzo < 24:
            obs["preus_properes_24h"] = obs["preus_properes_24h"][: self.horitzo]
            obs["co2_properes_24h"] = obs["co2_properes_24h"][: self.horitzo]
        if self.extres_obs:
            # Igual que el consum i la solar: lectures de l'hora t-1 (o de l'hora 0 al primer pas)
            fila = self.df.iloc[self.t - 1 if self.t > 0 else self.t]
            for camp in CAMPS_METEO:
                obs["noves_dades_5min"][camp] = float(fila[camp]) if camp in fila else 0.0
            obs["cap_max_kwh"] = float(self.cap)
            obs["power_max_kw"] = float(self.power)
        return obs
