"""
local_eval/dades.py
Construeix la taula horària del campus (load_kw, pv_kw, preu, co2) a partir de data/train/
i genera les setmanes de validació que imiten el test ocult (blocs de 7 dies per mes).
"""

from pathlib import Path

import pandas as pd

ARREL = Path(__file__).resolve().parent.parent
DIR_DADES = ARREL / "data" / "train"
CACHE = Path(__file__).resolve().parent / "cache" / "campus_hourly_v2.pkl"

COLS_BASE = ["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]
# Meteorologia (només la fa servir el Mode Difícil): nom intern -> text que identifica la columna
COLS_METEO = {
    "irradiancia_wm2": "IRRADIANCIA (W/m2)",
    "temp_ambient_c": "TEMPERATURA AMBIENTE",
    "temp_panel_c": "TEMPERATURA PANEL",
    "vent_ms": "VELOCIDAD VIENTO",
}

COL_CONTADOR = "Contador - POTENCIA ACTIVA EXPORTADA (kW)"
# Noms exactes: el filtre "POTENCIA ACTIVA" sense "FASE" de la web també agafa
# les columnes "SETPOINT RELATIVO POTENCIA ACTIVA (%)", que no són potència.
COLS_INVERSOR = [f"Inversor {i} - POTENCIA ACTIVA (kW)" for i in (1, 2, 3)]


def carregar_horari(refrescar: bool = False) -> pd.DataFrame:
    """Taula horària amb el mateix procediment que el tutorial oficial (mitjana horària + dropna)."""
    if CACHE.exists() and not refrescar:
        return pd.read_pickle(CACHE)

    c = pd.read_csv(DIR_DADES / "contador.csv", usecols=["Fecha", COL_CONTADOR],
                    parse_dates=["Fecha"]).set_index("Fecha")
    load = (-c[COL_CONTADOR]).clip(lower=0, upper=5000).resample("1h").mean()

    inv = pd.read_csv(DIR_DADES / "inversor.csv", usecols=["Fecha", *COLS_INVERSOR],
                      parse_dates=["Fecha"]).set_index("Fecha")
    pv = inv[COLS_INVERSOR].fillna(0).sum(axis=1).clip(lower=0).resample("1h").mean()

    preu = pd.read_csv(DIR_DADES / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
    co2 = pd.read_csv(DIR_DADES / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")

    df = pd.DataFrame({
        "load_kw": load,
        "pv_kw": pv,
        "precio_eur_per_kwh": preu["precio_eur_per_kwh"],
        "co2_kg_per_kwh": co2["co2_kg_per_kwh"],
    }).dropna()

    # Meteorologia: mitjana horària. No elimina cap fila (les hores avaluades són les mateixes
    # que sense meteo); els forats s'omplen amb l'últim valor conegut.
    capcalera = pd.read_csv(DIR_DADES / "meteo.csv", nrows=0).columns
    tria = {nom: next(c for c in capcalera if text in c) for nom, text in COLS_METEO.items()}
    met = pd.read_csv(DIR_DADES / "meteo.csv", usecols=["Fecha", *tria.values()],
                      parse_dates=["Fecha"]).set_index("Fecha")
    met = met.apply(pd.to_numeric, errors="coerce").resample("1h").mean()
    for nom, col in tria.items():
        df[nom] = met[col].reindex(df.index).ffill().fillna(0.0)
    df["irradiancia_wm2"] = df["irradiancia_wm2"].clip(lower=0)
    df.index.name = "Fecha"

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(CACHE)
    return df


# --- Esquemes de setmanes de validació -------------------------------------------------
# El test real són els últims 7 dies de cada mes. Al train cada mes té ~els dies 1..21/24,
# així que podem fer diversos "plecs" de 7 dies per mes:
#   final : els 7 últims dies disponibles de cada mes (el més semblant temporalment al test)
#   d01   : dies 1-7     d08 : dies 8-14     d15 : dies 15-21
PLECS = ("final", "d01", "d08", "d15")
MIN_HORES_SETMANA = 120  # descarta setmanes amb massa forats (p.ex. agost incomplet)


def setmanes(df: pd.DataFrame, plec: str) -> list[pd.DataFrame]:
    """Retorna la llista de blocs setmanals (un per mes) per a un plec donat."""
    blocs = []
    for _, mes in df.groupby(df.index.to_period("M")):
        dies = mes.index.normalize()
        if plec == "final":
            ultim = dies.max()
            inici, fi = ultim - pd.Timedelta(days=6), ultim
        else:
            d0 = int(plec[1:])
            inici = mes.index[0].normalize().replace(day=d0)
            fi = inici + pd.Timedelta(days=6)
        bloc = mes[(dies >= inici) & (dies <= fi)]
        if len(bloc) >= MIN_HORES_SETMANA:
            blocs.append(bloc)
    return blocs


def conjunt_validacio(df: pd.DataFrame, plec: str) -> pd.DataFrame:
    """Concatena les setmanes en una sola sèrie, igual que el test ocult (amb salts de temps).
    Afegeix la columna 'setmana' (índex del bloc) per desglossar resultats."""
    blocs = setmanes(df, plec)
    out = pd.concat([b.assign(setmana=i) for i, b in enumerate(blocs)])
    return out
