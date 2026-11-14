"""
entrenament/construir_parametres.py
Calcula, a partir de data/train/, les taules que fa servir l'agent del Mode Difícil
(submission_dificil/agent.py) i les desa a submission_dificil/parametres.json.

Aquest script NO es puja al servidor: només s'executa en local per generar el JSON.

Ús (des de l'arrel del repo):
    python entrenament/construir_parametres.py

Taules generades:
  - perfil_co2 / perfil_preu : valor típic per [mes][feiner o cap de setmana][hora]
  - envolupant_pv            : producció solar d'un dia clar (percentil 95) per [mes][hora], ja x12
  - perfil_consum            : consum típic del campus per [feiner o cap de setmana][hora]
  - co2_mitja / preu_mitja   : mitjanes ponderades per la importació, per posar CO2 i preu
                               a la mateixa escala que el Score
  - claredat_mitja           : fracció mitjana de la solar de dia clar que es produeix realment
Els mesos sense dades al train (novembre) es calculen com la mitjana dels dos mesos veïns.
"""

import json
import sys
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL / "local_eval"))
import dades  # noqa: E402

FACTOR_SOLAR = 12.0          # el Mode Difícil multiplica la planta solar per 12
SORTIDA = ARREL / "submission_dificil" / "parametres.json"


def omple_mesos(t: dict) -> dict:
    """Mesos sense dades = mitjana dels dos mesos veïns."""
    for m in range(1, 13):
        if m not in t:
            abans, despres = t[(m - 2) % 12 + 1], t[m % 12 + 1]
            t[m] = (0.5 * (np.array(abans) + np.array(despres))).tolist()
    return {str(m): t[m] for m in range(1, 13)}


def main():
    df = dades.carregar_horari().copy()
    df = df[df.load_kw > 0]                       # fora les hores d'avaria del comptador
    df["pv"] = df.pv_kw * FACTOR_SOLAR
    df["mes"] = df.index.month
    df["hora"] = df.index.hour
    df["cap"] = (df.index.dayofweek >= 5).astype(int)   # 0 feiner, 1 cap de setmana

    def taula_mes(col):
        t = {}
        for m in range(1, 13):
            sub = df[df.mes == m]
            if len(sub) < 24 * 5:
                continue
            t[m] = [[float(sub[(sub.cap == c) & (sub.hora == h)][col].mean()) for h in range(24)]
                    for c in (0, 1)]
        return omple_mesos(t)

    def envolupant_pv():
        t = {}
        for m in range(1, 13):
            sub = df[df.mes == m]
            if len(sub) < 24 * 5:
                continue
            t[m] = [float(sub[sub.hora == h].pv.quantile(0.95)) if (sub.hora == h).any() else 0.0
                    for h in range(24)]
        return omple_mesos(t)

    env = envolupant_pv()
    e = np.array([env[str(m)][h] for m, h in zip(df.mes, df.hora)])
    ok = e > 100
    claredat = float(np.clip(df.pv.values[ok] / e[ok], 0, 1.2).mean())
    perfil_consum = [[float(df[(df.cap == c) & (df.hora == h)].load_kw.mean()) for h in range(24)]
                     for c in (0, 1)]
    base = (df.load_kw - df.pv).clip(lower=0)

    params = {
        "co2_mitja": float(np.average(df.co2_kg_per_kwh, weights=base)),
        "preu_mitja": float(np.average(df.precio_eur_per_kwh, weights=base)),
        "perfil_co2": taula_mes("co2_kg_per_kwh"),
        "perfil_preu": taula_mes("precio_eur_per_kwh"),
        "perfil_consum": perfil_consum,
        "envolupant_pv": env,
        "claredat_mitja": claredat,
    }
    SORTIDA.parent.mkdir(parents=True, exist_ok=True)
    SORTIDA.write_text(json.dumps(params, indent=0), encoding="utf-8")
    print(f"Desat {SORTIDA.relative_to(ARREL)} ({SORTIDA.stat().st_size / 1024:.0f} KB)")
    print(f"  co2_mitja {params['co2_mitja']:.4f} | preu_mitja {params['preu_mitja']:.4f} | "
          f"claredat_mitja {claredat:.2f}")


if __name__ == "__main__":
    main()
