"""
local_eval/sostre.py
Calcula el MÀXIM POSSIBLE de puntuació (el "sostre") per a cada joc de setmanes: el pla que
faria un agent que conegués per endavant tot el que passarà (preus, CO2, consum i solar).
Cap agent real el pot superar; serveix per saber quant marge queda.

Ús (des de l'arrel del repo):
    python local_eval/sostre.py                  # mode normal
    python local_eval/sostre.py --mode dificil
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import eye, hstack, vstack, csr_matrix

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import avaluar  # noqa: E402
import dades  # noqa: E402
import modes  # noqa: E402

EPS = 1e-4


def pla_optim(conjunt, cap, pot, eta, soc0):
    """Programació lineal amb informació perfecta. Variables per hora: càrrega c, descàrrega d,
    energia guardada s i importació de xarxa g (g >= consum net + c - d, g >= 0).
    Minimitza el mateix que mesura el Score: sum(w * g)."""
    n = len(conjunt)
    net = (conjunt["load_kw"] - conjunt["pv_kw"]).values
    co2 = conjunt["co2_kg_per_kwh"].values
    preu = conjunt["precio_eur_per_kwh"].values
    base = np.maximum(net, 0.0)
    w = 0.7 * co2 / np.average(co2, weights=base) + 0.3 * preu / np.average(preu, weights=base)

    I = eye(n, format="csr")
    Z = csr_matrix((n, n))
    D = I - eye(n, k=-1, format="csr")                      # s_t - s_{t-1}
    A_eq = hstack([-eta * I, I / eta, D, Z]).tocsr()         # s_t - s_{t-1} = eta*c - d/eta
    b_eq = np.zeros(n)
    b_eq[0] = soc0
    A_ub = hstack([I, -I, Z, -I]).tocsr()                    # net + c - d - g <= 0
    b_ub = -net
    cost = np.concatenate([np.full(n, 1e-7), np.full(n, 1e-7), np.zeros(n), w])
    limits = [(0, pot)] * n + [(0, pot)] * n + [(0, cap)] * n + [(0, None)] * n
    r = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=limits, method="highs")
    if r.status != 0:
        raise RuntimeError(f"La LP no ha convergit: {r.message}")
    return r.x[:n] - r.x[n:2 * n]


def agent_repeticio(pla, cap, pot, eta):
    """Agent que executa un pla ja calculat, amb el clamp de seguretat."""
    class Repeticio:
        def __init__(self):
            self.i = 0

        def act(self, obs):
            p = float(pla[self.i])
            self.i += 1
            soc = float(obs["soc_kwh"])
            max_c = max(0.0, min(pot, (cap - soc) / eta - EPS))
            max_d = max(0.0, min(pot, soc * eta - EPS))
            return float(max(-max_d, min(max_c, p)))
    return Repeticio


def sostre(conjunt, mode):
    """Retorna les mètriques oficials del pla òptim per a un conjunt ja preparat per al mode."""
    env0 = modes.EnvMode(conjunt, mode=mode)
    pla = pla_optim(conjunt, env0.cap, env0.power, env0.eta_c, env0.soc_init)
    env, _, _ = avaluar.simular(agent_repeticio(pla, env0.cap, env0.power, env0.eta_c), conjunt, 0.0, mode)
    return avaluar.metriques(env)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Màxim possible de puntuació (informació perfecta)")
    ap.add_argument("--mode", default="normal", choices=list(modes.MODES))
    ap.add_argument("--plecs", default=",".join(dades.PLECS))
    args = ap.parse_args()

    df = dades.carregar_horari()
    print(f"\n=== Màxim possible (agent que coneix tot el futur) — mode {args.mode.upper()} ===")
    scores = []
    for plec in [p.strip() for p in args.plecs.split(",") if p.strip()]:
        conjunt = modes.preparar_conjunt(dades.conjunt_validacio(df, plec), args.mode)
        r = sostre(conjunt, args.mode)
        scores.append(r["score_brut"])
        print(f"[{plec:5s}] {avaluar.fmt(r)}  | export {r['kwh_exportats']:.0f} kWh")
    print(f"\n>>> MITJANA: {np.mean(scores):.2f}\n")


if __name__ == "__main__":
    main()
