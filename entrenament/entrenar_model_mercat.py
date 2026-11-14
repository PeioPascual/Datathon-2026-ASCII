"""
entrenament/entrenar_model_mercat.py
Entrena els models que prediuen el PREU i el CO2 de les hores +3..+23 per a l'agent del
Mode Difícil, i els desa a submission_dificil/model_mercat.npz.

Aquest script NO es puja al servidor. XGBoost només es fa servir aquí, per entrenar: els arbres
resultants es desen com a taules de números i l'agent els recorre amb numpy (més ràpid i sense
dependre de la versió de XGBoost del servidor).

Ús (des de l'arrel del repo; abans cal haver generat parametres.json):
    python entrenament/construir_parametres.py
    python entrenament/entrenar_model_mercat.py            # entrena el model final
    python entrenament/entrenar_model_mercat.py --honest   # a més, mesura la puntuació sense trampa

Com es construeixen els exemples
  Per a cada hora t del train i cada distància k (3..23) es reprodueix exactament el que sabria
  l'agent en aquell moment (les mateixes variables, calculades pel mateix codi: feats_mercat de
  agent.py) i s'apunta el valor real que va venir després. Tot va en relatiu al "nivell del dia".
  Un 15% dels exemples simulen una arrencada en fred (poc historial), com passa a l'inici de
  cada setmana de test.

Validació honesta (--honest)
  Per a cada joc de setmanes d'avaluació s'entrena un model SENSE aquelles setmanes (ni els dos
  dies del voltant) i es mesura la puntuació de l'agent amb aquell model. És el número fiable.
"""

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL / "local_eval"))
import avaluar  # noqa: E402
import dades  # noqa: E402
import modes  # noqa: E402

DIR_AGENT = ARREL / "submission_dificil"
HIPER = dict(n_estimators=150, learning_rate=0.10, max_depth=5, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=10, tree_method="hist", n_jobs=4, random_state=0)
FRACCIO_FRED = 0.15


def carrega_agent():
    spec = importlib.util.spec_from_file_location("agent_dificil", str(DIR_AGENT / "agent.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------
# Exemples d'entrenament
# --------------------------------------------------------------------------------------
class DiccionariLimitat(dict):
    """Diccionari que només 'veu' les claus entre lo i hi: imita el que l'agent coneix a l'hora t."""
    lo, hi = -10**18, 10**18

    def get(self, k, d=None):
        return d if (k > self.hi or k < self.lo) else dict.get(self, k, d)


def clau_hora(index, ns_per_hora):
    return (index.astype("datetime64[ns]").astype("int64") // ns_per_hora)


def construir_exemples(m):
    """Retorna X, Y (objectiu relatiu al nivell), B (predicció de la recepta) i T (dia de cada exemple)."""
    preu = pd.read_csv(dades.DIR_DADES / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
    co2 = pd.read_csv(dades.DIR_DADES / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")
    mk = pd.DataFrame({"co2": co2["co2_kg_per_kwh"], "preu": preu["precio_eur_per_kwh"]}).dropna()
    claus = clau_hora(mk.index, m.NS_PER_HORA)
    reals = {int(k): (float(a), float(b)) for k, a, b in zip(claus, mk.co2, mk.preu)}

    # Claredat solar tal com la mesura l'agent (solar real / solar de dia clar)
    hor = dades.carregar_horari()
    hor = hor[hor.load_kw > 0]
    claredat = {}
    for clau, ts, pv in zip(clau_hora(hor.index, m.NS_PER_HORA), hor.index, hor.pv_kw * 12.0):
        env = m.PARAMS["envolupant_pv"][str(ts.month)][ts.hour]
        if env > 100:
            claredat[int(clau)] = float(np.clip(pv / env, 0, 1.3))

    rng = np.random.default_rng(0)
    agent = m.BatteryAgent(usa_model=False)
    agent.mercat = DiccionariLimitat(reals)
    X, Y, B, T = {0: [], 1: []}, {0: [], 1: []}, {0: [], 1: []}, []
    for ts, clau in zip(mk.index, claus):
        clau = int(clau)
        if clau + 1 not in reals or clau + 2 not in reals:
            continue
        agent.mercat.hi = clau + 2
        fred = rng.random() < FRACCIO_FRED
        agent.mercat.lo = clau - int(rng.integers(0, 25)) if fred else -10**18
        agent.claredat, agent.clau_claredat = None, None
        for j in range(1, 7):
            if clau - j in claredat:
                agent.claredat, agent.clau_claredat = claredat[clau - j], clau - j
                break
        for idx in (0, 1):
            nivell, recepta, Xk = agent.feats_mercat(idx, clau, ts.hour, ts.dayofweek, ts.month, 24)
            for i, k in enumerate(range(3, 24)):
                real = reals.get(clau + k)
                if real is None:
                    continue
                X[idx].append(Xk[i])
                Y[idx].append(real[idx] / nivell)
                B[idx].append(recepta[k])
                if idx == 0:
                    T.append(ts.normalize())
    return ({i: np.array(X[i]) for i in X}, {i: np.array(Y[i]) for i in Y},
            {i: np.array(B[i]) for i in B}, pd.DatetimeIndex(T))


# --------------------------------------------------------------------------------------
# Entrenament i exportació dels arbres a taules de números
# --------------------------------------------------------------------------------------
def entrena(X, Y, mascara):
    models = {}
    for idx in (0, 1):
        r = xgb.XGBRegressor(**HIPER)
        r.fit(X[idx][mascara], Y[idx][mascara])
        models[idx] = r
    return models


def exporta_arbres(regressor, n_feats):
    """Converteix un model XGBoost en vectors plans. Les fulles apunten a si mateixes, de manera
    que l'agent només ha de baixar 'prof' nivells sense comprovar si ja és a una fulla."""
    booster = regressor.get_booster()
    df = booster.trees_to_dataframe()
    n_arbres, n_nodes = int(df.Tree.max()) + 1, int(df.Node.max()) + 1
    feat = np.zeros((n_arbres, n_nodes), np.int64)
    thr = np.full((n_arbres, n_nodes), np.inf, np.float32)
    val = np.zeros((n_arbres, n_nodes), np.float32)
    base = (np.arange(n_arbres) * n_nodes)[:, None]
    propi = base + np.arange(n_nodes)[None, :]
    yes, no, miss = propi.copy(), propi.copy(), propi.copy()
    num = lambda s: int(s.split("-")[1])  # noqa: E731
    for r in df.itertuples(index=False):
        if r.Feature == "Leaf":
            val[r.Tree, r.Node] = r.Gain
        else:
            feat[r.Tree, r.Node] = int(r.Feature[1:])
            thr[r.Tree, r.Node] = r.Split
            o = r.Tree * n_nodes
            yes[r.Tree, r.Node], no[r.Tree, r.Node], miss[r.Tree, r.Node] = o + num(r.Yes), o + num(r.No), o + num(r.Missing)
    P = dict(yes=yes.ravel(), no=no.ravel(), miss=miss.ravel(), feat=feat.ravel(), thr=thr.ravel(),
             val=val.ravel(), arrels=(np.arange(n_arbres) * n_nodes).astype(np.int64),
             prof=np.int32(HIPER["max_depth"]), base=np.float32(0.0))
    return P, booster


def desa_models(models, fitxer, m, n_feats):
    """Exporta els dos models, comprova que numpy dona el mateix que XGBoost i desa el .npz."""
    rng = np.random.default_rng(1)
    prova = (rng.random((500, n_feats)) * 2).astype(np.float32)
    prova[rng.random(prova.shape) < 0.15] = np.nan
    sortida, dif_max = {}, 0.0
    for idx, nom in enumerate(m.NOMS_MERCAT):
        P, booster = exporta_arbres(models[idx], n_feats)
        ref = booster.inplace_predict(prova)
        P["base"] = np.float32(np.mean(ref - m.prediu_arbres(P, prova)))   # valor base del model
        dif_max = max(dif_max, float(np.abs(m.prediu_arbres(P, prova) - ref).max()))
        for k, v in P.items():
            sortida[f"{nom}_{k}"] = v
    if dif_max > 1e-3:
        raise RuntimeError(f"La predicció amb numpy no coincideix amb XGBoost (dif. {dif_max:.2e})")
    fitxer.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fitxer, **sortida)
    return dif_max


# --------------------------------------------------------------------------------------
# Validació honesta
# --------------------------------------------------------------------------------------
def dies_plec(hor, plec, marge_dies=0):
    dies = set()
    for b in dades.setmanes(hor, plec):
        ini = b.index.min().normalize() - pd.Timedelta(days=marge_dies)
        fi = b.index.max().normalize() + pd.Timedelta(days=marge_dies)
        dies.update(pd.date_range(ini, fi))
    return list(dies)


def validacio_honesta(m, X, Y, B, T):
    hor = dades.carregar_horari()
    n_feats = X[0].shape[1]
    dir_tmp = ARREL / "local_eval" / "cache" / "models_honestos"
    print("\n=== Validació honesta: cada model s'entrena sense les setmanes amb què s'avalua ===")
    res = []
    for plec in dades.PLECS:
        fora = T.isin(dies_plec(hor, plec, marge_dies=2))
        dins = T.isin(dies_plec(hor, plec))
        models = entrena(X, Y, ~fora)
        errors = []
        for idx, nom in enumerate(m.NOMS_MERCAT):
            e_model = np.abs(models[idx].predict(X[idx][dins]) - Y[idx][dins]).mean()
            e_recepta = np.abs(B[idx][dins] - Y[idx][dins]).mean()
            errors.append(f"error {nom}: recepta {e_recepta:.3f} -> model {e_model:.3f}")
        fitxer = dir_tmp / plec / "model_mercat.npz"
        desa_models(models, fitxer, m, n_feats)
        arbres = m.carrega_models(fitxer.parent)

        conjunt = modes.preparar_conjunt(dades.conjunt_validacio(hor, plec), "dificil")
        punts = {}
        for nom, mods in (("recepta", {}), ("model", arbres)):
            def fabrica(mods=mods):
                a = m.BatteryAgent(usa_model=False)
                a.models = mods
                return a
            env, _, _ = avaluar.simular(fabrica, conjunt, 0.0, "dificil")
            punts[nom] = avaluar.metriques(env)["score_brut"]
        res.append(punts)
        print(f"[{plec:5s}] puntuació: recepta {punts['recepta']:.2f} -> model {punts['model']:.2f} "
              f"({punts['model'] - punts['recepta']:+.2f}) | " + " | ".join(errors), flush=True)
    mr, mm = np.mean([r["recepta"] for r in res]), np.mean([r["model"] for r in res])
    print(f">>> MITJANA honesta: recepta {mr:.2f} -> model {mm:.2f} ({mm - mr:+.2f})")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Entrena el model de preu i CO2 del Mode Difícil")
    ap.add_argument("--honest", action="store_true", help="mesura també la puntuació sense trampa")
    args = ap.parse_args()

    m = carrega_agent()
    X, Y, B, T = construir_exemples(m)
    print(f"Exemples d'entrenament: {len(T)} | variables: {X[0].shape[1]}")

    if args.honest:
        validacio_honesta(m, X, Y, B, T)

    models = entrena(X, Y, np.ones(len(T), bool))
    fitxer = DIR_AGENT / "model_mercat.npz"
    dif = desa_models(models, fitxer, m, X[0].shape[1])
    print(f"\nModel final (entrenat amb totes les dades) desat a {fitxer.relative_to(ARREL)} "
          f"({fitxer.stat().st_size / 1e6:.2f} MB). Diferència màxima numpy vs XGBoost: {dif:.1e}")


if __name__ == "__main__":
    main()
