"""
local_eval/avaluar.py
Avaluador local complet: carrega un agent (agent.py, carpeta o submission.zip) igual que el
servidor, passa l'escàner AST oficial, el simula amb el CampusBatteryEnv oficial sobre
setmanes de validació extretes del train i calcula el Score oficial.

Ús (des de l'arrel del repo):
    python local_eval/avaluar.py starter_kit/agent.py --nom plantilla
    python local_eval/avaluar.py submission.zip --plecs final
    python local_eval/avaluar.py agent.py --traca traca.csv     # desa el detall hora a hora

Cada execució s'afegeix a local_eval/resultats/historial.csv per comparar versions.
"""

import argparse
import csv
import hashlib
import importlib.util
import math
import os
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
ARREL = AQUI.parent
sys.path.insert(0, str(ARREL / "starter_kit"))
sys.path.insert(0, str(AQUI))

from environment import CampusBatteryEnv  # noqa: E402  (simulador oficial)
from check_submission import check_ast  # noqa: E402  (escàner AST oficial)
import dades  # noqa: E402

HORES_TEST = 2016
TIMEOUT_S = 60.0
MIDA_MAX_MB = 25.0
PENALITZACIONS = {"p020": 0.20, "p100": 1.00}  # la doc es contradiu: 0,20 (web/env) vs 1,00 (enunciat/FAQ)
FITXER_HISTORIAL = AQUI / "resultats" / "historial.csv"


# --------------------------------------------------------------------------------------
# Càrrega de l'agent (imitant el servidor)
# --------------------------------------------------------------------------------------
def preparar_paquet(origen: Path, desti: Path) -> list[str]:
    """Copia l'agent al directori temporal. Retorna avisos."""
    avisos = []
    if origen.is_dir():
        shutil.copytree(origen, desti, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.ipynb", ".ipynb_checkpoints"))
        avisos.append("Carpeta copiada sencera: al servidor només hi haurà el que poseu al .zip.")
    elif origen.suffix.lower() == ".zip":
        mida = origen.stat().st_size / 1e6
        if mida > MIDA_MAX_MB:
            avisos.append(f"El zip fa {mida:.1f} MB (> {MIDA_MAX_MB} MB): el servidor el rebutjarà.")
        with zipfile.ZipFile(origen) as z:
            z.extractall(desti)
    else:
        shutil.copy(origen, desti / "agent.py")
    return avisos


def carregar_classe(directori: Path):
    """Escàner AST + import de BatteryAgent. Retorna (classe, directori_agent, avisos)."""
    avisos = []
    agents = [f for f in directori.rglob("agent.py")
              if "__MACOSX" not in f.parts and not f.name.startswith("._")]
    if not agents:
        raise RuntimeError("No s'ha trobat cap 'agent.py' al paquet.")
    agent_py = agents[0]
    if agent_py.parent != directori:
        avisos.append("agent.py no és a l'arrel del zip (el servidor demana que hi sigui).")

    for py in directori.rglob("*.py"):
        if "__MACOSX" in py.parts or py.name.startswith("._"):
            continue
        err = check_ast(py)
        if err:
            raise RuntimeError(f"REBUTJAT PER SEGURETAT ({py.name}): {err}")
    avisos += revisar_imports(directori)

    sys.path.insert(0, str(agent_py.parent))
    nom_mod = f"agent_local_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(nom_mod, str(agent_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "BatteryAgent"):
        raise RuntimeError("agent.py no defineix la classe 'BatteryAgent'.")
    return mod.BatteryAgent, agent_py.parent, avisos


LLIBRERIES_ADMESES = {"numpy", "pandas", "scipy", "sklearn", "joblib", "xgboost", "lightgbm"}


def revisar_imports(directori: Path) -> list[str]:
    """Avisa d'imports que no siguin biblioteca estàndard, llibreries admeses o mòduls del paquet
    (p.ex. importar local_eval/dades.py o torch fallaria al servidor)."""
    import ast
    locals_ = {p.stem for p in directori.rglob("*.py")} | {p.name for p in directori.rglob("*") if p.is_dir()}
    avisos = []
    for py in directori.rglob("*.py"):
        if "__MACOSX" in py.parts or py.name.startswith("._"):
            continue
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            noms = []
            if isinstance(node, ast.Import):
                noms = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                noms = [node.module]
            for nom in noms:
                arrel = nom.split(".")[0]
                if arrel not in LLIBRERIES_ADMESES and arrel not in sys.stdlib_module_names \
                        and arrel not in locals_:
                    avisos.append(f"{py.name} importa '{arrel}', que no és al paquet ni a les llibreries admeses.")
    return avisos


def hash_paquet(origen: Path) -> str:
    h = hashlib.sha1()
    fitxers = sorted(origen.rglob("*")) if origen.is_dir() else [origen]
    for f in fitxers:
        if f.is_file() and "__pycache__" not in f.parts:
            h.update(f.read_bytes())
    return h.hexdigest()[:10]


# --------------------------------------------------------------------------------------
# Simulació i mètriques
# --------------------------------------------------------------------------------------
def simular(classe, conjunt: pd.DataFrame, preu_exportacio: float):
    """Una instància nova de l'agent recorre tot el conjunt (com fa el servidor)."""
    t0 = time.perf_counter()
    agent = classe()
    t_init = time.perf_counter() - t0

    env = CampusBatteryEnv(conjunt, export_price=preu_exportacio)
    obs = env.reset()
    done = False
    while not done:
        accio = agent.act(obs)
        try:
            valor = float(accio)
        except (TypeError, ValueError):
            raise RuntimeError(f"Pas {env.t}: act() ha retornat {type(accio).__name__}, no un número.")
        if not math.isfinite(valor):
            raise RuntimeError(f"Pas {env.t} ({obs['timestamp']}): act() ha retornat {valor}.")
        obs, done = env.step(valor)
    return env, t_init, time.perf_counter() - t0


def metriques(env: CampusBatteryEnv, mascara=None) -> dict:
    """Mateixa fórmula que environment.evaluate_results, però amb les dues penalitzacions."""
    n = len(env.grid)
    df = env.df.iloc[:n]
    m = np.ones(n, bool) if mascara is None else np.asarray(mascara)[:n]

    grid = np.asarray(env.grid)[m]
    exp = np.asarray(env.exports)[m]
    preu = df["precio_eur_per_kwh"].values[m]
    co2 = df["co2_kg_per_kwh"].values[m]
    base = np.maximum(df["load_kw"].values - df["pv_kw"].values, 0.0)[m]
    infr = int(np.asarray(env.infraccions_per_step)[m].sum())

    cost_ag = float(np.sum(grid * preu) - np.sum(exp * env.export_price))
    co2_ag = float(np.sum(grid * co2))
    cost_b = float(np.sum(base * preu))
    co2_b = float(np.sum(base * co2))
    d_co2 = 100 * (co2_b - co2_ag) / co2_b if co2_b > 0 else 0.0
    d_cost = 100 * (cost_b - cost_ag) / cost_b if cost_b > 0 else 0.0

    r = {"hores": int(m.sum()), "dco2_pct": d_co2, "dcost_pct": d_cost, "infraccions": infr,
         "co2_estalviat_kg": co2_b - co2_ag, "cost_estalviat_eur": cost_b - cost_ag,
         "kwh_exportats": float(exp.sum())}
    brut = 0.7 * d_co2 + 0.3 * d_cost
    r["score_brut"] = brut
    for k, p in PENALITZACIONS.items():
        r[f"score_{k}"] = max(0.0, brut - p * infr)
    return r


def meitats(env: CampusBatteryEnv):
    """Màscares de primera/segona meitat de cada setmana (aproximació del split públic/privat,
    que l'organització no especifica)."""
    s = env.df["setmana"]
    pos = s.groupby(s).cumcount()
    mida = s.map(s.value_counts())
    primera = (pos < mida / 2).values
    return primera, ~primera


def traca(env: CampusBatteryEnv) -> pd.DataFrame:
    n = len(env.grid)
    df = env.df.iloc[:n].copy()
    df["consigna_kw"] = env.actions
    df["soc_kwh"] = env.soc_series
    df["xarxa_kw"] = env.grid
    df["export_kw"] = env.exports
    df["base_kw"] = np.maximum(df["load_kw"] - df["pv_kw"], 0.0)
    df["infraccio"] = env.infraccions_per_step
    return df


# --------------------------------------------------------------------------------------
# Sortida
# --------------------------------------------------------------------------------------
def fmt(r: dict) -> str:
    return (f"score {r['score_p020']:6.2f}  (brut {r['score_brut']:6.2f})  "
            f"ΔCO2 {r['dco2_pct']:6.2f}%  ΔCost {r['dcost_pct']:6.2f}%  "
            f"infr {r['infraccions']:3d}")


def desar_historial(fila: dict):
    FITXER_HISTORIAL.parent.mkdir(parents=True, exist_ok=True)
    nou = not FITXER_HISTORIAL.exists()
    if not nou:
        with open(FITXER_HISTORIAL, newline="", encoding="utf-8") as f:
            capcalera = next(csv.reader(f), [])
        if set(capcalera) != set(fila):
            # columnes diferents (p.ex. altres plecs): reescriu amb la unió de columnes
            antic = pd.read_csv(FITXER_HISTORIAL)
            pd.concat([antic, pd.DataFrame([fila])]).to_csv(FITXER_HISTORIAL, index=False)
            return
    with open(FITXER_HISTORIAL, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nou:
            w.writeheader()
        w.writerow(fila)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Avaluador local del repte de la bateria")
    ap.add_argument("agent", help="agent.py, carpeta amb agent.py + models, o submission.zip")
    ap.add_argument("--nom", default="", help="etiqueta per a l'historial")
    ap.add_argument("--plecs", default=",".join(dades.PLECS),
                    help=f"plecs a avaluar, separats per comes ({', '.join(dades.PLECS)})")
    ap.add_argument("--preu-exportacio", type=float, default=0.0,
                    help="€/kWh de l'excedent (env i FAQ: 0; la web esmenta 0,04)")
    ap.add_argument("--traca", help="CSV on desar el detall hora a hora del primer plec")
    ap.add_argument("--setmanes", action="store_true", help="mostra el desglossament per setmana")
    ap.add_argument("--refrescar-dades", action="store_true", help="regenera la cache horària")
    ap.add_argument("--sense-historial", action="store_true")
    args = ap.parse_args()

    origen = Path(args.agent).resolve()
    if not origen.exists():
        sys.exit(f"No existeix: {origen}")
    plecs = [p.strip() for p in args.plecs.split(",") if p.strip()]
    for p in plecs:
        if p not in dades.PLECS:
            sys.exit(f"Plec desconegut '{p}'. Opcions: {dades.PLECS}")

    df = dades.carregar_horari(refrescar=args.refrescar_dades)
    print(f"\n=== Avaluació local: {origen.name}  [{hash_paquet(origen)}] ===")

    cwd = os.getcwd()
    resultats = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        avisos = preparar_paquet(origen, tmp)
        try:
            classe, dir_agent, av2 = carregar_classe(tmp)
        except Exception as e:
            sys.exit(f"[ERROR] {type(e).__name__}: {e}")
        avisos += av2
        for a in avisos:
            print(f"[AVÍS] {a}")
        print("[OK] Escàner AST superat i BatteryAgent importat.\n")

        os.chdir(dir_agent)  # el servidor executa des del directori de l'agent
        try:
            for i, plec in enumerate(plecs):
                conjunt = dades.conjunt_validacio(df, plec)
                n_set = conjunt["setmana"].nunique()
                try:
                    env, t_init, t_tot = simular(classe, conjunt, args.preu_exportacio)
                except Exception as e:
                    os.chdir(cwd)
                    sys.exit(f"[ERROR] plec '{plec}': {type(e).__name__}: {e}")

                r = metriques(env)
                a, b = meitats(env)
                r["meitat_A"] = metriques(env, a)["score_p020"]
                r["meitat_B"] = metriques(env, b)["score_p020"]
                r["temps_s"] = t_tot
                resultats[plec] = r

                t_est = t_tot * HORES_TEST / len(conjunt)
                avis_t = "  ⚠ A PROP DEL TIMEOUT" if t_est > TIMEOUT_S / 2 else ""
                print(f"[{plec:5s}] {n_set} setmanes, {len(conjunt)} h | {fmt(r)}")
                print(f"        meitat A {r['meitat_A']:6.2f} | meitat B {r['meitat_B']:6.2f} | "
                      f"score amb pen. 1,00: {r['score_p100']:6.2f} | export {r['kwh_exportats']:.0f} kWh | "
                      f"{t_tot:.1f} s (init {t_init * 1000:.0f} ms, ~{t_est:.1f} s per 2016 h){avis_t}")

                if args.setmanes:
                    for s in sorted(conjunt["setmana"].unique()):
                        m = (env.df["setmana"] == s).values
                        rs = metriques(env, m)
                        d0 = env.df.loc[m, "Fecha"].iloc[0]
                        print(f"          setm {s:2d} {d0:%Y-%m-%d} | {fmt(rs)}")

                if args.traca and i == 0:
                    traca(env).to_csv(args.traca, index=False)
                    print(f"        traça desada a {args.traca}")
        finally:
            os.chdir(cwd)

    if len(resultats) > 1:
        sc = np.array([r["score_p020"] for r in resultats.values()])
        print(f"\n>>> MITJANA {len(sc)} plecs: score {sc.mean():.2f}  (min {sc.min():.2f}, max {sc.max():.2f}, "
              f"desv {sc.std():.2f})")
    print()

    if not args.sense_historial:
        fila = {"data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "nom": args.nom,
                "fitxer": str(origen.relative_to(ARREL)) if origen.is_relative_to(ARREL) else str(origen),
                "hash": hash_paquet(origen), "preu_exp": args.preu_exportacio}
        for plec, r in resultats.items():
            fila[f"{plec}_score"] = round(r["score_p020"], 3)
            fila[f"{plec}_dco2"] = round(r["dco2_pct"], 3)
            fila[f"{plec}_dcost"] = round(r["dcost_pct"], 3)
            fila[f"{plec}_infr"] = r["infraccions"]
        fila["mitjana_score"] = round(float(np.mean([r["score_p020"] for r in resultats.values()])), 3)
        desar_historial(fila)
        print(f"Afegit a {FITXER_HISTORIAL.relative_to(ARREL)}")


if __name__ == "__main__":
    main()
