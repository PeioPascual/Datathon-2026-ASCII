"""
agent.py — Agent per al MODE DIFÍCIL (Repte UPC Sostenible, Datathon 2026).

Versió 5: "predir i després planificar", amb model d'arbres (entrenat amb XGBoost) per al
preu i el CO2, i tres ajustos de prudència (marge en descarregar, consum actual, excedent solar).
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

DIR = Path(__file__).parent
PARAMS = json.loads((DIR / "parametres.json").read_text(encoding="utf-8"))
NOMS_MERCAT = ("co2", "preu")
N_FEATS = 19


def carrega_models(directori):
    """Carrega els arbres de decisió de preu i CO2 (entrenats amb XGBoost i desats com a taules
    de números a model_mercat.npz). Si el fitxer no hi és o falla, retorna {} i l'agent fa servir
    la recepta senzilla."""
    models = {}
    try:
        fitxer = Path(directori) / "model_mercat.npz"
        if fitxer.exists():
            z = np.load(fitxer)
            for idx, nom in enumerate(NOMS_MERCAT):
                models[idx] = {k: z[f"{nom}_{k}"] for k in
                               ("yes", "no", "miss", "feat", "thr", "val", "arrels", "prof", "base")}
    except Exception:
        models = {}
    return models


def prediu_arbres(P, X):
    """Recorre tots els arbres alhora amb numpy: a cada nivell, cada (arbre, fila) baixa a
    l'esquerra o a la dreta segons el llindar; les fulles apunten a si mateixes."""
    X = np.asarray(X, np.float32)
    n, F = X.shape
    T = len(P["arrels"])
    node = np.repeat(P["arrels"], n)
    fila = np.tile(np.arange(n, dtype=np.int64) * F, T)
    xf = X.ravel()
    feat, thr, yes, no, miss = P["feat"], P["thr"], P["yes"], P["no"], P["miss"]
    hi_ha_nan = bool((xf != xf).any())
    for _ in range(int(P["prof"])):
        fx = xf[fila + feat[node]]
        seg = np.where(fx < thr[node], yes[node], no[node])
        if hi_ha_nan:
            falta = fx != fx
            seg[falta] = miss[node][falta]
        node = seg
    return float(P["base"]) + P["val"][node].reshape(T, n).sum(axis=0)


class BatteryAgent:
    def __init__(self, horitzo=24, pes_ahir=0.5, marge_descarrega=0.05, valor_final=0.0, pas_kwh=50.0,
                 usa_model=True, pes_persistencia=0.5, extra_excedent=50.0, llindar_excedent=200.0):
        self.H = horitzo
        self.pes_ahir = pes_ahir                  # pes de "ahir a la mateixa hora" en la predicció
        # Prudència en descarregar l'hora actual: es descarrega com a molt el 95% del consum net
        # previst. Passar-se regala energia; quedar-se curt només l'ajorna.
        self.marge_descarrega = marge_descarrega
        self.valor_final = valor_final            # valor de l'energia que queda al final del pla
        self.pas_kwh = pas_kwh                    # resolució de l'energia guardada al planificador
        # Consum de l'hora actual: barreja de "igual que fa una hora" i "perfil típic ajustat"
        self.pes_persistencia = pes_persistencia
        # Prop de l'excedent solar es suposen uns kW més de sobrant: si falla es compra una mica
        # al migdia (barat); si no es fes, es perdria energia gratuïta.
        self.extra_excedent = extra_excedent
        self.llindar_excedent = llindar_excedent
        self.consum_prev = None
        self.mercat = {}        # hora absoluta -> (co2, preu) reals ja vistos
        self.consum = {}        # hora absoluta -> consum real mesurat (kW)
        self.nivell_consum = 1.0
        self.claredat = None    # última claredat mesurada (solar real / solar de dia clar)
        self.clau_claredat = None
        self._dp = None
        self.models = carrega_models(DIR) if usa_model else {}

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
    def feats_mercat(self, idx, clau, hora, dow, mes, n):
        """Per a una variable de mercat (0=CO2, 1=preu) retorna:
             nivell : mitjana dels valors reals recents (escala d'avui)
             recepta: predicció senzilla de cada hora, relativa al nivell (perfil + ahir)
             X      : variables d'entrada del model per a les hores k >= 3 (una fila per hora)
        Tot es calcula en relatiu al nivell perquè el model no depengui de l'escala del dia."""
        perfil = PARAMS["perfil_" + NOMS_MERCAT[idx]][str(mes)]
        vals, profs = [], []
        for j in range(-21, 3):
            v = self.mercat.get(clau + j)
            if v is not None:
                hj, cj = self._hora_futura(hora, dow, j)
                vals.append(v[idx])
                profs.append(perfil[cj][hj])
        if not vals:
            return 1.0, np.ones(n), np.zeros((0, N_FEATS))
        nivell = sum(vals) / len(vals)
        niv_perfil = sum(profs) / len(profs)
        nan = float("nan")

        def rel(k):
            v = self.mercat.get(clau + k)
            return v[idx] / nivell if v is not None else nan

        x0, x1, x2 = rel(0), rel(1), rel(2)
        ahir3 = [rel(j - 24) for j in range(3)]
        avui3 = [x0, x1, x2]
        if any(a != a for a in ahir3 + avui3) or sum(ahir3) <= 0:
            tendencia = nan
        else:
            tendencia = sum(avui3) / sum(ahir3)
        vmin, vmax = min(vals) / nivell, max(vals) / nivell
        fresca = self.claredat is not None and clau - self.clau_claredat <= 6
        claredat = self.claredat if fresca else nan
        _, c0 = self._hora_futura(hora, dow, 0)

        recepta = np.zeros(n)
        files = []
        for k in range(n):
            hk, ck = self._hora_futura(hora, dow, k)
            prof_rel = perfil[ck][hk] / niv_perfil
            a1 = rel(k - 24)
            base = prof_rel if a1 != a1 else self.pes_ahir * a1 + (1 - self.pes_ahir) * prof_rel
            recepta[k] = base
            if k >= 3:
                ang = 2 * np.pi * hk / 24
                files.append([k, hk, np.sin(ang), np.cos(ang), ck, c0, x0, x1, x2, a1, rel(k - 48),
                              prof_rel, base, vmin, vmax, tendencia, claredat, len(vals), hora])
        return nivell, recepta, np.array(files, dtype=float).reshape(-1, N_FEATS)

    def prediu_mercat(self, clau, hora, dow, mes, n):
        """Prediu CO2 i preu de les properes n hores. Les hores ja conegudes es deixen tal qual;
        la resta surten dels arbres de decisió (si hi són) o de la recepta senzilla."""
        sortida = []
        for idx in (0, 1):
            nivell, recepta, X = self.feats_mercat(idx, clau, hora, dow, mes, n)
            pred = recepta.copy()
            model = self.models.get(idx)
            if model is not None and len(X):
                try:
                    pred[3:] = np.clip(prediu_arbres(model, X), 0.05, 5.0)
                except Exception:
                    pass
            out = pred * nivell
            for k in range(n):
                v = self.mercat.get(clau + k)
                if v is not None:
                    out[k] = v[idx]
            sortida.append(out)
        return sortida[0], sortida[1]

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
        if self.consum_prev is not None and n > 0:
            out[0] = (1 - self.pes_persistencia) * out[0] + self.pes_persistencia * self.consum_prev
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
        self.consum_prev = None if avaria else consum_prev
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
        if net[0] < self.llindar_excedent:
            net[0] -= self.extra_excedent
        if avaria:
            net[0] = 0.0                     # el simulador comptarà consum zero aquesta hora
        return self.planificar(w, net, soc, cap, pot)
