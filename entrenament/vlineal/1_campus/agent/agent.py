"""
agent.py — Agent per al MODE DIFÍCIL (Repte UPC Sostenible, Datathon 2026).

Versió 7: "predir i després planificar", tenint en compte la incertesa de les previsions.
  1. Prediu les properes 24 h de preu, CO2 (arbres de decisió entrenats amb XGBoost), consum
     del campus i producció solar (perfils del train + el que s'ha observat les darreres hores;
     model d'arbres per al consum de l'hora actual).
  2. Genera escenaris d'error del consum net (quantils dels errors observats en el train, segons
     l'horitzó, l'hora, el tipus de dia, la claredat del cel i l'altura del sol).
  3. Busca el pla de càrrega/descàrrega que minimitza el cost ESPERAT del que mesura el Score
     (0.7*CO2 + 0.3*cost de l'energia importada), sabent que l'excedent solar és gratuït.
     Programació dinàmica sobre l'energia guardada.
  4. Aplica només la consigna de l'hora actual i replanifica a l'hora següent.
Detalls pràctics:
  - Quan la simulació salta a una setmana nova, recupera de la taula del train (historial.npz)
    les 72 h anteriors (el train acaba just el dia abans de cada setmana de test).
  - De nit el comptador no s'actualitza (de 23 h a 5 h repeteix el valor de les 22 h).
  - Canvi d'hora (octubre i març): la solar i el perfil de CO2 es calculen en hora solar (UTC).
  - Calendari: festius i vacances de la universitat (4 tipus de dia per al consum) i festius
    estatals de data fixa per al preu (els peatges hi apliquen l'horari de cap de setmana).
Tots els fitxers de dades són opcionals: si en falta algun, l'agent fa servir la recepta senzilla.
"""

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided

ETA = 0.9 ** 0.5
EPS = 1e-4
NS_PER_HORA = 3_600_000_000_000
NAN = float("nan")

DIR = Path(__file__).parent
PARAMS = json.loads((DIR / "parametres.json").read_text(encoding="utf-8"))
NOMS_MERCAT = ("co2", "preu")
N_FEATS = 19
CAMPS_ARBRE = ("yes", "no", "miss", "feat", "thr", "val", "arrels", "prof", "base")
NIT = (23, 0, 1, 2, 3, 4, 5)     # hores en què el comptador repeteix la lectura de les 22 h


# ------------------------------------------------------------------ calendari
# Festius amb el campus tancat (estatals, Catalunya i locals de Barcelona), set 2025 - set 2026
FESTIUS = ("2025-09-11", "2025-09-24", "2025-10-12", "2025-11-01", "2025-12-06", "2025-12-08",
           "2025-12-25", "2025-12-26", "2026-01-01", "2026-01-06", "2026-04-03", "2026-04-06",
           "2026-05-01", "2026-05-25", "2026-06-24", "2026-08-15", "2026-09-11", "2026-09-24")
# Festius estatals de data fixa: els peatges del PVPC (2.0TD) hi apliquen l'horari de cap de setmana
NACIONALS = ("2025-10-12", "2025-11-01", "2025-12-06", "2025-12-08", "2025-12-25", "2026-01-01",
             "2026-01-06", "2026-05-01", "2026-08-15", "2026-10-12")
# Vacances de la universitat (feiners amb el campus a mig gas)
VACANCES = (("2025-12-24", "2026-01-06"), ("2026-03-28", "2026-04-06"), ("2026-08-01", "2026-08-31"))


def _dia(text):
    return (datetime.date.fromisoformat(text) - datetime.date(1970, 1, 1)).days


DIES_FESTIUS = frozenset(_dia(t) for t in FESTIUS)
DIES_NACIONALS = frozenset(_dia(t) for t in NACIONALS)
DIES_VACANCES = frozenset(d for a, b in VACANCES for d in range(_dia(a), _dia(b) + 1))


def tipus_dia(dia):
    """0 feiner, 1 dissabte, 2 diumenge o festiu, 3 feiner de vacances (dia = dies des de 1970)."""
    dow = (dia + 3) % 7
    if dia in DIES_FESTIUS or dow == 6:
        return 2
    if dow == 5:
        return 1
    return 3 if dia in DIES_VACANCES else 0


def tipus_consum(clau):
    """Tipus de dia que mana el consum de l'hora absoluta 'clau': les hores de 0 a 5 h segueixen
    el dia anterior (la nit de dilluns s'assembla a la de diumenge)."""
    return tipus_dia((clau - 6) // 24)


def cap_setmana_preu(clau):
    """1 si l'hora és de dissabte, diumenge o festiu estatal de data fixa (horari de peatges)."""
    dia = clau // 24
    return int((dia + 3) % 7 >= 5 or dia in DIES_NACIONALS)


def carrega_models(directori):
    """Carrega els fitxers de dades opcionals (cadascun pel seu compte: si un falla, els altres
    se segueixen fent servir):
      model_mercat.npz      arbres de preu i CO2
      model_campus.npz      arbres del consum de l'hora actual (consum0) i quantils de la
                            claredat de l'hora actual (solar0q0..q6)
      incert.json           quantils dels errors de consum i solar
      historial.npz         taula horària del train (per recuperar les hores anteriors)
      perfils_consum.json   perfil de consum (4 tipus de dia x 24 h) segons el calendari"""
    directori = Path(directori)
    models = {}
    try:
        z = np.load(directori / "model_mercat.npz")
        for idx, nom in enumerate(NOMS_MERCAT):
            models[idx] = {k: z[f"{nom}_{k}"] for k in CAMPS_ARBRE}
    except Exception:
        models = {}
    try:
        z = np.load(directori / "model_campus.npz")
        for nom in sorted({f[:-4] for f in z.files if f.endswith("_yes")}):
            models[nom] = {k: z[f"{nom}_{k}"] for k in CAMPS_ARBRE}
    except Exception:
        pass
    try:
        t = json.loads((directori / "incert.json").read_text(encoding="utf-8"))
        K = t["k_max"] + 1
        t["_L"] = np.array([t["L"][str(k)] for k in range(K)], dtype=float)      # (K, 2, 24, Q)
        t["_P"] = np.array([t["P"][str(k)] for k in range(K)], dtype=float)      # (K, nc, na, QP)
        models["incert"] = t
    except Exception:
        pass
    try:
        z = np.load(directori / "historial.npz")
        models["historial"] = {k: z[k] for k in z.files}
    except Exception:
        pass
    try:
        t = json.loads((directori / "perfils_consum.json").read_text(encoding="utf-8"))["perfil_consum_tipus"]
        if len(t) == 4 and all(len(f) == 24 and min(f) > 1.0 for f in t):
            models["perfil_tipus"] = [[float(v) for v in f] for f in t]
    except Exception:
        pass
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
    feat, thr, yes, no = P["feat"], P["thr"], P["yes"], P["no"]
    esq = P.get("_esq")
    if esq is None:      # nodes on un valor que falta va cap a l'esquerra (es calcula un sol cop)
        esq = P["_esq"] = P["miss"] == P["yes"]
    hi_ha_nan = bool((xf != xf).any())
    for _ in range(int(P["prof"])):
        fx = xf[fila + feat[node]]
        esquerra = fx < thr[node]
        if hi_ha_nan:
            esquerra |= (fx != fx) & esq[node]
        node = np.where(esquerra, yes[node], no[node])
    return float(P["base"]) + P["val"][node].reshape(T, n).sum(axis=0)


def bin_de(x, llindars):
    for i, l in enumerate(llindars):
        if x < l:
            return i
    return len(llindars)


def ultim_diumenge(any_, mes):
    d = datetime.date(any_, mes, 31)
    return 31 - (d.weekday() + 1) % 7


def desfasament_utc(any_, mes, dia, hora):
    """Hora oficial de Madrid: UTC+2 de l'últim diumenge de març (03:00) a l'últim d'octubre (03:00)."""
    if 3 < mes < 10:
        return 2
    if mes == 3:
        u = ultim_diumenge(any_, 3)
        return 2 if (dia > u or (dia == u and hora >= 3)) else 1
    if mes == 10:
        u = ultim_diumenge(any_, 10)
        return 2 if (dia < u or (dia == u and hora < 2)) else 1
    return 1


def data_de(clau):
    return datetime.datetime(1970, 1, 1) + datetime.timedelta(hours=int(clau))


class BatteryAgent:
    def __init__(self, horitzo=24, pes_ahir=0.5, marge_descarrega=0.05, valor_final=0.0, pas_kwh=50.0,
                 usa_model=True, pes_persistencia=0.5, extra_excedent=50.0, llindar_excedent=200.0,
                 escenaris=24, nq_futur=9, usa_consum0=True, usa_solar0=True,
                 hist_hores=72, nit_congelada=True, canvi_hora=True, cal_consum=True, cal_preu=True):
        self.H = horitzo
        self.pes_ahir = pes_ahir                  # pes de "ahir a la mateixa hora" en la predicció
        # Sense taules d'incertesa (recepta v5): es descarrega com a molt el 95% del consum net
        # previst i prop de l'excedent solar se suposen 50 kW més de sobrant.
        self.marge_descarrega = marge_descarrega
        self.extra_excedent = extra_excedent
        self.llindar_excedent = llindar_excedent
        self.valor_final = valor_final            # valor de l'energia que queda al final del pla
        self.pas_kwh = pas_kwh                    # resolució de l'energia guardada al planificador
        # Consum de l'hora actual: barreja de "igual que fa una hora" i "perfil típic ajustat"
        self.pes_persistencia = pes_persistencia
        # Incertesa del consum net: 0 = predicció puntual (v5); n >= 1 = cost esperat a les hores
        # 0..n-1 (a les futures amb nq_futur quantils, per rapidesa)
        self.escenaris = escenaris
        self.nq_futur = nq_futur
        self.usa_consum0, self.usa_solar0 = usa_consum0, usa_solar0
        self.hist_hores = hist_hores              # hores del train que es recuperen en un salt
        self.nit_congelada = nit_congelada
        self.canvi_hora = canvi_hora and "envolupant_pv_utc" in PARAMS
        # Calendari: tipus de dia per al consum (si hi ha perfils_consum.json) i festius per al preu
        self.cal_consum = cal_consum
        self.cal_preu = cal_preu and "perfil_preu_festius" in PARAMS

        self.consum_prev = None
        self.mercat = {}        # hora absoluta -> (co2, preu) reals ja vistos
        self.consum = {}        # hora absoluta -> consum real mesurat (kW)
        self.clar = {}          # hora absoluta -> claredat mesurada (solar real / solar de cel clar)
        self.nivell_consum = 1.0
        self.claredat = None    # última claredat mesurada
        self.clau_claredat = None
        self.claredat_mercat = None   # la mateixa, a l'escala amb què es va entrenar el model de mercat
        self.clau_claredat_mercat = None
        self.k_avui = None
        self.clau_anterior = None
        self.temp_prev = self.tpanel_prev = self.irr_prev = NAN
        self._mes = 1
        self._clau = 0
        self._cache_env = {}
        self._cache_dh = {}
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

    def delta_hora(self, clau):
        """Hores a sumar a l'hora de rellotge perquè coincideixi amb l'hora oficial que tenia el
        mes al train (el train d'octubre i març és anterior al canvi d'hora; el test, no)."""
        if not self.canvi_hora:
            return 0
        v = self._cache_dh.get(clau)
        if v is None:
            ts = data_de(clau)
            off_train = 2 if 4 <= ts.month <= 10 else 1
            v = 0 if ts.month == 11 else off_train - desfasament_utc(ts.year, ts.month, ts.day, ts.hour)
            if len(self._cache_dh) > 5000:
                self._cache_dh = {}
            self._cache_dh[clau] = v
        return v

    def env_hora(self, clau):
        """PV de cel clar (kW) de l'hora absoluta 'clau' (hora de rellotge). Amb canvi_hora es
        busca per hora UTC, de manera que la posició del sol no es desplaça amb el canvi d'hora."""
        v = self._cache_env.get(clau)
        if v is None:
            ts = data_de(clau)
            if self.canvi_hora:
                off = desfasament_utc(ts.year, ts.month, ts.day, ts.hour)
                v = PARAMS["envolupant_pv_utc"][str(ts.month)][(ts.hour - off) % 24]
            else:
                v = PARAMS["envolupant_pv"][str(ts.month)][ts.hour]
            if len(self._cache_env) > 5000:
                self._cache_env = {}
            self._cache_env[clau] = v
        return v

    def env_max(self, mes):
        taula = PARAMS["envolupant_pv_utc"] if self.canvi_hora else PARAMS["envolupant_pv"]
        return max(taula[str(mes)])

    # ------------------------------------------------------------------ mercat
    def feats_mercat(self, idx, clau, hora, dow, mes, n):
        """Per a una variable de mercat (0=CO2, 1=preu) retorna:
             nivell : mitjana dels valors reals recents (escala d'avui)
             recepta: predicció senzilla de cada hora, relativa al nivell (perfil + ahir)
             X      : variables d'entrada del model per a les hores k >= 3 (una fila per hora)
        Tot es calcula en relatiu al nivell perquè el model no depengui de l'escala del dia.
        Per al CO2, després del canvi d'hora el perfil es llegeix a l'hora del train (delta_hora)."""
        dh = (lambda kk: self.delta_hora(clau + kk)) if idx == 0 else (lambda kk: 0)
        if idx == 1 and self.cal_preu:
            # preu: els festius estatals de data fixa compten com a cap de setmana
            perfil = PARAMS["perfil_preu_festius"][str(mes)]
            cap = lambda kk: cap_setmana_preu(clau + kk)  # noqa: E731
        else:
            perfil = PARAMS["perfil_" + NOMS_MERCAT[idx]][str(mes)]
            cap = lambda kk: self._hora_futura(hora, dow, kk)[1]  # noqa: E731
        vals, profs = [], []
        for j in range(-21, 3):
            v = self.mercat.get(clau + j)
            if v is not None:
                hj, cj = (hora + j) % 24, cap(j)
                vals.append(v[idx])
                profs.append(perfil[cj][(hj + dh(j)) % 24])
        if not vals:
            return 1.0, np.ones(n), np.zeros((0, N_FEATS))
        nivell = sum(vals) / len(vals)
        niv_perfil = sum(profs) / len(profs)

        def rel(k):
            v = self.mercat.get(clau + k)
            return v[idx] / nivell if v is not None else NAN

        x0, x1, x2 = rel(0), rel(1), rel(2)
        ahir3 = [rel(j - 24) for j in range(3)]
        avui3 = [x0, x1, x2]
        if any(a != a for a in ahir3 + avui3) or sum(ahir3) <= 0:
            tendencia = NAN
        else:
            tendencia = sum(avui3) / sum(ahir3)
        vmin, vmax = min(vals) / nivell, max(vals) / nivell
        fresca = self.claredat_mercat is not None and clau - self.clau_claredat_mercat <= 6
        claredat = self.claredat_mercat if fresca else NAN
        c0 = cap(0)

        recepta = np.zeros(n)
        files = []
        for k in range(n):
            hk, ck = (hora + k + dh(k)) % 24, cap(k)
            prof_rel = perfil[ck][hk] / niv_perfil
            a1 = rel(k - 24)
            base = prof_rel if a1 != a1 else self.pes_ahir * a1 + (1 - self.pes_ahir) * prof_rel
            recepta[k] = base
            if k >= 3:
                ang = 2 * np.pi * hk / 24
                files.append([k, hk, np.sin(ang), np.cos(ang), ck, c0, x0, x1, x2, a1, rel(k - 48),
                              prof_rel, base, vmin, vmax, tendencia, claredat, len(vals), (hora + dh(0)) % 24])
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

    # ------------------------------------------------------------------ campus
    def perfil_tipus(self):
        """Perfil de consum per tipus de dia (4 x 24) o None (llavors feiner / cap de setmana)."""
        return self.models.get("perfil_tipus") if self.cal_consum else None

    def tipus(self, clau, hora, dow, k):
        """Tipus de dia de l'hora clau+k: calendari (0-3) o, sense perfils, 0 feiner / 1 cap de setmana."""
        if self.perfil_tipus() is not None:
            return tipus_consum(clau + k)
        return self._hora_futura(hora, dow, k)[1]

    def prediu_consum(self, clau, hora, dow, n):
        perfil = self.perfil_tipus() or PARAMS["perfil_consum"]
        out = np.zeros(n)
        for k in range(n):
            hk, ck = (hora + k) % 24, self.tipus(clau, hora, dow, k)
            pred = perfil[ck][hk] * self.nivell_consum
            # "ahir a la mateixa hora" només si ahir era un dia del mateix tipus
            ahir = self.consum.get(clau + k - 24)
            if k > 0 and ahir is not None and ahir > 1.0 and self.tipus(clau, hora, dow, k - 24) == ck:
                pred = 0.5 * ahir + 0.5 * pred
            out[k] = pred
        if self.consum_prev is not None and n > 0:
            out[0] = (1 - self.pes_persistencia) * out[0] + self.pes_persistencia * self.consum_prev
            model = self.models.get("consum0") if self.usa_consum0 else None
            if model is not None:
                try:
                    X = np.array([self.feats_consum0(clau, hora, dow, self._mes)], dtype=float)
                    out[0] = self.consum_prev + float(prediu_arbres(model, X)[0])
                except Exception:
                    pass
            # De nit el comptador repeteix l'última lectura: si l'última hora observada és de
            # 22 h a 4 h, les hores que queden fins a les 5 h valdran exactament el mateix.
            h_prev = (hora - 1) % 24
            if self.nit_congelada and (h_prev >= 22 or h_prev <= 4):
                for k in range(n):
                    if (hora + k) % 24 not in NIT:
                        break
                    out[k] = self.consum_prev
        return out

    def feats_consum0(self, clau, hora, dow, mes):
        """Variables del model de consum de l'hora actual (objectiu: consum(t) - consum(t-1))."""
        perfil = self.perfil_tipus() or PARAMS["perfil_consum"]

        def g(c):
            v = self.consum.get(c)
            return v if (v is not None and v > 1.0) else NAN
        l1, l2, l3, l24, l25 = g(clau - 1), g(clau - 2), g(clau - 3), g(clau - 24), g(clau - 25)
        h0, c0 = hora, self.tipus(clau, hora, dow, 0)
        hp, cp = (hora - 1) % 24, self.tipus(clau, hora, dow, -1)
        ang = 2 * np.pi * hora / 24
        return [hora, c0, mes, l1, l1 - l2, l2 - l3, l24 - l25, l24 - l1, perfil[c0][h0] / perfil[cp][hp],
                perfil[c0][h0], self.temp_prev, np.sin(ang), np.cos(ang)]

    def feats_solar0(self, clau, hora, mes, k_avui):
        """Variables dels models de quantils de la claredat de l'hora actual."""
        e0, e1 = self.env_hora(clau), self.env_hora(clau - 1)
        g = lambda c: self.clar.get(c, NAN)  # noqa: E731
        cl1, cl2, cl3 = g(clau - 1), g(clau - 2), g(clau - 3)
        avui = [self.clar[c] for c in range(clau - hora, clau) if c in self.clar]
        ahir = [self.clar[c] for c in range(clau - hora - 24, clau - hora) if c in self.clar]
        m_avui = sum(avui) / len(avui) if avui else NAN
        m_ahir = sum(ahir) / len(ahir) if ahir else NAN
        ang = 2 * np.pi * hora / 24
        return [hora, mes, e0, e0 / max(e1, 1.0), e0 / self.env_max(mes), k_avui, cl1, cl2, cl3, cl1 - cl2,
                m_avui, min(avui) if avui else NAN, max(avui) if avui else NAN, len(avui), m_ahir,
                self.irr_prev, self.tpanel_prev - self.temp_prev, np.sin(ang), np.cos(ang)]

    def prediu_solar(self, clau, hora, mes, n):
        mitja = PARAMS["claredat_mitja"]
        if self.claredat is None or clau - self.clau_claredat > 30:
            k_avui = mitja
        elif clau - self.clau_claredat <= 2:
            k_avui = self.claredat
        else:
            k_avui = 0.5 * self.claredat + 0.5 * mitja
        self.k_avui = k_avui
        k_dema = 0.5 * k_avui + 0.5 * mitja
        out = np.zeros(n)
        for k in range(n):
            out[k] = self.env_hora(clau + k) * (k_avui if hora + k < 24 else k_dema)
        return out

    def registra_solar(self, clau_h, solar_kw):
        """Guarda la claredat mesurada a l'hora absoluta clau_h (si el sol és prou alt)."""
        env = self.env_hora(clau_h)
        if env > 100.0:
            self.claredat = float(np.clip(solar_kw / env, 0.0, 1.3))
            self.clau_claredat = clau_h
            self.clar[clau_h] = self.claredat
        # la variable de claredat del model de mercat es manté a l'escala amb què es va entrenar
        ts = data_de(clau_h)
        env_m = PARAMS["envolupant_pv"][str(ts.month)][(ts.hour + self.delta_hora(clau_h)) % 24]
        if env_m > 100.0:
            self.claredat_mercat = float(np.clip(solar_kw / env_m, 0.0, 1.3))
            self.clau_claredat_mercat = clau_h

    def precarrega(self, clau):
        """Salt de temps: copia a la memòria les hores [clau - hist_hores, clau - 1] de la taula
        del train (només hores passades). Retorna la fila de l'hora clau-1 si hi és, si no None."""
        H = self.models.get("historial")
        if H is None or self.hist_hores <= 0:
            return None
        a = int(np.searchsorted(H["clau"], clau - self.hist_hores))
        b = int(np.searchsorted(H["clau"], clau))
        fila = None
        for i in range(a, b):
            k = int(H["clau"][i])
            c, p = float(H["co2"][i]), float(H["preu"][i])
            ld, pv = float(H["consum"][i]), float(H["solar"][i])
            if c == c and p == p and k not in self.mercat:
                self.mercat[k] = (c, p)
            if ld == ld and ld > 1.0:
                self.consum[k] = ld
                pv = pv if pv == pv else 0.0
                if k == clau - 1:
                    fila = {"ultim_consum_kw": ld, "ultima_solar_kw": pv}
                    for nom in ("irradiancia_wm2", "temp_ambient_c", "temp_panel_c"):
                        if nom in H and H[nom][i] == H[nom][i]:
                            fila[nom] = float(H[nom][i])
                else:
                    self.registra_solar(k, pv)
        return fila

    # ------------------------------------------------------------------ incertesa del consum net
    def escenaris_error(self, clau, hora, dow, mes, m, consum, solar):
        """Matriu (S, m) d'errors possibles del consum net (real - previst) per a les hores 0..m-1.
        Errors de consum (relatius) i de solar (relatius al cel clar) combinats com a independents."""
        t = self.models.get("incert")
        if t is None:
            return None
        ks = np.arange(m)
        hs = (hora + ks) % 24
        if self.perfil_tipus() is not None:   # festius com a cap de setmana; vacances com a feiner
            cs = np.array([int(tipus_consum(clau + k) in (1, 2)) for k in range(m)])
        else:
            cs = (((dow + (hora + ks) // 24) % 7) >= 5).astype(int)
        kk = np.minimum(ks, t["k_max"])
        ql = t["_L"][kk, cs, hs] * consum[:m, None]                                   # (m, Q)
        kc = self.k_avui if self.k_avui is not None else PARAMS["claredat_mitja"]
        bc = bin_de(kc, t["llindars_clar"])
        e = np.array([self.env_hora(clau + k) for k in range(m)])
        be = np.digitize(e / self.env_max(mes), t["llindars_alt"])
        qp = t["_P"][kk, bc, be] * e[:, None]                                         # (m, QP)
        qp[e <= t["env_min"]] = 0.0
        # Hora actual: quantils de la claredat donats pels models d'arbres (si hi són)
        if self.usa_solar0 and "solar0q0" in self.models and e[0] > t["env_min"] and self.k_avui is not None:
            try:
                X = np.array([self.feats_solar0(clau, hora, mes, self.k_avui)], dtype=float)
                qs = []
                i = 0
                while ("solar0q%d" % i) in self.models:
                    qs.append(float(prediu_arbres(self.models["solar0q%d" % i], X)[0]))
                    i += 1
                qs = np.clip(np.sort(qs), 0.0, 1.5) * e[0] - solar[0]
                if len(qs) == qp.shape[1]:
                    qp[0] = qs
            except Exception:
                pass
        return (ql[:, :, None] - qp[:, None, :]).reshape(m, -1).T

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
        self._dp = {"cap": cap, "pot": pot, "graella": graella, "baixa": baixa, "puja": puja,
                    "potencia": potencia, "mem": mem, "finestres": finestres,
                    "tmp": np.empty((n_graons + 1, baixa + puja + 1))}

    def _valor(self, w, imp):
        """Valor de cada nivell d'energia a l'inici de l'hora 1 (recursió enrere), per a un sol
        vector de pesos w. imp[k, j] = energia importada (esperada) a l'hora k amb el moviment j."""
        d = self._dp
        graella, mem, tmp = d["graella"], d["mem"], d["tmp"]
        centre = slice(d["baixa"], d["baixa"] + len(graella))
        V = -self.valor_final * float(np.mean(w)) * graella       # valor de l'energia al final
        for k in range(len(w) - 1, 0, -1):
            mem[centre] = V
            np.add(d["finestres"], w[k] * imp[k], out=tmp)
            V = tmp.min(axis=1)
        return V

    def planificar(self, w, net, soc, cap, pot, E=None):
        """Tria la consigna de l'hora actual. w: pes del Score per hora (previsió puntual);
        net: consum net previst; E: escenaris d'error del consum net (S, m) o None."""
        if self._dp is None or self._dp["cap"] != cap or self._dp["pot"] != pot:
            self._prepara_dp(cap, pot)
        d = self._dp
        graella, potencia = d["graella"], d["potencia"]
        n = len(w)
        m = 0 if E is None else E.shape[1]
        # Energia importada a cada hora per a cada moviment; a les hores amb escenaris, la mitjana
        # sobre nq_futur quantils de l'error.
        imp = np.maximum(net[:, None] + potencia[None, :], 0.0)
        if m > 1:
            Ef = np.sort(E[:, 1:], axis=0)
            S = Ef.shape[0]
            if S > self.nq_futur:
                Ef = Ef[((np.arange(self.nq_futur) + 0.5) * S / self.nq_futur).astype(int)]
            x = net[1:m, None, None] + Ef.T[:, :, None] + potencia[None, None, :]
            imp[1:m] = np.maximum(x, 0.0).mean(axis=1)
        V = self._valor(w, imp)

        # Hora actual: accions candidates contínues (el SOC real no cau sobre la graella)
        max_c = max(0.0, min(pot, (cap - soc) / ETA - EPS))
        if m > 0:
            # cost esperat sobre tots els escenaris; no es descarrega més que el consum net màxim
            sup = float(np.max(net[0] + E[:, 0]))
            max_d = max(0.0, min(pot, soc * ETA - EPS, max(0.0, sup)))
            cand = np.concatenate([np.linspace(-max_d, max_c, 241), [0.0]])
            cost0 = w[0] * np.maximum(net[0] + E[:, :1] + cand[None, :], 0.0).mean(axis=0)
        else:
            max_d = max(0.0, min(pot, soc * ETA - EPS, max(0.0, net[0] * (1.0 - self.marge_descarrega))))
            cand = np.concatenate([np.linspace(-max_d, max_c, 241), [0.0, min(max_c, max(0.0, -net[0]))]])
            cost0 = w[0] * np.maximum(net[0] + cand, 0.0)
        soc1 = soc + np.where(cand >= 0, cand * ETA, cand / ETA)
        cost = cost0 + np.interp(soc1, graella, V) + 1e-6 * np.abs(cand)
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
        self._mes, self._clau = mes, clau
        preus, co2s = list(obs["preus_properes_24h"]), list(obs["co2_properes_24h"])
        for j, (c, p) in enumerate(zip(co2s, preus)):
            self.mercat[clau + j] = (float(c), float(p))

        # --- telemetria de l'hora anterior
        dades = dict(obs.get("noves_dades_5min") or {})
        salt = self.clau_anterior is None or clau != self.clau_anterior + 1
        self.clau_anterior = clau
        if salt:
            # inici de setmana: la telemetria rebuda és de l'última hora de la setmana anterior;
            # si la taula del train té l'hora anterior, se'n fan servir els valors
            try:
                fila = self.precarrega(clau)
            except Exception:
                fila = None
            if fila is not None:
                dades.update(fila)
        consum_prev = float(dades.get("ultim_consum_kw", 0.0))
        solar_prev = float(dades.get("ultima_solar_kw", 0.0))

        def lectura(nom):
            v = dades.get(nom)
            return float(v) if v is not None else NAN
        self.temp_prev, self.tpanel_prev, self.irr_prev = (lectura("temp_ambient_c"), lectura("temp_panel_c"),
                                                           lectura("irradiancia_wm2"))
        avaria = consum_prev <= 1.0          # el comptador no envia dades
        h_prev = (hora - 1) % 24
        self.consum_prev = None if avaria else consum_prev
        if not avaria:
            self.consum[clau - 1] = consum_prev
            perfil = self.perfil_tipus() or PARAMS["perfil_consum"]
            self.nivell_consum = consum_prev / perfil[self.tipus(clau, hora, dow, -1)][h_prev]
            self.registra_solar(clau - 1, solar_prev)

        # --- previsions (si el vector de mercat s'escurça, s'acaba la simulació)
        n = self.H if len(preus) >= 3 else len(preus)
        co2_f, preu_f = self.prediu_mercat(clau, hora, dow, mes, n)
        w = 0.7 * co2_f / PARAMS["co2_mitja"] + 0.3 * preu_f / PARAMS["preu_mitja"]
        consum = self.prediu_consum(clau, hora, dow, n)
        solar = self.prediu_solar(clau, hora, mes, n)
        net = consum - solar

        # --- escenaris d'error
        E = None
        if self.escenaris > 0 and not avaria:
            E = self.escenaris_error(clau, hora, dow, mes, min(self.escenaris, n), consum, solar)
        if E is None and net[0] < self.llindar_excedent:
            net[0] -= self.extra_excedent    # recepta v5 (sense taules d'incertesa)
        if avaria:
            net[0] = 0.0                     # el simulador comptarà consum zero aquesta hora
        return self.planificar(w, net, soc, cap, pot, E)
