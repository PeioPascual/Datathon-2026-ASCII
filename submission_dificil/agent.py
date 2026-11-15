"""
agent.py — Agent per al MODE DIFÍCIL (Repte UPC Sostenible, Datathon 2026).

Versió "lineaE1" (lineal 2.2 entrenada amb totes les dades): "predir i després planificar", tenint en compte la incertesa de les previsions.
  1. Prediu les properes 24 h:
       - preu i CO2 de les hores +3..+23 (les 3 primeres ja es coneixen) amb una regressió lineal
         regularitzada (ridge) que prediu les dues corbes senceres alhora, a partir de
         log(valor / perfil típic) de les darreres 48 h i del calendari (pesos a nt_mercat.npz; mitjana
         dels pesos entrenats amb 7 regularitzacions, 100..3000, perquè no depengui d'una sola tria);
         per al CO2, la corba de la regressió se substitueix per la d'un filtre de Kalman (kf_mercat.npz:
         nivell, forma del migdia, forma del vespre i terme curt, ajustat per màxima versemblança),
         que segueix millor el CO2 de les primeres hores; per al preu es manté la regressió;
       - consum del campus i producció solar amb perfils del train + el que s'ha observat les
         darreres hores (arbres de decisió per al consum i la claredat de l'hora actual,
         model_campus.npz).
  2. Genera escenaris d'error del consum net (quantils dels errors observats en el train, segons
     l'horitzó, l'hora, el tipus de dia, la claredat del cel i l'altura del sol).
  3. Busca el pla de càrrega/descàrrega que minimitza el cost ESPERAT del que mesura el Score
     (0.7*CO2 + 0.3*cost de l'energia importada), sabent que l'excedent solar és gratuït.
     Programació dinàmica sobre l'energia guardada. El valor de l'energia que quedarà a la bateria
     d'aquí a 3 hores és la mitjana (50/50) del que calcula el pla i del que prediu una xarxa neuronal
     petita (nt_valor.npz) entrenada amb les solucions òptimes del train (amb informació perfecta);
     la xarxa només rep l'estat que l'agent veu en aquell moment.
  4. Aplica només la consigna de l'hora actual i replanifica a l'hora següent.
Detalls pràctics:
  - Quan la simulació salta a una setmana nova, recupera de la taula del train (historial.npz)
    les 72 h anteriors (el train acaba just el dia abans de cada setmana de test). Les hores sense inversor
    (17-24 d'agost) hi tenen el consum del comptador i la solar estimada a partir de la irradiància.
  - Vacances d'agost de l'1 al 23: el dilluns 24 el campus ja consumeix com un dia feiner (dades del train).
  - Valor de l'energia: xarxa neuronal amb 15 membres (3 llavors x 5), entrenada amb totes les dades.
  - De nit el comptador no s'actualitza (de 23 h a 5 h repeteix el valor de les 22 h).
  - Canvi d'hora (octubre i març): la solar i el perfil de CO2 es calculen en hora solar (UTC).
  - Calendari (llistes de festius 2025-2027): festius i vacances de la universitat per al consum
    (4 tipus de dia), festius estatals de data fixa per al preu (els peatges hi apliquen l'horari
    de cap de setmana) i festius estatals + Divendres Sant per al CO2 (a tot l'Estat la demanda hi
    baixa com un diumenge i es crema menys gas).
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
SOC_NODES = np.linspace(0.0, 1.0, 13)    # valor apres de l'energia: 12 trams d'energia guardada
SOC_MIG = 0.5 * (SOC_NODES[1:] + SOC_NODES[:-1])
NIT = (23, 0, 1, 2, 3, 4, 5)     # hores en què el comptador repeteix la lectura de les 22 h


# ------------------------------------------------------------------ calendari
# Festius amb el campus tancat (estatals, Catalunya i locals de Barcelona), set 2025 - set 2026
# Festius de 2025 a 2027 (consum del campus): estatals, de Catalunya i locals de Barcelona.
FESTIUS = ("2025-09-11", "2025-09-24", "2025-10-12", "2025-11-01", "2025-12-06", "2025-12-08",
           "2025-12-25", "2025-12-26", "2026-01-01", "2026-01-06", "2026-04-03", "2026-04-06",
           "2026-05-01", "2026-05-25", "2026-06-24", "2026-08-15", "2026-09-11", "2026-09-24",
           "2026-10-12", "2026-11-01", "2026-12-06", "2026-12-08", "2026-12-25", "2026-12-26",
           "2027-01-01", "2027-01-06", "2027-03-26", "2027-03-29", "2027-05-01", "2027-05-17",
           "2027-06-24", "2027-08-15", "2027-09-11", "2027-09-24", "2027-10-12", "2027-11-01",
           "2027-12-06", "2027-12-08", "2027-12-25", "2027-12-26")
# Festius estatals de data fixa: els peatges del PVPC (2.0TD) hi apliquen l'horari de cap de setmana
NACIONALS = ("2025-10-12", "2025-11-01", "2025-12-06", "2025-12-08", "2025-12-25", "2026-01-01",
             "2026-01-06", "2026-05-01", "2026-08-15", "2026-10-12", "2026-11-01", "2026-12-06",
             "2026-12-08", "2026-12-25", "2027-01-01", "2027-01-06", "2027-05-01", "2027-08-15",
             "2027-10-12", "2027-11-01", "2027-12-06", "2027-12-08", "2027-12-25")
# Festius estatals per al CO2 (els de data fixa + Divendres Sant): a tot l'Estat la demanda hi baixa
# com un diumenge i es crema menys gas, per això el CO2 es llegeix amb el perfil de cap de setmana.
ESTATALS_CO2 = NACIONALS + ("2026-04-03", "2027-03-26")
# Vacances de la universitat (feiners amb el campus a mig gas)
VACANCES = (("2025-12-24", "2026-01-06"), ("2026-03-28", "2026-04-06"), ("2026-08-01", "2026-08-23"))


def _dia(text):
    return (datetime.date.fromisoformat(text) - datetime.date(1970, 1, 1)).days


DIES_FESTIUS = frozenset(_dia(t) for t in FESTIUS)
DIES_ESTATALS_CO2 = frozenset(_dia(t) for t in ESTATALS_CO2)
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


def cap_setmana_co2(clau):
    """1 si l'hora és de dissabte, diumenge o festiu estatal (perfil de cap de setmana per al CO2)."""
    dia = clau // 24
    return int((dia + 3) % 7 >= 5 or dia in DIES_ESTATALS_CO2)


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
    try:
        z = np.load(directori / "nt_valor.npz")
        models["valor"] = {k: z[k] for k in z.files}
    except Exception:
        pass
    try:
        z = np.load(directori / "nt_valor24.npz")
        models["valor24"] = {k: z[k] for k in z.files}
    except Exception:
        pass
    for nom in ("nt_mercat", "nt_consum0", "nt_solar0", "kf_mercat"):
        try:
            z = np.load(directori / (nom + ".npz"))
            models[nom] = {k: z[k] for k in z.files}
        except Exception:
            pass
    return models


def carrega_models_incert(fitxer):
    t = json.loads(Path(fitxer).read_text(encoding="utf-8"))
    K = t["k_max"] + 1
    t["_L"] = np.array([t["L"][str(k)] for k in range(K)], dtype=float)
    t["_P"] = np.array([t["P"][str(k)] for k in range(K)], dtype=float)
    return t


def mlp_forward(P, X):
    """Xarxa densa: capes W{i}, b{i}; tanh a les ocultes; sortida lineal. Diversos membres (seeds)
    es fan servir com a mitjana. X ja estandarditzada."""
    out = 0.0
    for m in range(int(P["n_membres"])):
        h = X
        L = int(P["n_capes"])
        for i in range(L):
            h = h @ P["W%d_%d" % (m, i)] + P["b%d_%d" % (m, i)]
            if i < L - 1:
                h = np.tanh(h)
        out = out + h
    return out / float(P["n_membres"])


def prediu_nt(P, x):
    """Model sense arbres (lineal o xarxa) sobre un vector de variables x (estandarditzat aqui)."""
    z = (np.asarray(x, float) - P["mu"]) / P["sd"]
    if str(P["tipus"]) == "lin":
        return P["b"] + z @ P["W"]
    return P["ymu"] + P["ysd"] * mlp_forward(P, z[None, :])[0]


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


def _softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def disseny_valor(f):
    """Variables del model de valor -> vector de disseny (NaN a 0 + indicador, hora en one-hot)."""
    f = np.asarray(f, float)
    nan = f != f
    x = np.where(nan, 0.0, f)
    hh = np.zeros(24)
    hh[int(f[0]) % 24] = 1.0
    return np.concatenate([x, [float(nan.any())], hh])


def prediu_valor(P, f):
    """Valor marginal (relatiu al nivell de w) de l'energia guardada als 12 trams; no creixent."""
    t = str(P["tipus"])
    if t == "arb":
        nb = len(SOC_MIG)
        X = np.concatenate([np.repeat(np.asarray(f, float)[None, :], nb, axis=0), SOC_MIG[:, None]], axis=1)
        lam = prediu_arbres(P, X)
        return np.minimum.accumulate(np.maximum(lam, 0.0))
    z = (disseny_valor(f) - P["mu"]) / P["sd"]
    if t == "lin":
        return np.minimum.accumulate(np.maximum(P["b"] + z @ P["W"], 0.0))
    out = 0.0
    for m in range(int(P["n_membres"])):
        h = z
        L = int(P["n_capes"])
        for i in range(L):
            h = h @ P["W%d_%d" % (m, i)] + P["b%d_%d" % (m, i)]
            if i < L - 1:
                h = np.tanh(h)
        sp = _softplus(h)                       # [decrements bins 0..10, nivell del tram 11]
        out = out + np.cumsum(sp[::-1])[::-1]
    return float(P["ysc"]) * out / float(P["n_membres"])


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
                 hist_hores=72, nit_congelada=True, canvi_hora=True, cal_consum=True, cal_preu=True, cal_co2=True,
                 mercat_nt=1.0, consum_nt=False, solar_nt=False, nt_biaix=True, nt_kmin=3,
                 alfa_valor=0.5, mode_valor="t3", alfa_term=1.0,
                 kf_pes=0.0, kf_biaix=True, kf_w=((1.0,) * 21, (0.0,) * 21)):
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
        self.cal_co2 = cal_co2      # CO2: festius estatals amb el perfil de cap de setmana

        # Models sense arbres: mercat_nt = pes del model nou en la barreja amb els arbres (1 = nomes el
        # model nou; si no hi ha arbres, el model nou sol); consum_nt / solar_nt: substitueixen consum0 / solar0q*
        self.mercat_nt, self.consum_nt, self.solar_nt, self.nt_biaix = mercat_nt, consum_nt, solar_nt, nt_biaix
        self.nt_kmin = nt_kmin
        # Valor apres de l'energia guardada (entrenat amb solucions optimes del train): "t3" = barreja
        # (alfa_valor) amb el cost futur del pla a l'hora 3; "term" = valor al final del pla (alfa_valor x)
        self.alfa_valor, self.mode_valor, self.alfa_term = alfa_valor, mode_valor, alfa_term
        # Filtre de Kalman del mercat (agent Q3): kf_pes = pes de la seva previsio a les hores 3..23;
        # kf_w = llista (2, 21) de pesos per variable i hora (si es dona, substitueix kf_pes)
        self.kf_pes, self.kf_biaix = kf_pes, kf_biaix
        self.kf_w = None if kf_w is None else np.asarray(kf_w, float).reshape(2, 21)
        self.kf_a = self.kf_P = self.kf_t = None
        self._kf_prof = {}
        self.registre = None
        self._ctx = None
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
        ts = data_de(clau)
        if ts.month == 11:
            return 0
        off_train = 2 if 4 <= ts.month <= 10 else 1
        return off_train - desfasament_utc(ts.year, ts.month, ts.day, ts.hour)

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
        elif idx == 0 and self.cal_co2:
            perfil = PARAMS["perfil_co2"][str(mes)]
            cap = lambda kk: cap_setmana_co2(clau + kk)  # noqa: E731
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

    def perfils_mercat(self, idx, clau, hora, dow, mes, j0=-45):
        """Perfil (mes x tipus de dia x hora) de la variable idx per a les hores relatives j=j0..23."""
        if idx == 1 and self.cal_preu:
            perfil = PARAMS["perfil_preu_festius"][str(mes)]
            return np.array([perfil[cap_setmana_preu(clau + j)][(hora + j) % 24] for j in range(j0, 24)])
        perfil = PARAMS["perfil_" + NOMS_MERCAT[idx]][str(mes)]
        if idx == 0 and self.cal_co2:
            return np.array([perfil[cap_setmana_co2(clau + j)][(hora + j + self.delta_hora(clau + j)) % 24]
                             for j in range(j0, 24)])
        return np.array([perfil[self._hora_futura(hora, dow, j)[1]][(hora + j + self.delta_hora(clau + j)) % 24]
                         for j in range(j0, 24)])

    def feats_nt(self, clau, hora, dow, mes, ext=False):
        """Variables del model de mercat sense arbres (les dues variables alhora):
             an: log(valor / perfil) de les hores j=-45..2 (NaN si no es coneix), centrat pel nivell
             lev: nivell = mitjana de log(valor/perfil) de les ultimes <=24 hores conegudes
        Retorna (x, prof (2, 24) per a k=0..23, lev (2,)) o None si no es coneix res."""
        j0 = -93 if ext else -45
        anl = np.full((2, 3 - j0), NAN)
        prof = np.zeros((2, 24))
        for idx in (0, 1):
            p = self.perfils_mercat(idx, clau, hora, dow, mes, j0)
            prof[idx] = p[-24:]
            for j in range(j0, 3):
                v = self.mercat.get(clau + j)
                if v is not None and v[idx] > 0 and p[j - j0] > 0:
                    anl[idx, j - j0] = np.log(v[idx] / p[j - j0])
        an = anl[:, -48:]
        rec = an[:, 24:]
        if np.isnan(rec).all(axis=1).any():
            return None
        lev = np.nanmean(rec, axis=1)
        d = an - lev[:, None]
        falta = np.isnan(d)
        d[falta] = 0.0
        fresca = self.claredat_mercat is not None and clau - self.clau_claredat_mercat <= 6
        hh = np.zeros(24); hh[hora] = 1.0
        dd = np.zeros(7); dd[dow] = 1.0
        ang = 2 * np.pi * hora / 24
        harm = np.array([np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)])
        inter = (d[:, -3:].reshape(-1, 1) * harm[None, :]).ravel()          # 24: hores conegudes x hora
        angm = 2 * np.pi * (mes - 1) / 12
        x = np.concatenate([d.ravel(), lev, [falta[:, :45].mean(), np.sin(angm), np.cos(angm),
                            self.claredat_mercat if fresca else 0.0, float(fresca), float(cap_setmana_preu(clau)),
                            float(cap_setmana_preu(clau + 24))], hh, dd, inter, lev[0] * harm, lev[1] * harm])
        if ext:
            agg = []
            for a, b in ((0, 24), (24, 48)):
                blk = anl[:, a:b]
                for idx in (0, 1):
                    ok = ~np.isnan(blk[idx])
                    agg += [float(blk[idx][ok].mean() - lev[idx]) if ok.any() else 0.0, float(not ok.any())]
            x = np.concatenate([x, agg, hh * np.sin(angm), hh * np.cos(angm)])
        return x, prof, lev

    def prediu_nt_mercat(self, clau, hora, dow, mes):
        """(2, 24) previsio del model sense arbres (hores 3..23; 0..2 = perfil*nivell) o None."""
        P = self.models.get("nt_mercat")
        if P is None:
            return None
        fl = int(P["kfx"]) if "kfx" in P else 0          # hibrid: previsio (1), estat (2) o tots dos (3) del Kalman
        nx = (42 if fl in (1, 3) else 0) + (8 if fl in (2, 3) else 0)
        f = self.feats_nt(clau, hora, dow, mes, ext=P["mu"].size - nx > 168)
        if f is None:
            return None
        x, prof, lev = f
        if fl:
            kfp = self.prev_kf(clau)
            if kfp is None:
                return None
            ex = [x]
            if fl in (1, 3):
                ex.append((np.log(kfp[:, 3:] / prof[:, 3:]) - lev[:, None]).ravel())
            if fl in (2, 3):
                ex.append(self.kf_a)
            x = np.concatenate(ex)
        y = prediu_nt(P, x).reshape(2, 21)
        if self.nt_biaix:
            y = y + P["s2"].reshape(2, 21) * 0.5
        out = prof * np.exp(lev[:, None])
        out[:, 3:] = prof[:, 3:] * np.exp(lev[:, None] + np.clip(y, -2.5, 1.6))
        return out

    # ------------------------------------------------------------------ filtre de Kalman (mercat)
    def kf_perfil(self, k):
        """Perfil (CO2, preu) de l'hora absoluta k, amb el mes, el tipus de dia i l'hora de la mateixa k."""
        v = self._kf_prof.get(k)
        if v is None:
            ts = data_de(k)
            h, mes = ts.hour, str(ts.month)
            cs = int(ts.weekday() >= 5)
            pc = PARAMS["perfil_co2"][mes][cs][(h + self.delta_hora(k)) % 24]
            if self.cal_preu:
                pp = PARAMS["perfil_preu_festius"][mes][cap_setmana_preu(k)][h]
            else:
                pp = PARAMS["perfil_preu"][mes][cs][h]
            v = (pc, pp)
            if len(self._kf_prof) > 5000:
                self._kf_prof = {}
            self._kf_prof[k] = v
        return v

    def kf_zs(self, P):
        """Matrius d'observacio Z (24 hores del dia, 2, 8), calculades un sol cop."""
        Zs = P.get("_Z")
        if Zs is None:
            lam = P["lam"]
            Zs = np.zeros((24, 2, 8))
            for s in (0, 1):
                Zs[:, s, s] = 1.0
                Zs[:, s, 2 + s] = lam[s, 0]
                Zs[:, s, 4 + s] = lam[s, 1]
                Zs[:, s, 6 + s] = 1.0
            P["_Z"] = Zs
        return Zs

    def kf_z(self, P, k):
        return self.kf_zs(P)[data_de(k).hour]

    def kf_estat(self, P, fins):
        """Porta l'estat filtrat fins a l'hora 'fins' (inclosa) amb els valors reals coneguts. Si és el primer cop,
        hi ha hagut un salt de temps o un forat de més de 200 h, torna a començar des de la distribució estacionària
        96 h abans (hi entren les 72 h precarregades)."""
        phi, Q, H, mu = P["phi"], P["Q"], P["H"], P["mu"]
        PP = np.outer(phi, phi)
        if self.kf_t is None or fins < self.kf_t or fins - self.kf_t > 200:
            self.kf_t = fins - 97
            self.kf_a = np.zeros(8)
            self.kf_P = Q / (1.0 - PP)
        a, Pm = self.kf_a, self.kf_P
        for k in range(self.kf_t + 1, fins + 1):
            a = phi * a
            Pm = PP * Pm + Q
            v = self.mercat.get(k)
            if v is None or not (v[0] > 0 and v[1] > 0):
                continue
            pc, pp = self.kf_perfil(k)
            if not (pc > 0 and pp > 0):
                continue
            Z = self.kf_z(P, k)
            e = np.array([np.log(v[0] / pc), np.log(v[1] / pp)]) - mu - Z @ a
            PZ = Pm @ Z.T
            F = Z @ PZ + np.diag(H)
            K = PZ @ np.linalg.inv(F)
            a = a + K @ e
            Pm = Pm - K @ PZ.T
            Pm = 0.5 * (Pm + Pm.T)
        self.kf_a, self.kf_P, self.kf_t = a, Pm, fins
        return a, Pm

    def prev_kf(self, clau):
        """prediu_kf amb memoria per a la mateixa hora (la fan servir el model hibrid i la barreja)."""
        c = getattr(self, "_kfc", None)
        if c is not None and c[0] == clau:
            return c[1]
        try:
            p = self.prediu_kf(clau)
            if p is not None and not np.isfinite(p[:, 3:]).all():
                p = None
        except Exception:
            p = None
            self.kf_t = None
        self._kfc = (clau, p)
        return p

    def prediu_kf(self, clau):
        """(2, 24) previsió del filtre de Kalman per a les hores 3..23 (les 0..2 es deixen a NaN) o None."""
        P = self.models.get("kf_mercat")
        if P is None:
            return None
        a, Pm = self.kf_estat(P, clau + 2)
        phi, Q, H, mu = P["phi"], P["Q"], P["H"], P["mu"]
        h = np.arange(1, 22)
        pw = phi[None, :] ** h[:, None]                                  # (21, 8)
        Z = self.kf_zs(P)[(data_de(clau).hour + 2 + h) % 24]               # (21, 2, 8)
        m = mu[None, :] + np.einsum("hsk,hk->hs", Z, pw * a[None, :])
        if self.kf_biaix:
            PP = np.outer(phi, phi)
            PPh = PP[None] ** h[:, None, None]
            V = PPh * Pm[None] + Q[None] * (1.0 - PPh) / (1.0 - PP)[None]   # covariancia de l'estat a cada horitzo
            m = m + 0.5 * (np.einsum("hsk,hkr,hsr->hs", Z, V, Z) + H[None, :])
        pr = np.array([self.kf_perfil(clau + k) for k in range(3, 24)])     # (21, 2)
        out = np.full((2, 24), NAN)
        out[:, 3:] = (pr * np.exp(np.minimum(m, 1.6))).T
        return out

    def prediu_mercat(self, clau, hora, dow, mes, n):
        """Prediu CO2 i preu de les properes n hores. Les hores ja conegudes es deixen tal qual;
        la resta surten dels arbres de decisió (si hi són) o de la recepta senzilla."""
        sortida = []
        nt = None
        kfp = None
        if self.kf_pes > 0 or self.kf_w is not None:
            kfp = self.prev_kf(clau)
        if self.mercat_nt > 0:
            try:
                nt = self.prediu_nt_mercat(clau, hora, dow, mes)
            except Exception:
                nt = None
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
            if nt is not None and n == 24:
                a = np.full(21, self.mercat_nt if model is not None else 1.0)
                a[:max(0, self.nt_kmin - 3)] = 1.0          # hores properes: nomes el model nou
                out[3:] = (1 - a) * out[3:] + a * nt[idx, 3:]
            if kfp is not None and n == 24:
                b = self.kf_w[idx] if self.kf_w is not None else self.kf_pes
                out[3:] = (1 - b) * out[3:] + b * kfp[idx, 3:]
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
            Pnt = self.models.get("nt_consum0") if self.consum_nt else None
            if Pnt is not None:
                try:
                    y = float(prediu_nt(Pnt, self.feats_nt_consum0(clau, hora, dow, self._mes))[0])
                    out[0] = self.consum_prev * (1.0 + float(np.clip(y, -0.6, 1.5)))
                    model = None
                except Exception:
                    pass
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

    def feats_nt_consum0(self, clau, hora, dow, mes):
        """Variables del model lineal del consum de l'hora actual (objectiu: consum(t)/consum(t-1) - 1)."""
        perfil = self.perfil_tipus() or PARAMS["perfil_consum"]

        def g(c):
            v = self.consum.get(c)
            return v if (v is not None and v > 1.0) else NAN
        l1, l2, l3, l24, l25, l23 = g(clau - 1), g(clau - 2), g(clau - 3), g(clau - 24), g(clau - 25), g(clau - 23)
        c0, cp = self.tipus(clau, hora, dow, 0), self.tipus(clau, hora, dow, -1)
        hp = (hora - 1) % 24
        f = lambda v: 0.0 if v != v else float(np.clip(v, -1.0, 1.0))  # noqa: E731
        r12, r23, r24 = f(l1 / l2 - 1), f(l2 / l3 - 1), f(l24 / l25 - 1)
        rly, rnext = f(l24 / l1 - 1), f(l23 / l24 - 1)
        pstep = float(np.log(perfil[c0][hora] / perfil[cp][hp]))
        dev = f(l1 / perfil[cp][hp] - 1)
        t = self.temp_prev if self.temp_prev == self.temp_prev else 18.0
        ang = 2 * np.pi * hora / 24
        lab = int(c0 in (0, 3))
        hh = np.zeros(48); hh[hora + 24 * lab] = 1.0
        x = [r12, r23, r24, rly, rnext, pstep, dev, float(l2 != l2), float(l24 != l24), (t - 18) / 10,
             (t - 18) / 10 * np.sin(ang), (t - 18) / 10 * np.cos(ang), abs(t - 18) / 10, pstep * lab, r24 * lab,
             dev * pstep, float(c0 == 3), float(c0 != cp)]
        h24 = hh[:24] + hh[24:]
        return np.concatenate([x, hh, hh * r24, h24 * pstep, h24 * r12, h24 * dev, h24 * rly, hh * x[9]])

    def feats_nt_solar0(self, clau, hora, mes, k_avui):
        """Variables del model de quantils lineal de la claredat de l'hora actual (a partir de feats_solar0)."""
        (h, _, e0, er, ealt, kav, cl1, cl2, cl3, d12, mav, mn, mx, na, mah, irr, dtp, s1, c1) =             self.feats_solar0(clau, hora, mes, k_avui)
        nz = lambda v, alt: alt if v != v else v  # noqa: E731
        c1v = nz(cl1, kav)
        c2v = nz(cl2, c1v)
        e1 = self.env_hora(clau - 1)
        irr_r = nz(irr, 0.0) / max(e1, 1.0) * 1000.0 / 1000.0
        return np.array([kav, c1v, c2v, nz(cl3, c2v), c1v - c2v, nz(mav, kav), nz(mn, kav), nz(mx, kav), nz(mah, kav),
                         float(cl1 != cl1), float(mah != mah), min(na, 12) / 12, ealt, min(er, 3.0), c1v * ealt,
                         c1v * min(er, 3.0), nz(dtp, 0.0) / 20, irr_r, s1, c1, float(hora < 12), c1v * float(hora < 12),
                         (nz(mx, kav) - nz(mn, kav)), c1v * c1v])

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
        Pq = self.models.get("nt_solar0") if self.solar_nt else None
        if Pq is not None and e[0] > t["env_min"] and self.k_avui is not None:
            try:
                qs = prediu_nt(Pq, self.feats_nt_solar0(clau, hora, mes, self.k_avui))
                qs = np.clip(np.sort(qs), 0.0, 1.5) * e[0] - solar[0]
                if len(qs) == qp.shape[1]:
                    qp[0] = qs
            except Exception:
                pass
        elif self.usa_solar0 and "solar0q0" in self.models and e[0] > t["env_min"] and self.k_avui is not None:
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

    def _valor(self, w, imp, fins=1, V=None, des_de=None):
        """Valor de cada nivell d'energia a l'inici de l'hora 1 (recursió enrere), per a un sol
        vector de pesos w. imp[k, j] = energia importada (esperada) a l'hora k amb el moviment j."""
        d = self._dp
        graella, mem, tmp = d["graella"], d["mem"], d["tmp"]
        centre = slice(d["baixa"], d["baixa"] + len(graella))
        if V is None:
            V = -self.valor_final * float(np.mean(w)) * graella       # valor de l'energia al final
        for k in range(len(w) - 1 if des_de is None else des_de, fins - 1, -1):
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
        V = None
        if n >= 24 and (self.registre is not None or (self.alfa_valor > 0 and "valor" in self.models)):
            try:
                V = self._valor_apres(w, net, E, imp, cap)
            except Exception:
                V = None
        if V is None:
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

    # ------------------------------------------------------------------ valor apres de l'energia
    def feats_valor(self, w, net, E, Vp3):
        """Variables (independents del SOC) de l'estat a l'origen. Tot relatiu al nivell recent de w."""
        clau, hora, dow, mes = self._ctx
        def wabs(c):
            v = self.mercat.get(c)
            return NAN if v is None else 0.7 * v[0] / PARAMS["co2_mitja"] + 0.3 * v[1] / PARAMS["preu_mitja"]
        ws = np.array([wabs(clau + j) for j in range(-23, 3)])
        ok = ws == ws
        nivell = float(ws[ok].mean()) if ok.any() else float(np.mean(w))
        ang, angm = 2 * np.pi * hora / 24, 2 * np.pi * (mes - 1) / 12
        fresca = self.claredat is not None and clau - self.clau_claredat <= 6
        tc = tipus_consum(clau) if self.perfil_tipus() is not None else int(dow >= 5)
        sd = float(np.std(E, axis=0).mean()) / 1000 if E is not None else NAN
        g = self._dp["graella"]
        nodes = np.round(SOC_NODES * (len(Vp3) - 1)).astype(int)
        lam_p = -np.diff(Vp3[nodes]) / np.diff(g[nodes]) / nivell
        f = [hora, np.sin(ang), np.cos(ang), dow, float(tc in (1, 2)), float(tc == 3), np.sin(angm), np.cos(angm),
             nivell, float(ok.sum()), self.nivell_consum, (self.claredat if fresca else NAN), sd,
             float(np.sum(np.maximum(net[3:], 0))) / 1000, float(np.sum(np.minimum(net[3:], 0))) / 1000,
             float(np.mean(w[3:12])) / nivell, float(np.mean(w[12:])) / nivell]
        return np.concatenate([f, ws / nivell, w[3:24] / nivell, net[:24] / 1000, lam_p]), nivell

    def corba_valor(self, lam, nivell, cap):
        """Cost futur C(s) sobre la graella (C(0) = 0) a partir del valor marginal per trams."""
        graella = self._dp["graella"]
        nb = len(SOC_MIG)
        lam = lam * nivell
        mig = 0.5 * (graella[1:] + graella[:-1]) / cap
        idx = np.minimum((mig * nb).astype(int), nb - 1)
        return np.concatenate([[0.0], -np.cumsum(lam[idx] * np.diff(graella))])

    def _valor_apres(self, w, net, E, imp, cap):
        Vp3 = self._valor(w, imp, fins=3)
        f, nivell = self.feats_valor(w, net, E, Vp3)
        if self.registre is not None:
            self.registre.append((self._ctx[0], f, nivell))
        P = self.models.get("valor")
        if P is None or self.alfa_valor <= 0:
            return None
        C = self.corba_valor(prediu_valor(P, f), nivell, cap)
        a = self.alfa_valor
        if self.mode_valor == "term":
            return self._valor(w, imp, V=a * C)
        P24 = self.models.get("valor24")
        if self.mode_valor == "both" and P24 is not None:    # valor final apres + barreja a l'hora 3
            Vp3 = self._valor(w, imp, fins=3, V=self.alfa_term * self.corba_valor(prediu_valor(P24, f), nivell, cap))
        V = a * C + (1.0 - a) * (Vp3 - Vp3[0])
        return self._valor(w, imp, fins=1, V=V, des_de=2)

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
        self._ctx = (clau, hora, dow, mes)
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

        # --- previsions: sempre 24 h (les hores sense valor conegut les prediu el model)
        n = self.H
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
