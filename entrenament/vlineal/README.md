# Entrenament de l'agent del Mode Difícil (vlineaE1)

Aquesta carpeta conté tot el codi que genera els fitxers de dades de l'agent que hi ha a
`submission_dificil/` (versió **vlineaE1**). L'agent (`submission_dificil/agent.py`) no entrena res al
servidor: només llegeix els pesos ja calculats i fa els càlculs amb numpy.

Tot s'executa **des de l'arrel del repositori** amb l'entorn `.venv`.

## Com funciona l'agent, en una línia

Prediu les properes 24 h (preu, CO2, consum i solar), busca amb programació dinàmica el pla de
càrrega/descàrrega que minimitza el cost esperat del que mesura el Score (0,7·CO2 + 0,3·cost de
l'energia importada), aplica només la consigna de l'hora actual i torna a planificar a l'hora següent.

## Fitxers del paquet i d'on surten

| Fitxer | Què és | Pas que el genera |
|---|---|---|
| `agent.py` | L'agent | escrit a mà (versió vlineaE1) |
| `nt_mercat.npz` | Model lineal (ridge multisortida) del preu i el CO2 de les hores +3..+23 | 3_mercat |
| `kf_mercat.npz` | Filtre de Kalman del CO2 (nivell, forma del migdia, del vespre i terme curt) | 5_kalman |
| `nt_valor.npz` | Xarxa neuronal del valor de l'energia guardada d'aquí a 3 h (15 membres) | 4_valor |
| `historial.npz` | Taula horària del train (72 h que es recuperen en començar cada setmana) | 2_historial |
| `parametres.json` | Perfils típics de preu, CO2, solar i consum, i mitjanes de normalització | 1_campus (v7a) |
| `perfils_consum.json` | Consum per tipus de dia (feiner, dissabte, diumenge/festiu, feiner de vacances) | 1_campus (v7a) |
| `incert.json` | Quantils dels errors de consum i solar (per planificar sobre el cost esperat) | 1_campus (v7a) |
| `model_campus.npz` | Arbres del consum i de la claredat solar de l'hora actual | 1_campus (v7a) |

Tots els models finals s'entrenen amb **totes** les dades del train. Excepcions deliberades: les 15 hores
amb el comptador avariat (consum 0) i les hores sense una resposta completa (just abans d'un forat de
dades). El mercat té 282 dies (novembre inclòs); el campus, 250 dies (els sensors no tenen novembre).

## Reconstruir-ho tot

```bash
bash entrenament/vlineal/construir_vlineaE1.sh
```

Refà l'historial, el model del mercat, el valor de l'energia i el Kalman, i munta el paquet a
`entrenament/vlineal/sortida/vlineaE1/`. Els fitxers del campus es copien de `submission_dificil/`
(vegeu el pas 1). Tarda uns 15–25 minuts. Els fitxers intermedis (`ag_tot/`, `art/`, `data/`,
`sortida/`) no es desen a git.

## Els passos

**1_campus** — scripts de la v7 (`build/`, amb l'agent que feien servir a `agent/` i l'ablació original a
`README_v7.md`). `build/construeix.sh <carpeta agent> <sortida>` genera l'historial, els perfils de consum
per tipus de dia, la taula d'errors (executant l'agent sobre el train), el model de consum de l'hora
actual i els quantils de la claredat solar, per a cada plec de validació i per a tot el train.
`parametres.json` surt de `entrenament/construir_parametres.py` (v5), ampliat a la v6 i a la v7
(`build/parametres_v7.py`: preu amb festius estatals i novembre real).

**2_historial** — `historial.py <sortida>`: taula horària del train. Omple les hores del 17 al 24 d'agost
de 2026 (hi ha comptador però no inversor) amb el consum del comptador i la solar estimada a partir de la
irradiància.

**3_mercat** — `ds_mkt.py tot` construeix, per a cada hora del train, el que veuria l'agent (log del
valor dividit pel perfil típic de les darreres 48 h, nivell de les darreres 24 h, calendari) i el que va
passar les 21 hores següents. `aavg.py` ajusta un ridge multisortida amb 7 regularitzacions (100..3000) i
en fa la mitjana dels pesos. `train_mkt.py` és la versió amb validació interna per triar la regularització.

**4_valor** — `collect.py tot` executa l'agent sobre tot el train i desa, a cada hora, l'estat que veu;
`train_valor.py tot 3 mlp <llavor>` calcula amb informació perfecta (programació dinàmica sobre el train)
quant valia l'energia guardada d'aquí a 3 h i entrena una xarxa de 32 neurones (5 membres) que l'estima a
partir de l'estat. `ajunta.py` ajunta 3 llavors en una xarxa de 15 membres. L'agent fa la mitjana 50/50
d'aquest valor i del que calcula el pla. Les solucions amb informació perfecta només serveixen per
entrenar: el model és una funció de l'estat, no guarda res per dates.

**5_kalman** — `series.py tot` construeix la sèrie horària de log(valor/perfil) del CO2 i el preu;
`fit.py tot ml` ajusta el model d'estat (8 components) per màxima versemblança; `exporta.py` en treu
`kf_mercat.npz`.

## Validació honesta (com s'han triat les peces)

Cada idea s'ha mesurat en 4 grups de setmanes (`final`, `d01`, `d08`, `d15`, unes 11 setmanes cadascun)
amb models entrenats **sense** aquelles setmanes ni 2 dies abans i després, i simulant la bateria amb
`local_eval/avaluar.py --mode dificil`. Mitjana de la nota local:

| Versió | Nota local honesta |
|---|---|
| v7a (arbres per al mercat) | 19,59 |
| vlineal1 (model lineal per al mercat) | 20,03 |
| + valor de l'energia amb xarxa (v2.1) | 20,13 |
| + Kalman per al CO2 (v2.2) | 20,16 |
| sostre amb informació perfecta | ~21,2 |

Normes que es compleixen: res que depengui de quan acaba la simulació ni de les dates del test; cap
dada externa; l'agent sempre planifica 24 h; el límit de seguretat de la consigna no es toca.
