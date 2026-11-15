# v7 = v6 + calendari (F) + perfils de mercat de novembre (A) + valor après de l'energia (E)

## Ablació honesta (4 plecs; artefactes de cada plec construïts sense els dies del plec ±2)
| variant | d01 | d08 | d15 | final | mitjana | Δ vs pas anterior | temps/2016 h vs v5 (back-to-back, d08) |
|---|---|---|---|---|---|---|---|
| v5 | 20.160 | 19.450 | 17.687 | 20.069 | 19.341 | | 1× (7–9 s) |
| v6 | 20.496 | 19.855 | 17.973 | 20.617 | 19.735 | | ~1.2× |
| v6 + F (calendari), novembre antic (`var_sensenov`) | 20.813 | 19.946 | 18.087 | 20.684 | 19.883 | +0.148 | |
| **v7a** = v6 + F + A (novembre real) → `final_a/` | 20.778 | 20.014 | 18.058 | 20.688 | 19.885 | +0.150 vs v6 (A: +0.002, soroll) | ~1.2× |
| v7a + E (valor après, 250 arbres, alfa 0.5) | 20.885 | 20.035 | 18.147 | 20.834 | 19.975 | +0.090 | ~2.0× |
| v7a + E (250 arbres, alfa 0.3; NO triat: seria ajustar sobre els plecs) | 20.899 | 20.058 | 18.159 | 20.824 | 19.985 | | |
| **v7 = v7a + E (125 arbres, lr 0.12, alfa 0.5) → `final/`** | **20.911** | **20.030** | **18.124** | **20.861** | **19.982** | **+0.097** (+0.247 vs v6) | ~1.7–2.0× |
| v7 + J (errors en línia, recentrar k=0; n=48 / n=168) — REBUTJAT | 20.887 / 20.893 | 20.018 / 20.022 | 18.113 / 18.107 | 20.855 / 20.857 | 19.968 / 19.970 | −0.014 / −0.012 (negatiu als 4 plecs) | |

Soroll (re-entrenar amb una altra llavor): ±0.05 a la mitjana, ±0.2 per plec. F: positiu als 4 plecs; E: positiu als 4
(d08 +0.016, dins del soroll). Optimitzacions exactes de velocitat (cache de delta_hora, recorregut d'arbres amb els
valors que falten precalculats) donen puntuacions idèntiques.

Calendari equivocat (plec final, v7): 5 feiners marcats com a festius 20.861 → 20.755 (−0.106);
7 feiners marcats com a vacances → 20.837 (−0.024). (`sens_calendari.py`)

## Paquets (només fitxers de la submissió)
- `final_a/`: v7a. agent.py, parametres.json, model_mercat.npz, model_campus.npz, incert.json, historial.npz,
  perfils_consum.json.
- `final/`: v7 = el mateix + model_valor.npz (agent.py amb el valor après).
Fitxers opcionals (provat amb `fallback.py`, també amb fitxers corromputs): sense perfils_consum.json el consum
torna a v6 (feiner/cap de setmana); sense model_valor.npz el planificador és el de v7a; sense
`perfil_preu_festius` a parametres.json el preu torna a v6. parametres.json és obligatori (com a v5/v6).

## Canvis respecte de v6 (agent.py)
- Calendari (llistes FESTIUS, NACIONALS, VACANCES): `tipus_dia`, `tipus_consum` (hores 0–5 = dia anterior),
  `cap_setmana_preu` (festius estatals de data fixa = cap de setmana per al preu).
- Consum: perfil 4 tipus × 24 h (perfils_consum.json), "ahir" només si ahir era del mateix tipus, nivell amb el perfil
  del tipus; variables de consum0 amb el tipus; taules d'error amb "cap de setmana o festiu".
- Preu: perfil `perfil_preu_festius` i indicador amb festius estatals (també a les variables del model). CO2 i canvi
  d'hora sense canvis.
- Novembre: perfil_co2 / perfil_preu / perfil_preu_festius de novembre refets amb els CSV reals (abans mitjana oct/des).
- Valor après (agent_E): DP exacta sobre les 3 hores conegudes; valor final a l'hora 3 = 0.5·(model d'arbres del valor
  marginal de l'energia, entrenat amb solucions de previsió perfecta) + 0.5·(cost futur del pla de 24 h de v6).

## Reconstruir artefactes (des de l'arrel del repo, camins `C:/...`)
    .venv/Scripts/python.exe -W ignore <v7c>/build/parametres_v7.py <v6c>/final/parametres.json <agent_F>/paquet/parametres.json <v7c>/final/parametres.json
    bash <v7c>/build/construeix.sh <v7c>/final <v7c>/b_nou
→ `b_nou/<plec>/` honestos (final d01 d08 d15) i `b_nou/tot/` (paquet: copiar model_mercat.npz, model_campus.npz,
incert.json, historial.npz, perfils_consum.json, model_valor.npz a final/). Passos: historial.py, perfils.py, recull.py,
llarg.py, train_now.py, mk_incert.py, train_s0.py, train_mercat.py, train_valor.py (`3 125,0.12`).
Els artefactes honestos usats per avaluar són a `art/<plec>/` (= b1/<plec> + b_v125/<plec>/model_valor.npz).
(Comprovat: `build/construeix.sh` sobre `final/` → `b_rebuild/` (~8 min) reprodueix bit a bit tots els artefactes de art/ i final/.)

## Avaluació honesta
    .venv/Scripts/python.exe <v7c>/avalua_honest.py [dir_agent] [dir_artefactes] [--sense-v5v6]
(per defecte final/ i art/; imprimeix v5, v6 i v7 per plec i mitjana; 4 processos alhora).
Resultat: v5 19.341, v6 19.735, v7 19.982 (final 20.861, d01 20.911, d08 20.030, d15 18.124), 0 infraccions.

## Directoris de desenvolupament
agent/ (v7a), agent_e/ (v7), agent_j/ (J, rebutjat), var_sensenov/ (ablació novembre), art_* (artefactes de cada
variant), res/ (sortides), dev/ (pedaços aplicats).
