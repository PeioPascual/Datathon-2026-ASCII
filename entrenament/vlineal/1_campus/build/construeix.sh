#!/bin/bash
# Construeix tots els fitxers de dades de l'agent v7 a partir de data/train/:
#   <out>/<plec>/  (plecs final d01 d08 d15): honestos, sense els dies del plec (+-2 dies)
#   <out>/tot/     paquet final (tot el train)
# Executar des de l'arrel del repo (camins tipus C:/..., no /c/...):
#   bash construeix.sh <dir_agent> <out> ['<json kwargs agent per recollir>'] ['<json hores k0_nit>']
# Requisit: <dir_agent>/parametres.json ja construït amb parametres_v7.py.
set -e
B="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
PY=".venv/Scripts/python.exe -W ignore"
A="$1"; OUT="$2"; KW="${3:-{\}}"; NIT="${4:-[23, 0, 1, 2, 3, 4, 5]}"
mkdir -p "$OUT/tot"
# 0) taula historial i perfils de consum per tipus de dia (per plec, sense els dies del plec; i complets)
for p in final d01 d08 d15; do mkdir -p "$OUT/$p"; $PY "$B/historial.py" "$OUT/$p" $p; done
$PY "$B/historial.py" "$OUT/tot"
$PY "$B/perfils.py" "$A" "$OUT"
# 1) prediccions puntuals de l'agent sobre el train -> taula llarga d'errors
$PY "$B/recull.py" "$A" "$OUT/tot" "$OUT/rec.pkl" "$KW"
$PY "$B/llarg.py" "$OUT/rec.pkl" "$OUT/llarg.pkl"
# 2) model de consum de l'hora actual (+ residus fora de mostra)
$PY "$B/train_now.py" "$A" "$OUT"
# 3) taules de quantils d'error (k=0: residus del model consum0, excepte les hores de nit)
$PY "$B/mk_incert.py" "$OUT/llarg.pkl" "$OUT" "{\"oof\": \"$OUT/oof_consum0.pkl\", \"k0_nit\": $NIT}"
# 4) models de quantils de la claredat de l'hora actual
$PY "$B/train_s0.py" "$A" "$OUT/rec.pkl" "$OUT"
# (Els passos 5-6 de la v7, arbres del mercat i del valor, ja no formen part de la vlineaE1: el mercat
#  és el model lineal de 3_mercat i el valor la xarxa de 4_valor.)
echo "fet: $OUT"
