#!/bin/bash
# Reconstrueix els fitxers de dades de la vlineaE1 a partir de data/train/ i munta el paquet a
# entrenament/vlineal/sortida/vlineaE1/. Executar des de l'arrel del repo:
#     bash entrenament/vlineal/construir_vlineaE1.sh
# L'agent és submission_dificil/agent.py. Els fitxers del campus (parametres.json, perfils_consum.json,
# incert.json, model_campus.npz) vénen de la v7a (pas 1, vegeu el README); aquí es copien de
# submission_dificil/ i només es refan el mercat, el valor de l'energia, el Kalman i l'historial.
# Temps aproximat: 15-25 min (el Kalman és el pas més lent).
set -e
PY=".venv/Scripts/python.exe -W ignore"
B=entrenament/vlineal
SUB=submission_dificil
OUT=$B/sortida/vlineaE1
BASE="parametres.json perfils_consum.json incert.json model_campus.npz"
mkdir -p $OUT
prepara() {   # prepara <carpeta del pas>: agent + parametres a ag_tot/, carpeta art/tot/ i data/
  mkdir -p $1/ag_tot $1/art/tot $1/data
  cp $SUB/agent.py $SUB/parametres.json $1/ag_tot/
}
echo "== 2) historial (72 h que l'agent recupera en començar cada setmana)"
$PY $B/2_historial/historial.py $B/2_historial/out

echo "== 3) model lineal del preu i el CO2"
prepara $B/3_mercat
$PY $B/3_mercat/src/ds_mkt.py tot
$PY $B/3_mercat/src/aavg.py $B/3_mercat/data/mkt_tot.npz $B/3_mercat/art/tot/nt_mercat_ridge.npz

echo "== 4) valor de l'energia guardada (xarxa neuronal, 3 llavors x 5 membres)"
prepara $B/4_valor
for f in $BASE; do cp $SUB/$f $B/4_valor/art/tot/; done
cp $B/2_historial/out/historial.npz $B/3_mercat/art/tot/nt_mercat_ridge.npz $B/4_valor/art/tot/
$PY $B/4_valor/src/collect.py tot
for s in 0 100 200; do $PY $B/4_valor/src/train_valor.py tot 3 mlp $s; done
$PY $B/4_valor/src/ajunta.py $B/4_valor/art/tot/nt_valor.npz $B/4_valor/art/tot/nt_valor_mlp_K3.npz \
    $B/4_valor/art/tot/nt_valor_mlp_K3_s100.npz $B/4_valor/art/tot/nt_valor_mlp_K3_s200.npz

echo "== 5) filtre de Kalman del CO2"
prepara $B/5_kalman
$PY $B/5_kalman/src/series.py tot
$PY $B/5_kalman/src/fit.py tot ml
$PY $B/5_kalman/src/exporta.py $B/5_kalman/data/kf_tot_ml.npz $B/5_kalman/art/tot/kf_mercat.npz

echo "== paquet"
cp $SUB/agent.py $OUT/
for f in $BASE; do cp $SUB/$f $OUT/; done
cp $B/2_historial/out/historial.npz $OUT/
cp $B/3_mercat/art/tot/nt_mercat_ridge.npz $OUT/nt_mercat.npz
cp $B/4_valor/art/tot/nt_valor.npz $OUT/
cp $B/5_kalman/art/tot/kf_mercat.npz $OUT/
echo "fet: $OUT"
