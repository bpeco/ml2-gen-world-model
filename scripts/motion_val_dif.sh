#!/usr/bin/env bash
# Diferencias pareadas de FVD val de los GPT nuevos contra gpt_s_kd: el rollout val de gpt_s_kd (del paper) no quedó en
# disco ni en la caché de features, así que se regenera con la misma receta (T 0,9, fp16, graphs, semilla 0) con otro
# nombre (no pisa results/video_gpt_s_kd.json). Corre después de la cola AB.
set -u
PY=.venv/Scripts/python.exe
until grep -q "fin AB" logs/motion_AB.log 2>/dev/null; do sleep 60; done
n=gpt_s_kd_val_rep
[ -f results/rollouts/$n.npy ] || $PY -m src.rollouts --clips eval --name $n --dtype fp16 --graphs --model gpt_s_kd --temperature 0.9 > logs/val_$n.log 2>&1
$PY -m src.metrics_video --name $n >> logs/val_$n.log 2>&1
grep -i fvd logs/val_$n.log | tail -1
for m in kdB_s0 kd3x768_B kdOP_ctrl kdB_s1; do $PY -m src.estadistica diferencia --a $m --b $n 2>&1 | grep -v -i warn | tail -1; done
echo "== $(date +%H:%M) fin val_dif"
