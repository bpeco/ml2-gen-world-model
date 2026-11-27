#!/usr/bin/env bash
# Evaluación oficial (val del paper: 400 clips / 100 segmentos, FVD a 1/2/3 s + LPIPS) de los modelos nuevos de
# motion-experiments, con la misma receta que los del paper (GPT T 0,9; MaskGIT T 1,0 con 4 y 8 pasos; fp16 + graphs).
# Escribe results/video_<name>.json, que es lo que lee el probador y la tabla del paper.
# Uso: bash scripts/motion_val.sh   (desde la raíz del repo)
set -u
PY=.venv/Scripts/python.exe
mkdir -p logs
uno() {  # nombre args-de-rollouts...
  local n=$1; shift
  echo "== $(date +%H:%M) $n"
  [ -f results/rollouts/$n.npy ] || $PY -m src.rollouts --clips eval --name $n --dtype fp16 --graphs "$@" > logs/val_$n.log 2>&1 || { echo "falló $n"; return; }
  $PY -m src.metrics_video --name $n >> logs/val_$n.log 2>&1 || echo "falló metrics_video $n"
  grep -i fvd logs/val_$n.log | tail -1
}
for m in kdB_s0 kd3x768_B kdOP_ctrl kdB_s1; do uno $m --model $m --temperature 0.9; done
for s in 4 8; do uno mgA_pc08_p$s --model mgA_pc08 --steps $s --temperature 1.0; done
for m in kdB_s0 kd3x768_B kdOP_ctrl; do $PY -m src.estadistica diferencia --a $m --b gpt_s_kd 2>&1 | grep -v -i warn | tail -1; done
for s in 4 8; do $PY -m src.estadistica diferencia --a mgA_pc08_p$s --b maskgit_s_abs_p$s 2>&1 | grep -v -i warn | tail -1; done
echo "== $(date +%H:%M) fin val"
