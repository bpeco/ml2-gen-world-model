#!/usr/bin/env bash
# Extras de la etapa 3 (plan, sección 5): barrido de contexto, presupuesto 10x para el mejor modelo barato
# y control del learning rate de los tamaños M. Salta lo que ya existe.
#
#   bash scripts/extras.sh quality   # entrenamientos + CE + rollouts + LPIPS/FVD
#   bash scripts/extras.sh bench     # latencias del barrido de contexto (con la GPU libre)
set -e
PY=${PY:-.venv/Scripts/python}
CK=${CK:-checkpoints}
mkdir -p results/logs

train() {  # train <corrida> <args de src.train...>
  local name=$1; shift
  [ -f "$CK/$name/final.pt" ] && return
  echo "== train $name"; mkdir -p "$CK/$name"
  $PY -u -m src.train --run "$name" --bs 16 --seed 0 --warmup 200 --ckpt-dir "$CK" "$@" > "$CK/$name/train.out" 2>&1
}
ce() { [ -f "results/eval_$1.json" ] || { echo "== CE $1"; $PY -u -m src.evaluate --model "$1" --name "$1" > "results/logs/eval_$1.out" 2>&1; }; }
roll() {  # roll <nombre> <args de src.rollouts...>: rollouts + LPIPS/FVD
  local name=$1; shift
  [ -f "results/rollouts/$name.npy" ] || { echo "== rollouts $name"; $PY -u -m src.rollouts --name "$name" "$@" > "results/logs/roll_$name.out" 2>&1; }
  [ -f "results/video_$name.json" ] || { echo "== video $name"; $PY -u -m src.metrics_video --name "$name" > "results/logs/video_$name.out" 2>&1; }
}

quality() {
  # barrido de contexto en inferencia: los modelos se entrenaron con ventanas de 20 frames y la CE por
  # posición muestra que se las arreglan con pocos frames; menos contexto = prefill más corto
  for k in 4 8; do
    roll gpt_s_ctx$k --model gpt_s --dtype fp16 --graphs --ctx-frames $k
    roll maskgit_s_s2_ctx$k --model maskgit_s --steps 2 --dtype fp16 --graphs --ctx-frames $k
    roll maskgit_s_s4_ctx$k --model maskgit_s --steps 4 --dtype fp16 --graphs --ctx-frames $k
  done
  # control: ¿el tamaño M necesitaba el LR de S? (con 6e-4 queda igual o peor que S)
  train gpt_m_lr1e3 --kind gpt --size m --lr 1e-3 --tokens 100e6 --config '{"pos": "frame"}'
  ce gpt_m_lr1e3
  return 0
}

benchctx() {
  local W="--warmup 20 --frames 100"
  for k in 4 8; do
    $PY -u -m src.bench --model gpt_s --dtype fp16 --graphs --ctx-frames $k $W | tail -1
    for s in 2 4; do $PY -u -m src.bench --model maskgit_s --steps $s --dtype fp16 --graphs --ctx-frames $k $W | tail -1; done
  done
}

case "${1:-quality}" in
  quality) quality ;;
  bench) benchctx ;;
  *) echo "uso: $0 quality|bench"; exit 1 ;;
esac
