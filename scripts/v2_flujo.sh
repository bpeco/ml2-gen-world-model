#!/usr/bin/env bash
# v2: módulo de flujo (motioncb completo) contra el MaskGIT-S del paper, con dos semillas cada uno.
#   flow : motion_cb + transformer chico (2 capas, 128) sobre las últimas 4 diferencias del codebook de cada celda
#   base : maskgit_s_abs (el del paper; la semilla 0 ya existe, acá se entrena la 1)
# Mismo presupuesto que el paper. Cada corrida se evalúa con 4 y 8 pasos (CE, rollouts, LPIPS/FVD, movimiento) y al
# final se calcula la diferencia pareada de FVD (src.estadistica diferencia) flow - base por semilla.
# Uso: bash scripts/v2_flujo.sh   (desde la raíz del repo)
set -u
PY=.venv/Scripts/python.exe
mkdir -p logs results/v2
FLOW="--set config.motion_cb=true --set config.flow_k=4"

correr() {  # variante semilla [--set ...]
  local v=$1 s=$2; shift 2
  local run=maskgit_s_$v; [ "$s" != 0 ] && run=${run}_s$s
  echo "== $(date +%H:%M) $run"
  $PY -m src.cli maskgit entrenar --tamaño s --variante $v --semilla $s "$@" > logs/$run.log 2>&1 || { echo "falló el entrenamiento de $run"; return; }
  $PY -m src.cli maskgit evaluar --tamaño s --variante $v --semilla $s --pasos 4,8 > logs/${run}_eval.log 2>&1 || echo "falló la evaluación de $run"
  $PY -m src.metrics_motion --name ${run}_p4 --name ${run}_p8
}

correr v2_flow 0 $FLOW
correr abs 1
correr v2_flow 1 $FLOW

for p in 4 8; do
  for par in "maskgit_s_v2_flow_p$p maskgit_s_abs_p$p" "maskgit_s_v2_flow_s1_p$p maskgit_s_abs_s1_p$p"; do
    set -- $par
    echo "== diferencia FVD $1 - $2"
    $PY -m src.cli estadistica diferencia --a $1 --b $2 2>&1 | tail -8 | tee -a results/v2/flujo_diferencias.txt
  done
done
echo "== $(date +%H:%M) listo"
