#!/usr/bin/env bash
# v2: el módulo de flujo como ajuste fino sobre motioncb ya entrenado (desde cero se quedaba trabado en la meseta inicial).
# Las dos ramas arrancan del mismo checkpoint (maskgit_s_v2_motioncb, 2438 pasos) y entrenan 1000 pasos más con la misma
# receta (lr pico 3e-4, warmup 100, coseno), los mismos datos y la misma semilla; la única diferencia es el flujo:
#   ft_flow : motion_cb + transformer de flujo (flow_k=4), el flujo inicializado en cero
#   ft_cb   : motion_cb solo (control: separa el efecto del flujo del efecto de entrenar 1000 pasos más)
# Dos semillas de ajuste fino (orden de datos distinto). Al final, diferencia pareada de FVD ft_flow - ft_cb.
# Uso: bash scripts/v2_ajuste.sh   (desde la raíz del repo)
set -u
PY=.venv/Scripts/python.exe
mkdir -p logs results/v2
FT="--kind maskgit --size s --init-from maskgit_s_v2_motioncb --presupuesto 320000 --lr 3e-4 --warmup 100"

correr() {  # run semilla config
  local run=$1 s=$2 cfg=$3
  echo "== $(date +%H:%M) $run"
  $PY -m src.train $FT --run $run --seed $s --config "$cfg" > logs/$run.log 2>&1 || { echo "falló el entrenamiento de $run"; return; }
  $PY -m src.cli maskgit evaluar --run $run --pasos 4,8 > logs/${run}_eval.log 2>&1 || echo "falló la evaluación de $run"
  $PY -m src.metrics_motion --name ${run}_p4 --name ${run}_p8
}

for s in 0 1; do
  suf=""; [ $s != 0 ] && suf=_s$s
  correr maskgit_s_v2_ft_flow$suf $s '{"motion_cb":true,"flow_k":4}'
  correr maskgit_s_v2_ft_cb$suf $s '{"motion_cb":true}'
done

for p in 4 8; do
  for suf in "" _s1; do
    a=maskgit_s_v2_ft_flow${suf}_p$p b=maskgit_s_v2_ft_cb${suf}_p$p
    echo "== diferencia FVD $a - $b"
    $PY -m src.cli estadistica diferencia --a $a --b $b 2>&1 | tail -8 | tee -a results/v2/ajuste_diferencias.txt
  done
done
echo "== $(date +%H:%M) listo"
