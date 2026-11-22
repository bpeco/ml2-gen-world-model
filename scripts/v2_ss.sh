#!/usr/bin/env bash
# v2: flujo + scheduled sampling, como ajuste fino de 1000 pasos sobre maskgit_s_v2_motioncb (igual que v2_ajuste.sh).
# Diseño 2x2 con semilla 0: {sin flujo, con flujo} x {sin SS, con SS (p_self 0.25)}, todas con flow_limpio (el
# movimiento de los frames pasados se calcula sin máscaras, como el contexto en inferencia; antes el entrenamiento lo
# veía con ~64 % de tokens enmascarados).
#   ft_l_cb        : motion_cb                      ft_l_flow     : motion_cb + flujo
#   ft_l_cb_ss     : motion_cb + SS                 ft_l_flow_ss  : motion_cb + flujo + SS
# Al final, diferencias pareadas de FVD contra ft_cb (el control de v2_ajuste.sh) y entre las ramas.
set -u
PY=.venv/Scripts/python.exe
mkdir -p logs results/v2
FT="--kind maskgit --size s --init-from maskgit_s_v2_motioncb --presupuesto 320000 --lr 3e-4 --warmup 100 --seed 0"

correr() {  # run config
  local run=$1 cfg=$2
  echo "== $(date +%H:%M) $run $cfg"
  $PY -m src.train $FT --run $run --config "$cfg" > logs/$run.log 2>&1 || { echo "falló el entrenamiento de $run"; return; }
  $PY -m src.cli maskgit evaluar --run $run --pasos 4,8 > logs/${run}_eval.log 2>&1 || echo "falló la evaluación de $run"
  $PY -m src.metrics_motion --name ${run}_p4 --name ${run}_p8
}

correr maskgit_s_v2_ft_l_flow_ss '{"motion_cb":true,"flow_k":4,"flow_limpio":true,"p_self":0.25}'
correr maskgit_s_v2_ft_l_cb_ss   '{"motion_cb":true,"flow_limpio":true,"p_self":0.25}'
correr maskgit_s_v2_ft_l_flow    '{"motion_cb":true,"flow_k":4,"flow_limpio":true}'
correr maskgit_s_v2_ft_l_cb      '{"motion_cb":true,"flow_limpio":true}'

for p in 4 8; do
  for par in "ft_l_flow_ss ft_cb" "ft_l_flow_ss ft_l_cb_ss" "ft_l_flow ft_l_cb" "ft_l_flow ft_flow" "ft_l_cb_ss ft_cb" "ft_l_cb ft_cb"; do
    set -- $par
    a=maskgit_s_v2_$1_p$p b=maskgit_s_v2_$2_p$p
    echo "== diferencia FVD $a - $b"
    $PY -m src.cli estadistica diferencia --a $a --b $b 2>&1 | tail -8 | tee -a results/v2/ss_diferencias.txt
  done
done
echo "== $(date +%H:%M) listo"
