#!/usr/bin/env bash
# Fase 3 de motion: condicionamiento por pose del GPT (Config.pose_dim = 6; tests/test_pose_gpt.py).
#   kdP_s<k>  : FT de kdB_s0, 1000 pasos, KD + secuencias del maestro (la receta de kdB), lr 3e-4, CON poses
#   kdPc_s<k> : lo mismo SIN poses (control: separa el efecto de entrenar 1000 pasos más)
# Única diferencia entre los dos: pose_dim. Las ventanas del maestro entran "sin pose" (no tienen pose real).
# Evaluación en clips_motion: kdP con --poses none / hold / real, kdPc sin poses; giro + FVD + diferencias pareadas.
# Semillas 0 y 1 (mismo init kdB_s0; cambia orden de datos y dropout).
# Espera a que termine la evaluación val (scripts/motion_val.sh) para no compartir la GPU.
# Uso: bash scripts/motion_F3.sh   (desde la raíz del repo)
set -u
PY=.venv/Scripts/python.exe
mkdir -p logs results/motion
until grep -q "fin val" logs/motion_val.log 2>/dev/null; do sleep 60; done
echo "== $(date +%H:%M) arranca F3"
SEQ="--seq-maestro data/maestro_seq.npy --seq-frac 0.5"
FT="--kind gpt --size s --init-from kdB_s0 --kd --presupuesto 320000 --lr 3e-4 --bs 8 --accum 2 $SEQ"
entrenar() {  # run args...
  local run=$1; shift
  echo "== $(date +%H:%M) entrenar $run"
  [ -f checkpoints/$run/final.pt ] || $PY -m src.train --run $run "$@" > logs/$run.log 2>&1 || { echo "falló $run"; return 1; }
  grep "^listo" logs/$run.log | tail -1
}
uno() {  # nombre args-de-rollouts...
  local n=$1; shift
  echo "== $(date +%H:%M) $n"
  [ -f results/rollouts/$n.npy ] || $PY -m src.rollouts --clips motion --name $n --dtype fp16 --graphs "$@" > logs/$n.log 2>&1 || { echo "falló $n"; return; }
  $PY -m src.metrics_video --clips motion --name $n >> logs/$n.log 2>&1 || echo "falló metrics_video $n"
  grep -i fvd logs/$n.log | tail -1
  $PY -m src.metrics_flow --clips motion --name $n 2>&1 | grep -v -i warn | tail -1
}
dif() {
  echo "== diferencia FVD $1 - $2"
  $PY -m src.estadistica diferencia --clips motion --a $1 --b $2 2>&1 | grep -v -i warn | tail -1 | tee -a results/motion/diferencias.txt
}
for s in 0 1; do
  entrenar kdP_s$s $FT --seed $s --config '{"pose_dim":6}' && \
    for m in hold real none; do uno kdP_s${s}_${m}_motion --model kdP_s$s --temperature 0.9 --poses $m; done
  entrenar kdPc_s$s $FT --seed $s && uno kdPc_s${s}_motion --model kdPc_s$s --temperature 0.9
  for m in hold real none; do dif kdP_s${s}_${m}_motion kdPc_s${s}_motion; done
  dif kdPc_s${s}_motion kdB_s0_motion
  $PY -m src.metrica_giro --names kdP_s${s}_hold_motion kdP_s${s}_real_motion kdP_s${s}_none_motion kdPc_s${s}_motion kdB_s0_motion 2>&1 | grep -v -i warn
done
echo "== $(date +%H:%M) fin F3"
