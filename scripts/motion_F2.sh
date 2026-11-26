#!/usr/bin/env bash
# Fase 2 revisada (experiments/motion/plan_accion.md), una sola cola de GPU:
#  2B' destilación por secuencias: GPT2M genera 20 frames desde 19 reales (1250 ventanas de train del shard 0, ~25 k
#      frames, src/maestro_seq.py); la mitad de cada batch son ventanas de esas secuencias (--seq-maestro).
#      kdB_s0 vs kdOP_ctrl, kdB_s1 vs kdR1 (FT de 500 pasos de gpt_s_kd con KD; único cambio: las secuencias)
#      mgB vs mgT_ctrl (FT de 1000 pasos de maskgit_s_abs sin maestro; único cambio: las secuencias, CE sobre sus tokens)
#  2A' capacidad: GPT 3x768 (24,8 M) con KD desde cero, la receta de gpt_s_kd (780 k frames, lr 1e-3 constante,
#      batch efectivo 16 = 8 x 2 por VRAM), snapshots en 813 / 1626 / 2438 pasos; control gpt_s_kd.
# Uso: bash scripts/motion_F2.sh   (desde la raíz del repo)
set -u
PY=.venv/Scripts/python.exe
mkdir -p logs results/motion
uno() {  # nombre [--fvd] args-de-rollouts...
  local n=$1; shift
  local fvd=0; [ "$1" = "--fvd" ] && { fvd=1; shift; }
  echo "== $(date +%H:%M) $n"
  [ -f results/rollouts/$n.npy ] || $PY -m src.rollouts --clips motion --name $n --dtype fp16 --graphs "$@" > logs/$n.log 2>&1 || { echo "falló $n"; return; }
  [ $fvd = 1 ] && { $PY -m src.metrics_video --clips motion --name $n >> logs/$n.log 2>&1 || echo "falló metrics_video $n"; }
  $PY -m src.metrics_flow --clips motion --name $n 2>&1 | grep -v -i warn | tail -1
}
dif() {
  echo "== diferencia FVD $1 - $2"
  $PY -m src.estadistica diferencia --clips motion --a $1 --b $2 2>&1 | grep -v -i warn | tail -1 | tee -a results/motion/diferencias.txt
}
entrenar() {  # run args...
  local run=$1; shift
  echo "== $(date +%H:%M) entrenar $run"
  [ -f checkpoints/$run/final.pt ] || $PY -m src.train --run $run "$@" > logs/$run.log 2>&1 || { echo "falló $run"; return 1; }
  grep "^listo" logs/$run.log | tail -1
}

# ---- 2B'
echo "== $(date +%H:%M) generar secuencias del maestro"
$PY -u -m src.maestro_seq --n 1250 --gen 20 --bs 12 > logs/maestro_seq.log 2>&1 || { echo "falló maestro_seq"; exit 1; }
tail -1 logs/maestro_seq.log
SEQ="--seq-maestro data/maestro_seq.npy --seq-frac 0.5"
GFT="--kind gpt --size s --init-from gpt_s_kd --kd --presupuesto 160000 --lr 3e-4 --bs 8 --accum 2"
entrenar kdB_s0 $GFT --seed 0 $SEQ && uno kdB_s0_motion --fvd --model kdB_s0 --temperature 0.9
entrenar kdB_s1 $GFT --seed 1 $SEQ && uno kdB_s1_motion --fvd --model kdB_s1 --temperature 0.9
MFT="--kind maskgit --size s --init-from maskgit_s_abs --presupuesto 320000 --lr 3e-4 --warmup 100 --seed 0 --bs 8 --accum 2"
if entrenar mgB $MFT $SEQ; then
  uno mgB_p4_motion --fvd --model mgB --steps 4 --temperature 1.0
  uno mgB_p8_motion --model mgB --steps 8 --temperature 1.0
fi
dif kdB_s0_motion kdOP_ctrl_motion
dif kdB_s1_motion kdR1_motion
dif mgB_p4_motion mgT_ctrl_p4_motion
$PY -m src.metrica_giro --names kdB_s0_motion kdB_s1_motion mgB_p4_motion mgB_p8_motion 2>&1 | grep -v -i warn
echo "== $(date +%H:%M) fin 2B"

# ---- 2A'
FT="--kind gpt --size s --config {\"n_layer\":3,\"dim\":768,\"n_head\":12} --kd --seed 0 --bs 8 --accum 2 --run kd3x768"
for s in 813 1626 2438; do
  echo "== $(date +%H:%M) 2A entrenar kd3x768 hasta el paso $s"
  $PY -m src.train $FT --max-steps $s >> logs/kd3x768.log 2>&1 || { echo "falló kd3x768 en $s"; exit 1; }
  grep "^listo" logs/kd3x768.log | tail -1
  cp checkpoints/kd3x768/final.pt checkpoints/kd3x768/snap_$s.pt
  if [ $s = 2438 ]; then uno kd3x768_${s}_motion --fvd --model checkpoints/kd3x768/snap_$s.pt --temperature 0.9
  else uno kd3x768_${s}_motion --model checkpoints/kd3x768/snap_$s.pt --temperature 0.9; fi
done
dif kd3x768_2438_motion gpt_s_kd_motion
$PY -m src.bench --model checkpoints/kd3x768/snap_2438.pt --dtype fp16 --graphs --frames 60 --warmup 10 --name kd3x768 2>&1 | grep -v -i warn | tail -1
$PY -m src.metrica_giro --names kd3x768_813_motion kd3x768_1626_motion kd3x768_2438_motion 2>&1 | grep -v -i warn
echo "== $(date +%H:%M) fin de la cola F2"
