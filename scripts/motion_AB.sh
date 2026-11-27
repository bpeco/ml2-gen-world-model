#!/usr/bin/env bash
# Después de la fase 3 (espera "fin F3" en logs/motion_F3.log).
#  A  kdB_1b    : la receta de kdB_s0 (FT 500 pasos, KD + secuencias del maestro, lr 3e-4, semilla 0) sobre kdS1 (1b: gpt_s_kd
#                 + 780 k frames más de KD) en vez de gpt_s_kd. ¿Una base con más datos + el FT de kdB mejora? Decide si
#                 vale la pena entrenar con el shard completo. Control: kdB_s0.
#  B  kd4x512   : alumno 4×512 (más profundo que 3×768, 46,8 ms en la 4060), KD desde cero con la receta de kd3x768
#                 (780 k frames, bs 8 × 2, semilla 0); después kd4x512_B = el FT de kdB encima. Controles: kd3x768(_B), kdB_s0.
# Uso: bash scripts/motion_AB.sh   (desde la raíz del repo)
set -u
PY=.venv/Scripts/python.exe
mkdir -p logs results/motion
until grep -q "fin F3" logs/motion_F3.log 2>/dev/null; do sleep 60; done
echo "== $(date +%H:%M) arranca AB"
SEQ="--seq-maestro data/maestro_seq.npy --seq-frac 0.5"
GFT="--kind gpt --kd --presupuesto 160000 --lr 3e-4 --bs 8 --accum 2 --seed 0 $SEQ"
entrenar() {
  local run=$1; shift
  echo "== $(date +%H:%M) entrenar $run"
  [ -f checkpoints/$run/final.pt ] || $PY -m src.train --run $run "$@" > logs/$run.log 2>&1 || { echo "falló $run"; return 1; }
  grep "^listo" logs/$run.log | tail -1
}
uno() {
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
# ---- A
entrenar kdB_1b $GFT --size s --init-from kdS1 && uno kdB_1b_motion --model kdB_1b --temperature 0.9
dif kdB_1b_motion kdB_s0_motion
dif kdB_1b_motion kdS1_2438_motion
$PY -m src.metrica_giro --names kdB_1b_motion kdB_s0_motion kdS1_2438_motion 2>&1 | grep -v -i warn
# ---- B
C4='{"n_layer":4,"dim":512,"n_head":8}'
entrenar kd4x512 --kind gpt --size s --config "$C4" --kd --seed 0 --bs 8 --accum 2 && uno kd4x512_motion --model kd4x512 --temperature 0.9
entrenar kd4x512_B $GFT --size s --config "$C4" --init-from kd4x512 && uno kd4x512_B_motion --model kd4x512_B --temperature 0.9
$PY -m src.bench --model kd4x512_B --dtype fp16 --graphs --frames 60 --warmup 10 --name kd4x512 2>&1 | grep -v -i warn | tail -1
dif kd4x512_motion kd3x768_2438_motion
dif kd4x512_B_motion kd3x768_B_motion
dif kd4x512_B_motion kdB_s0_motion
$PY -m src.metrica_giro --names kd4x512_motion kd4x512_B_motion kd3x768_2438_motion kd3x768_B_motion kdB_s0_motion 2>&1 | grep -v -i warn
echo "== $(date +%H:%M) fin AB"
