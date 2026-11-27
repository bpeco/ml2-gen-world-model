#!/usr/bin/env bash
# Paso 1 tras la fase 2: el FT de kdB_s0 (500 pasos, KD + secuencias del maestro, lr 3e-4, semilla 0) sobre kd3x768.
# ¿El FT corto le destraba el giro al modelo grande como se lo destrabó a kd? Control: kd3x768_2438 (sin FT).
# Uso: bash scripts/motion_F2c.sh   (desde la raíz del repo)
set -u
PY=.venv/Scripts/python.exe
echo "== $(date +%H:%M) entrenar kd3x768_B"
$PY -m src.train --run kd3x768_B --kind gpt --size s --config '{"n_layer":3,"dim":768,"n_head":12}' --init-from kd3x768 \
  --kd --presupuesto 160000 --lr 3e-4 --seed 0 --bs 8 --accum 2 --seq-maestro data/maestro_seq.npy --seq-frac 0.5 \
  > logs/kd3x768_B.log 2>&1 || { echo "falló kd3x768_B"; exit 1; }
grep "^listo" logs/kd3x768_B.log | tail -1
echo "== $(date +%H:%M) kd3x768_B_motion"
$PY -m src.rollouts --clips motion --name kd3x768_B_motion --dtype fp16 --graphs --model kd3x768_B --temperature 0.9 > logs/kd3x768_B_motion.log 2>&1 || echo "falló rollout"
$PY -m src.metrics_video --clips motion --name kd3x768_B_motion >> logs/kd3x768_B_motion.log 2>&1
grep FVD logs/kd3x768_B_motion.log | tail -1
$PY -m src.metrics_flow --clips motion --name kd3x768_B_motion 2>&1 | grep -v -i warn | tail -1
for b in kd3x768_2438_motion kdB_s0_motion; do
  $PY -m src.estadistica diferencia --clips motion --a kd3x768_B_motion --b $b 2>&1 | grep -v -i warn | tail -1 | tee -a results/motion/diferencias.txt
done
$PY -m src.metrica_giro --names kd3x768_B_motion kd3x768_2438_motion kdB_s0_motion 2>&1 | grep -v -i warn
echo "== $(date +%H:%M) fin F2c"
