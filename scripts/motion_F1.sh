#!/usr/bin/env bash
# Fase 1 del plan (experiments/motion/plan_accion.md): 1c bench de GPTs, 1a giro de modelos ya entrenados,
# 1b curva de escala de la destilación de GPT-S. (1d va en scripts/motion_F1d.sh, después de esta cola.)
# Uso: bash scripts/motion_F1.sh   (desde la raíz del repo)
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

# ---- 1c: latencia de GPTs (pesos al azar, fp16 + graphs, B=1, 4060). Referencia: 6x384 = GPT-S
echo "== $(date +%H:%M) 1c bench"
for cfg in 6x384 4x512 4x640 4x768 6x512 6x640 8x384 3x768 4x1024; do
  L=${cfg%x*}; D=${cfg#*x}; H=$((D / 64))
  $PY -m src.bench --model random:gpt_s --dtype fp16 --graphs --frames 60 --warmup 10 --name gptbench_${cfg} \
    --opt n_layer=$L --opt dim=$D --opt n_head=$H 2>&1 | grep -v -i warn | tail -1
done

# ---- 1a: más datos (MaskGIT con el shard completo) y más params (Genie-M), sin maestro
uno mgS0_p4_motion --fvd --model maskgit_s_abs_shard0 --steps 4 --temperature 1.0
uno mgS0_p8_motion --fvd --model maskgit_s_abs_shard0 --steps 8 --temperature 1.0
uno genie_m_p4_motion --fvd --model genie_m --steps 4 --temperature 1.0

# ---- 1b: KD desde gpt_s_kd, +780k frames (2438 pasos), lr 3e-4 constante, semilla 1; giro en 813, 1626 y 2438
FT="--kind gpt --size s --init-from gpt_s_kd --kd --presupuesto 780160 --lr 3e-4 --seed 1 --bs 8 --accum 2 --run kdS1"
for s in 813 1626 2438; do
  echo "== $(date +%H:%M) 1b entrenar kdS1 hasta el paso $s"
  $PY -m src.train $FT --max-steps $s >> logs/kdS1.log 2>&1 || { echo "falló kdS1 en $s"; exit 1; }
  grep "^listo" logs/kdS1.log | tail -1
  cp checkpoints/kdS1/final.pt checkpoints/kdS1/snap_$s.pt
  if [ $s = 2438 ]; then uno kdS1_${s}_motion --fvd --model checkpoints/kdS1/snap_$s.pt --temperature 0.9
  else uno kdS1_${s}_motion --model checkpoints/kdS1/snap_$s.pt --temperature 0.9; fi
done
echo "== $(date +%H:%M) fin de la cola F1"
