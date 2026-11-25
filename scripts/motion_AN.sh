#!/usr/bin/env bash
# motion-experiments: ¿el giro de kd mejoró por annealing? (sigue a scripts/motion_OP.sh, donde el control kdOP_ctrl
# -- 500 pasos más de KD a lr 3e-4 constante -- subió el giro 2-3 s de 0,03 a 0,21)
#  d050_ctrl  : el mismo FT que kdOP_ctrl pero sobre gpt_s_d050 y sin maestro (¿el FT solo da giro sin destilación?)
#  kdAN_cos1k : FT de gpt_s_kd con KD, 1000 pasos, lr coseno 3e-4 -> 0 (annealing completo); se compara con kdOP_ctrl
#  Todos: semilla 0, batch 8 x accum 2 (VRAM de la 4060), rollouts en clips_motion a T 0,9.
# Uso: bash scripts/motion_AN.sh   (desde la raíz del repo)
set -u
PY=.venv/Scripts/python.exe
mkdir -p logs results/motion
COMUN="--kind gpt --size s --lr 3e-4 --seed 0 --bs 8 --accum 2"

uno() {
  local n=$1; shift
  echo "== $(date +%H:%M) $n"
  [ -f results/rollouts/$n.npy ] || $PY -m src.rollouts --clips motion --name $n --dtype fp16 --graphs "$@" > logs/$n.log 2>&1 || { echo "falló $n"; return; }
  $PY -m src.metrics_video --clips motion --name $n >> logs/$n.log 2>&1 || echo "falló metrics_video $n"
  $PY -m src.metrics_flow --clips motion --name $n 2>&1 | grep -v -i warn | tail -1
}

entrenar() {  # run args...
  local run=$1; shift
  echo "== $(date +%H:%M) entrenar $run"
  [ -f checkpoints/$run/last.pt ] || $PY -m src.train $COMUN --run $run "$@" > logs/$run.log 2>&1 || { echo "falló $run"; return 1; }
  grep "^listo" logs/$run.log
  uno ${run}_motion --model $run --temperature 0.9
}

dif() {
  echo "== diferencia FVD $1 - $2"
  $PY -m src.estadistica diferencia --clips motion --a $1 --b $2 2>&1 | grep -v -i warn | tail -1 | tee -a results/motion/diferencias.txt
}

entrenar d050_ctrl --init-from gpt_s_d050 --presupuesto 160000
entrenar kdAN_cos1k --init-from gpt_s_kd --kd --presupuesto 320000 \
  --set 'receta.lr_segs=[{"nombre":"cosine","forma":"cosine","ini":3e-4,"min_frac":0.0}]'
dif d050_ctrl_motion gpt_s_d050_motion
dif kdAN_cos1k_motion kdOP_ctrl_motion
dif kdAN_cos1k_motion gpt_s_kd_motion
echo "== $(date +%H:%M) fin de la cola AN"
