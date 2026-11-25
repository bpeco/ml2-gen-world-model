#!/usr/bin/env bash
# motion-experiments, prioridad 4: destilación on-policy de kd.
#  kdOP_ctrl : ajuste fino de gpt_s_kd con KD normal, 500 pasos (control: separa el efecto de entrenar más)
#  kdOP_k8   : lo mismo + on-policy: la mitad del batch con sus últimos 8 frames generados por el alumno (T 0,9) y solo
#              KL(maestro || alumno) ahí; mezcla 50/50 con KD normal (src/models/gpt.py, onpolicy_k)
#  Ambos: presupuesto 160000 frames (500 pasos de 16 x 20), lr 3e-4 constante, semilla 0, batch 8 x accum 2 (con
#  batch 16 la VRAM de la 4060 desborda a memoria compartida y cada paso tarda ~65 s en vez de ~2-5 s).
#  Rollouts en clips_motion a T 0,9 (como gpt_s_kd_motion), FVD/LPIPS, flow y diferencias pareadas.
# Uso: bash scripts/motion_OP.sh   (desde la raíz del repo)
set -u
PY=.venv/Scripts/python.exe
mkdir -p logs results/motion
FT="--kind gpt --size s --init-from gpt_s_kd --kd --presupuesto 160000 --lr 3e-4 --seed 0 --bs 8 --accum 2"

uno() {
  local n=$1; shift
  echo "== $(date +%H:%M) $n"
  [ -f results/rollouts/$n.npy ] || $PY -m src.rollouts --clips motion --name $n --dtype fp16 --graphs "$@" > logs/$n.log 2>&1 || { echo "falló $n"; return; }
  $PY -m src.metrics_video --clips motion --name $n >> logs/$n.log 2>&1 || echo "falló metrics_video $n"
  $PY -m src.metrics_flow --clips motion --name $n 2>&1 | grep -v -i warn | tail -1
}

entrenar() {  # run config
  local run=$1 cfg=$2
  echo "== $(date +%H:%M) entrenar $run $cfg"
  [ -f checkpoints/$run/last.pt ] || $PY -m src.train $FT --run $run --config "$cfg" > logs/$run.log 2>&1 || { echo "falló $run"; return 1; }
  grep "^listo" logs/$run.log
  uno ${run}_motion --model $run --temperature 0.9
}

dif() {
  echo "== diferencia FVD $1 - $2"
  $PY -m src.estadistica diferencia --clips motion --a $1 --b $2 2>&1 | grep -v -i warn | tail -1 | tee -a results/motion/diferencias.txt
}

entrenar kdOP_ctrl '{}'
entrenar kdOP_k8 '{"onpolicy_k":8}'
dif kdOP_k8_motion kdOP_ctrl_motion
dif kdOP_ctrl_motion gpt_s_kd_motion
dif kdOP_k8_motion gpt_s_kd_motion
echo "== $(date +%H:%M) fin de la cola OP"
