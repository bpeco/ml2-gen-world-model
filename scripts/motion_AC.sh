#!/usr/bin/env bash
# motion-experiments, prioridades A y C (cola única de GPU).
#  A: ajuste fino de maskgit_s_abs, 1000 pasos (presupuesto 320000 tokens, lr 3e-4, warmup 100, semilla 0, como v2):
#     mgA_ctrl : sin cambios de receta (control: separa el efecto de entrenar 1000 pasos más)
#     mgA_pc08 : p_clean 0,8 (cada frame queda limpio con prob. 0,8)
#     mgA_ctxl : ctx_limpio (un solo frame enmascarado por clip; su contexto siempre limpio, como al generar)
#     Rollouts en clips_motion con 4 y 8 pasos, T 1,0 (como la fase 1).
#  C: temperatura de gpt_s_kd (T 0,9 ya está: gpt_s_kd_motion): T 1,0, T 1,1, T 1,0 + top-p 0,95.
# Al final, diferencias pareadas de FVD (bootstrap por clip) contra los controles.
# Uso: bash scripts/motion_AC.sh   (desde la raíz del repo)
set -u
PY=.venv/Scripts/python.exe
mkdir -p logs results/motion
FT="--kind maskgit --size s --init-from maskgit_s_abs --presupuesto 320000 --lr 3e-4 --warmup 100 --seed 0"

uno() {  # nombre args-de-rollouts...
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
  tail -1 logs/$run.log
  for s in 8 4; do uno ${run}_p${s}_motion --model $run --steps $s --temperature 1.0; done
}

# C primero: no entrena y responde rápido
uno kdC_t10_motion --model gpt_s_kd --temperature 1.0
uno kdC_t11_motion --model gpt_s_kd --temperature 1.1
uno kdC_t10_p95_motion --model gpt_s_kd --temperature 1.0 --opt top_p=0.95

entrenar mgA_ctrl '{}'
entrenar mgA_ctxl '{"ctx_limpio":true}'
entrenar mgA_pc08 '{"p_clean":0.8}'

dif() { echo "== diferencia FVD $1 - $2"; $PY -m src.cli estadistica diferencia --clips motion --a $1 --b $2 2>&1 | tail -8 | tee -a results/motion/diferencias.txt; }
for t in kdC_t10 kdC_t11 kdC_t10_p95; do dif ${t}_motion gpt_s_kd_motion; done
for v in ctxl pc08; do for s in 8 4; do dif mgA_${v}_p${s}_motion mgA_ctrl_p${s}_motion; done; done
for s in 8 4; do dif mgA_ctrl_p${s}_motion mg_p${s}_motion; done
echo "== $(date +%H:%M) listo"
