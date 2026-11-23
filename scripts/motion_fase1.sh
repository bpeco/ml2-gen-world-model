#!/usr/bin/env bash
# motion-experiments, fase 1: rollouts sobre results/clips_motion.json (100 rectas + 100 curvas del val con poses),
# FVD/LPIPS y flow (RAFT) de cada uno. Temperaturas = las de la evaluación del paper (results/rollouts/*_r.json, *_p*.json).
# gpt2m va al final y con --reanudar (checkpoint tras cada batch).
set -u
PY=.venv/Scripts/python.exe
mkdir -p logs
uno() {  # nombre args-de-rollouts...
  local n=$1; shift
  echo "== $(date +%H:%M) $n"
  [ -f results/rollouts/$n.npy ] || $PY -m src.rollouts --clips motion --name $n --dtype fp16 --graphs "$@" > logs/$n.log 2>&1 || { echo "falló $n"; return; }
  $PY -m src.metrics_video --clips motion --name $n >> logs/$n.log 2>&1 || echo "falló metrics_video $n"
  $PY -m src.metrics_flow --clips motion --name $n 2>&1 | grep -v -i warn | tail -1
}
for s in 1 2 4 8; do uno mg_p${s}_motion --model maskgit_s_abs --steps $s --temperature 1.0; done
uno genie_s_p4_motion --model genie_s --steps 4 --temperature 1.0
uno gpt_s_kd_motion --model gpt_s_kd --temperature 0.9
uno gpt_s_d050_motion --model gpt_s_d050 --temperature 0.9
uno gpt2m_motion --model gpt2m --temperature 1.0 --bs 20 --reanudar
echo "== $(date +%H:%M) listo"
