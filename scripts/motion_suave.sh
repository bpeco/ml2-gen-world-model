#!/usr/bin/env bash
# Sensibilidad de la métrica de giro al umbral de curva: 50 curvas de 10-20° (results/clips_motion_suave.json, rutas
# fuera de clips_motion.json). Rollouts de los modelos de la fase 1 (mismas temperaturas) y flow; sin FVD.
# Espera a que termine scripts/motion_AC.sh (una cola de GPU a la vez).
# Uso: bash scripts/motion_suave.sh   (desde la raíz del repo)
set -u
PY=.venv/Scripts/python.exe
until grep -qE "^== [0-9:]+ listo" logs/motion_AC.log 2>/dev/null; do sleep 60; done
uno() {  # nombre args-de-rollouts...
  local n=$1; shift
  echo "== $(date +%H:%M) $n"
  [ -f results/rollouts/$n.npy ] || $PY -m src.rollouts --clips motion_suave --name $n --dtype fp16 --graphs "$@" > logs/$n.log 2>&1 || { echo "falló $n"; return; }
  $PY -m src.metrics_flow --clips motion_suave --name $n 2>&1 | grep -v -i warn | tail -1
}
for s in 1 2 4 8; do uno mg_p${s}_suave --model maskgit_s_abs --steps $s --temperature 1.0; done
uno genie_s_p4_suave --model genie_s --steps 4 --temperature 1.0
uno gpt_s_kd_suave --model gpt_s_kd --temperature 0.9
uno gpt_s_d050_suave --model gpt_s_d050 --temperature 0.9
uno gpt2m_suave --model gpt2m --temperature 1.0 --bs 25 --reanudar
echo "== $(date +%H:%M) listo"
