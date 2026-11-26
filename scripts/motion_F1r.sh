#!/usr/bin/env bash
# Fase 1, réplica de semilla (después de scripts/motion_F1d.sh): 1b dio giro 2-3 s que BAJA con más KD (0,07 / 0,03 / 0,02
# en 813 / 1626 / 2438 pasos, semilla 1) mientras kdOP_ctrl (500 pasos, semilla 0) dio 0,21. Esta corrida es kdOP_ctrl
# con semilla 1 (único cambio): separa "0,21 fue suerte de semilla/ventanas" de "el giro sube y después decae".
# Uso: bash scripts/motion_F1r.sh   (desde la raíz del repo)
set -u
PY=.venv/Scripts/python.exe
until grep -q "fin de la cola F1d" logs/motion_F1d.log 2>/dev/null; do sleep 60; done
FT="--kind gpt --size s --init-from gpt_s_kd --kd --presupuesto 160000 --lr 3e-4 --seed 1 --bs 8 --accum 2 --run kdR1"
echo "== $(date +%H:%M) entrenar kdR1"
$PY -m src.train $FT > logs/kdR1.log 2>&1 || { echo "falló kdR1"; exit 1; }
grep "^listo" logs/kdR1.log | tail -1
echo "== $(date +%H:%M) kdR1_motion"
$PY -m src.rollouts --clips motion --name kdR1_motion --dtype fp16 --graphs --model kdR1 --temperature 0.9 > logs/kdR1_motion.log 2>&1 || echo "falló rollout"
$PY -m src.metrics_video --clips motion --name kdR1_motion >> logs/kdR1_motion.log 2>&1
$PY -m src.metrics_flow --clips motion --name kdR1_motion 2>&1 | grep -v -i warn | tail -1
$PY -m src.metrica_giro --names kdR1_motion kdOP_ctrl_motion 2>&1 | grep -v -i warn
echo "== $(date +%H:%M) fin de la cola F1r"
