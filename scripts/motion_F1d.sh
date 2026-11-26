#!/usr/bin/env bash
# Fase 1 (resto), después de scripts/motion_F1.sh (una cola de GPU a la vez):
#  1c bench de latencia de GPTs más grandes (pesos al azar, B=1, fp16 + graphs; con la GPU libre)
#  1d MaskGIT-S con maestro (etiquetas blandas en teacher forcing, src/models/maskgit.py) contra su control:
#     mgT_ctrl : FT de maskgit_s_abs sin maestro, 1000 pasos (receta de mgA), batch 8 x accum 2
#     mgT_kd   : lo mismo + --kd (alpha 0,5, tau 1)
#     (el batch 8 x 2 es por la VRAM con el maestro; el control usa el mismo para que el único cambio sea el maestro)
# Uso: bash scripts/motion_F1d.sh   (desde la raíz del repo)
set -u
PY=.venv/Scripts/python.exe
until grep -q "fin de la cola F1" logs/motion_F1.log 2>/dev/null; do sleep 60; done

uno() {
  local n=$1; shift
  local fvd=0; [ "$1" = "--fvd" ] && { fvd=1; shift; }
  echo "== $(date +%H:%M) $n"
  [ -f results/rollouts/$n.npy ] || $PY -m src.rollouts --clips motion --name $n --dtype fp16 --graphs "$@" > logs/$n.log 2>&1 || { echo "falló $n"; return; }
  [ $fvd = 1 ] && { $PY -m src.metrics_video --clips motion --name $n >> logs/$n.log 2>&1 || echo "falló metrics_video $n"; }
  $PY -m src.metrics_flow --clips motion --name $n 2>&1 | grep -v -i warn | tail -1
}

echo "== $(date +%H:%M) 1c bench"
CFGS="6x384 4x512 4x640 4x768 6x512 6x640 8x384 3x768 3x1024 4x1024"
$PY scripts/gpt_bench_cfgs.py $CFGS 2>&1 | grep -v -i warn
for c in $CFGS; do
  $PY -m src.bench --model checkpoints/_bench/gpt_$c.pt --dtype fp16 --graphs --frames 60 --warmup 10 --name gptbench_$c 2>&1 | grep -v -i warn | tail -1
done

FT="--kind maskgit --size s --init-from maskgit_s_abs --presupuesto 320000 --lr 3e-4 --warmup 100 --seed 0 --bs 8 --accum 2"
for v in ctrl kd; do
  echo "== $(date +%H:%M) 1d entrenar mgT_$v"
  extra=""; [ $v = kd ] && extra="--kd"
  [ -f checkpoints/mgT_$v/final.pt ] || $PY -m src.train $FT $extra --run mgT_$v > logs/mgT_$v.log 2>&1 || { echo "falló mgT_$v"; continue; }
  grep "^listo" logs/mgT_$v.log
  uno mgT_${v}_p4_motion --fvd --model mgT_$v --steps 4 --temperature 1.0
  uno mgT_${v}_p8_motion --model mgT_$v --steps 8 --temperature 1.0
done
echo "== diferencia FVD mgT_kd_p4 - mgT_ctrl_p4"
$PY -m src.estadistica diferencia --clips motion --a mgT_kd_p4_motion --b mgT_ctrl_p4_motion 2>&1 | grep -v -i warn | tail -1 | tee -a results/motion/diferencias.txt
echo "== $(date +%H:%M) fin de la cola F1d"
