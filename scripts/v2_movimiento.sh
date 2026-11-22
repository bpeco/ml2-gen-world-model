#!/usr/bin/env bash
# v2: experimentos de movimiento con MaskGIT-S, mismo presupuesto que maskgit_s_abs (el del paper).
#   peso3   : CE con peso 3 en los tokens que cambian respecto del frame anterior
#   pself   : scheduled sampling, la mitad de los frames de entrada son reconstrucciones del propio modelo
#   motion  : embedding de "igual / distinto al frame anterior" en cada token visible
#   motioncb: proyección de la diferencia de vectores del codebook de commaVQ con el frame anterior
# Cada uno se evalúa con 4 y 8 pasos (CE, rollouts, LPIPS/FVD) y con la tasa de cambio de tokens.
# Uso: bash scripts/v2_movimiento.sh   (desde la raíz del repo; espera a que la GPU no tenga otro barrido corriendo)
set -u
PY=.venv/Scripts/python.exe
mkdir -p logs

# esperar a que termine un barrido de src.temperatura (si hay uno corriendo; se busca solo entre los python)
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*src.temperatura*' }) { exit 0 } else { exit 1 }"; do
  sleep 60
done

for exp in "peso3 config.peso_cambio=3" "pself config.p_self=0.5" "motion config.motion_in=true" "motioncb config.motion_cb=true"; do
  set -- $exp
  v=v2_$1
  echo "== $(date +%H:%M) $v ($2)"
  $PY -m src.cli maskgit entrenar --tamaño s --variante $v --set $2 > logs/maskgit_$v.log 2>&1 || { echo "falló el entrenamiento de $v"; continue; }
  $PY -m src.cli maskgit evaluar --tamaño s --variante $v --pasos 4,8 > logs/maskgit_${v}_eval.log 2>&1 || echo "falló la evaluación de $v"
  $PY -m src.metrics_motion --name maskgit_s_${v}_p4 --name maskgit_s_${v}_p8
done
# temperatura por debajo de 1 en el modelo del shard completo (en el original lo mejor fue 0,8-0,9)
$PY -m src.temperatura --model maskgit_s_abs_shard0 --steps 8 --dtype fp16 --temperaturas 0.8,0.9 --out results/extra/ct1_bajo > logs/temp_shard0_bajo.log 2>&1
echo "== $(date +%H:%M) listo"
