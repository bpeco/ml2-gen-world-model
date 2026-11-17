#!/usr/bin/env bash
# Evaluación de todas las configuraciones: CE (protocolo de comma), rollouts sobre los 400 clips fijos,
# LPIPS/FVD y latencia con batch 1. Salta lo que ya existe en results/.
#
#   bash scripts/eval_all.sh quality   # CE + rollouts + métricas de video (se puede correr con la GPU ocupada)
#   bash scripts/eval_all.sh bench     # latencias: correr con la GPU libre (nada más corriendo)
#   bash scripts/eval_all.sh latent    # E5: probes, sorpresa, acciones y CEM de jepa/lewm (src.eval_latent)
#   bash scripts/eval_all.sh bench-latent  # E5: latencias de jepa_s y lewm_s (token y latente), GPU libre
#
# Rollouts de calidad: transformers en fp16 con CUDA graphs (mismos tokens que eager); ConvLSTM y
# baselines en fp32. Temperatura 1 y semillas fijas para todos (plan, sección 2.4).
set -e
PY=${PY:-.venv/Scripts/python}
CK=${CK:-checkpoints}
mkdir -p results/logs

ce() {  # ce <spec> [nombre]: CE en val, fp32 (spec: baseline, gpt2m o nombre de corrida)
  local name=${2:-$1}
  [ -f "results/eval_$name.json" ] && return
  echo "== CE $name"; $PY -u -m src.evaluate --model "$1" --name "$name" > "results/logs/eval_$name.out" 2>&1
}
roll() {  # roll <nombre> <args de src.rollouts...>: rollouts + LPIPS/FVD
  local name=$1; shift
  if [ ! -f "results/rollouts/$name.npy" ]; then
    echo "== rollouts $name"; $PY -u -m src.rollouts --name "$name" "$@" > "results/logs/roll_$name.out" 2>&1
  fi
  if [ ! -f "results/video_$name.json" ]; then
    echo "== video $name"; $PY -u -m src.metrics_video --name "$name" > "results/logs/video_$name.out" 2>&1
  fi
}
bench() {  # bench <args de src.bench...>
  echo "== bench $*"; $PY -u -m src.bench "$@" 2>&1 | tail -1
}

quality() {
  # referencias
  roll real --model real
  roll frozen --model copy_mix --temperature 0     # copy_mix a temperatura 0 = último frame congelado
  for b in unigram copy copy_mix; do ce $b; done
  roll unigram --model unigram
  roll copy --model copy
  # E0
  ce gpt2m
  [ -n "$SKIP_GPT2M" ] || roll gpt2m --model gpt2m --dtype fp16 --graphs --bs 16
  # E1, E2, E2b (GPT en fp16 con CUDA graphs)
  ce convlstm; roll convlstm --model convlstm
  for r in gpt_s gpt_m gpt_s_kd gpt_m_kd gpt_l gpt_s_abs; do
    [ -f "$CK/$r/final.pt" ] || continue
    ce $r; roll "$r" --model $r --dtype fp16 --graphs
  done
  # E3: MaskGIT, barrido de pasos (las ablaciones solo con 8 pasos)
  for r in maskgit_s maskgit_m maskgit_s_genie maskgit_s_nopc maskgit_s_abs; do
    [ -f "$CK/$r/final.pt" ] || continue
    ce $r
    case $r in maskgit_s|maskgit_m) S="1 2 4 8" ;; *) S="8" ;; esac
    for s in $S; do roll "${r}_s$s" --model $r --steps $s --dtype fp16 --graphs; done
  done
  # H3: fp32 vs fp16 en calidad (CE y video)
  for r in gpt_s maskgit_s; do
    [ -f "results/eval_${r}_fp16.json" ] || $PY -u -m src.evaluate --model $r --name $r --dtype fp16 > "results/logs/eval_${r}_fp16.out" 2>&1
  done
  roll gpt_s_fp32 --model gpt_s
  roll maskgit_s_s8_fp32 --model maskgit_s --steps 8
  # E5: world models latentes (jepa, lewm), en val sin poses (acción nula). fp16 + CUDA graphs como los transformers.
  for r in $LATENT; do
    [ -f "$CK/$r/final.pt" ] || continue
    ce $r
    case $r in lewm_s_l003|lewm_s_l03) continue ;; esac  # barrido de λ: alcanza con CE y eval_latent (ahorra ~25 min)
    roll "$r" --model $r --dtype fp16 --graphs
  done
  for r in jepa_s lewm_s; do  # rollout latente: ẑ entra al contexto sin re-codificar el frame muestreado
    [ -f "$CK/$r/final.pt" ] && roll "${r}_latent" --model $r --dtype fp16 --graphs --opt rollout=latent
  done
  return 0
}

LATENT="jepa_s jepa_s_ce jepa_s_sigreg jepa_s_codebook lewm_s lewm_s_ce lewm_s_ema lewm_s_l003 lewm_s_l03 jepa_xs jepa_xxs lewm_xs"

latent() {
  # evaluaciones estilo LeWM (probes, sorpresa, rectitud, acciones, CEM) en el holdout separado por ruta
  for r in $LATENT; do
    [ -f "$CK/$r/final.pt" ] || continue
    [ -f "results/latent_$r.json" ] && continue
    # CEM de JEPA: ~20 s por plan; en los controles de JEPA no aporta a H6 y se saltea (ahorra ~35 min)
    case $r in jepa_s_ce|jepa_s_codebook|jepa_xs|jepa_xxs) NP=--no-plan ;; *) NP= ;; esac
    echo "== latente $r"; $PY -u -m src.eval_latent --model $r $NP > "results/logs/latent_$r.out" 2>&1
  done
  for r in gpt_s maskgit_s; do  # referencias de tokens: solo el probe de pose
    [ -f "results/latent_$r.json" ] && continue
    echo "== probe $r"; $PY -u -m src.eval_latent --model $r --probes-only > "results/logs/latent_$r.out" 2>&1
  done
}

benchlatent() {
  # con la GPU libre; los controles y ablaciones comparten arquitectura (y latencia) con jepa_s / lewm_s
  local W="--warmup 20 --frames 100"
  for m in jepa_s lewm_s; do
    bench --model $m --dtype fp32 $W
    bench --model $m --dtype fp16 $W
    bench --model $m --dtype fp16 --graphs $W
    bench --model $m --name "${m}_latent" --opt rollout=latent --dtype fp16 --graphs $W
  done
  for m in jepa_xs jepa_xxs lewm_xs; do  # E5b, versiones chicas: solo la configuración de la tabla
    [ -f "$CK/$m/final.pt" ] || continue
    [ -f "results/bench/${m}__fp16_graphs.json" ] || bench --model $m --dtype fp16 --graphs $W
  done
}

benchall() {
  # Con la GPU libre. Los modelos entrenados se miden con su checkpoint (la latencia no depende de los
  # pesos, pero sí de la arquitectura); el tamaño L, sin entrenar. Scratch y destilado comparten arquitectura.
  local W="--warmup 20 --frames 100"
  bench --decoder --dtype fp32 $W; bench --decoder --dtype fp16 $W
  bench --model copy $W
  bench --model gpt2m --dtype fp32 --warmup 3 --frames 20
  bench --model gpt2m --dtype fp16 --warmup 5 --frames 30
  bench --model gpt2m --dtype fp16 --graphs $W
  for m in gpt_s gpt_m random:gpt_l; do
    bench --model $m --dtype fp32 $W
    bench --model $m --dtype fp16 $W
    bench --model $m --dtype fp16 --graphs $W
  done
  for m in maskgit_s maskgit_m random:maskgit_l; do
    for s in 1 2 4 8; do
      bench --model $m --steps $s --dtype fp32 $W
      bench --model $m --steps $s --dtype fp16 --graphs $W
    done
    bench --model $m --steps 8 --dtype fp16 $W
  done
  bench --model convlstm --dtype fp32 $W
  bench --model convlstm --dtype fp16 $W
  # CPU: fp32 vs int8 dinámico (H3)
  for m in gpt_s convlstm; do
    bench --model $m --device cpu --warmup 3 --frames 20
    bench --model $m --device cpu --int8 --warmup 3 --frames 20
  done
  bench --model maskgit_s --steps 4 --device cpu --warmup 3 --frames 20
  bench --model maskgit_s --steps 4 --device cpu --int8 --warmup 3 --frames 20
}

case "${1:-quality}" in
  quality) quality ;;
  bench) benchall ;;
  latent) latent ;;
  bench-latent) benchlatent ;;
  *) echo "uso: $0 quality|bench|latent|bench-latent"; exit 1 ;;
esac
