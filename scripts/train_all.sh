#!/usr/bin/env bash
# Entrenamientos de E1, E2, E2b y E3 con el mismo presupuesto: 100M tokens, batch 16 x 20 frames,
# semilla 0. Scratch y destilado comparten tamaño, datos, orden de batches, inicialización y LR (H1).
# Los transformers propios usan embedding de posición factorizado (frame + celda, pos="frame"): con el
# absoluto, el GPT empeora en los últimos frames de la ventana, justo donde genera. MaskGIT además deja
# la mitad de los frames sin máscara al entrenar (p_clean=0.5): en inferencia el contexto está limpio, y
# sin eso la CE del último frame se dispara con contextos largos (ver Docs/resultados.md).
# Uso, desde la raíz del repo: bash scripts/train_all.sh [ckpt-dir]
set -e
PY=${PY:-.venv/Scripts/python}
CK=${1:-checkpoints}
COMMON="--tokens 100e6 --bs 16 --seed 0 --warmup 200 --ckpt-dir $CK"
FPOS='{"pos": "frame"}'
MG='{"pos": "frame", "p_clean": 0.5}'
run() {  # run <nombre> <args...>: salta las corridas que ya tienen final.pt
  local name=$1; shift
  if [ -f "$CK/$name/final.pt" ]; then echo "== $name ya está"; return; fi
  echo "== $name: $*"
  mkdir -p "$CK/$name"
  $PY -u -m src.train --run "$name" $COMMON "$@" > "$CK/$name/train.out" 2>&1
  tail -1 "$CK/$name/train.out"
  # se anota en results/checkpoints.json (sha256, config, params): después se carga con load_model("<nombre>")
  [ "$CK" = checkpoints ] && $PY -m src.store register "$name" > /dev/null
  return 0
}
# E1, E2, E3 (de cero)
run gpt_s           --kind gpt --size s --lr 1e-3 --config "$FPOS"
run maskgit_s       --kind maskgit --size s --lr 1e-3 --config "$MG"
run gpt_m           --kind gpt --size m --lr 6e-4 --config "$FPOS"
run maskgit_m       --kind maskgit --size m --lr 6e-4 --config "$MG"
# ablaciones baratas (tamaño S): máscara de Genie (tasa uniforme en [0,5; 1]), sin p_clean y posición absoluta
run maskgit_s_genie --kind maskgit --size s --lr 1e-3 --config '{"pos": "frame", "p_clean": 0.5, "mask_schedule": "uniform"}'
run maskgit_s_nopc  --kind maskgit --size s --lr 1e-3 --config "$FPOS"
run gpt_s_abs       --kind gpt --size s --lr 1e-3
run maskgit_s_abs   --kind maskgit --size s --lr 1e-3
# E2b: destilados (H1), mismos tamaños, datos, semilla y LR que los de cero
run gpt_s_kd        --kind gpt --size s --lr 1e-3 --config "$FPOS" --kd --alpha 0.5 --tau 1.0
run gpt_m_kd        --kind gpt --size m --lr 6e-4 --config "$FPOS" --kd --alpha 0.5 --tau 1.0
# Los world models latentes (LeWM, V-JEPA 2) y el ConvLSTM de píxeles se entrenan con la cola de entrenamiento
# (receta por modelo), no con este script.
