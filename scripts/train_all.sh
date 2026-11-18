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
run convlstm        --kind convlstm --lr 1e-3
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
# E5: world models latentes (Docs/plan-jepa.md), mismos batches, semilla y 100M tokens. Las poses entran como
# acciones con dropout (p_null=0,25): sin poses (val) usan la acción nula. Necesitan data/train_poses.npy
# (python -m src.data --poses-only). lr elegido en corridas de humo de 400 pasos: jepa 1e-3, lewm 3e-4 (wd 1e-3
# como en el paper). jepa = V-JEPA (latente por celda + EMA); lewm = LeWorldModel (un vector por frame + SIGReg).
run jepa_s          --kind jepa --size s --lr 1e-3
run jepa_s_ce       --kind jepa --size s --lr 1e-3 --config '{"objective": "ce"}'
run jepa_s_sigreg   --kind jepa --size s --lr 1e-3 --config '{"objective": "sigreg"}'
run jepa_s_codebook --kind jepa --size s --lr 1e-3 --config '{"objective": "codebook"}'
run lewm_s          --kind lewm --lr 3e-4 --wd 1e-3
run lewm_s_ce       --kind lewm --lr 3e-4 --wd 1e-3 --config '{"objective": "ce"}'
run lewm_s_ema      --kind lewm --lr 3e-4 --wd 1e-3 --config '{"objective": "ema"}'
run lewm_s_l003     --kind lewm --lr 3e-4 --wd 1e-3 --config '{"lam": 0.03}'
run lewm_s_l03      --kind lewm --lr 3e-4 --wd 1e-3 --config '{"lam": 0.3}'
# E5b: versiones más chicas (mismo lr que su familia), para ver si alguna entra en la zona rápida del Pareto
# (entre la Copia, 0,47 ms, y ConvLSTM, 1,1 ms). Latencia sin entrenar: jepa_xs 1,7 ms, jepa_xxs 1,0, lewm_xs 0,7.
run jepa_xs         --kind jepa --lr 1e-3 --config '{"n_layer": 4, "n_head": 4, "dim": 256, "n_enc": 1}'
run jepa_xxs        --kind jepa --lr 1e-3 --config '{"n_layer": 3, "n_head": 3, "dim": 192, "n_enc": 1}'
run lewm_xs         --kind lewm --lr 3e-4 --wd 1e-3 --config '{"enc_layers": 6, "proj_hidden": 1024, "pred_dim": 256, "pred_layers": 4, "pred_heads": 8, "pred_mlp": 1024, "dec_layers": 2}'
# E5c: LeWM desde píxeles, como el paper: ViT sobre parches de 16x16 de la imagen RGB en vez de los vectores del codebook.
# El dataset solo trae tokens, así que los frames se generan al vuelo con el decoder de comma congelado (fp16). Misma
# arquitectura que lewm_xs / lewm_s salvo la entrada: compara tokens contra píxeles. En inferencia no necesita el encoder de comma.
run lewm_px_xs      --kind lewm --lr 3e-4 --wd 1e-3 --config '{"input": "pixels", "enc_layers": 6, "proj_hidden": 1024, "pred_dim": 256, "pred_layers": 4, "pred_heads": 8, "pred_mlp": 1024, "dec_layers": 2}'
run lewm_px         --kind lewm --lr 3e-4 --wd 1e-3 --config '{"input": "pixels"}'
