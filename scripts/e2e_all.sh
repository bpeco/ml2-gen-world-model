#!/usr/bin/env bash
# Inferencia de punta a punta (Docs/plan-inferencia.md): video -> encoder -> world model -> decoder -> video, batch 1.
# Salta lo que ya existe en results/infer/. Correr con la GPU libre (nada más corriendo).
#
#   bash scripts/e2e_all.sh codec     # fidelidad y latencia del codec en este dispositivo
#   bash scripts/e2e_all.sh bench     # latencia de la cadena de cada configuración, en línea e imaginación
#   bash scripts/e2e_all.sh paced     # algunas configuraciones a ritmo de cámara (un frame cada 50 ms)
#   bash scripts/e2e_all.sh quality   # calidad a 1 paso en línea, deriva de ConvLSTM y verificación del camino nuevo
#   bash scripts/e2e_all.sh new       # rollouts y LPIPS/FVD de lo que no estaba medido (MaskGIT con 12 pasos)
#   bash scripts/e2e_all.sh all       # codec + bench + paced (lo que hay que correr en cada dispositivo)
#
# RANDOM_MODELS=1: modelos sin entrenar con la misma arquitectura (random:<corrida>), para medir latencia sin bajar
# los checkpoints (Colab, laptop): la latencia no depende de los pesos. TAG=<nombre> fija el nombre del dispositivo.
set -e
PY=${PY:-.venv/Scripts/python}
[ -x "$PY" ] || PY=python
mkdir -p results/infer results/logs
TAGARG=${TAG:+--tag $TAG}
TAGV=${TAG:-$($PY -c "import torch; from src.infer.bench_codec import device_tag; print(device_tag('cuda' if torch.cuda.is_available() else 'cpu'))")}

spec() { [ -n "$RANDOM_MODELS" ] && echo "random:$1" || echo "$1"; }
have() {  # have <corrida>: hay checkpoint local o registrado (o se usan modelos sin entrenar)
  [ -n "$RANDOM_MODELS" ] || [ -f "checkpoints/$1/final.pt" ] || \
    $PY -c "import sys; from src import store; sys.exit('$1' not in store.load_manifest())" 2>/dev/null
}

e2e() {  # e2e <nombre> <modo> <pace> <args de bench_e2e...>
  local name=$1 mode=$2 pace=$3; shift 3
  local pn=$([ "$pace" = 0 ] && echo closed || echo "pace$pace")
  [ -f "results/infer/e2e_${name}__${mode}__${pn}__${TAGV}.json" ] && return 0
  $PY -u -m src.infer.bench_e2e --name "${name%_s[0-9]*}" --mode "$mode" --pace "$pace" --frames "$FRAMES" $TAGARG "$@" \
    2>&1 | grep -v -i warn || true
}
FRAMES=${FRAMES:-3000}  # frames medidos por configuración (en Colab, FRAMES=1000 para que entre en una sesión)

CONFIGS=(  # nombre|modelo|args
  "codec|none|"
  "convlstm|convlstm|"
  "maskgit_s_s1|maskgit_s|--steps 1" "maskgit_s_s2|maskgit_s|--steps 2" "maskgit_s_s4|maskgit_s|--steps 4"
  "maskgit_s_s8|maskgit_s|--steps 8" "maskgit_s_s12|maskgit_s|--steps 12"
  "maskgit_m_s1|maskgit_m|--steps 1" "maskgit_m_s2|maskgit_m|--steps 2" "maskgit_m_s4|maskgit_m|--steps 4"
  "maskgit_m_s8|maskgit_m|--steps 8" "maskgit_m_s12|maskgit_m|--steps 12"
  "jepa_s|jepa_s|" "jepa_xs|jepa_xs|" "jepa_xxs|jepa_xxs|"
  "lewm_s|lewm_s|" "lewm_xs|lewm_xs|"
  # LeWM desde píxeles (E5c): la cámara entra directo al modelo, sin el encoder de comma; imagina en el latente
  "lewm_px_xs|lewm_px_xs|--opt rollout=latent" "lewm_px|lewm_px|--opt rollout=latent"
  "gpt_s|gpt_s|--frames 300 --warmup 10"
)

codec() {
  [ -f "results/infer/codec_${TAGV}.json" ] || $PY -u -m src.infer.bench_codec $TAGARG 2>&1 | grep -v -i warn
}

bench() {
  [ -z "$RANDOM_MODELS" ] && e2e copy online 0 --model copy && e2e copy imagine 0 --model copy
  for c in "${CONFIGS[@]}"; do
    IFS="|" read -r name model args <<< "$c"
    m=$([ "$model" = none ] && echo none || spec "$model")
    [ "$model" = none ] || have "$model" || continue
    for mode in online imagine; do
      [ "$model" = none ] && [ "$mode" = imagine ] && continue
      e2e "$name" "$mode" 0 --model "$m" $args
    done
  done
  for k in 5 20; do  # en línea, prediciendo más adelante (k pasos del modelo, un solo decode)
    for c in "convlstm|convlstm|" "maskgit_s_s4|maskgit_s|--steps 4" "jepa_xxs|jepa_xxs|"; do
      IFS="|" read -r name model args <<< "$c"
      [ -f "results/infer/e2e_${name}__online_k${k}__closed__${TAGV}.json" ] && continue
      $PY -u -m src.infer.bench_e2e --name "${name%_s[0-9]*}" --model "$(spec $model)" $args --k $k --frames "$FRAMES" \
        $TAGARG 2>&1 | grep -v -i warn || true
    done
  done
}

paced() {
  for c in "codec|none|" "convlstm|convlstm|" "maskgit_s_s4|maskgit_s|--steps 4" "maskgit_m_s8|maskgit_m|--steps 8" \
           "jepa_s|jepa_s|" "lewm_xs|lewm_xs|"; do
    IFS="|" read -r name model args <<< "$c"
    m=$([ "$model" = none ] && echo none || spec "$model")
    mode=$([ "$model" = none ] && echo online || echo imagine)
    e2e "$name" "$mode" 50 --model "$m" $args
  done
}

quality() {
  # calidad a 1 paso en línea (50 segmentos de val x 400 frames), temperatura 1 y 0
  for c in "convlstm|" "maskgit_s|--steps 4" "maskgit_m|--steps 8" "jepa_s|" "jepa_xxs|" "lewm_xs|" "lewm_px_xs|" "lewm_px|" "copy|"; do
    IFS="|" read -r model args <<< "$c"
    [ "$model" = copy ] || have "$model" || continue
    for t in 1 0; do
      name=$model$(echo "$args" | sed -E 's/--steps ([0-9]+)/_s\1/; s/ //g')
      sfx=$([ "$t" = 0 ] && echo _t0 || echo "")
      [ -f "results/infer/online_${name}${sfx}.json" ] && continue
      $PY -u -m src.infer.quality online --model $model $args --temperature $t 2>&1 | grep -v -i warn
    done
  done
  # deriva de ConvLSTM en streams de 1 minuto: estado continuo vs. recalculado con la ventana de 19 frames
  [ -f results/infer/online_convlstm_f1200.json ] || $PY -u -m src.infer.quality online --model convlstm --segments 20 --frames 1200
  [ -f results/infer/online_convlstm_r20_f1200.json ] || $PY -u -m src.infer.quality online --model convlstm --segments 20 --frames 1200 --reset-every 20
  # el camino nuevo (Stream) contra los rollouts de la tabla: tokens y LPIPS/FVD
  for c in "maskgit_s_s4|maskgit_s|--steps 4" "convlstm|convlstm|"; do
    IFS="|" read -r name model args <<< "$c"
    [ -f "results/infer/imagine_check_${name}.json" ] || $PY -u -m src.infer.quality imagine --model $model $args --name $name 2>&1 | grep -v -i warn
    [ -f "results/video_${name}_stream.json" ] || $PY -u -m src.metrics_video --name "${name}_stream" > "results/logs/video_${name}_stream.out" 2>&1
  done
}

new() {
  for r in maskgit_s maskgit_m; do
    n=${r}_s12
    [ -f "results/rollouts/$n.npy" ] || $PY -u -m src.rollouts --name $n --model $r --steps 12 --dtype fp16 --graphs > "results/logs/roll_$n.out" 2>&1
    [ -f "results/video_$n.json" ] || $PY -u -m src.metrics_video --name $n > "results/logs/video_$n.out" 2>&1
  done
}

case "${1:-all}" in
  codec) codec ;;
  bench) bench ;;
  paced) paced ;;
  quality) quality ;;
  new) new ;;
  all) codec; bench; paced ;;
  *) echo "uso: $0 codec|bench|paced|quality|new|all"; exit 1 ;;
esac
