# ml2-gen-world-model

World models de manejo sobre [commaVQ](https://github.com/commaai/commavq). Trabajo práctico de Aprendizaje Automático II (MIA, UdeSA).

Pregunta del proyecto: ¿cuánto se puede achicar y acelerar un world model de video de manejo sin perder calidad?

## Setup

En Mac conviene un Python arm64 nativo: con uno x86_64 (Rosetta) solo se consigue torch 2.2.2.

```bash
uv venv --python cpython-3.12-macos-aarch64-none .venv
uv pip install --python .venv/bin/python -r requirements.txt
git clone --depth 1 https://github.com/commaai/commavq.git third_party/commavq   # código del VQ-VAE y del GPT de comma
```

## Datos

```bash
.venv/bin/python -m src.data   # shard 0 (train), val oficial, decoder y pesos del GPT de comma -> data/
.venv/bin/python -m src.viz    # frames de val decodificados -> results/
```

- Train: `data-0000.tar.gz`, 2.500 segmentos de 1 min (1200 x 8 x 16 tokens).
- Val: los 100 segmentos de `val.zip` de una revisión anterior del dataset. La lista queda en `results/val_split.json`. No comparten rutas con el shard de train.

## E0: referencia

```bash
.venv/bin/python -m src.evaluate --model gpt2m   # -> results/eval_gpt2m.json
```

`src/evaluate.py` replica el protocolo de `notebooks/eval.ipynb` de commaVQ (commit `a1adab98`):

- 59 ventanas de 20 frames (2.580 tokens) por segmento, con BOS antes de cada frame.
- `ce_all`: cross-entropy media en nats, incluyendo los targets BOS. Es comparable con el 2,02 que reporta comma.
- `ce_tok`, `bits_tok`, `acc_tok`: las mismas métricas sin los targets BOS.

Resultado: `ce_all = 2,026` sobre los 100 segmentos.

## Slides

`pre-entrega/index.html`: presentación de avance. Se navega con las flechas.
