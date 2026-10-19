# ml2-gen-world-model

Repositorio para el proyecto de modelo generativo del mundo (ML-II).

## Setup

Usar un Python **arm64 nativo** en Mac. Los Python x86_64 bajo Rosetta solo consiguen torch 2.2.2.

```bash
uv venv --python cpython-3.12-macos-aarch64-none .venv
uv pip install --python .venv/bin/python -r requirements.txt
git clone --depth 1 https://github.com/commaai/commavq.git third_party/commavq   # decoder y gpt2m de comma
```

## Pipeline

```bash
.venv/bin/python -m src.data          # baja shard 0, val oficial, decoder y gpt2m -> data/
.venv/bin/python -m src.baselines     # baselines de conteo (unigrama, markov temporal)
.venv/bin/python -m src.viz           # figuras de datos
.venv/bin/python -m src.train --minutes 30                         # GPT mini -> checkpoints/gpt_mini.pt
.venv/bin/python -m src.evaluate --model checkpoints/gpt_mini.pt   # benchmark -> results/eval_gpt_mini.json
.venv/bin/python -m src.generate --model checkpoints/gpt_mini.pt --name gpt_mini
.venv/bin/python -m src.generate --plot gpt_mini                   # horizonte y tira de frames
```

## Benchmark

`src/evaluate.py` replica el protocolo de `notebooks/eval.ipynb` de comma (commit `a1adab98`):

- Val: los 100 segmentos oficiales de `val.zip`. La lista está en `results/val_split.json`.
- Ventanas: 59 por segmento, de 20 frames (2.580 tokens), con BOS antes de cada frame.
- `ce_all`: cross-entropy media en nats, con los targets BOS incluidos. Es comparable con el 2,02 que comma reporta para gpt2m.
- `ce_tok`, `bits_tok`, `acc_tok`: las mismas métricas sin los targets BOS.

`--model` acepta `unigram`, `markov`, `gpt2m` o la ruta a un checkpoint propio.
