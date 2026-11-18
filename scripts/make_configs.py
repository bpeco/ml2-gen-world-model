"""Arma results/configs.json (lo que lee src.report): una entrada por configuración, con sus JSON de
CE (eval), video (LPIPS/FVD) y latencia (bench). Las que no tienen resultados se saltan.

La latencia no depende de los pesos: los destilados y las ablaciones usan el bench del modelo de cero de
su misma arquitectura, medido con batch 1 en la misma GPU. Configuración de latencia de referencia:
fp16 + CUDA graphs para los transformers, la más rápida de las medidas para el resto.

Uso: python scripts/make_configs.py
"""
import json
from pathlib import Path

R = Path("results")


def exists(p):
  return Path(p).as_posix() if p and Path(p).exists() else None


def params(run):
  p = R / f"train_{run}.json"
  return json.loads(p.read_text())["params"] if p.exists() else None


def best_bench(prefix):
  """El bench más rápido entre los que empiezan con prefix (p. ej. random_gpt_l__)."""
  cands = sorted(R.glob(f"bench/{prefix}*.json"))
  cands = [c for c in cands if "_cpu" not in c.name and "_int8" not in c.name]
  if not cands:
    return None
  return min(cands, key=lambda c: json.loads(c.read_text())["median_ms"]).as_posix()


def e2e(name):
  """Latencia de punta a punta (src/infer/bench_e2e.py), modo imaginación, bucle cerrado, en la GPU de la tabla."""
  return R / f"infer/e2e_{name}__imagine__closed__rtx3080ti.json"


def entry(label, family, eval_=None, video=None, bench=None, params_=None, autoregressive=False, horizon=True,
          group="principal", e2e_=None, **extra):
  """extra: short (etiqueta corta para las figuras), size, steps, kd. e2e_: JSON de bench_e2e (misma arquitectura)."""
  e = {"label": label, "family": family, "eval": exists(eval_), "video": exists(video), "bench": exists(bench),
       "params": params_, "autoregressive": autoregressive, "horizon": horizon, "group": group,
       "e2e": exists(e2e_), **extra}
  return e if (e["eval"] or e["video"]) else None


def main():
  C = []
  add = lambda e: e and C.append(e)
  # referencias
  add(entry("Unigrama", "baseline", R / "eval_unigram.json", R / "video_unigram.json", horizon=False, short="Unigrama"))
  # "copy" es el nombre viejo de la tabla de transición: los JSON de resultados ya medidos conservan ese nombre
  add(entry("Transición por celda", "baseline", R / "eval_copy.json", R / "video_copy.json",
            R / "bench/copy__fp32.json", autoregressive=True, short="Transición", e2e_=e2e("copy")))
  add(entry("Frame congelado", "baseline", None, R / "video_frozen.json", short="Frame congelado"))
  add(entry("gpt2m (comma)", "gpt2m", R / "eval_gpt2m.json", R / "video_gpt2m.json",
            R / "bench/gpt2m__fp16_graphs.json", 307e6, autoregressive=True, short="gpt2m (comma)"))
  # GPT alumnos (E2, E2b)
  for size in ("s", "m", "l"):
    for kd in ("", "_kd"):
      run = f"gpt_{size}{kd}"
      add(entry(f"GPT-{size.upper()}" + (" destilado" if kd else ""), "gpt_kd" if kd else "gpt",
                R / f"eval_{run}.json", R / f"video_{run}.json", R / f"bench/gpt_{size}__fp16_graphs.json",
                params(run), autoregressive=True, short=f"GPT-{size.upper()}" + (" KD" if kd else ""),
                size=size, kd=bool(kd), e2e_=e2e(f"gpt_{size}")))
  # MaskGIT (E3): un punto por cantidad de pasos (12: agregado con la inferencia de punta a punta)
  for size in ("s", "m"):
    for s in (1, 2, 4, 8, 12):
      run = f"maskgit_{size}"
      add(entry(f"MaskGIT-{size.upper()} {s} pasos", "maskgit", R / f"eval_{run}.json", R / f"video_{run}_s{s}.json",
                R / f"bench/maskgit_{size}__fp16_graphs_s{s}.json", params(run), horizon=s in (1, 8),
                short=f"MaskGIT-{size.upper()}", size=size, steps=s, e2e_=e2e(f"maskgit_{size}_s{s}")))
  # ablaciones
  add(entry("GPT-S, posición absoluta", "ablación", R / "eval_gpt_s_abs.json", R / "video_gpt_s_abs.json",
            R / "bench/gpt_s__fp16_graphs.json", params("gpt_s_abs"), autoregressive=True, horizon=False,
            group="ablación"))
  add(entry("MaskGIT-S 8 pasos, posición absoluta", "ablación", R / "eval_maskgit_s_abs.json",
            R / "video_maskgit_s_abs_s8.json", R / "bench/maskgit_s__fp16_graphs_s8.json",
            params("maskgit_s_abs"), horizon=False, group="ablación"))
  add(entry("MaskGIT-S 8 pasos, máscara Genie", "ablación", R / "eval_maskgit_s_genie.json",
            R / "video_maskgit_s_genie_s8.json", R / "bench/maskgit_s__fp16_graphs_s8.json",
            params("maskgit_s_genie"), horizon=False, group="ablación"))
  add(entry("MaskGIT-S 8 pasos, sin p_clean", "ablación", R / "eval_maskgit_s_nopc.json",
            R / "video_maskgit_s_nopc_s8.json", R / "bench/maskgit_s__fp16_graphs_s8.json",
            params("maskgit_s_nopc"), horizon=False, group="ablación"))
  # extras: presupuesto 10x y barrido de contexto (contexto más corto = prefill más corto)
  for k in (4, 8):
    add(entry(f"GPT-S, contexto {k}", "contexto", None, R / f"video_gpt_s_ctx{k}.json",
              R / f"bench/gpt_s__fp16_graphs_ctx{k}.json", params("gpt_s"), horizon=False, group="extra"))
    for s in (2, 4):
      add(entry(f"MaskGIT-S {s} pasos, contexto {k}", "contexto", None, R / f"video_maskgit_s_s{s}_ctx{k}.json",
                R / f"bench/maskgit_s__fp16_graphs_s{s}_ctx{k}.json", params("maskgit_s"), horizon=False,
                group="extra"))
  (R / "configs.json").write_text(json.dumps(C, indent=1, ensure_ascii=False), encoding="utf-8")
  print(f"{len(C)} configuraciones -> {R / 'configs.json'}")


if __name__ == "__main__":
  main()
