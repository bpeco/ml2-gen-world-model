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
  """El bench más rápido entre los que empiezan con prefix (p. ej. random_convlstm__)."""
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
  add(entry("Copia (tabla por celda)", "baseline", R / "eval_copy.json", R / "video_copy.json",
            R / "bench/copy__fp32.json", autoregressive=True, short="Copia", e2e_=e2e("copy")))
  add(entry("Frame congelado", "baseline", None, R / "video_frozen.json", short="Frame congelado"))
  add(entry("gpt2m (comma)", "gpt2m", R / "eval_gpt2m.json", R / "video_gpt2m.json",
            R / "bench/gpt2m__fp16_graphs.json", 307e6, autoregressive=True, short="gpt2m (comma)"))
  add(entry("ConvLSTM", "convlstm", R / "eval_convlstm.json", R / "video_convlstm.json",
            best_bench("convlstm__"), params("convlstm"), autoregressive=True, short="ConvLSTM", e2e_=e2e("convlstm")))
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
  add(entry("ConvLSTM, 1B tokens", "convlstm", R / "eval_convlstm_1b.json", R / "video_convlstm_1b.json",
            best_bench("convlstm__"), params("convlstm_1b"), autoregressive=True, group="extra",
            short="ConvLSTM, 1B tokens"))
  for k in (4, 8):
    add(entry(f"GPT-S, contexto {k}", "contexto", None, R / f"video_gpt_s_ctx{k}.json",
              R / f"bench/gpt_s__fp16_graphs_ctx{k}.json", params("gpt_s"), horizon=False, group="extra"))
    for s in (2, 4):
      add(entry(f"MaskGIT-S {s} pasos, contexto {k}", "contexto", None, R / f"video_maskgit_s_s{s}_ctx{k}.json",
                R / f"bench/maskgit_s__fp16_graphs_s{s}_ctx{k}.json", params("maskgit_s"), horizon=False,
                group="extra"))
  # E5: world models latentes. jepa_s y lewm_s compiten en la tabla (mismo presupuesto de 100M tokens); los
  # controles, el 2×2 y el barrido de λ van en el grupo "latente" (no compiten). Los controles y ablaciones usan
  # el bench del principal de su familia (misma arquitectura). No entran en H4 (autoregressive=False): su CE
  # es por celda independiente dado el pasado, como la de ConvLSTM, pero H4 quedó fijada con el set original.
  for run, label, fam, short in (("jepa_s", "JEPA-S (V-JEPA, latente por celda)", "jepa", "JEPA-S"),
                                 ("lewm_s", "LeWM (LeWorldModel, un vector por frame)", "lewm", "LeWM")):
    add(entry(label, fam, R / f"eval_{run}.json", R / f"video_{run}.json", R / f"bench/{run}__fp16_graphs.json",
              params(run), horizon=True, short=short, e2e_=e2e(run)))
  # E5b: versiones más chicas de cada familia. Compiten en la tabla, como GPT-M y MaskGIT-M.
  for run, label, fam, short in (("jepa_xs", "JEPA-XS (4 capas, D 256)", "jepa", "JEPA-XS"),
                                 ("jepa_xxs", "JEPA-XXS (3 capas, D 192)", "jepa", "JEPA-XXS"),
                                 ("lewm_xs", "LeWM-XS (ViT de 6 capas, predictor de 4)", "lewm", "LeWM-XS")):
    add(entry(label, fam, R / f"eval_{run}.json", R / f"video_{run}.json", R / f"bench/{run}__fp16_graphs.json",
              params(run), horizon=False, short=short, e2e_=e2e(run)))
  # E5c: LeWM desde píxeles (como el paper): misma arquitectura que lewm_xs / lewm_s salvo la entrada. Compiten.
  for run, label, short in (("lewm_px_xs", "LeWM-XS desde píxeles", "LeWM-XS px"),
                            ("lewm_px", "LeWM desde píxeles", "LeWM px")):
    add(entry(label, "lewm", R / f"eval_{run}.json", R / f"video_{run}.json", R / f"bench/{run}__fp16_graphs.json",
              params(run), horizon=False, short=short, e2e_=e2e(run)))
  latent = [("jepa_s_ce", "JEPA-S, control CE", "jepa", "jepa_s"),
            ("jepa_s_sigreg", "JEPA-S con SIGReg", "jepa", "jepa_s"),
            ("jepa_s_codebook", "JEPA-S, target codebook", "jepa", "jepa_s"),
            ("lewm_s_ce", "LeWM, control CE", "lewm", "lewm_s"),
            ("lewm_s_ema", "LeWM con EMA", "lewm", "lewm_s"),
            ("lewm_s_l003", "LeWM, λ = 0,03", "lewm", "lewm_s"),
            ("lewm_s_l03", "LeWM, λ = 0,3", "lewm", "lewm_s")]
  for run, label, fam, base in latent:
    add(entry(label, fam, R / f"eval_{run}.json", R / f"video_{run}.json", R / f"bench/{base}__fp16_graphs.json",
              params(run), horizon=False, group="latente", short=label))
  for run, label, fam in (("jepa_s", "JEPA-S, rollout latente", "jepa"), ("lewm_s", "LeWM, rollout latente", "lewm"),
                          ("lewm_px_xs", "LeWM-XS desde píxeles, rollout latente", "lewm"),
                          ("lewm_px", "LeWM desde píxeles, rollout latente", "lewm")):
    add(entry(label, fam, None, R / f"video_{run}_latent.json", R / f"bench/{run}_latent__fp16_graphs.json",
              params(run), horizon=False, group="latente", short=label))
  (R / "configs.json").write_text(json.dumps(C, indent=1, ensure_ascii=False), encoding="utf-8")
  print(f"{len(C)} configuraciones -> {R / 'configs.json'}")


if __name__ == "__main__":
  main()
