"""Arma results/configs.json (lo que lee src.report): una fila por modelo (corrida) con resultados, derivada de lo que hay
en results/. El nombre del modelo es el de sus JSON: video_<nombre>.json, eval_<run>.json, latent_<run>.json,
bench/<nombre>__<variante>.json e infer/e2e_<nombre>__*.json. Un dato distinto es otro modelo (variante) y convive.
Lo medido en otro dispositivo (--tag, p. ej. bench/<nombre>__<variante>__radeon860m.json) convive pero no desplaza a la RTX:
la fila apunta a los archivos de la RTX (es_ref) y solo usa los de otro dispositivo si es lo único que hay.

Los sufijos _p<pasos> (pasos de decodificación), _ctx<N>, _t<T> y _fp32/_fp16 identifican la evaluación; el resto es la corrida
(la CE de eval_<run>.json y los parámetros de train_<run>.json son de la corrida). La latencia no depende de los pesos: si el
modelo evaluado no tiene bench propio se usa el de su misma arquitectura (la corrida sin las variantes _kd ni _s<semilla>).

  python -m src.configs        (la CLI lo corre sola al terminar un entrenar, evaluar o bench)
"""
import argparse
import json
import re
from pathlib import Path

R = Path("results")
SUF = re.compile(r"(_p\d+)?(_ctx\d+)?(_t[\d.]+)?(_fp(?:16|32))?(_stream)?$")
BASELINES = {"unigram": "Unigrama", "transicion": "Transición por celda", "congelado": "Frame congelado", "real": "Real (piso)"}
AUTOREG = ("gpt", "gpt_kd", "gpt2m", "transicion")   # el destilado también genera de a un token
REF = "rtx3080ti"   # dispositivo de referencia: lo que se mide en otro (--tag) convive en results/ pero no lo desplaza


def exists(p):
  return Path(p).as_posix() if p and Path(p).exists() else None


def params(run):
  p = R / f"train_{run}.json"
  try:
    return json.loads(p.read_text(encoding="utf-8"))["params"] if p.exists() else None
  except (OSError, KeyError, json.JSONDecodeError):
    return None


def run_de(nombre):
  """La corrida de un nombre de evaluación (sin _p8, _ctx4, _t0.8, _fp32 ni _stream)."""
  return SUF.sub("", nombre)


def es_ref(p):
  """¿Es de la RTX, el dispositivo de referencia? Un bench sin etiqueta (bench/<nombre>__<variante>.json) o una latencia
  infer/..__rtx3080ti.json. Lo medido en otro dispositivo lleva su etiqueta (--tag) y no es de la referencia."""
  p = Path(p)
  partes = p.stem.split("__")
  return partes[-1] == REF or (p.parent.name == "bench" and len(partes) == 2)


def preferir_ref(fs):
  """Los archivos de la RTX si hay alguno; si no, todos. Así un resultado de otro dispositivo nunca desplaza al de la RTX
  en configs.json ni en los cuadernos."""
  fs = list(fs)
  return [p for p in fs if es_ref(p)] or fs


def mas_rapido(patron):
  c = preferir_ref(x for x in sorted(R.glob(patron)) if "_cpu" not in x.name and "_int8" not in x.name)
  if not c:
    return None
  return min(c, key=lambda x: json.loads(x.read_text(encoding="utf-8")).get("median_ms", 1e18)).as_posix()


def nombres():
  """Nombres de modelo con algún resultado (video, eval, latent o bench)."""
  out = set()
  for pat, pre in (("video_*.json", "video_"), ("eval_*.json", "eval_"), ("latent_*.json", "latent_")):
    out |= {p.stem[len(pre):] for p in R.glob(pat)}
  out |= {p.name.split("__")[0] for p in R.glob("bench/*__*.json")}
  return sorted(n for n in out if not n.endswith("_stream") and not re.search(r"_seg\d+$", n))


def familia(run):
  if run in BASELINES:
    return "baseline"
  base = run.split("_")[0]
  if run == "gpt2m":
    return "gpt2m"
  if base == "gpt":
    return "gpt_kd" if run.endswith("_kd") else "gpt"
  return base if base in ("genie", "maskgit", "convlstm", "lewm", "vjepa2") else "otro"


def entrada(nombre):
  run = run_de(nombre)
  partes = run.split("_")
  size = partes[1] if len(partes) > 1 and partes[1] in ("s", "m", "l") else None
  m = re.search(r"_p(\d+)", nombre)
  arq = re.sub(r"(_kd|_s\d+)+$", "", run)   # la misma arquitectura sin variante de entrenamiento ni semilla
  bench = mas_rapido(f"bench/{nombre}__*.json") or mas_rapido(f"bench/{run}__*.json") or mas_rapido(f"bench/{arq}__*.json")
  e2e = next(iter(preferir_ref(sorted(R.glob(f"infer/e2e_{nombre}__*.json")) or sorted(R.glob(f"infer/latencia_{nombre}__*.json")))), None)
  ev = exists(R / f"eval_{run}.json") or exists(R / f"latent_{run}.json")
  vd = exists(R / f"video_{nombre}.json")
  if not (ev or vd):
    return None
  fam = familia(run)
  etiqueta = BASELINES.get(run, nombre)
  return {"label": etiqueta, "family": fam, "eval": ev, "video": vd, "bench": bench, "params": params(run),
          "autoregressive": fam in AUTOREG or run in AUTOREG, "horizon": True,
          "group": "ablación" if fam == "maskgit" else ("semilla" if re.search(r"_s\d+$", run) else "principal"), "e2e": exists(e2e), "short": etiqueta, "size": size,
          "steps": int(m.group(1)) if m else None, "kd": run.endswith("_kd")}


def main(argv=None):
  argparse.ArgumentParser(description="Arma results/configs.json con los resultados que existen").parse_args(argv)
  C = [e for e in (entrada(n) for n in nombres()) if e]
  R.mkdir(parents=True, exist_ok=True)
  (R / "configs.json").write_text(json.dumps(C, indent=1, ensure_ascii=False), encoding="utf-8")
  print(f"{len(C)} configuraciones -> {R / 'configs.json'}")
  return C


if __name__ == "__main__":
  main()
