"""Exporta a ONNX una parte de un modelo latente y compara onnxruntime (CPU) contra PyTorch.

  python -m src.export_latente --model <run | ruta.pt | random:kind> --parte {encoder,predictor} [--out ruta.onnx]

Qué se exporta de cada modelo (la función de export vive con el modelo, aquí solo se la llama y se compara):
  lewm        encoder (un frame uint8 (B, 128, 256, 3) -> embedding) y predictor (un paso: z (B, 3, D) y poses crudas
              (B, 3, 30) sin NaN -> ẑ del paso siguiente)                              src/models/lewm.py exportar_onnx
  vjepa2      encoder (clip normalizado (B, 3, T, H, W) -> tokens del target_encoder con layer_norm)
                                                                                       src/models/vjepa/pretrain.py exportar_onnx
  vjepa2_ac   predictor (z (B, T, N, D), acciones y estados (B, T, A) -> frame siguiente, con layer_norm); su encoder es el
              de vjepa2 y se exporta con ese modelo                                    src/models/vjepa/action.py exportar_onnx
La atención va con ops explícitas (sin SDPA), como en src/export_onnx.py. Escribe el .onnx (por defecto
checkpoints/onnx/<nombre>_<parte>.onnx) y results/export/<nombre>_<parte>.json con el error máximo ORT contra PyTorch.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.models import load_model
from src.models.lewm import LeWM
from src.models.lewm import exportar_onnx as exportar_lewm
from src.models.vjepa.action import VJEPA2AC
from src.models.vjepa.action import exportar_onnx as exportar_ac
from src.models.vjepa.pretrain import VJEPA2
from src.models.vjepa.pretrain import exportar_onnx as exportar_vjepa2

EXPORTAR = {LeWM: (exportar_lewm, ("encoder", "predictor")), VJEPA2: (exportar_vjepa2, ("encoder",)),
            VJEPA2AC: (exportar_ac, ("predictor",))}
TOLERANCIA = 1e-3  # error absoluto máximo aceptado entre ORT y PyTorch (fp32)


def comparar(ruta, entradas, salida):
  """Corre el .onnx con onnxruntime en CPU y devuelve el error absoluto máximo contra la salida de PyTorch."""
  import onnx
  import onnxruntime as ort
  onnx.checker.check_model(onnx.load(str(ruta)))
  sess = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
  out = sess.run(None, entradas)[0]
  assert out.shape == salida.shape, (out.shape, salida.shape)
  return float(np.abs(out - salida).max())


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--model", required=True, help="corrida, ruta .pt o random:<kind> (lewm, vjepa2, vjepa2_ac)")
  p.add_argument("--parte", required=True, choices=["encoder", "predictor"])
  p.add_argument("--out", help="ruta del .onnx (por defecto checkpoints/onnx/<nombre>_<parte>.onnx)")
  p.add_argument("--results", default="results", help="carpeta del JSON de la comparación")
  p.add_argument("--opset", type=int, default=17)
  a = p.parse_args(argv)
  model = load_model(a.model, "cpu").eval()
  if type(model) not in EXPORTAR:
    raise SystemExit(f"{a.model}: se exportan lewm, vjepa2 y vjepa2_ac, no {type(model).__name__}")
  fn, partes = EXPORTAR[type(model)]
  if a.parte not in partes:
    raise SystemExit(f"{type(model).__name__} exporta {' y '.join(partes)}, no {a.parte}"
                     + (": el encoder de vjepa2_ac es el de vjepa2 (exportalo con esa corrida)" if isinstance(model, VJEPA2AC) else ""))
  out = Path(a.out) if a.out else Path("checkpoints/onnx") / f"{model.name}_{a.parte}.onnx"
  ej = fn(model, a.parte, out, opset=a.opset)
  err = comparar(out, ej["entradas"], ej["salida"])
  res = {"model": a.model, "name": model.name, "parte": a.parte, "onnx": str(out), "bytes": out.stat().st_size,
         "entradas": {k: list(v.shape) for k, v in ej["entradas"].items()}, "salida": list(ej["salida"].shape),
         "max_abs_ort_vs_torch": err, "tolerancia": TOLERANCIA, "ok": err <= TOLERANCIA, "opset": a.opset}
  d = Path(a.results) / "export"
  d.mkdir(parents=True, exist_ok=True)
  (d / f"{model.name}_{a.parte}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
  print(f"{out}: {res['bytes'] / 1e6:.1f} MB, ORT (CPU) contra PyTorch: error máximo {err:.2e} ({'ok' if res['ok'] else 'FALLA'})")
  if not res["ok"]:
    raise SystemExit(1)
  return res


if __name__ == "__main__":
  main()
