"""Paso final del humo (`<modelo> entrenar --humo`): compara los entrenamientos cortos y escribe la elección.

El humo entrena 300 pasos con cinco lr: el de la receta (fuente), ese lr escalado por raíz y lineal según
batch efectivo / batch de la fuente (Recipe.batch_fuente), y x3 y x10 el de la fuente. Aquí se lee final_monitor_ce de cada
results/train_<corrida>.json y se escribe results/humo_<kind>.json con el candidato de menor loss de monitoreo finita.
La elección entra a la receta por un commit, no por un override a mano. Si la elegida no es la fuente, el JSON lo dice
(`forzado`) para declararla en la receta como forzada por presupuesto.

Modelos latentes (--colapso: vjepa2, vjepa2_ac, lewm): una loss baja puede ser un colapso. Si el log de la corrida
(checkpoints/<corrida>/log.jsonl) trae alguna medida de dispersión de los embeddings (claves *_std: z_std, target_std,
pred_std), se descartan los candidatos cuya dispersión final es menos de 0,1 veces la mediana de los candidatos (o < 1e-4).
Si el log no trae ninguna, el JSON dice "sin chequeo de colapso".

  python -m src.humo --kind gpt --cand fuente:gpt_humo_fuente:1e-3 --cand raiz:gpt_humo_raiz:2e-3
"""
import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

REL_COLAPSO = 0.1   # dispersión < REL_COLAPSO x mediana de los candidatos = colapsado
ABS_COLAPSO = 1e-4


def elegir(candidatos):
  """El candidato de menor monitor_ce (los NaN, faltantes y colapsados no compiten). None si ninguno sirve."""
  validos = [c for c in candidatos if c.get("monitor_ce") is not None and math.isfinite(c["monitor_ce"]) and not c.get("colapsado")]
  return min(validos, key=lambda c: c["monitor_ce"]) if validos else None


def dispersion(ckpt_dir, run):
  """{clave: valor} de las claves *_std del último registro de checkpoints/<run>/log.jsonl ({} si no hay)."""
  f = Path(ckpt_dir) / run / "log.jsonl"
  if not f.exists():
    return {}
  ult = {}
  for ln in f.read_text(encoding="utf-8").splitlines():
    try:
      r = json.loads(ln)
    except json.JSONDecodeError:
      continue
    if any(k.endswith("_std") for k in r):
      ult = r
  return {k: v for k, v in ult.items() if k.endswith("_std") and isinstance(v, (int, float))}


def marcar_colapsados(cands):
  """Pone cand['dispersion'] y cand['colapsado']. Devuelve True si hubo alguna medida para chequear."""
  hay = [c for c in cands if c["dispersion"]]
  if not hay:
    return False
  claves = set.intersection(*[set(c["dispersion"]) for c in hay])
  for c in cands:
    c["colapsado"] = False
  for k in claves:
    ref = statistics.median(c["dispersion"][k] for c in hay)
    for c in hay:
      v = c["dispersion"][k]
      if not math.isfinite(v) or v < ABS_COLAPSO or v < REL_COLAPSO * ref:
        c["colapsado"] = True
  return True


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--kind", required=True)
  ap.add_argument("--cand", action="append", required=True, metavar="nombre:corrida:lr",
                  help="candidato: su nombre (fuente|raiz|lineal|x3|x10), la corrida de humo y el lr pico que se probó")
  ap.add_argument("--pasos", type=int, default=300)
  ap.add_argument("--batch", type=int, help="batch efectivo (muestras) de la receta")
  ap.add_argument("--batch-fuente", type=int, help="batch efectivo de la fuente (muestras); sin él no hay candidatos raíz ni lineal")
  ap.add_argument("--alias", action="append", default=[], metavar="nombre=otro",
                  help="candidato que tiene el mismo lr que otro y no se entrenó (p. ej. raiz=fuente)")
  ap.add_argument("--colapso", action="store_true", help="modelo latente: descartar los candidatos colapsados (si el log lo permite)")
  ap.add_argument("--results-dir", default="results")
  ap.add_argument("--ckpt-dir", default="checkpoints")
  a = ap.parse_args(argv)
  res_dir = Path(a.results_dir)
  cands = []
  for c in a.cand:
    nombre, run, lr = c.split(":")
    f = res_dir / f"train_{run}.json"
    ce = json.loads(f.read_text(encoding="utf-8")).get("final_monitor_ce") if f.exists() else None
    cands.append({"nombre": nombre, "run": run, "lr": float(lr), "monitor_ce": ce, "dispersion": dispersion(a.ckpt_dir, run)})
  nota = []
  chequeo = None
  if a.colapso:
    chequeo = marcar_colapsados(cands)
    if not chequeo:
      nota.append("sin chequeo de colapso: los logs de las corridas no traen ninguna medida de dispersión de los embeddings (*_std)")
  mejor = elegir(cands)
  fuente = next((c for c in cands if c["nombre"] == "fuente"), None)
  if a.batch_fuente is None:
    nota.append("la receta no declara batch_fuente: sin candidatos raíz ni lineal")
  if any(c.get("colapsado") for c in cands):
    nota.append("descartados por colapso: " + ", ".join(c["nombre"] for c in cands if c.get("colapsado")))
  if mejor is None:
    nota.append("ningún candidato terminó con una loss de monitoreo finita")
  forzado = None
  if mejor and fuente and mejor["nombre"] != "fuente":
    forzado = {"campo": "lr", "receta": fuente["lr"], "elegido": mejor["lr"], "candidato": mejor["nombre"], "categoria": "presupuesto",
               "motivo": f"el humo de {a.pasos} pasos elige lr {mejor['lr']:.3g} ({mejor['nombre']}) y no el de la fuente ({fuente['lr']:.3g}): "
                         "declararlo como forzado por presupuesto en la receta"}
  out = {"kind": a.kind, "pasos": a.pasos, "batch_efectivo": a.batch, "batch_fuente": a.batch_fuente,
         "candidatos": cands, "alias": dict(x.split("=") for x in a.alias), "elegido": mejor["nombre"] if mejor else None, "lr_elegido": mejor["lr"] if mejor else None,
         "es_la_fuente": bool(mejor and mejor["nombre"] == "fuente"), "forzado": forzado,
         "chequeo_colapso": chequeo if a.colapso else "no aplica",
         "nota": "; ".join(nota), "fecha": datetime.now().isoformat(timespec="seconds")}
  res_dir.mkdir(parents=True, exist_ok=True)
  (res_dir / f"humo_{a.kind}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
  for c in cands:
    print(f"  {c['nombre']:7s} lr {c['lr']:.3g}  monitor {c['monitor_ce']}" + ("  [colapsado]" if c.get("colapsado") else ""))
  print(f"elegido: {out['elegido']} (lr {out['lr_elegido']})" + (f"  [{out['nota']}]" if out["nota"] else ""))
  if forzado:
    print("forzado: " + forzado["motivo"])
  if mejor is None:
    raise SystemExit(1)
  return out


if __name__ == "__main__":
  main()
