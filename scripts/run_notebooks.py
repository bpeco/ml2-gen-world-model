"""Ejecuta todos los notebooks del repo con nbclient, en CPU y con un recorte chico, y falla si alguno da error.

  python scripts/run_notebooks.py                     todos, con --limite 3
  python scripts/run_notebooks.py --limite 5 --solo modelos/genie demo_clase

Cada notebook corre en su propio kernel, con CUDA_VISIBLE_DEVICES=-1 y ML2_DEVICE=cpu (CPU aunque haya GPU). `--limite` va a la
variable de entorno ML2_NB_LIMITE, que las celdas leen como LIMITE (pasos, segmentos o frames de cada acción) y que activa el modo
prueba de demo_clase.ipynb (pesos sin entrenar y clips cortos). Los resultados y checkpoints que no existen se muestran como
*pendiente*: no es un error. No guarda las salidas: los .ipynb del repo quedan sin ejecutar.
"""
import argparse
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
NB = RAIZ / "notebooks"


def lista(solo=()):
  todos = sorted(p for p in NB.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)
  if not solo:
    return todos
  return [p for p in todos if any(s in p.relative_to(NB).as_posix() for s in solo)]


def correr(ruta, timeout):
  import asyncio
  if sys.platform == "win32":   # zmq no usa el loop Proactor de Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
  import nbformat
  from nbclient import NotebookClient
  from nbclient.exceptions import CellExecutionError
  nb = nbformat.read(str(ruta), as_version=4)
  cli = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(ruta.parent)}},
                       allow_errors=False)
  t0 = time.time()
  try:
    cli.execute()
    return time.time() - t0, None, nb
  except CellExecutionError as e:
    return time.time() - t0, str(e)[-1800:], nb


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--limite", type=int, default=3, help="recorte de cada acción (ML2_NB_LIMITE); default 3")
  ap.add_argument("--solo", nargs="*", default=[], help="solo los notebooks cuya ruta (dentro de notebooks/) contiene alguno de estos textos")
  ap.add_argument("--timeout", type=int, default=900, help="segundos por celda")
  a = ap.parse_args(argv)
  os.environ.update({"CUDA_VISIBLE_DEVICES": "-1", "ML2_DEVICE": "cpu", "ML2_NB_LIMITE": str(a.limite), "PYTHONIOENCODING": "utf-8"})
  fallas, t0 = [], time.time()
  nbs = lista(a.solo)
  for p in nbs:
    dur, err, _ = correr(p, a.timeout)
    rel = p.relative_to(NB).as_posix()
    print(f"{'ok   ' if err is None else 'FALLA'} {rel:32s} {dur:7.1f} s", flush=True)
    if err:
      fallas.append(rel)
      print(err, flush=True)
  print(f"\n{len(nbs) - len(fallas)}/{len(nbs)} notebooks sin errores en {time.time() - t0:.0f} s (limite {a.limite}, CPU)")
  return 1 if fallas else 0


if __name__ == "__main__":
  sys.exit(main())
