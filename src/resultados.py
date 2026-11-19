"""Estado de cada resultado de results/: vigente, protocolo anterior o pendiente.

results/estado.json (versionado) mapea la ruta de cada archivo de resultados a {estado, motivo, fecha}:
  vigente             medido con el protocolo y la receta actuales
  protocolo anterior  medido antes de la revisión de fidelidad (protocolo y/o receta anteriores)
  pendiente           esperado pero todavía no medido (el archivo puede no existir; la clave es su ruta o nombre)
src.report solo grafica los vigentes y rotula cada fila de la tabla con su estado.

  python -m src.resultados                          # lista agrupado por estado (y avisa de archivos sin estado)
  python -m src.resultados --init                   # registra como "protocolo anterior" los resultados sin estado
  python -m src.resultados --set ARCHIVO ESTADO --motivo TEXTO

API para otros scripts: estado(ruta), marcar(ruta, estado, motivo), pendiente(nombre, motivo), vigente(ruta).
"""
import argparse
import json
from datetime import date
from pathlib import Path

ESTADOS = ("vigente", "protocolo anterior", "pendiente")
SIN_ESTADO = "sin estado"
ARCHIVO = Path("results/estado.json")
MOTIVO_ANTERIOR = "medido antes de la revisión de fidelidad (protocolo y/o receta anteriores)"
# los JSON de resultados: medidos y derivados (estadística, piso, temperatura, humo), no manifiestos (checkpoints,
# configs, estado, val_split, clips_*), figuras ni cachés. Todo lo que matchea tiene estado y pasa por `bitacora verificar`
PATRONES = ("eval_*.json", "video_*.json", "latent_*.json", "bench/*.json", "infer/*.json", "train_*.json",
            "ce_por_posicion.json", "semillas_*.json", "diferencia_*.json", "piso_*.json", "temperatura_*.json",
            "humo_*.json")


def clave(ruta) -> str:
  """Ruta normalizada (posix, relativa al cwd si está dentro de él) que se usa como clave."""
  p = Path(ruta)
  if p.is_absolute():
    try:
      p = p.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
      pass
  return p.as_posix()


def cargar(archivo=ARCHIVO) -> dict:
  archivo = Path(archivo)
  return json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else {}


def guardar(d, archivo=ARCHIVO):
  archivo = Path(archivo)
  archivo.parent.mkdir(parents=True, exist_ok=True)
  archivo.write_text(json.dumps(dict(sorted(d.items())), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def estado(ruta, registro=None) -> str:
  """Estado de un archivo de resultados ("sin estado" si no está registrado)."""
  d = cargar() if registro is None else registro
  return d.get(clave(ruta), {}).get("estado", SIN_ESTADO)


def vigente(ruta, registro=None) -> bool:
  return estado(ruta, registro) == "vigente"


def marcar(ruta, nuevo, motivo="", archivo=ARCHIVO) -> dict:
  """Registra (o cambia) el estado de `ruta`. Devuelve la entrada."""
  if nuevo not in ESTADOS:
    raise ValueError(f"estado {nuevo!r} inválido; opciones: {ESTADOS}")
  d = cargar(archivo)
  d[clave(ruta)] = e = {"estado": nuevo, "motivo": motivo, "fecha": date.today().isoformat()}
  guardar(d, archivo)
  return e


def pendiente(nombre, motivo="", archivo=ARCHIVO) -> dict:
  """Anota un resultado esperado que todavía no existe (nombre o ruta); lo llaman los paquetes que lo van a medir.
  Cuando se mida, marcar(ruta, "vigente", ...) reemplaza la entrada."""
  return marcar(nombre, "pendiente", motivo, archivo)


def existentes(raiz="results"):
  raiz = Path(raiz)
  return sorted({clave(p) for pat in PATRONES for p in raiz.glob(pat) if p.is_file()})


def inicializar(raiz="results", archivo=ARCHIVO) -> int:
  """Registra como "protocolo anterior" los resultados existentes que no tienen estado. Devuelve cuántos."""
  d = cargar(archivo)
  hoy = date.today().isoformat()
  nuevos = [k for k in existentes(raiz) if k not in d]
  for k in nuevos:
    d[k] = {"estado": "protocolo anterior", "motivo": MOTIVO_ANTERIOR, "fecha": hoy}
  if nuevos:
    guardar(d, archivo)
  return len(nuevos)


def listar(archivo=ARCHIVO, raiz="results") -> dict:
  """{estado: [claves]} con los registrados y, bajo "sin estado", los archivos existentes sin registrar."""
  d = cargar(archivo)
  out = {e: sorted(k for k, v in d.items() if v["estado"] == e) for e in ESTADOS}
  out[SIN_ESTADO] = [k for k in existentes(raiz) if k not in d]
  return out


def main(argv=None):
  ap = argparse.ArgumentParser(description="Estado (vigente / protocolo anterior / pendiente) de los resultados")
  ap.add_argument("--set", nargs=2, metavar=("ARCHIVO", "ESTADO"), help=f"cambia el estado; uno de {ESTADOS}")
  ap.add_argument("--motivo", default="", help="motivo del cambio (con --set)")
  ap.add_argument("--init", action="store_true", help='registra los resultados sin estado como "protocolo anterior"')
  ap.add_argument("--estado-file", default=str(ARCHIVO))
  a = ap.parse_args(argv)
  if a.set:
    arch, nuevo = a.set
    marcar(arch, nuevo, a.motivo, a.estado_file)
    print(f"{clave(arch)}: {nuevo}")
  if a.init:
    print(f"{inicializar(archivo=a.estado_file)} resultados registrados como protocolo anterior")
  res = listar(a.estado_file)
  d = cargar(a.estado_file)
  for e, ks in res.items():
    if not ks and e == SIN_ESTADO:
      continue
    print(f"\n{e} ({len(ks)})")
    motivos = {}
    for k in ks:
      motivos.setdefault(d.get(k, {}).get("motivo", ""), []).append(k)
    for m, kk in motivos.items():
      if m:
        print(f"  -- {m}")
      for k in kk:
        print(f"  {k}")
  return res


if __name__ == "__main__":
  main()
