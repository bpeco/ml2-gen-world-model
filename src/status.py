"""Estado de las corridas: train.py escribe checkpoints/<run>/status.json en cada log; esto lo lee.

  python -m src.status                 una fila por corrida de checkpoints/
  python -m src.status --watch [seg]   lo repite cada `seg` segundos (default 10), Ctrl-C para salir

Una corrida "en curso" cuyo proceso ya no existe, o cuyo estado tiene más de 3 min, se muestra "colgada".
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

STALE_S = 180  # un estado más viejo que esto es de una corrida colgada o muerta


def write_status(run_dir, **fields):
  """Escribe status.json de forma atómica (un lector nunca ve un archivo a medias)."""
  run_dir = Path(run_dir)
  fields.setdefault("pid", os.getpid())
  fields.setdefault("hora", time.time())
  tmp = run_dir / "status.json.tmp"
  tmp.write_text(json.dumps(fields, indent=1, ensure_ascii=False))
  os.replace(tmp, run_dir / "status.json")


def pid_alive(pid):
  """¿Existe el proceso? (en Windows no se puede usar os.kill(pid, 0): termina el proceso)."""
  if not pid:
    return False
  if sys.platform == "win32":
    import ctypes
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
      return False
    try:
      code = ctypes.c_ulong()
      return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
    finally:
      k32.CloseHandle(h)
  try:
    os.kill(int(pid), 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  return True


def effective_state(st, now=None, alive=pid_alive):
  """Estado a mostrar: 'en curso' pasa a 'colgada' si el pid murió o el archivo está viejo."""
  now = time.time() if now is None else now
  estado = st.get("estado", "?")
  if estado == "en curso" and (not alive(st.get("pid")) or now - st.get("hora", 0) > STALE_S):
    return "colgada"
  return estado


def _dur(s):
  if s is None:
    return "-"
  s = int(s)
  h, r = divmod(s, 3600)
  return f"{h}h{r // 60:02d}m" if h else f"{r // 60}m{r % 60:02d}s"


def collect(ckpt_dir="checkpoints"):
  """[(run, status dict | None)] ordenado por nombre; sin las carpetas _descartados."""
  root = Path(ckpt_dir)
  out = []
  if not root.exists():
    return out
  for d in sorted(root.iterdir()):
    if not d.is_dir() or d.name.startswith("_"):
      continue
    f = d / "status.json"
    st = None
    if f.exists():
      try:
        st = json.loads(f.read_text())
      except (OSError, json.JSONDecodeError):
        st = None
    if st is None and not any((d / n).exists() for n in ("last.pt", "final.pt", "log.jsonl")):
      continue  # carpeta que no es una corrida
    out.append((d.name, st))
  return out


def rows(ckpt_dir="checkpoints", now=None, alive=pid_alive):
  head = ("run", "estado", "%", "paso/total", "loss", "tramo", "ETA")
  out = [head]
  for run, st in collect(ckpt_dir):
    if st is None:
      done = (Path(ckpt_dir) / run / "final.pt").exists()
      out.append((run, "lista" if done else "?", "-", "-", "-", "-", "-"))
      continue
    estado = effective_state(st, now, alive)
    eta = _dur(st.get("eta_s")) + (f" ({st['eta_hora'][11:16]})" if st.get("eta_hora") else "") if estado == "en curso" else "-"
    loss = st.get("loss")
    out.append((run, estado, f"{st.get('porcentaje', 0):.1f}", f"{st.get('paso', 0)}/{st.get('total', 0)}",
                f"{loss:.4f}" if isinstance(loss, (int, float)) else "-", st.get("tramo", "-"), eta))
  return out


def render(ckpt_dir="checkpoints", now=None, alive=pid_alive):
  r = rows(ckpt_dir, now, alive)
  w = [max(len(str(x[i])) for x in r) for i in range(len(r[0]))]
  return "\n".join("  ".join(str(c).ljust(w[i]) for i, c in enumerate(x)).rstrip() for x in r)


def hora_txt(ts):
  return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def main(argv=None):
  p = argparse.ArgumentParser(prog="python -m src.status", description=__doc__.split("\n")[0])
  p.add_argument("--watch", nargs="?", const=10.0, type=float, metavar="seg")
  p.add_argument("--ckpt-dir", default="checkpoints")
  a = p.parse_args(argv)
  if a.watch is None:
    print(render(a.ckpt_dir))
    return
  try:
    while True:
      print("\033[2J\033[H" + time.strftime("%H:%M:%S") + "\n" + render(a.ckpt_dir), flush=True)
      time.sleep(a.watch)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()
