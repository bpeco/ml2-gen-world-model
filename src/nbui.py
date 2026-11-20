"""Piezas visuales para los notebooks: entrenamiento en vivo, paneles de la cola y la bitácora, tablas modelo -> performance,
opciones de una acción con su comando de CLI, y video embebido. Solo IPython.display y matplotlib.

Regla de todas las funciones: si falta lo que muestran (sin JSON, sin log, sin checkpoint) muestran "pendiente" y no fallan.
Las corridas salen siempre por `Toolkit` (src/toolkit.py), así que quedan en results/bitacora.jsonl con su línea de CLI;
`lanzar` las pone en un hilo y el kernel no se bloquea (el trabajo en sí es un subproceso, ver src/ejecutor.py).

  from src import nbui
  nbui.opciones("genie", "entrenar", tamano="s")             # opciones con sus defaults y el comando equivalente
  t = nbui.lanzar("genie", "entrenar", dry_run=False, tamano="s")   # en segundo plano
  nbui.en_vivo("genie_s")                                     # barra + curvas hasta que termine
  nbui.panel_cola(); nbui.panel_bitacora()
  nbui.tabla_video(); nbui.tabla_latente(); nbui.estado_modelos()
"""
import base64
import html
import io
import json
import re
import shlex
import threading
import time
from datetime import datetime
from pathlib import Path

from src import rutas

PENDIENTE = "pendiente"
ESTADOS_RUN = ("en curso", "lista", "fallada", "colgada")


# ---------------------------------------------------------------- display

def _en_notebook():
  """¿Hay un kernel de IPython? Fuera de uno (tests, scripts) todo se imprime como texto."""
  try:
    from IPython import get_ipython
    return get_ipython() is not None
  except ImportError:
    return False


def _display(obj):
  if _en_notebook():
    from IPython.display import display
    display(obj)
  else:
    print(f"<{type(obj).__name__}>")


def mostrar_html(texto):
  """Muestra HTML en un notebook; fuera de uno imprime el texto sin etiquetas."""
  if _en_notebook():
    from IPython.display import HTML, display
    display(HTML(texto))
  else:
    print(re.sub(r"<[^>]+>", " ", texto).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").strip())


def mostrar_md(texto):
  if _en_notebook():
    from IPython.display import Markdown, display
    display(Markdown(texto))
  else:
    print(texto)


def pendiente(que="", detalle=""):
  """Celda 'pendiente': lo que se pidió todavía no existe. Devuelve el texto (y lo muestra)."""
  txt = f"{PENDIENTE}: {que}" + (f" ({detalle})" if detalle else "") if que else PENDIENTE
  mostrar_html(f"<div style='opacity:.75;font-style:italic;padding:2px 0'>{html.escape(txt)}</div>")
  return txt


def _fmt(x, d=3, coma=True):
  if x is None or (isinstance(x, float) and x != x):
    return "–"
  s = f"{x:.{d}f}"
  return s.replace(".", ",") if coma else s


def _ic(v, lo=None, hi=None, d=0):
  if v is None:
    return "–"
  return _fmt(v, d) + (f" [{_fmt(lo, d)}, {_fmt(hi, d)}]" if lo is not None and hi is not None else "")


def tabla_html(cols, filas, texto_hasta=1):
  """Tabla HTML sin dependencias (hereda los colores del tema). Las primeras `texto_hasta` columnas a la izquierda."""
  def celda(tag, i, v):
    al = "left" if i < texto_hasta else "right"
    return f"<{tag} style='text-align:{al};padding:3px 10px;border-bottom:1px solid rgba(128,128,128,.35)'>{html.escape(str(v))}</{tag}>"
  h = "<tr>" + "".join(celda("th", i, c) for i, c in enumerate(cols)) + "</tr>"
  b = "".join("<tr>" + "".join(celda("td", i, c) for i, c in enumerate(f)) + "</tr>" for f in filas)
  return f"<table style='border-collapse:collapse;font-size:13px'>{h}{b}</table>"


def mostrar_tabla(cols, filas, vacio="sin filas", texto_hasta=1):
  if not filas:
    return pendiente(vacio)
  mostrar_html(tabla_html(cols, filas, texto_hasta))
  return filas


def leer_json(ruta):
  """El JSON de `ruta`, o None si no existe o está cortado."""
  try:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))
  except (OSError, ValueError):
    return None


def _get(d, *keys):
  for k in keys:
    try:
      d = d[k]
    except (KeyError, IndexError, TypeError):
      return None
  return d


# ---------------------------------------------------------------- entrenamiento en vivo

def leer_status(run, ckpt=None):
  """checkpoints/<run>/status.json (lo escribe train.py en cada log) o None."""
  return leer_json(Path(ckpt or rutas.ckpt()) / run / "status.json")


def leer_log(run, ckpt=None):
  """Registros de checkpoints/<run>/log.jsonl; las líneas cortadas o ilegibles se saltan. [] si no existe."""
  f = Path(ckpt or rutas.ckpt()) / run / "log.jsonl"
  out = []
  try:
    for linea in f.read_text(encoding="utf-8").splitlines():
      try:
        out.append(json.loads(linea))
      except ValueError:
        pass
  except OSError:
    pass
  return out


def estado_efectivo(st):
  """'en curso', 'lista', 'fallada' o 'colgada' (en curso sin proceso o sin actualizar hace más de 3 min)."""
  if not st:
    return None
  try:
    from src.status import effective_state
    return effective_state(st)
  except Exception:  # noqa: BLE001
    return st.get("estado")


def _dur(s):
  if s is None:
    return "–"
  s = int(s)
  h, r = divmod(s, 3600)
  return f"{h} h {r // 60:02d} min" if h else f"{r // 60} min {r % 60:02d} s"


def barra_html(st, ancho=420):
  """Barra de porcentaje con paso, loss, lr, tramo y ETA a partir de un status.json."""
  pct = max(0.0, min(100.0, float(st.get("porcentaje") or 0)))
  estado = estado_efectivo(st) or "?"
  color = {"en curso": "#2a78d6", "lista": "#1baf7a", "fallada": "#d03030", "colgada": "#c08a00"}.get(estado, "#888")
  loss, lr = st.get("loss"), st.get("lr")
  partes = [f"{estado}", f"paso {st.get('paso', 0)}/{st.get('total', 0)}"]
  if isinstance(loss, (int, float)):
    partes.append(f"loss {loss:.4g}")
  if isinstance(lr, (int, float)):
    partes.append(f"lr {lr:.3g}")
  if st.get("tramo"):
    partes.append(f"tramo {st['tramo']}")
  if estado == "en curso":
    partes.append(f"ETA {_dur(st.get('eta_s'))}" + (f" ({st['eta_hora'][11:16]})" if st.get("eta_hora") else ""))
    if st.get("frames_s"):
      partes.append(f"{st['frames_s']:.0f} frames/s")
  if st.get("error"):
    partes.append(f"error: {st['error'][:120]}")
  return (f"<div style='font-size:13px'><b>{html.escape(str(st.get('run', '')))}</b> "
          f"<div style='width:{ancho}px;height:16px;border:1px solid rgba(128,128,128,.6);border-radius:4px;overflow:hidden;display:inline-block;vertical-align:middle'>"
          f"<div style='width:{pct:.1f}%;height:100%;background:{color}'></div></div> <b>{pct:.1f} %</b><br>"
          f"<span style='opacity:.8'>{html.escape(' · '.join(partes))}</span></div>")


def mostrar_barra(run, ckpt=None):
  """Barra de progreso de una corrida; 'pendiente' si todavía no tiene status.json."""
  st = leer_status(run, ckpt)
  if st is None:
    return pendiente(f"la corrida {run} no empezó", f"no hay {Path(ckpt or rutas.ckpt()) / run / 'status.json'}")
  st = {"run": run, **st}
  mostrar_html(barra_html(st))
  return st


def _receta_de(run, st, resultados=None):
  """Recipe de la corrida: la del train_<run>.json si terminó; si no, la del modelo (por el kind del status y el tamaño del nombre)."""
  from src import recipes as R
  d = leer_json(Path(resultados or rutas.results()) / f"train_{run}.json")
  if d and d.get("receta"):
    return R.Recipe.from_dict(d["receta"])
  kind = (st or {}).get("kind")
  if not kind:
    return None
  try:
    from src.models import model_class
    m = re.search(r"_(s|m|l)(?:_|$)", run)
    return model_class(kind).recipe(m.group(1) if m else None)
  except Exception:  # noqa: BLE001
    return None


def tramos(run, registros=None, st=None, ckpt=None, resultados=None):
  """[(nombre, paso_ini, paso_fin)] del schedule de lr de la corrida, para sombrearlos. [] si no se puede saber."""
  registros = leer_log(run, ckpt) if registros is None else registros
  st = leer_status(run, ckpt) if st is None else st
  pasos = sorted({r["step"] for r in registros if "step" in r and "lr" in r})
  total = (st or {}).get("total") or (max(pasos) if pasos else 0)
  r = _receta_de(run, st, resultados)
  if r is None or not total:
    return []
  out = []
  try:
    for s in list(range(0, total, max(1, total // 400))) + [total - 1]:
      n = r.tramo(s, total)
      if out and out[-1][0] == n:
        out[-1][2] = s + 1
      else:
        out.append([n, s, s + 1])
  except Exception:  # noqa: BLE001
    return []
  return [tuple(t) for t in out]


def curvas(run, ckpt=None, resultados=None, figsize=(11, 3.2), mostrar=True):
  """Figura con loss, lr y monitoreo (monitor_ce) de la corrida, con el tramo del schedule sombreado.
  Devuelve la figura, o None (y muestra 'pendiente') si todavía no hay log."""
  import matplotlib
  import matplotlib.pyplot as plt
  reg = leer_log(run, ckpt)
  tr = [r for r in reg if "lr" in r and "step" in r]
  mon = [r for r in reg if "monitor_ce" in r]
  if not reg:
    pendiente(f"curvas de {run}", "todavía no hay log.jsonl")
    return None
  st = leer_status(run, ckpt)
  tb = tramos(run, reg, st, ckpt, resultados)
  fig, axs = plt.subplots(1, 3, figsize=figsize)
  sombras = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]

  def sombrear(ax):
    for i, (n, a, b) in enumerate(tb):
      ax.axvspan(a, b, color=sombras[i % len(sombras)], alpha=0.10, lw=0)

  claves = [k for k in (tr[0] if tr else {}) if k not in ("step", "tokens", "lr", "tok_s", "elapsed")]
  for k in claves[:4]:
    axs[0].plot([r["step"] for r in tr if k in r], [r[k] for r in tr if k in r], label=k, lw=1.5)
  axs[0].set_title("loss")
  if len(claves) > 1:
    axs[0].legend(fontsize=7)
  axs[1].plot([r["step"] for r in tr], [r["lr"] for r in tr], color="#2a78d6", lw=1.5)
  axs[1].set_title("lr")
  if mon:
    axs[2].plot([r["step"] for r in mon], [r["monitor_ce"] for r in mon], "o-", color="#eb6834", lw=1.5, ms=4)
  else:
    axs[2].text(0.5, 0.5, "pendiente", ha="center", va="center", transform=axs[2].transAxes, alpha=.6)
  axs[2].set_title("monitoreo (CE en el holdout)")
  for ax in axs:
    sombrear(ax)
    ax.set_xlabel("paso")
    ax.grid(alpha=.3)
  if tb:
    axs[1].set_title("lr  (" + " | ".join(n for n, _, _ in tb) + ")")
  fig.tight_layout()
  if mostrar:
    _display(fig)
    plt.close(fig)
  return fig


def en_vivo(run, cada=3.0, max_seg=None, curvas_cada=4, ckpt=None):
  """Barra y curvas de una corrida, actualizadas cada `cada` s hasta que queda lista, fallada o colgada (o pasan `max_seg` s).
  Sin status.json muestra 'pendiente' y vuelve en el acto. Se puede interrumpir el kernel sin afectar al entrenamiento."""
  t0, n = time.time(), 0
  while True:
    st = leer_status(run, ckpt)
    if _en_notebook():
      from IPython.display import clear_output
      clear_output(wait=True)
    if st is None:
      return pendiente(f"la corrida {run} no empezó", "lanzala con nbui.lanzar(...) o python -m src.cli <modelo> entrenar")
    st = {"run": run, **st}
    mostrar_html(barra_html(st))
    if n % curvas_cada == 0 or estado_efectivo(st) != "en curso":
      curvas(run, ckpt)
    n += 1
    if estado_efectivo(st) != "en curso" or (max_seg is not None and time.time() - t0 > max_seg):
      return st
    time.sleep(cada)


# ---------------------------------------------------------------- cola y bitácora

def panel_cola():
  """Candado de GPU y trabajos de results/queue.json en una tabla. 'GPU libre' y 'cola vacía' si no hay nada."""
  from src import cola
  cand = cola.leer_candado()
  if cand:
    from src.status import pid_alive
    vivo = pid_alive(cand.get("pid")) if cand.get("pid") else False
    mostrar_html(f"<div><b>GPU ocupada</b> por {html.escape(str(cand.get('run')))} / {html.escape(str(cand.get('accion')))} "
                 f"(pid {cand.get('pid')}, desde {html.escape(str(cand.get('hora')))})" + ("" if vivo else " <i>[candado huérfano]</i>") + "</div>")
  else:
    mostrar_html("<div><b>GPU libre</b></div>")
  items = cola.cargar_cola()
  filas = [(i["id"], i.get("estado", "?"), (i.get("encolada") or "")[:19], "python -m src.cli " + " ".join(i["argv"]),
            i.get("codigo", "")) for i in items]
  mostrar_tabla(("id", "estado", "encolada", "comando", "código"), filas, vacio="la cola está vacía", texto_hasta=4)
  return {"candado": cand, "cola": items}


def panel_bitacora(n=15):
  """Las últimas `n` acciones de results/bitacora.jsonl: fecha, resultado, duración y la línea de CLI exacta."""
  from src import bitacora
  try:
    es = bitacora.leer()[-n:]
  except Exception:  # noqa: BLE001
    es = []
  filas = []
  for e in reversed(es):
    est = "encolada" if e.get("encolada") else ("rechazada" if e.get("rechazada") else f"salida {e.get('salida')}")
    filas.append((e.get("id", "?"), (e.get("fecha") or "")[:19], est, f"{e.get('duracion_s') or 0:.0f} s",
                  e.get("argv_texto", ""), "" if e.get("limpio") else "árbol sucio"))
  mostrar_tabla(("id", "fecha", "resultado", "duración", "comando", "nota"), filas, vacio="la bitácora está vacía", texto_hasta=6)
  return es


def estado_corridas(ckpt=None):
  """Una fila por corrida de checkpoints/ (como `python -m src.status`)."""
  from src import status
  filas = status.rows(str(ckpt or rutas.ckpt()))[1:]
  mostrar_tabla(status.rows(str(ckpt or rutas.ckpt()))[0], filas, vacio="no hay corridas en checkpoints/", texto_hasta=2)
  return filas


# ---------------------------------------------------------------- lanzar una acción

class Trabajo:
  """Una acción de Toolkit en un hilo. `.mostrar()` imprime lo que lleva; `.esperar()` bloquea hasta que termina."""

  def __init__(self, modelo, accion, opts):
    self.modelo, self.accion, self.opts = modelo, accion, opts
    self.lineas, self.resultado, self.error = [], None, None
    self.hilo = None

  def salida(self, *a):
    self.lineas.append(" ".join(str(x) for x in a))

  @property
  def terminado(self):
    return self.hilo is None or not self.hilo.is_alive()

  @property
  def estado(self):
    if self.error:
      return "error"
    if not self.terminado:
      return "corriendo"
    return getattr(self.resultado, "estado", "?") if self.resultado else "?"

  @property
  def ok(self):
    return self.terminado and not self.error and bool(self.resultado) and self.resultado.codigo == 0

  def esperar(self, max_seg=None):
    if self.hilo is not None:
      self.hilo.join(max_seg)
    return self

  def mostrar(self, ultimas=40):
    print("\n".join(self.lineas[-ultimas:]) or "(sin salida todavía)")
    if self.error:
      print("error:", self.error)
    print(f"[{self.modelo} {self.accion}: {self.estado}]")
    return self


def comando_cli(modelo, accion, **opts):
  """La línea de `python -m src.cli` equivalente a estas opciones."""
  from src import ejecutor
  from src.toolkit import CONTROL, argv_canonico
  o = {k: v for k, v in opts.items() if k not in CONTROL or k == "permitir_sucio"}
  return ejecutor.argv_texto(argv_canonico(modelo, accion, o)).replace("'--tamaño'", "--tamaño")


def lanzar(modelo, accion, dry_run=True, segundo_plano=True, extra=(), mostrar_salida=True, **opts):
  """Corre `Toolkit(modelo).accion(accion, ...)`. Con dry_run=True (el default de los notebooks) solo imprime los comandos.
  Con dry_run=False lanza el trabajo en un hilo (el subproceso no bloquea el kernel); la acción queda en la bitácora.
  Devuelve un Trabajo. Nunca lanza una excepción por un pedido inválido: queda en `.error` y se muestra."""
  from src.toolkit import Toolkit
  t = Trabajo(modelo, accion, opts)
  t.salida("$ " + comando_cli(modelo, accion, dry_run=dry_run, **opts) if not dry_run else "# " + comando_cli(modelo, accion, **opts))

  def correr():
    try:
      t.resultado = Toolkit(modelo).accion(accion, extra=tuple(extra), dry_run=dry_run, salida=t.salida, **opts)
    except Exception as e:  # noqa: BLE001  (ToolkitError, un --set inválido...)
      t.error = f"{type(e).__name__}: {e}"

  if dry_run or not segundo_plano:
    correr()
    if mostrar_salida:
      t.mostrar()
  else:
    t.hilo = threading.Thread(target=correr, daemon=True, name=f"nbui-{modelo}-{accion}")
    t.hilo.start()
    print(f"lanzada en segundo plano: {comando_cli(modelo, accion, **opts)}\n"
          "seguila con nbui.en_vivo(<corrida>), nbui.panel_cola() o t.mostrar()")
  return t


# ---------------------------------------------------------------- opciones de una acción

def _defaults_receta(modelo, tamano=None):
  """{opción: valor por defecto} que salen de la receta del modelo."""
  try:
    from src.toolkit import _clase, _es_sized
    cls = _clase(modelo)
    r = cls.recipe((tamano or "s") if _es_sized(modelo) else None)
    return r, {"presupuesto": f"{r.presupuesto:,} frames".replace(",", ".")}
  except Exception:  # noqa: BLE001
    return None, {}


def opciones(modelo, accion, mostrar=True, **valores):
  """Opciones de una acción con sus defaults (los de la receta del modelo) y el comando de CLI equivalente a `valores`.
  Devuelve el comando."""
  from src.toolkit import CONTROL, OPCIONES, Toolkit, ToolkitError
  aviso = None
  try:
    tk = Toolkit(modelo)
    if accion not in tk.acciones and accion != "estado":
      return pendiente(f"{modelo} {accion}", f"{modelo} no tiene esa acción; declara: {', '.join(tk.acciones)}")
    tk.plan(accion, **{k: v for k, v in valores.items() if k not in CONTROL})
  except ToolkitError as e:
    aviso = str(e)
  except Exception as e:  # noqa: BLE001
    aviso = f"{type(e).__name__}: {e}"
  r, defs = _defaults_receta(modelo, valores.get("tamano"))
  base = {"run": "<modelo>_<tamaño>", "tamano": "s", "semilla": "0", "device": "el que elige el repo (cuda si hay)",
          "dtype": "fp16 en CUDA, fp32 en CPU", "temperatura": "1.0 (o la elegida en el holdout)", "ctx_frames": "20 (1 s)"}
  filas = []
  for o in OPCIONES:
    if o.acciones and accion not in o.acciones:
      continue
    v = valores.get(o.nombre)
    filas.append((o.flags[0], "" if v in (None, False, []) else str(v), str(defs.get(o.nombre, base.get(o.nombre, "–"))), o.ayuda))
  cmd = comando_cli(modelo, accion, **valores)
  if mostrar:
    mostrar_html(f"<div style='font-size:13px'><b>{html.escape(modelo)} {html.escape(accion)}</b></div>")
    mostrar_tabla(("opción", "valor pedido", "default", "qué hace"), filas, texto_hasta=4)
    if r is not None and accion in ("receta", "entrenar", "continuar"):
      mostrar_html(f"<div style='font-size:12px;opacity:.85'>receta: {html.escape(r.resumen())}</div>")
    mostrar_html(f"<pre style='font-size:13px'>{html.escape(cmd)}</pre>")
    if aviso:
      mostrar_html(f"<div style='font-size:12px;color:#c08a00'>aviso: {html.escape(aviso)}</div>")
  return cmd


def receta(modelo, tamano=None, mostrar=True):
  """Receta del modelo (la de su fuente), con los valores forzados y su categoría. Devuelve la Recipe o None."""
  r, _ = _defaults_receta(modelo, tamano)
  if r is None:
    return pendiente(f"receta de {modelo}", "el modelo no tiene Recipe (baselines, gpt2m, codec)") and None
  if mostrar:
    d = r.to_dict()
    filas = [(k, json.dumps(d[k], ensure_ascii=False) if not isinstance(d[k], str) else d[k])
             for k in ("name", "fuente", "optimizer", "betas", "lr_segs", "wd_ini", "wd_fin", "wd_on", "ema_segs", "grad_clip", "amp",
                       "batch", "frames", "frameskip", "accum", "presupuesto", "batch_fuente", "extras") if k in d and d[k] is not None]
    mostrar_tabla(("campo", "valor"), filas, texto_hasta=1)
    if r.forzado:
      mostrar_tabla(("forzado", "motivo", "categoría"), [tuple(f) for f in r.forzado], texto_hasta=3)
  return r


def fidelidad(modelo, mostrar=True):
  """Tabla de fidelidad (decisión, fuente, aquí, categoría) del modelo. [] y 'pendiente' si el módulo no la declara."""
  try:
    from src import recipes as R
    nombre = modelo
    lista = R.fidelidad(nombre)
  except Exception:  # noqa: BLE001
    lista = []
  filas = [(f.get("campo", ""), f.get("fuente_valor", f.get("paper", "")), f.get("aquí", ""), f.get("categoría", ""), f.get("fuente", ""))
           for f in lista]
  if mostrar:
    mostrar_tabla(("decisión", "fuente / paper", "aquí", "categoría", "fuente"), filas, vacio=f"fidelidad de {modelo}", texto_hasta=5)
  return filas


# ---------------------------------------------------------------- resultados: modelo -> performance

SUF = re.compile(r"(_p\d+)?(_ctx\d+)?(_t[\d.]+)?(_fp(?:16|32))?(_stream)?$")
BASELINES = ("unigram", "transicion", "congelado", "real")
# corridas que la cola produce (plan, decisiones del 17-18/11): lo que el notebook de resultados espera ver
ESPERADOS = ("unigram", "transicion", "congelado", "gpt2m", "real", "gpt_s_kd", "genie_s", "convlstm_px", "lewm", "vjepa2_s", "vjepa2_ac")
LATENTES = ("lewm", "vjepa2_s", "vjepa2_ac")
FUERA_DE_COLA = ("gpt_s", "gpt_m", "genie_m", "maskgit_s")
FAMILIAS = {"gpt": "GPT", "gpt2m": "gpt2m (comma)", "maskgit": "MaskGIT propio", "genie": "Genie (MaskGIT)", "convlstm": "ConvLSTM",
            "lewm": "LeWM", "vjepa2": "V-JEPA 2", "baseline": "referencia"}


def run_de(nombre):
  return SUF.sub("", nombre)


def familia(run):
  if run in BASELINES:
    return "baseline"
  if run == "gpt2m":
    return "gpt2m"
  return run.split("_")[0] if run.split("_")[0] in FAMILIAS else "otro"


def escanear(raiz=None):
  """{nombre: {"video", "eval", "latent", "bench", "e2e"}} con las rutas (Path) de lo que existe en results/. Sin results/ da {}."""
  R = Path(raiz or rutas.results())
  out = {}

  def nodo(n):
    return out.setdefault(n, {})
  if not R.exists():
    return out
  for p in R.glob("video_*.json"):
    n = p.stem[len("video_"):]
    if not n.endswith("_stream") and not re.search(r"_seg\d+$", n):
      nodo(n)["video"] = p
  for p in R.glob("eval_*.json"):
    n = p.stem[len("eval_"):]
    if not re.search(r"_seg\d+$", n):
      nodo(n)["eval"] = p
  for p in R.glob("latent_*.json"):
    nodo(p.stem[len("latent_"):])["latent"] = p
  from src import configs   # los resultados de la RTX (dispositivo de referencia) mandan sobre los de otro dispositivo
  benchs, e2es = {}, {}
  for p in sorted(R.glob("bench/*__*.json")):
    n = p.name.split("__")[0]
    nodo(n)
    if "cpu" not in p.name and "int8" not in p.name:
      benchs.setdefault(n, []).append(p)
  for n, fs in benchs.items():
    nodo(n)["bench"] = min(configs.preferir_ref(fs), key=lambda p: (leer_json(p) or {}).get("median_ms") or 1e18)
  for pat in ("infer/latencia_*.json", "infer/e2e_*.json"):
    for p in sorted(R.glob(pat)):
      e2es.setdefault(p.name.split("__")[0].split("_", 1)[1], []).append(p)
  for n, fs in e2es.items():
    nodo(n)["e2e"] = configs.preferir_ref(fs)[0]
  return out


def _estado_de(path, registro):
  from src import resultados
  if path is None:
    return PENDIENTE
  try:
    return resultados.estado(path, registro) if registro else "vigente"
  except Exception:  # noqa: BLE001
    return "vigente"


def _registro(raiz):
  from src import resultados
  return resultados.cargar(Path(raiz or rutas.results()) / "estado.json")


def _fvd_de(vd, h):
  return _get(vd, "fvd", h, "value"), _get(vd, "fvd", h, "lo"), _get(vd, "fvd", h, "hi")


def _elegido(n, solo):
  return not solo or run_de(n) in solo or n in solo


def filas_video(raiz=None, solo_vigentes=False, solo=None):
  """Una fila (dict) por modelo con resultados de video, ya leídas de los JSON actuales. Más el piso de FVD de real contra real."""
  R = Path(raiz or rutas.results())
  reg = _registro(R)
  piso = leer_json(R / "piso_fvd.json")
  filas = []
  for n, f in sorted(escanear(R).items()):
    if ("video" not in f and "eval" not in f) or not _elegido(n, solo):
      continue
    vd = leer_json(f["video"]) if "video" in f else None
    ev = leer_json(f["eval"]) if "eval" in f else None
    if ev is None and run_de(n) != n:   # la CE es de la corrida, no de la variante de evaluación
      ev = leer_json(R / f"eval_{run_de(n)}.json")
    bn = leer_json(f["bench"]) if "bench" in f else None
    if bn is None:
      a = leer_json(R / "bench" / f"{run_de(n)}__fp16_graphs.json")
      bn = a
    e2e = leer_json(f["e2e"]) if "e2e" in f else None
    st = _estado_de(f.get("video") or f.get("eval"), reg)
    if solo_vigentes and st != "vigente":
      continue
    f1, l1, h1 = _fvd_de(vd, "1")
    filas.append({"modelo": n, "familia": familia(run_de(n)), "estado": st, "ce": _get(ev, "ce_all"), "ce_ultimo": _get(ev, "ce_last"),
                  "lpips1": _get(vd, "lpips", "at", "1", "mean"), "fvd1": f1, "fvd1_lo": l1, "fvd1_hi": h1,
                  "fvd3": _fvd_de(vd, "3")[0], "piso1": _get(vd, "piso", "1", "value") if _get(vd, "piso", "1", "value") is not None
                  else _get(piso, "piso", "1", "value") if piso else None,
                  "ms": _get(bn, "median_ms"), "p99": _get(bn, "p99_ms"), "params": _get(bn, "params"),
                  "e2e_p50": _get(e2e, "cadena_ritmo", "p50") or _get(e2e, "steady", "p50"),
                  "e2e_p99": _get(e2e, "cadena_ritmo", "p99") or _get(e2e, "steady", "p99"),
                  "lpips_curva": _get(vd, "lpips", "curve"), "fvd_n": _get(vd, "n_clips")})
  return filas


def tabla_video(raiz=None, solo=None):
  """Tabla modelo -> performance de video: CE, LPIPS y FVD a 1 s (con IC por segmento) y a 3 s, piso de FVD y latencia.
  Una fila por modelo, solo con lo que existe; 'pendiente' si no hay nada."""
  fs = filas_video(raiz, solo=solo)
  pisos = [f["piso1"] for f in fs if f["piso1"] is not None]
  filas = [(f["modelo"], f["estado"], _fmt(f["ce"]), _fmt(f["lpips1"]), _ic(f["fvd1"], f["fvd1_lo"], f["fvd1_hi"]), _fmt(f["fvd3"], 0),
            _fmt(f["ms"], 1), _fmt(f["p99"], 1), _fmt(f["e2e_p99"], 1)) for f in fs]
  r = mostrar_tabla(("modelo", "estado", "CE", "LPIPS 1 s", "FVD 1 s [IC por segmento]", "FVD 3 s", "ms p50", "ms p99", "cadena p99"),
                    filas, vacio="no hay resultados de video todavía (python -m src.cli <modelo> evaluar)", texto_hasta=2)
  if filas:
    mostrar_html(f"<div style='font-size:12px;opacity:.8'>piso de FVD (real contra real): "
                 f"{_fmt(pisos[0], 0) if pisos else 'pendiente (python -m src.cli real evaluar)'}</div>")
  return r


def _pareto(xs, ys):
  n = len(xs)
  ok = [xs[i] is not None and ys[i] is not None for i in range(n)]
  return [ok[i] and not any(ok[j] and xs[j] <= xs[i] and ys[j] <= ys[i] and (xs[j] < xs[i] or ys[j] < ys[i]) for j in range(n))
          for i in range(n)]


def pareto(raiz=None, x="ms", figsize=(6.5, 4.2), mostrar=True):
  """Calidad (FVD a 1 s) contra latencia por frame, con la frontera de Pareto entre los modelos generativos, y el piso de FVD.
  Solo datos actuales (sin 'protocolo anterior'). 'pendiente' si no hay ningún modelo con las dos cosas."""
  import matplotlib.pyplot as plt
  fs = [f for f in filas_video(raiz) if f["estado"] != "protocolo anterior" and f["familia"] not in ("baseline",)
        and f["fvd1"] is not None and f[x] is not None]
  if not fs:
    pendiente("Pareto calidad-latencia", "hace falta un modelo con FVD y con latencia")
    return None
  ms = _pareto([f[x] for f in fs], [f["fvd1"] for f in fs])
  fig, ax = plt.subplots(figsize=figsize)
  for f, p in zip(fs, ms):
    ax.errorbar(f[x], f["fvd1"], yerr=None if f["fvd1_lo"] is None else [[f["fvd1"] - f["fvd1_lo"]], [f["fvd1_hi"] - f["fvd1"]]],
                fmt="o" if p else "s", mfc="none" if not p else None, capsize=2)
    ax.annotate(f["modelo"], (f[x], f["fvd1"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
  pisos = [f["piso1"] for f in filas_video(raiz) if f["piso1"] is not None]
  if pisos:
    ax.axhline(pisos[0], ls="--", color="gray", lw=1)
    ax.annotate("piso de FVD (real contra real)", (ax.get_xlim()[0], pisos[0]), textcoords="offset points", xytext=(4, 4), fontsize=8, color="gray")
  ax.axvline(50, ls=":", color="gray", lw=1)
  ax.set_xscale("log")
  ax.set_xlabel("ms por frame (p50)" if x == "ms" else "ms por frame (cadena, p50)")
  ax.set_ylabel("FVD a 1 s (menos es mejor)")
  ax.grid(alpha=.3)
  fig.tight_layout()
  if mostrar:
    _display(fig)
    plt.close(fig)
  return fig


def horizonte(raiz=None, figsize=(7, 3.6), mostrar=True, solo=None):
  """LPIPS por frame generado (curva por horizonte) de cada modelo con resultados de video."""
  import matplotlib.pyplot as plt
  fs = [f for f in filas_video(raiz, solo=solo) if f["lpips_curva"] and f["estado"] != "protocolo anterior"]
  if not fs:
    pendiente("curvas de LPIPS por horizonte")
    return None
  fig, ax = plt.subplots(figsize=figsize)
  for f in fs:
    c = f["lpips_curva"]
    ax.plot([(i + 1) / 20 for i in range(len(c))], c, label=f["modelo"], lw=1.6)
  ax.set_xlabel("segundos generados")
  ax.set_ylabel("LPIPS contra el video real")
  ax.legend(fontsize=7)
  ax.grid(alpha=.3)
  fig.tight_layout()
  if mostrar:
    _display(fig)
    plt.close(fig)
  return fig


def filas_latente(raiz=None, solo=None):
  """Una fila por modelo latente: error contra copiar el último latente, probes contra encoder sin entrenar, sorpresa, CEM."""
  R = Path(raiz or rutas.results())
  reg = _registro(R)
  out = []
  for n, f in sorted(escanear(R).items()):
    if "latent" not in f or not _elegido(n, solo):
      continue
    d = leer_json(f["latent"]) or {}
    mse = _get(d, "horizonte", "mse")
    razon = _get(mse, "razon_modelo_sobre_copiar") if mse else None
    r2 = _get(d, "probes", "obs", "modelo", "velocidad", "r2", "valor")
    r2s = _get(d, "probes", "obs", "sin_entrenar", "velocidad", "r2", "valor")
    r2p = _get(d, "probes", "obs", "pixeles", "velocidad", "r2", "valor")
    cem = d.get("cem") if isinstance(d.get("cem"), dict) else {}
    h = next((cem[k] for k in sorted(cem) if re.fullmatch(r"h\d+", k)), None) if cem else None
    out.append({"modelo": n, "estado": _estado_de(f["latent"], reg),
                "razon_ultimo": _get(razon, -1, "valor"), "razon_ic": _get(razon, -1, "ic95"),
                "r2": r2, "r2_sin_entrenar": r2s, "r2_pixeles": r2p,
                "sorpresa": _get(d, "sorpresa", "teletransporte", "modelo", "razon", "valor"),
                "sorpresa_ref": _get(d, "sorpresa", "teletransporte", "referencia_copiar_ultimo", "razon", "valor"),
                "cem_mae": _get(h, "velocidad", "mae_cem", "valor"), "cem_ref": _get(h, "velocidad", "mae_repetir_ultima", "valor"),
                "ms_plan": _get(h, "ms_por_plan"), "params": d.get("params")})
  return out


def tabla_latente(raiz=None, solo=None):
  """Tabla latente: cada columna con su referencia al lado. Los latentes no entran en la tabla de video."""
  fs = filas_latente(raiz, solo=solo)
  filas = [(f["modelo"], f["estado"], _fmt(f["razon_ultimo"], 2), _fmt(f["r2"], 2), _fmt(f["r2_sin_entrenar"], 2), _fmt(f["r2_pixeles"], 2),
            _fmt(f["sorpresa"], 2), _fmt(f["sorpresa_ref"], 2), _fmt(f["cem_mae"], 3), _fmt(f["cem_ref"], 3), _fmt(f["ms_plan"], 0)) for f in fs]
  return mostrar_tabla(("modelo", "estado", "error / copiar último (h máx.)", "probe R² velocidad", "…encoder sin entrenar", "…píxeles",
                        "sorpresa (razón)", "…ref. copiar", "CEM MAE", "…repetir última", "ms por plan"),
                       filas, vacio="no hay evaluación latente todavía (python -m src.cli lewm evaluar)", texto_hasta=2)


def estado_modelos(raiz=None, ckpt=None, esperados=ESPERADOS):
  """Qué tiene cada modelo esperado: checkpoint, evaluación, latencia. 'hecho' o 'pendiente' en cada columna."""
  R = Path(raiz or rutas.results())
  C = Path(ckpt or rutas.ckpt())
  es = escanear(R)
  filas = []
  for n in esperados:
    f = es.get(n, {})
    entrena = n not in ("unigram", "transicion", "congelado", "gpt2m", "real")
    ck = "n/a" if not entrena else ("hecho" if (C / n / "final.pt").exists() else PENDIENTE)
    ev = "hecho" if ("video" in f or "latent" in f or "eval" in f) else PENDIENTE
    ln = "hecho" if ("bench" in f or "e2e" in f) else PENDIENTE
    filas.append((n, ck, ev, ln))
  mostrar_tabla(("modelo", "entrenado", "evaluado", "latencia"), filas, texto_hasta=1)
  return filas


def tabla_fidelidad(modelos, mostrar=True):
  """Todas las tablas de fidelidad juntas: una fila por decisión de cada modelo, con su categoría."""
  filas = []
  for m in modelos:
    for f in fidelidad(m, mostrar=False):
      filas.append((m, f[0], f[1], f[2], f[3]))
  if mostrar:
    mostrar_tabla(("modelo", "decisión", "fuente / paper", "aquí", "categoría"), filas, vacio="fidelidad", texto_hasta=5)
  return filas


# ---------------------------------------------------------------- figuras y video

def figura(ruta, ancho=None):
  """Muestra un PNG de results/; 'pendiente' si no existe."""
  p = Path(ruta)
  if not p.exists():
    return pendiente(p.name)
  if _en_notebook():
    from IPython.display import Image
    _display(Image(filename=str(p), width=ancho))
  else:
    print(p)
  return p


def _gif_de_mp4(ruta, cada=2, max_frames=200, ancho=None):
  """mp4 -> bytes de un GIF animado (el mp4v de OpenCV no se reproduce en todos los navegadores)."""
  import cv2
  from PIL import Image
  cap = cv2.VideoCapture(str(ruta))
  fps = cap.get(cv2.CAP_PROP_FPS) or 20
  imgs, i = [], 0
  while len(imgs) < max_frames:
    ok, f = cap.read()
    if not ok:
      break
    if i % cada == 0:
      f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
      if ancho and f.shape[1] > ancho:
        f = cv2.resize(f, (ancho, int(f.shape[0] * ancho / f.shape[1])), interpolation=cv2.INTER_AREA)
      imgs.append(Image.fromarray(f))
    i += 1
  cap.release()
  if not imgs:
    return None
  b = io.BytesIO()
  imgs[0].save(b, "GIF", save_all=True, append_images=imgs[1:], duration=int(1000 * cada / fps), loop=0)
  return b.getvalue()


def video_html(ruta, titulo="", ancho=None, cada=2):
  """HTML con el video embebido (GIF animado en base64, a la velocidad real). '' si no se puede leer."""
  p = Path(ruta)
  if not p.exists():
    return ""
  try:
    datos = _gif_de_mp4(p, cada=cada, ancho=ancho)
  except Exception:  # noqa: BLE001
    datos = None
  if not datos:
    return ""
  w = f" width='{ancho}'" if ancho else ""
  return (f"<figure style='margin:4px;display:inline-block'><img src='data:image/gif;base64,{base64.b64encode(datos).decode()}'{w}>"
          f"<figcaption style='font-size:12px'>{html.escape(titulo or p.name)}</figcaption></figure>")


def video(ruta, titulo="", ancho=None, cada=2):
  """Muestra un mp4 embebido en el notebook (lo que escribe `demo`). 'pendiente' si falta."""
  h = video_html(ruta, titulo, ancho, cada)
  if not h:
    return pendiente(Path(ruta).name, "el video no existe todavía (correlo con la acción demo)")
  mostrar_html(h)
  return Path(ruta)


def lado_a_lado(items, ancho=420):
  """items: [(titulo, mp4, ms_por_frame | None)]. Los muestra en fila, con la latencia por frame de cada uno en el título."""
  partes = []
  for t, p, ms in items:
    h = video_html(p, t + (f" · {ms:.1f} ms por frame" if ms else ""), ancho)
    partes.append(h or f"<div style='opacity:.7;margin:4px'>{html.escape(t)}: pendiente</div>")
  mostrar_html("<div style='display:flex;flex-wrap:wrap'>" + "".join(partes) + "</div>")
  return items


def info_demo(mp4):
  """El .json que escribe `demo` junto al mp4 (latencia por frame generado, fuente, modo) o None."""
  return leer_json(Path(mp4).with_suffix(".json"))


# ---------------------------------------------------------------- modelos, backends y datos de la clase

def corridas_disponibles(ckpt=None, manifiesto=None):
  """{corrida: 'entrenada' | 'sin entrenar todavía'}: las del manifiesto results/checkpoints.json y las de checkpoints/<run>/final.pt."""
  C = Path(ckpt or rutas.ckpt())
  man = leer_json(manifiesto or (rutas.results() / "checkpoints.json")) or {}
  out = {k: PENDIENTE for k in man if not k.startswith("_")}
  if C.exists():
    for d in sorted(C.iterdir()):
      if d.is_dir() and not d.name.startswith("_") and (d / "final.pt").exists():
        out[d.name] = "entrenada"
  for k in list(out):
    if (C / k / "final.pt").exists():
      out[k] = "entrenada"
    else:
      out[k] = "sin entrenar todavía"
  return out


def modelo_de_corrida(run):
  """El nombre de modelo del Toolkit al que pertenece una corrida (gpt_s_kd -> gpt, vjepa2_s -> vjepa2, lewm -> lewm) o None."""
  from src.toolkit import modelos
  ms = modelos()
  m = max((x for x in ms if run == x or run.startswith(x + "_")), key=len, default=None)
  return m


def backends(mostrar=True):
  """Backends de inferencia detectados (cuda, cpu, rocm, dml, npu...). Usa src.infer.backends; si no existe, una detección mínima."""
  try:
    from src.infer import backends as B
    filas = [(t["backend"], "sí" if t["disponible"] else "no", ",".join(t["motores"]), t["motivo"] if t["disponible"] else (t["instalar"] or t["motivo"]))
             for t in B.tabla()]
    disp = list(B.disponibles())
  except Exception:  # noqa: BLE001
    disp = ["cpu"]
    filas = [("cpu", "sí", "torch", "")]
    try:
      import torch
      if torch.cuda.is_available():
        disp.insert(0, "rocm" if torch.version.hip else "cuda")
        filas.append((disp[0], "sí", "torch", torch.cuda.get_device_name(0)))
    except Exception:  # noqa: BLE001
      pass
    try:
      import torch
      if torch.backends.mps.is_available():
        disp.append("mps")
        filas.append(("mps", "sí", "torch", "GPU de Apple Silicon"))
    except Exception:  # noqa: BLE001
      pass
    try:
      import onnxruntime as ort
      prov = ort.get_available_providers()
      for n, p in (("dml", "DmlExecutionProvider"), ("npu", "VitisAIExecutionProvider")):
        filas.append((n, "sí" if p in prov else "no", "ort", ""))
        if p in prov:
          disp.append(n)
    except Exception:  # noqa: BLE001
      pass
  if mostrar:
    mostrar_tabla(("backend", "disponible", "motores", "detalle"), filas, texto_hasta=4)
  return disp


def device_inicial(disp=None):
  """El dispositivo que usa el repo por defecto: cuda, rocm o mps (Mac con Apple Silicon) si hay GPU de torch; si no, cpu."""
  disp = disp if disp is not None else backends(mostrar=False)
  for d in ("cuda", "rocm", "mps"):
    if d in disp:
      return d
  return "cpu"


def device_kw(device):
  """Opciones de Toolkit para un dispositivo: {"device": ...}. En 'mps' va vacío: la CLI no lo fuerza (--device no lo lista) y
  src.common.device() ya elige MPS solo cuando no hay CUDA; con 'cpu' sí se fuerza (ML2_DEVICE=cpu)."""
  if device in (None, ""):
    return {}
  if device == "mps":
    try:
      from src.toolkit import POR_NOMBRE
      if "mps" in (POR_NOMBRE["device"].choices or ()):
        return {"device": "mps"}
    except Exception:  # noqa: BLE001
      pass
    return {}
  return {"device": device}


def ruta_demo(run, modo="imagine", pasos=None, k=10, latente=False, mascaras=False, carpeta=None):
  """Dónde escribe `demo` su resultado: el mp4 (video) o el png (latentes) de results/demo/."""
  c = Path(carpeta or (rutas.results() / "demo"))
  if latente:
    return c / f"{run}_{'mascaras' if mascaras else 'latente'}.png"
  return c / (f"{run}{f'_s{pasos}' if pasos else ''}_{modo}{f'_k{k}' if modo == 'online' else ''}.mp4")


def fuente_video_extra(fuente, segmento=0, inicio=0, mp4=None):
  """Los argumentos de `demo` (van después de `--`) para elegir la fuente: 'ejemplo' (video de comma), 'val' (segmento de val) o 'mp4'."""
  if fuente == "val":
    return ["--segment", str(segmento), "--start", str(inicio)]
  if fuente == "mp4":
    if not mp4:
      raise ValueError("fuente 'mp4': indicá la ruta del video propio")
    return ["--video", str(mp4), "--start", str(inicio)]
  return ["--start", str(inicio)]


def hora():
  return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------- secciones propias de cada modelo

def _fig(fig, mostrar):
  import matplotlib.pyplot as plt
  if mostrar:
    _display(fig)
    plt.close(fig)
  return fig


def destapar_frame(pasos=8, choice_temp=4.5, temperatura=1.0, semilla=0, mostrar=True):
  """Cómo se destapa un frame en la decodificación MaskGIT: corre `decode_frame` de src/models/genie.py (el real, con su calendario
  coseno y su ruido de confianza) sobre un modelo de juguete con logits sintéticos (celdas con confianza distinta), y dibuja la
  rejilla de 8 x 16 tokens en cada paso: en oscuro lo ya fijado, en claro lo que sigue tapado. Sin pesos ni GPU."""
  import matplotlib.pyplot as plt
  import numpy as np
  import torch
  from src.common import MASK
  from src.models import genie as G
  g = torch.Generator().manual_seed(semilla)
  escala = torch.rand(1, G.CODES, 1, generator=g) * 6          # confianza propia de cada celda
  base = torch.randn(1, G.CODES, 1024, generator=g)
  fijados = []

  def logits_fn(cur):
    fijados.append((cur != MASK)[0].reshape(8, 16).numpy().copy())
    return base * escala

  cur = torch.full((1, G.CODES), MASK, dtype=torch.long)
  G.decode_frame(logits_fn, pasos, temperatura, choice_temp, cur, generator=g)
  n = [G.CODES] + [G.n_masked(s, pasos) for s in range(1, pasos + 1)]
  fig, axs = plt.subplots(1, pasos + 1, figsize=(1.7 * (pasos + 1), 1.9))
  for i, ax in enumerate(np.atleast_1d(axs)):
    m = fijados[i] if i < pasos else np.ones((8, 16), bool)
    ax.imshow(m.astype(float), cmap="Blues", vmin=0, vmax=1.4)
    ax.set_title(f"paso {i + 1}\n{int((~m).sum())} tapados" if i < pasos else "listo\n0 tapados", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
  fig.suptitle(f"{pasos} pasos, choice_temp {choice_temp:g}, T {temperatura:g}; tapados tras cada paso (coseno): {n}", fontsize=8)
  fig.tight_layout()
  return _fig(fig, mostrar)


def calendario_destapado(pasos=(1, 2, 4, 8, 12), mostrar=True):
  """Tokens que quedan tapados tras cada paso (calendario coseno de MaskGIT) para varios números de pasos."""
  import matplotlib.pyplot as plt
  from src.models import genie as G
  fig, ax = plt.subplots(figsize=(5, 3))
  for S in pasos:
    ax.plot(range(S + 1), [G.n_masked(s, S) if s else G.CODES for s in range(S + 1)], "o-", ms=3, label=f"{S} pasos")
  ax.set_xlabel("paso")
  ax.set_ylabel("tokens tapados (de 128)")
  ax.legend(fontsize=7)
  ax.grid(alpha=.3)
  fig.tight_layout()
  return _fig(fig, mostrar)


def mascaras_vjepa2(semilla=0, mostrar=True):
  """Máscaras multibloque de V-JEPA 2 (src/models/vjepa/masks.py) sobre la rejilla de un clip de 16 frames: los 8 bloques cortos de
  escala 0,15 (arriba) y los 2 largos de 0,7 (abajo). Oscuro = tapado (lo que el predictor tiene que adivinar). Sin pesos ni GPU."""
  import matplotlib.pyplot as plt
  import numpy as np
  import torch
  from src.models.vjepa import masks as M
  g = torch.Generator().manual_seed(semilla)
  enc, pred = M.multiseq_multiblock3d(1, generator=g)
  T, H, W = 8, 8, 16
  fig, axs = plt.subplots(2, 4, figsize=(11, 3.4))
  for fila, (p, nombre) in enumerate(zip(pred, ("8 bloques cortos (0,15)", "2 bloques largos (0,7)"))):
    m = np.zeros(T * H * W)
    m[p[0].cpu().numpy()] = 1
    m = m.reshape(T, H, W)
    for j, t in enumerate((0, 3, 7)):
      axs[fila, j].imshow(m[t], cmap="Greys", vmin=0, vmax=1.6)
      axs[fila, j].set_title(f"{nombre}, t={t}", fontsize=8)
    axs[fila, 3].imshow(m.mean(0), cmap="Greys", vmin=0, vmax=1)
    axs[fila, 3].set_title(f"{100 * len(p[0]) / (T * H * W):.0f} % tapado", fontsize=8)
  for ax in axs.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
  fig.tight_layout()
  return _fig(fig, mostrar)


def nitidez_video(mp4, mostrar=True):
  """Nitidez por horizonte: varianza del laplaciano de cada frame de un mp4 de `demo` (real a la izquierda, predicho a la derecha).
  Un modelo de píxeles con MSE desenfoca a medida que se aleja del contexto: la curva predicha cae por debajo de la real."""
  import cv2
  import matplotlib.pyplot as plt
  import numpy as np
  if not Path(mp4).exists():
    return pendiente("nitidez por horizonte", f"falta {Path(mp4).name}; correr la demo (sección anterior) con DRY_RUN = False")
  cap = cv2.VideoCapture(str(mp4))
  real, pred = [], []
  while True:
    ok, f = cap.read()
    if not ok:
      break
    w = f.shape[1] // 2
    for lado, lista in ((f[:, :w], real), (f[:, w:], pred)):
      lista.append(float(cv2.Laplacian(cv2.cvtColor(lado, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()))
  cap.release()
  if not real:
    return pendiente("nitidez por horizonte", "el video está vacío")
  fig, ax = plt.subplots(figsize=(6, 3))
  ax.plot(np.arange(len(real)) / 20, real, label="real")
  ax.plot(np.arange(len(pred)) / 20, pred, label="predicho")
  ax.set_xlabel("segundos")
  ax.set_ylabel("nitidez (varianza del laplaciano)")
  ax.legend()
  ax.grid(alpha=.3)
  fig.tight_layout()
  return _fig(fig, mostrar)


def curva_pasos(run, raiz=None, mostrar=True):
  """FVD a 1 s (con IC) contra el número de pasos de decodificación, de los resultados video_<run>_p<pasos>.json."""
  import matplotlib.pyplot as plt
  R = Path(raiz or rutas.results())
  pts = []
  for p in sorted(R.glob(f"video_{run}_p*.json")):
    m = re.fullmatch(rf"video_{re.escape(run)}_p(\d+)", p.stem)
    d = leer_json(p)
    if m and _get(d, "fvd", "1", "value") is not None:
      pts.append((int(m.group(1)), d["fvd"]["1"]["value"], d["fvd"]["1"].get("lo"), d["fvd"]["1"].get("hi")))
  if not pts:
    return pendiente(f"FVD contra pasos de {run}", f"python -m src.cli <modelo> evaluar --run {run} --pasos 1,2,4,8,12")
  pts.sort()
  fig, ax = plt.subplots(figsize=(5, 3))
  ax.errorbar([a for a, *_ in pts], [b for _, b, *_ in pts],
              yerr=[[b - lo if lo is not None else 0 for _, b, lo, _h in pts], [hi - b if hi is not None else 0 for _, b, _l, hi in pts]],
              fmt="o-", capsize=2)
  ax.set_xlabel("pasos de decodificación")
  ax.set_ylabel("FVD a 1 s")
  ax.grid(alpha=.3)
  fig.tight_layout()
  return _fig(fig, mostrar)


def ce_por_posicion(raiz=None, mostrar=True):
  """CE por posición del frame en la ventana (results/ce_por_posicion.json, scripts/ce_por_posicion.py)."""
  import matplotlib.pyplot as plt
  d = leer_json(Path(raiz or rutas.results()) / "ce_por_posicion.json")
  pf = (d or {}).get("por_frame") or {}
  if not pf:
    return pendiente("CE por posición del frame", "python scripts/ce_por_posicion.py")
  fig, ax = plt.subplots(figsize=(6, 3))
  for k, v in pf.items():
    ax.plot(range(len(v)), v, label=k)
  ax.set_xlabel("posición del frame en la ventana")
  ax.set_ylabel("CE")
  ax.legend(fontsize=7)
  ax.grid(alpha=.3)
  fig.tight_layout()
  return _fig(fig, mostrar)


def _latente(run, raiz=None):
  return leer_json(Path(raiz or rutas.results()) / f"latent_{run}.json")


def horizonte_latente(run, raiz=None, medida="mse", mostrar=True):
  """Error de predicción por horizonte (segundos) del modelo contra las referencias 'copiar el último latente' y 'acción nula'."""
  import matplotlib.pyplot as plt
  d = _latente(run, raiz)
  h = _get(d, "horizonte", medida)
  if not h or "modelo" not in h:
    return pendiente(f"error por horizonte de {run}", f"python -m src.cli <modelo> evaluar --run {run}")
  seg = d["horizonte"].get("segundos") or list(range(1, len(h["modelo"]) + 1))
  fig, ax = plt.subplots(figsize=(5.5, 3.2))
  for k, et in (("modelo", run), ("copiar_ultimo", "copiar el último latente"), ("accion_nula", "acción nula")):
    if k in h:
      ax.plot(seg, [x["valor"] for x in h[k]], "o-", ms=3, label=et)
      ax.fill_between(seg, [x["ic95"][0] for x in h[k]], [x["ic95"][1] for x in h[k]], alpha=.15)
  ax.set_xlabel("segundos")
  ax.set_ylabel(f"error latente ({medida})")
  ax.legend(fontsize=7)
  ax.grid(alpha=.3)
  fig.tight_layout()
  return _fig(fig, mostrar)


def sorpresa_fig(run, raiz=None, mostrar=True):
  """Error por paso con el video original y con el corte (teletransporte o salto de 2 s): la sorpresa es el pico en el corte."""
  import matplotlib.pyplot as plt
  d = _latente(run, raiz)
  s = _get(d, "sorpresa")
  if not s or "teletransporte" not in s:
    return pendiente(f"sorpresa de {run}", f"python -m src.cli <modelo> evaluar --run {run}")
  fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
  for ax, k in zip(axs, ("teletransporte", "salto_2s")):
    e = s.get(k, {})
    ax.plot(e.get("por_paso_original", []), label="original")
    ax.plot(e.get("por_paso_perturbado", []), label="con corte")
    if "paso_del_corte" in s:
      ax.axvline(s["paso_del_corte"] - 1, ls=":", color="gray")
    ax.set_title(f"{k}: razón {_fmt(_get(e, 'modelo', 'razon', 'valor'), 2)} (copiar: {_fmt(_get(e, 'referencia_copiar_ultimo', 'razon', 'valor'), 2)})", fontsize=8)
    ax.set_xlabel("paso (0,25 s)")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
  fig.tight_layout()
  return _fig(fig, mostrar)


def probes_fig(run, raiz=None, mostrar=True):
  """R² de los probes de pose (velocidad y giro) sobre el encoder entrenado, el encoder sin entrenar y los píxeles."""
  import matplotlib.pyplot as plt
  d = _latente(run, raiz)
  pr = _get(d, "probes", "obs")
  if not pr:
    return pendiente(f"probes de {run}", f"python -m src.cli <modelo> evaluar --run {run}")
  dims = ("velocidad", "giro")
  fig, ax = plt.subplots(figsize=(5.5, 3))
  w = 0.25
  for i, (k, et) in enumerate((("modelo", run), ("sin_entrenar", "encoder sin entrenar"), ("pixeles", "píxeles"))):
    ax.bar([j + i * w for j in range(2)], [_get(pr, k, dm, "r2", "valor") or 0 for dm in dims], w, label=et)
  ax.set_xticks([j + w for j in range(2)])
  ax.set_xticklabels(dims)
  ax.set_ylabel("R² en val")
  ax.legend(fontsize=7)
  ax.grid(alpha=.3)
  fig.tight_layout()
  return _fig(fig, mostrar)


def cem_tabla(run, raiz=None):
  """Recuperación de acciones con CEM: error de la pose planeada contra repetir la última y contra la media de train."""
  cem = (_latente(run, raiz) or {}).get("cem")
  filas = []
  if isinstance(cem, dict):
    for k in sorted(k for k in cem if re.fullmatch(r"h\d+", k)):
      for dim in ("velocidad", "giro"):
        e = cem[k].get(dim, {})
        filas.append((k, dim, _fmt(_get(e, "mae_cem", "valor")), _fmt(_get(e, "mae_repetir_ultima", "valor")),
                      _fmt(_get(e, "mae_media_train", "valor")), _fmt(cem[k].get("ms_por_plan"), 0)))
  return mostrar_tabla(("horizonte", "dimensión", "MAE CEM", "repetir la última", "media de train", "ms por plan"), filas,
                       vacio=f"plan con CEM de {run}", texto_hasta=2)


def codec_resultados(raiz=None):
  """Fidelidad fp32 contra fp16 y latencia por dispositivo del codec (results/infer/codec_<dispositivo>.json)."""
  R = Path(raiz or rutas.results()) / "infer"
  filas = []
  for p in sorted(R.glob("codec_*.json")) if R.exists() else []:
    d = leer_json(p) or {}
    fid, lat = d.get("fidelity") or {}, d.get("latency") or {}
    lat_txt = json.dumps({k: (round(v.get("p50", v.get("median_ms", 0)), 2) if isinstance(v, dict) else v) for k, v in lat.items()}) if isinstance(lat, dict) else ""
    filas.append((d.get("tag", p.stem), _fmt(fid.get("tokens_fp16_vs_fp32"), 4), _fmt(fid.get("decoder_psnr_fp16_vs_fp32_db"), 1),
                  _fmt(fid.get("lpips_recon_fp16_vs_fp32"), 4), _fmt(fid.get("lpips_recon_fp32_vs_input"), 4), lat_txt))
  return mostrar_tabla(("dispositivo", "tokens fp16 = fp32", "PSNR decoder (dB)", "LPIPS fp16 vs fp32", "LPIPS recon vs entrada", "latencia p50 (ms)"),
                       filas, vacio="el codec todavía no se midió (python -m src.cli codec bench)", texto_hasta=1)


def latencia_etapas(raiz=None):
  """Latencia de la cadena (cámara -> encoder -> modelo -> decoder) de results/infer/latencia_*.json: p50 y p99 de cada columna del
  protocolo y la mediana por etapa."""
  R = Path(raiz or rutas.results()) / "infer"
  filas = []
  for p in sorted(R.glob("latencia_*.json")) if R.exists() else []:
    d = leer_json(p) or {}
    for col in ("modelo_solo", "cadena_cerrada", "cadena_ritmo"):
      c = d.get(col)
      if not c:
        continue
      et = c.get("stages_p50_ms") or {}
      filas.append((d.get("name", p.stem), d.get("tag", ""), col, _fmt(c.get("p50"), 2), _fmt(c.get("p99"), 2),
                    ", ".join(f"{k} {v:.2f}" for k, v in et.items()) or "–"))
  return mostrar_tabla(("modelo", "dispositivo", "columna", "p50 ms", "p99 ms", "mediana por etapa"), filas,
                       vacio="sin latencias de la cadena (python -m src.cli <modelo> bench --cadena)", texto_hasta=3)


def nodos_cpu(raiz=None):
  """Nodos que ONNX Runtime ejecuta en cada provider (results/infer/ort_*.json con --profile): los que caen a la CPU son el fallback."""
  R = Path(raiz or rutas.results()) / "infer"
  filas = []
  for p in sorted(R.glob("ort_*.json")) if R.exists() else []:
    d = leer_json(p) or {}
    for parte, prov in (d.get("placement") or {}).items():
      tot = sum(v.get("nodes", 0) for v in prov.values()) or 1
      for nombre, v in prov.items():
        filas.append((d.get("tag", p.stem), parte, nombre, v.get("nodes"), f"{100 * v.get('nodes', 0) / tot:.0f} %", _fmt(v.get("ms"), 2)))
  return mostrar_tabla(("corrida", "parte", "provider", "nodos", "% de los nodos", "ms"), filas,
                       vacio="sin perfil de nodos (python -m src.infer.bench_ort --profile)", texto_hasta=3)


# ---------------------------------------------------------------- demo para la clase

# corrida -> (modelo del Toolkit, opciones de `entrenar` que la producen): las de la cola y las que quedan implementadas fuera de ella
CORRIDAS = {
  "genie_s": ("genie", dict(tamano="s")), "gpt_s_kd": ("gpt", dict(tamano="s", destilar=True, variante="kd")),
  "convlstm_px": ("convlstm_px", {}), "lewm": ("lewm", {}), "vjepa2_s": ("vjepa2", dict(tamano="s")),
  "vjepa2_ac": ("vjepa2_ac", dict(encoder="vjepa2_s")),
  "gpt_s": ("gpt", dict(tamano="s")), "gpt_m": ("gpt", dict(tamano="m")), "genie_m": ("genie", dict(tamano="m")),
  "maskgit_s": ("maskgit", dict(tamano="s")),
}
CON_PASOS = ("maskgit", "genie")


def modelo_y_opciones(run):
  """(modelo del Toolkit, opciones de entrenar) de una corrida; (None, {}) si no se sabe."""
  if run in CORRIDAS:
    return CORRIDAS[run]
  return modelo_de_corrida(run), {}


def comando_entrenar(run):
  """La línea de CLI que entrena esa corrida (la que se muestra cuando todavía no hay checkpoint)."""
  m, o = modelo_y_opciones(run)
  return comando_cli(m, "entrenar", **o) if m else f"python -m src.cli <modelo> entrenar --run {run}"


def catalogo_corridas(candidatas=("genie_s", "gpt_s_kd", "convlstm_px", "lewm", "vjepa2_s", "vjepa2_ac"), mostrar=True):
  """Corridas para elegir: las candidatas y las que haya en checkpoints/ o en results/checkpoints.json, con su estado
  ('entrenada' o 'sin entrenar todavía') y, si falta, el comando que la entrena. Devuelve {corrida: estado}."""
  disp = corridas_disponibles()
  todas = {r: disp.get(r, "sin entrenar todavía") for r in candidatas}
  todas.update({r: e for r, e in disp.items() if r not in todas})
  if mostrar:
    mostrar_tabla(("corrida", "estado", "para entrenarla"), [(r, e, "" if e == "entrenada" else comando_entrenar(r)) for r, e in todas.items()],
                  texto_hasta=3)
  return todas


def ultimo(patron="*.mp4", carpeta=None):
  """El archivo más nuevo de results/demo/ que coincide con `patron` (una ruta que no existe si no hay ninguno)."""
  c = Path(carpeta or (rutas.results() / "demo"))
  fs = sorted(c.glob(patron), key=lambda p: p.stat().st_mtime) if c.exists() else []
  return fs[-1] if fs else c / "(ninguno todavía)"


def demo(run, device=None, fuente="ejemplo", segmento=0, inicio=0, mp4=None, ctx=20, horizon=60, frames=None, pasos=None,
         aleatorio=False, reusar=True, mostrar=True, titulo=None):
  """La acción `demo` de Toolkit sobre una corrida, de verdad (no dry-run), y el resultado embebido en el notebook.

  Video (gpt, genie, maskgit, convlstm_px): mp4 con el real a la izquierda y el predicho a la derecha; los `ctx` frames de contexto van con
  borde gris y los imaginados con borde naranja ('imagina +k'). Latentes (lewm, vjepa2, vjepa2_ac): la figura de src.infer.demo_latente.
  Sin checkpoint no falla: dice qué comando entrena la corrida. Con aleatorio=True usa los pesos sin entrenar (solo para probar).
  reusar=True no vuelve a correr una demo con los mismos parámetros que ya tiene su archivo.
  Devuelve {"run", "ruta", "ms", "ok", "latente"} o None si no se pudo."""
  from src.toolkit import Toolkit
  modelo, _ = modelo_y_opciones(run)
  if modelo is None:
    pendiente(f"modelo de {run}", "no sé a qué modelo pertenece la corrida")
    return None
  entrenada = (rutas.ckpt() / run / "final.pt").exists()
  if not entrenada and not aleatorio:
    pendiente(f"{run}: sin entrenar todavía", f"entrenalo con: {comando_entrenar(run)}")
    return None
  tk = Toolkit(modelo)
  latente = tk.evaluacion == "latente"
  frames = frames or (ctx + horizon)
  etiqueta = f"{run}{'_aleatorio' if aleatorio else ''}"
  firma = f"{fuente}{segmento if fuente == 'val' else ''}_{inicio}_{ctx}_{horizon}_{frames}" + (f"_s{pasos}" if pasos else "")
  out = (rutas.results() / "demo" / f"clase_{etiqueta}_{firma}").with_suffix(".png" if latente else ".mp4")
  opts = dict(run=run, **device_kw(device or device_inicial()))
  if aleatorio:
    opts["aleatorio"] = True
  if latente:
    if fuente != "val":
      print(f"{modelo} usa los segmentos de val con poses: uso el segmento {segmento}")
    extra = ["--segment", str(segmento), "--start", str(max(inicio, 300)), "--pasos", str(max(1, horizon // 5)), "--out", str(out), "--mp4"]
  else:
    if pasos and modelo in CON_PASOS:
      opts["pasos"] = str(pasos)
    opts["limite"] = frames
    extra = ["--ctx", str(ctx), "--horizon", str(horizon), "--out", str(out)] + fuente_video_extra(fuente, segmento, inicio, mp4)
  if not (reusar and out.exists()):
    out.parent.mkdir(parents=True, exist_ok=True)
    t = lanzar(modelo, "demo", dry_run=False, segundo_plano=False, extra=extra, mostrar_salida=False, **opts)
    if not t.ok:
      print("\n".join(t.lineas[-12:]), t.error or "")
  ok = out.exists()
  info = info_demo(out) if not latente else None
  ms = (info or {}).get("ms_per_frame_p50")
  if mostrar:
    nombre = titulo or etiqueta + (" (pesos sin entrenar)" if aleatorio else "")
    if not ok:
      pendiente(f"demo de {run}", "no se generó el archivo; mirá la salida de arriba")
    elif latente:
      figura(out)
      mp = out.with_suffix(".mp4")
      if mp.exists():
        video(mp, nombre)
    else:
      video(out, nombre + (f" · {ms:.1f} ms por frame generado" if ms else ""))
  return {"run": run, "ruta": out if ok else None, "ms": ms, "ok": ok, "latente": latente}


def comparar(runs, **kw):
  """Lo mismo que `demo` para dos o tres corridas sobre el mismo clip (mismos parámetros), lado a lado y con la latencia por frame de cada una.
  Las corridas sin entrenar se saltan con su comando. Los latentes muestran su figura aparte."""
  res = [demo(r, mostrar=False, **kw) for r in runs]
  vid = [x for x in res if x and x["ok"] and not x["latente"]]
  if vid:
    lado_a_lado([(x["run"] + (" (pesos sin entrenar)" if kw.get("aleatorio") else ""), x["ruta"], x["ms"]) for x in vid])
    mostrar_tabla(("corrida", "ms por frame generado (p50)"), [(x["run"], _fmt(x["ms"], 1)) for x in vid], texto_hasta=1)
  for x in res:
    if x and x["ok"] and x["latente"]:
      figura(x["ruta"])
  if not any(res):
    pendiente("comparación", "ninguna de las corridas tiene checkpoint")
  return res


def imaginar(run, ctx=20, horizon=60, **kw):
  """El modelo ve `ctx` frames reales (1 s) e imagina `horizon` (3 s): en el video, borde gris = contexto real y borde naranja con
  'imagina +k' = imaginado (el real sigue a la izquierda para comparar). En los latentes, `horizon // 5` pasos de 0,25 s."""
  mostrar_html("<div style='font-size:13px'>Borde gris = el modelo ve el frame real &middot; borde naranja = lo imagina "
               "(el real sigue a la izquierda para comparar).</div>")
  return demo(run, ctx=ctx, horizon=horizon, frames=ctx + horizon, **kw)


def general(nombre, dry_run=True, extra=()):
  """reporte, resultados o estadistica (scripts de CPU sin modelo): `python -m src.cli <nombre>`. Imprime los comandos con dry_run."""
  from src import ejecutor
  from src.toolkit import plan_general
  return ejecutor.ejecutar(plan_general(nombre, extra=tuple(extra)), dry_run=dry_run, salida=print)


def muestras(n=3, carpeta=None, ancho=360):
  """Los `n` videos más nuevos de results/demo/ (real a la izquierda, predicho a la derecha), embebidos. 'pendiente' si no hay."""
  c = Path(carpeta or (rutas.results() / "demo"))
  fs = sorted(c.glob("*.mp4"), key=lambda p: p.stat().st_mtime)[-n:] if c.exists() else []
  if not fs:
    return pendiente("muestras", "todavía no se generó ninguna demo (python -m src.cli <modelo> demo, o notebooks/demo_clase.ipynb)")
  lado_a_lado([(p.stem, p, (info_demo(p) or {}).get("ms_per_frame_p50")) for p in fs], ancho=ancho)
  return fs


def ort_perfil(enc="cpu", dec="cpu", frames=100, dry_run=True):
  """Mide el codec con ONNX Runtime y cuenta los nodos por provider (`python -m src.infer.bench_ort --enc X --dec Y --profile`):
  los que caen a la CPU son el fallback. Va por subprocesos con el mismo intérprete del kernel (sys.executable)."""
  import subprocess
  import sys
  argv = [sys.executable, "-m", "src.infer.bench_ort", "--enc", enc, "--dec", dec, "--frames", str(frames), "--profile"]
  txt = "python -m src.infer.bench_ort " + " ".join(argv[3:])
  if dry_run:
    print("#", txt)
    return None
  print("$", txt)
  r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
  print((r.stdout or "")[-1500:] + (r.stderr or "")[-600:])
  return r.returncode
