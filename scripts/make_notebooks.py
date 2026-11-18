"""Genera los notebooks del repo: uno por modelo (notebooks/modelos/<modelo>.ipynb) y los generales (resultados, latentes,
inferencia, demo_clase, colab). Los .ipynb salen sin salidas guardadas y sin números: lo que muestran lo leen de results/ al correr.

  python scripts/make_notebooks.py            escribe los que cambiaron
  python scripts/make_notebooks.py --check    falla (código 1) si algún notebook del disco difiere del que se generaría

Un notebook de modelo tiene, en este orden: receta y fidelidad, entrenar o continuar, evaluar, latencia, demo y exportar (los
mismos nombres, el mismo orden y los mismos defaults que la CLI: `python -m src.cli <modelo> <acción>`), y una sección propia.
Las acciones van con DRY_RUN = True (imprimen los comandos); con DRY_RUN = False corren de verdad por Toolkit, en segundo plano,
y quedan en results/bitacora.jsonl. La lista de modelos sale del registro (src/toolkit.py): un modelo nuevo aparece al regenerar.
"""
import argparse
import re
import sys
from pathlib import Path

import nbformat

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
NB = RAIZ / "notebooks"

ORDEN = ("receta", "entrenar", "continuar", "evaluar", "bench", "demo", "exportar")


# ---------------------------------------------------------------- celdas

def md(t):
  return nbformat.v4.new_markdown_cell(t.strip("\n"))


def sin_eco(texto):
  """Agrega ';' a la última línea si es una llamada suelta a nbui: IPython no muestra su valor de retorno (una lista o un dict) debajo de la celda."""
  lineas = texto.split("\n")
  m = re.match(r"^(nbui\.\S.*?)(\s+#.*)?$", lineas[-1])
  if m and m.group(1).rstrip().endswith(")"):
    lineas[-1] = m.group(1).rstrip() + ";" + (m.group(2) or "")
  return "\n".join(lineas)


def code(t):
  return nbformat.v4.new_code_cell(sin_eco(t.strip("\n")))


def lit(v):
  """Literal de Python con comillas dobles, para los dicts de opciones."""
  if isinstance(v, str):
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
  return repr(v)


def dic(d, extra=""):
  """dict(a=1, b="x") (+ extra, texto ya armado, p. ej. **DEV)."""
  partes = [f"{k}={lit(v)}" for k, v in d.items()] + ([extra] if extra else [])
  return "dict(" + ", ".join(partes) + ")"


def armar(celdas, titulo):
  nb = nbformat.v4.new_notebook()
  nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}, "title": titulo}
  nb.cells = list(celdas)
  for i, c in enumerate(nb.cells):
    c["id"] = f"c{i:03d}"   # ids estables: el mismo script da siempre el mismo archivo
  nbformat.validate(nb)
  return nbformat.writes(nb) + "\n"


SETUP = '''
%matplotlib inline
import os, sys
from pathlib import Path

# la raíz del repo: sube desde donde esté abierto el notebook hasta encontrar src/common.py
RAIZ = next((d for d in [Path.cwd().resolve(), *Path.cwd().resolve().parents] if (d / "src" / "common.py").exists()), None)
if RAIZ is None:
  raise SystemExit("abrí el notebook dentro del repo (en Colab, corré antes la primera celda de notebooks/colab.ipynb)")
os.chdir(RAIZ)
sys.path.insert(0, str(RAIZ))

from src import nbui
from src.toolkit import Toolkit

# scripts/run_notebooks.py pone ML2_NB_LIMITE para correr todo chico; a mano queda en None (sin recorte)
LIMITE = int(os.environ.get("ML2_NB_LIMITE", "0")) or None
'''


# ---------------------------------------------------------------- perfil de cada modelo

PERFIL = {
  "gpt": dict(titulo="GPT de comma, escalado", tam="s", entrena=dict(tamano="s", destilar=True, variante="kd"),
              que="GPT autoregresivo sobre los tokens del VQ-VAE de comma (diseño de comma escalado, posición absoluta). "
                  "El de la cola es S destilado del gpt2m de comma (`gpt_s_kd`); el de cero (`gpt_s`) y M quedan implementados fuera de la cola."),
  "maskgit": dict(titulo="MaskGIT propio (ablación)", tam="s", entrena=dict(tamano="s"),
                  que="MaskGIT con receta propia (el paper no publica el entrenamiento): decodificación paralela por confianza con calendario coseno. "
                      "El MaskGIT fiel de video es `genie`; este es la ablación propia."),
  "genie": dict(titulo="Genie: dinámica con decodificación MaskGIT", tam="s", entrena=dict(tamano="s"),
                que="La dinámica de Genie (ST-transformer con atención espacial, temporal causal y FFW) entrenada con máscara Bernoulli y "
                    "decodificada al estilo MaskGIT. Es el MaskGIT fiel del proyecto; sin modelo de acciones latentes."),
  "convlstm_px": dict(titulo="ConvLSTM de píxeles (Shi et al. 2015)", tam=None, entrena={},
                      que="ConvLSTM encoder-forecaster con peepholes, entrada y salida en píxeles, loss L2. Es determinista (sin temperatura ni CE) "
                          "y no usa el codec de comma."),
  "lewm": dict(titulo="LeWM: LeWorldModel desde píxeles", tam=None, entrena={},
               que="ViT-Tiny + predictor autoregresivo con SIGReg contra el colapso, fiel al repo oficial. Predice el embedding del próximo "
                   "paso (4 Hz) y planea con CEM. No genera video: su decoder de píxeles se entrena aparte, solo para figuras."),
  "vjepa2": dict(titulo="V-JEPA 2: preentrenamiento autosupervisado", tam="s", entrena=dict(tamano="s"),
                 que="Encoder ViT que aprende prediciendo, en el espacio latente, los tubelets tapados por máscaras multibloque 3D "
                     "(target EMA, L1, 3D-RoPE). No genera nada: se juzga por probes y por lo que sirve de encoder al 2-AC."),
  "vjepa2_ac": dict(titulo="V-JEPA 2-AC: predictor condicionado por acciones", tam=None, entrena=dict(encoder="vjepa2_s"),
                    que="Predictor causal por bloques con acciones sobre un encoder V-JEPA 2 congelado; planea con CEM (800 muestras, top-10, L1). "
                        "Necesita un `vjepa2` entrenado: `--encoder vjepa2_s`."),
  "codec": dict(titulo="Codec de comma (VQ-VAE)", tam=None, entrena=None,
                que="El encoder y decoder del VQ-VAE de comma que convierte cámara en 128 tokens por frame y viceversa. No se entrena acá: "
                    "se mide (fidelidad fp32 contra fp16 y latencia por dispositivo) y se exporta a ONNX."),
  "gpt2m": dict(titulo="gpt2m de comma (referencia y maestro)", tam=None, entrena=None,
                que="El GPT de 307M de comma, sin cambios: referencia de calidad y maestro de la destilación."),
  "unigram": dict(titulo="Baseline unigrama", tam=None, entrena=None, que="Frecuencia de cada código en train: cota inferior."),
  "transicion": dict(titulo="Baseline de transición por celda", tam=None, entrena=None,
                     que="p(y | x) con x el código de la misma celda en el frame anterior (antes `copy`)."),
  "congelado": dict(titulo="Baseline de frame congelado", tam=None, entrena=None,
                    que="Repite el último frame (la mezcla copiar/unigrama a T = 0; antes `copy_mix`)."),
  "real": dict(titulo="Real contra real: el piso de FVD", tam=None, entrena=None,
               que="Los propios clips reales evaluados como si fueran un modelo: el FVD que ni un modelo perfecto puede bajar."),
}
CON_PASOS = ("maskgit", "genie")
LATENTES = ("lewm", "vjepa2", "vjepa2_ac")


def perfil(m):
  from src import toolkit as T
  p = dict(PERFIL.get(m, {}))
  kind = m in T.kinds()
  p.setdefault("titulo", m)
  p.setdefault("que", "Modelo registrado en `src/models/__init__.py` (ver su guía en `Docs/modelos/`).")
  p.setdefault("entrena", {} if kind else None)
  p.setdefault("tam", None)
  return p


def guia(m):
  """La guía del modelo en Docs/modelos/ (la de las referencias, o el índice, si no tiene una propia)."""
  d = RAIZ / "Docs" / "modelos"
  for f in (f"{m}.md", "referencias.md", "README.md"):
    if (d / f).exists() and (f != "referencias.md" or m in ("gpt2m", "unigram", "transicion", "congelado", "real")):
      return f"Docs/modelos/{f}"
  return "Docs/modelos/"


def run_de(m, opts):
  from src.toolkit import Toolkit
  return Toolkit(m).nombre_run(opts or {})


# ---------------------------------------------------------------- secciones comunes

def seccion_receta(m, p, acciones):
  c = [md("## 1. Receta y fidelidad\nLa receta de entrenamiento es la de la fuente del modelo (con lo forzado por datos o presupuesto marcado) y la tabla "
          "de fidelidad dice qué es fiel, qué lo fuerzan los datos o el presupuesto y qué es propio. CLI: `python -m src.cli " + m + " receta`.")]
  if "receta" in acciones:
    tam = f"TAMANO" if p["tam"] else "None"
    c += [code(f'''
r = nbui.receta(MODELO, {tam})        # la Recipe (optimizador, lr por tramos, wd, EMA, clip, batch, presupuesto) y lo forzado
nbui.fidelidad(MODELO)                # tabla de fidelidad: decisión, fuente, lo que hace el repo, categoría
'''), code(f'''
nbui.opciones(MODELO, "receta", {"tamano=TAMANO" if p["tam"] else ""})   # opciones de la acción y el comando equivalente
t = nbui.lanzar(MODELO, "receta", dry_run=DRY_RUN, {"tamano=TAMANO" if p["tam"] else ""})   # lo que imprime `python -m src.recipes`
''')]
  else:
    c += [code('nbui.fidelidad(MODELO)'), md(f"`{m}` no tiene receta de entrenamiento: no se entrena en el repo.")]
  return c


def seccion_entrenar(m, p, acciones):
  ent = p["entrena"]
  c = [md("## 2. Entrenar o continuar\nEntrenar de cero con la receta de la fuente, probar el lr en un humo corto, seguir la corrida en vivo y "
          "continuarla o extenderla. CLI: `entrenar` (con `--humo`), `continuar`, `estado`.")]
  if ent is None or "entrenar" not in acciones:
    return c + [md(f"No aplica: `{m}` no se entrena en el repo.")]
  humo = {k: v for k, v in ent.items() if k != "variante"}
  c += [code(f'''
OPC_ENTRENAR = {dic(ent, "limite=LIMITE")}     # los mismos nombres que la CLI: --tamaño, --destilar, --variante, --limite...
nbui.opciones(MODELO, "entrenar", **OPC_ENTRENAR)
'''), code(f'''
# humo: 300 pasos con el lr de la fuente, escalado por raíz, lineal, x3 y x10; elige el que mejor baja (--humo)
t = nbui.lanzar(MODELO, "entrenar", dry_run=DRY_RUN, humo=True, **{dic(humo)})
'''), code('''
# entrenamiento de verdad (con DRY_RUN = False corre en segundo plano: el kernel sigue libre)
t = nbui.lanzar(MODELO, "entrenar", dry_run=DRY_RUN, **OPC_ENTRENAR)
'''), code('''
# seguimiento: barra con porcentaje y ETA, y curvas de loss, lr y monitoreo con el tramo del schedule sombreado
nbui.mostrar_barra(RUN)
nbui.curvas(RUN)
# nbui.en_vivo(RUN)     # se actualiza solo hasta que la corrida termina (en una corrida real)
'''), code('''
nbui.estado_corridas()                                 # una fila por corrida de checkpoints/ (= python -m src.status)
nbui.panel_cola()                                      # si la GPU estaba ocupada, la acción quedó en la cola
''')]
  if "continuar" in acciones:
    c += [md("**Continuar.** Retomar una corrida cortada (`continuar --run`) o extenderla con más presupuesto (`--presupuesto 2x`): la extensión es "
             "una corrida nueva `<run>_x<k>` con `parent`, sha y receta. Las recetas con coseno solo se extienden con `--restart`."),
          code('''
OPC_CONTINUAR = dict(run=RUN, limite=LIMITE)                       # retomar desde last.pt
OPC_EXTENDER = dict(run=RUN, presupuesto="2x", limite=LIMITE)      # extender: corrida nueva <run>_x2
nbui.opciones(MODELO, "continuar", **OPC_CONTINUAR)
t = nbui.lanzar(MODELO, "continuar", dry_run=DRY_RUN, **OPC_CONTINUAR)
t = nbui.lanzar(MODELO, "continuar", dry_run=DRY_RUN, **OPC_EXTENDER)
''')]
  return c


def seccion_evaluar(m, p, acciones):
  latente = m in LATENTES
  c = [md("## 3. Evaluar\n" + ("Evaluación latente: error de predicción por horizonte contra copiar el último latente, probes contra el encoder sin "
                              "entrenar y contra píxeles, sorpresa y plan con CEM. " if latente else
                              "Métricas propias de video: CE, rollouts de 3 s, LPIPS y FVD con IC por segmento, y el piso de FVD (real contra real) al lado. ")
                      + f"CLI: `python -m src.cli {m} evaluar`.")]
  if "evaluar" not in acciones:
    return c + [md(f"No aplica: `{m}` no declara la acción `evaluar`.")]
  pasos = ', pasos="8"' if m in CON_PASOS else ""
  c += [code(f'''
OPC_EVALUAR = dict(run=RUN{pasos}, limite=LIMITE, **DEV)
nbui.opciones(MODELO, "evaluar", **OPC_EVALUAR)
t = nbui.lanzar(MODELO, "evaluar", dry_run=DRY_RUN, **OPC_EVALUAR)
''')]
  if latente:
    c += [code('''
nbui.tabla_latente(solo=[RUN])        # una fila con las referencias al lado; "pendiente" hasta que exista latent_<run>.json
''')]
  else:
    c += [code('''
nbui.tabla_video(solo=[RUN])          # una fila: CE, LPIPS, FVD con IC por segmento, piso de FVD y latencia; "pendiente" sin JSON
''')]
    if m not in ("real", "congelado"):
      c += [md("Más opciones de `evaluar`: `--cadena` (calidad en línea y con la cadena completa), `--barrer-temperatura` (T elegida en el holdout), "
               "`--temperatura`, `--ctx-frames`, `--dtype`."),
            code('''
t = nbui.lanzar(MODELO, "evaluar", dry_run=DRY_RUN, run=RUN, cadena=True, limite=LIMITE, **DEV)               # calidad con la cadena de punta a punta
t = nbui.lanzar(MODELO, "evaluar", dry_run=DRY_RUN, run=RUN, barrer_temperatura=True, limite=LIMITE, **DEV)    # barrido de T en el holdout
''')]
  return c


def seccion_bench(m, p, acciones):
  c = [md("## 4. Latencia\nUn protocolo único: un paso por frame en batch 1, fp16 con CUDA graphs donde anda, p50 y p99; con `--cadena` mide la "
          "cadena cámara → encoder → modelo → decoder, en bucle cerrado y a ritmo de cámara (50 ms). " + f"CLI: `python -m src.cli {m} bench`.")]
  if "bench" not in acciones:
    return c + [md(f"No aplica: `{m}` no declara la acción `bench`.")]
  from src import toolkit as T
  kind = m in T.kinds()
  c += [code('''
OPC_BENCH = dict(run=RUN, limite=LIMITE, **DEV)
nbui.opciones(MODELO, "bench", **OPC_BENCH)
t = nbui.lanzar(MODELO, "bench", dry_run=DRY_RUN, **OPC_BENCH)                                   # el modelo solo
t = nbui.lanzar(MODELO, "bench", dry_run=DRY_RUN, cadena=True, ritmo=50, **OPC_BENCH)            # la cadena a ritmo de cámara
t = nbui.lanzar(MODELO, "bench", dry_run=DRY_RUN, cadena=True, diagnostico=True, **OPC_BENCH)    # el piso de ~20 ms a ritmo de cámara
''')] if m != "codec" else [code('''
OPC_BENCH = dict(limite=LIMITE, **DEV)
nbui.opciones(MODELO, "bench", **OPC_BENCH)
t = nbui.lanzar(MODELO, "bench", dry_run=DRY_RUN, **OPC_BENCH)        # latencia del codec y fidelidad fp32 contra fp16
''')]
  if kind:
    c += [md("La latencia no depende de los pesos: con `aleatorio=True` se mide la arquitectura sin entrenar."),
          code('t = nbui.lanzar(MODELO, "bench", dry_run=DRY_RUN, aleatorio=True, limite=LIMITE, **DEV)')]
  c += [code('nbui.latencia_etapas()      # p50 y p99 de cada columna del protocolo y mediana por etapa, de results/infer/')]
  return c


def seccion_demo(m, p, acciones):
  c = [md("## 5. Demo\n" + ("Decodificación del latente con el decoder de píxeles, máscaras (V-JEPA 2) y curva de error. " if m in LATENTES else
                          "Un mp4 con el video real a la izquierda y lo que predice el modelo a la derecha: 1 s de contexto real y después imagina. ")
                      + f"CLI: `python -m src.cli {m} demo`.")]
  if "demo" not in acciones:
    return c + [md(f"No aplica: `{m}` no declara la acción `demo`.")]
  if m in LATENTES:
    c += [code('''
OPC_DEMO = dict(run=RUN, limite=LIMITE, **DEV)
nbui.opciones(MODELO, "demo", **OPC_DEMO)
t = nbui.lanzar(MODELO, "demo", dry_run=DRY_RUN, **OPC_DEMO)          # figura y curvas de error (src.infer.demo_latente)
nbui.figura(nbui.ultimo("*_latente.png"))
'''), code('''
# el decoder de píxeles se entrena aparte (encoder congelado, MSE): solo para figuras
t = nbui.lanzar(MODELO, "demo", dry_run=DRY_RUN, entrenar_decoder=True, limite=LIMITE, **DEV, **({"encoder": "vjepa2_s"} if MODELO == "vjepa2_ac" else {}))
''')]
  else:
    pas = ', pasos="8"' if m in CON_PASOS else ""
    c += [code(f'''
OPC_DEMO = dict(run=RUN{pas}, limite=LIMITE, **DEV)
nbui.opciones(MODELO, "demo", **OPC_DEMO)
# lo que va después de -- es de src.infer.demo: --ctx (frames de contexto), --horizon (frames imaginados), --segment, --video...
t = nbui.lanzar(MODELO, "demo", dry_run=DRY_RUN, extra=["--ctx", "20", "--horizon", "60"], **OPC_DEMO)
nbui.video(nbui.ruta_demo(RUN{', pasos=8' if m in CON_PASOS else ''}), "real (izquierda) contra predicho (derecha)")
''')]
  return c


def seccion_exportar(m, p, acciones):
  c = [md("## 6. Exportar\nONNX, donde aplica (`python -m src.cli " + m + " exportar`). El export verifica que ONNX Runtime coincida con PyTorch.")]
  if "exportar" not in acciones:
    return c + [md(f"No aplica: `{m}` no declara la acción `exportar`.")]
  c += [code('''
OPC_EXPORTAR = dict(run=RUN) if MODELO != "codec" else dict()
nbui.opciones(MODELO, "exportar", **OPC_EXPORTAR)
t = nbui.lanzar(MODELO, "exportar", dry_run=DRY_RUN, **OPC_EXPORTAR)
t = nbui.lanzar(MODELO, "exportar", dry_run=DRY_RUN, dtype="fp16", **OPC_EXPORTAR)      # la versión fp16
''')]
  return c


# ---------------------------------------------------------------- sección propia de cada modelo

def propia(m, p):
  r = "## 7. Sección propia: "
  if m == "gpt":
    return [md(r + "de cero contra destilado\nEl GPT de la cola es S destilado del gpt2m de comma (`--destilar`): la pérdida suma la CE de los datos "
                   "y la del maestro. Se lo compara con el mismo S de cero (`gpt_s`, fuera de la cola) y con el propio maestro."),
            code('''
OPC_CERO = dict(tamano="s", limite=LIMITE)                                         # de cero
OPC_KD = dict(tamano="s", destilar=True, variante="kd", limite=LIMITE)             # destilado de gpt2m
for nombre, o in (("de cero", OPC_CERO), ("destilado", OPC_KD)):
  print("##", nombre)
  t = nbui.lanzar(MODELO, "entrenar", dry_run=DRY_RUN, **o)
'''), code('''
for corrida in ("gpt_s", "gpt_s_kd"):            # las curvas de las dos corridas, si existen
  nbui.curvas(corrida)
'''), code('nbui.tabla_video(solo=["gpt_s", "gpt_s_kd", "gpt2m"])      # de cero, destilado y maestro, con el piso de FVD'),
            md("**CE por posición del frame.** Con posición absoluta la CE cambia con el lugar del frame en la ventana: se mide con "
               "`scripts/ce_por_posicion.py`."),
            code('nbui.ce_por_posicion()')]
  if m in CON_PASOS:
    quien = "genie" if m == "genie" else "maskgit"
    return [md(r + "pasos de decodificación, choice_temp y cómo se destapa un frame\nEn cada paso el modelo muestrea todas las celdas tapadas, "
                   "puntúa su confianza (log p más ruido Gumbel de `choice_temp·(1 − s/S)`) y fija las más confiables hasta dejar tapadas "
                   "`floor(128·cos(π/2·s/S))` celdas. Con un paso es argmax; con más pasos el frame se destapa de a poco."),
            code('nbui.calendario_destapado((1, 2, 4, 8, 12))      # cuántas celdas siguen tapadas tras cada paso'),
            md("**Un frame, paso a paso.** Es el `decode_frame` real del modelo sobre un modelo de juguete (logits sintéticos con confianza distinta por "
               "celda): oscuro = ya fijado. Con `choice_temp` alto el orden se mezcla al principio; con 0 se fijan siempre las más confiables."),
            code('''
for ct in (0.0, 4.5):
  nbui.destapar_frame(pasos=8, choice_temp=ct, temperatura=1.0)
'''), md("**Barrido de pasos** (1, 2, 4, 8, 12): cuesta un paso del modelo por paso de decodificación, así que la latencia crece con S "
        "mientras el FVD mejora hasta cierto punto."),
            code('''
t = nbui.lanzar(MODELO, "evaluar", dry_run=DRY_RUN, run=RUN, pasos="1,2,4,8,12", limite=LIMITE, **DEV)
t = nbui.lanzar(MODELO, "bench", dry_run=DRY_RUN, run=RUN, pasos="1,2,4,8,12", limite=LIMITE, **DEV)
nbui.curva_pasos(RUN)
'''), md("**Temperatura.** `--temperatura` cambia el muestreo de cada celda y queda en el nombre del resultado (`_t<T>`). El barrido de "
        "`choice_temp` sobre el modelo entrenado necesita una opción propia del Toolkit (hoy solo se puede pedir con `--opt choice_temp=...` "
        "por `--`, que pisaría el resultado de la misma corrida): acá se lo ve en el destapado."),
            code('''
for T in (0.8, 1.0):
  t = nbui.lanzar(MODELO, "evaluar", dry_run=DRY_RUN, run=RUN, pasos="8", temperatura=T, limite=LIMITE, **DEV)
''')]
  if m == "convlstm_px":
    return [md(r + "nitidez por horizonte y cadena sin codec\nUn modelo de píxeles entrenado con MSE promedia futuros posibles y desenfoca a "
                   "medida que se aleja del contexto. La nitidez (varianza del laplaciano) del video predicho contra la del real lo muestra. "
                   "Como no usa el codec, su cadena es cámara → ConvLSTM → píxeles."),
            code('nbui.nitidez_video(nbui.ruta_demo(RUN))       # sale del mp4 de la sección 5: pendiente hasta que exista'),
            code('nbui.horizonte(solo=[RUN])                      # LPIPS por segundo generado'),
            code('''
t = nbui.lanzar(MODELO, "bench", dry_run=DRY_RUN, run=RUN, cadena=True, ritmo=50, limite=LIMITE, **DEV)   # la cadena sin codec
nbui.latencia_etapas()
''')]
  if m == "lewm":
    return [md(r + "SIGReg, sorpresa y CEM\n**SIGReg** (λ 0,09, 17 nodos, 1024 proyecciones) empuja los embeddings hacia una gaussiana isótropa y "
                   "evita el colapso: en las curvas se ve junto con el error de predicción y el desvío de los latentes (`z_std`). La **sorpresa** es "
                   "el error de predicción cuando el video se corta (teletransporte o salto de 2 s). El **CEM** planea las acciones que llevan al "
                   "latente de un frame objetivo."),
            code('nbui.curvas(RUN)     # loss, pred_mse, sigreg y z_std (si el log los trae)'),
            code('nbui.sorpresa_fig(RUN)'), code('nbui.horizonte_latente(RUN)'), code('nbui.cem_tabla(RUN)'),
            code('''
t = nbui.lanzar(MODELO, "evaluar", dry_run=DRY_RUN, run=RUN, limite=LIMITE, **DEV)           # la evaluación completa
t = nbui.lanzar(MODELO, "evaluar", dry_run=DRY_RUN, run=RUN, sin_plan=True, limite=LIMITE, **DEV)   # sin el CEM (más rápido)
''')]
  if m == "vjepa2":
    return [md(r + "máscaras multibloque y probes contra el encoder sin entrenar\nCada muestra tapa ~90 % de los tubelets con 8 bloques cortos (0,15) "
                   "y 2 largos (0,7), con aspecto 0,75–1,5 y todo el tiempo: el predictor solo ve lo visible y adivina lo tapado. Las máscaras no "
                   "dependen de los pesos."),
            code('nbui.mascaras_vjepa2(semilla=0)'),
            code('''
# la misma figura sobre un clip real, por la CLI (con pesos aleatorios alcanza: las máscaras no usan los pesos)
t = nbui.lanzar(MODELO, "demo", dry_run=DRY_RUN, aleatorio=True, run=RUN, limite=LIMITE, **DEV)
nbui.figura(nbui.ultimo("*_mascaras.png"))
'''), md("**Probes.** Un ridge sobre el encoder entrenado predice velocidad y giro; se compara con el mismo encoder sin entrenar y con píxeles: si el "
         "entrenamiento sirvió, el R² del modelo gana a los dos."),
            code('nbui.probes_fig(RUN)'), code('nbui.tabla_latente(solo=[RUN])')]
  if m == "vjepa2_ac":
    return [md(r + "error por horizonte y recuperación de acciones\nEl rollout autoregresivo con las acciones reales se compara con *copiar el último "
                   "latente* y con la *acción nula*: si la acción importa, la nula es peor que la real. El CEM recupera las acciones minimizando el L1 "
                   "contra el latente objetivo."),
            code('nbui.horizonte_latente(RUN)'), code('nbui.cem_tabla(RUN)'),
            code('''
t = nbui.lanzar(MODELO, "evaluar", dry_run=DRY_RUN, run=RUN, limite=LIMITE, **DEV)
nbui.tabla_latente(solo=[RUN])
''')]
  if m == "codec":
    return [md(r + "fidelidad y latencia\nEl decoder en fp16 tiene que dar los mismos tokens y casi la misma imagen que en fp32; la latencia se mide "
                   "por dispositivo (cuda, cpu, rocm, mps, dml, npu) con el mismo protocolo."),
            code('nbui.backends()'),
            code('''
t = nbui.lanzar("codec", "bench", dry_run=DRY_RUN, limite=LIMITE, **DEV)
nbui.codec_resultados()
''')]
  if m in ("gpt2m", "unigram", "transicion", "congelado", "real"):
    return [md(r + "su fila de la tabla\nUna referencia de la tabla modelo → performance."), code(f'nbui.tabla_video(solo=[MODELO])')]
  return [md(r + "propia\nEste modelo todavía no tiene una sección propia: agregala en `scripts/make_notebooks.py` (función `propia`).")]


# ---------------------------------------------------------------- notebook de un modelo

def notebook_modelo(m):
  from src import toolkit as T
  p = perfil(m)
  try:
    acciones = tuple(T.acciones_de(m))
  except T.ToolkitError:
    acciones = ()
  opts = p["entrena"] if p["entrena"] is not None else {}
  run = run_de(m, opts) if m in T.kinds() else m
  tam = p["tam"]
  cab = md(f"# {p['titulo']}\n`python -m src.cli {m} <acción>` en un notebook: cada sección es una acción de la CLI, en el mismo orden y con los "
           f"mismos defaults, más una sección propia del modelo. {p['que']}\n\n"
           f"Guía completa: `{guia(m)}`. Acciones de `{m}`: {', '.join(f'`{a}`' for a in acciones) or 'ninguna'}.\n\n"
           "**Cómo se usa.** Con `DRY_RUN = True` (lo de la primera celda) cada acción solo imprime los comandos que correría, igual que `--dry-run`. "
           "Con `DRY_RUN = False` corre de verdad por `Toolkit`, en segundo plano (el kernel sigue libre), con candado de GPU y cola, y queda en "
           "`results/bitacora.jsonl` con su línea de CLI exacta. Lo que todavía no existe (resultados, checkpoints) se muestra como *pendiente*.")
  cfg = code(f'''
MODELO = "{m}"
{f'TAMANO = "{tam}"        # tamaño del modelo' if tam else ''}
RUN = "{run}"       # nombre de la corrida: <modelo>_<tamaño>[_<variante>][_s<semilla>]
DRY_RUN = True      # True: solo imprime los comandos (= --dry-run). False: los corre de verdad, en segundo plano
DEVICE = nbui.device_inicial()      # el que detecta el repo: cuda, rocm, mps, cpu (dml y npu: con ONNX Runtime)
DEV = nbui.device_kw(DEVICE)        # = --device
tk = Toolkit(MODELO)
print(MODELO, "| acciones:", ", ".join(tk.acciones), "| corrida:", RUN, "| device:", DEVICE, "| dry-run:", DRY_RUN)
'''.replace("\n\n", "\n"))
  celdas = [cab, code(SETUP), cfg]
  celdas += seccion_receta(m, p, acciones)
  celdas += seccion_entrenar(m, p, acciones)
  celdas += seccion_evaluar(m, p, acciones)
  celdas += seccion_bench(m, p, acciones)
  celdas += seccion_demo(m, p, acciones)
  celdas += seccion_exportar(m, p, acciones)
  celdas += propia(m, p)
  return armar(celdas, p["titulo"])


# ---------------------------------------------------------------- notebooks generales

def nb_resultados():
  c = [md("""
# Resultados: world models de conducción sobre commaVQ
La tabla **modelo → performance**: una fila por modelo y solo con datos actuales, leída de `results/`. Si un resultado todavía no existe, la
celda dice *pendiente* (nunca muestra un número viejo). Para producirlo: `python -m src.cli <modelo> evaluar` o la cola (`python -m src.cli cola correr`);
cada corrida queda en `results/bitacora.jsonl` con su comando exacto. El informe escrito está en `Docs/resultados.md`.
"""), code(SETUP),
       md("## 1. Estado de cada modelo\nQué se entrenó, evaluó y midió. Una fila por modelo de la cola; *pendiente* hasta que su resultado exista."),
       code("nbui.estado_modelos()"),
       md("## 2. Tabla de video\nCE, LPIPS y FVD a 1 s con IC por segmento (se remuestrean segmentos, no clips: los 4 clips de un segmento comparten ruta), "
          "FVD a 3 s, latencia del modelo (p50, p99) y p99 de la cadena. El **piso de FVD** (real contra real) es el valor que ningún modelo puede "
          "bajar; hay que leer la tabla contra él. Los latentes no entran acá: tienen su tabla propia."),
       code("nbui.tabla_video()"),
       md("## 3. Pareto calidad–latencia\nFVD a 1 s contra ms por frame, solo entre modelos generativos y con datos actuales; la línea punteada vertical "
          "son los 50 ms de la cámara y la horizontal, el piso de FVD."),
       code("nbui.pareto()"),
       md("## 4. Calidad según el horizonte\nLPIPS contra el video real en cada frame generado (3 s)."),
       code("nbui.horizonte()"),
       md("## 5. Muestras\nLos últimos videos que generó `demo` (real a la izquierda, predicho a la derecha). Se generan desde `notebooks/demo_clase.ipynb` "
          "o con `python -m src.cli <modelo> demo`."),
       code("nbui.muestras(3)"),
       md("## 6. Tabla latente\nLos modelos latentes (LeWM, V-JEPA 2, V-JEPA 2-AC) no generan video: se miden por error latente contra *copiar el último "
          "latente*, probes contra el encoder sin entrenar y contra píxeles, sorpresa y plan con CEM. Cada columna lleva su referencia al lado."),
       code("nbui.tabla_latente()"),
       md("## 7. Tabla de fidelidad\nPara cada modelo, cada decisión con su fuente y su categoría: *fiel*, *datos* (lo fuerza el dato de comma), "
          "*presupuesto* (lo fuerza el cómputo) o *propio*."),
       code("""
from src import recipes
nbui.tabla_fidelidad(recipes.fidelidad_kinds())
"""),
       md("## 8. Corridas, cola y bitácora"),
       code("nbui.estado_corridas()\nnbui.panel_cola()\nnbui.panel_bitacora(10)"),
       md("## 9. Regenerar las tablas y figuras\n`python -m src.cli reporte` arma `results/configs.json` y las figuras con lo que hay en `results/`. "
          "Con `DRY_RUN = False` lo corre de verdad (es CPU y queda en la bitácora)."),
       code("""
DRY_RUN = True
nbui.general("reporte", dry_run=DRY_RUN)
""")]
  return armar(c, "Resultados")


def nb_latentes():
  c = [md("""
# Modelos latentes: V-JEPA 2, V-JEPA 2-AC y LeWM
Tres modelos que **no generan video**: aprenden un espacio de embeddings y predicen en él. No tiene sentido medirlos con FVD, así que se los compara con
otras cosas: error de predicción contra *copiar el último latente*, probes de pose contra un encoder sin entrenar y contra píxeles, sorpresa ante un
corte del video y recuperación de acciones con CEM. Todo lo que sigue sale de `results/latent_<corrida>.json`; sin él, *pendiente*.
"""), code(SETUP),
       md("""
## 1. Qué es cada uno
| | V-JEPA 2 | V-JEPA 2-AC | LeWM |
|---|---|---|---|
| Qué aprende | un encoder ViT que predice los tubelets tapados | un predictor de latentes condicionado por acciones, sobre un encoder congelado | un encoder + predictor de un embedding por frame |
| Objetivo | L1 contra un target EMA con stop-grad | L1 con teacher forcing y rollout de 2 pasos | error de predicción + SIGReg (evita el colapso) |
| Acciones | no | sí (pose y diferencia de pose) | sí (5 poses apiladas) |
| Planifica | no | CEM (800 muestras, top-10) | CEM |
| CLI | `vjepa2` | `vjepa2_ac` (necesita un `vjepa2`) | `lewm` |
"""),
       md("## 2. Receta y fidelidad de cada uno"),
       code("""
from src import recipes
nbui.tabla_fidelidad(["vjepa2", "vjepa2_ac", "lewm"])
"""),
       md("## 3. Máscaras de V-JEPA 2\nEl predictor solo ve los tokens visibles de un clip tapado en ~90 % con bloques 3D; tiene que adivinar el resto en el espacio "
          "latente. No dependen de los pesos."),
       code("nbui.mascaras_vjepa2(semilla=0)"),
       md("## 4. Comparación\nUna fila por modelo, con sus referencias al lado."),
       code("nbui.tabla_latente()"),
       md("### Error por horizonte\nContra copiar el último latente y contra la acción nula (si la acción importa, la nula es peor que la real)."),
       code("""
for run in ("lewm", "vjepa2_ac"):
  nbui.horizonte_latente(run)
"""),
       md("### Probes de pose\nR² de velocidad y giro sobre el encoder entrenado, el encoder sin entrenar y los píxeles."),
       code("""
for run in ("lewm", "vjepa2_s"):
  nbui.probes_fig(run)
"""),
       md("### Sorpresa\nEl error de predicción salta cuando el video se corta (teletransporte o salto de 2 s) si el modelo entiende la escena."),
       code("""
for run in ("lewm", "vjepa2_ac"):
  nbui.sorpresa_fig(run)
"""),
       md("### Plan con CEM\nError de la pose planeada contra repetir la última acción y contra la media de train."),
       code("""
for run in ("lewm", "vjepa2_ac"):
  nbui.cem_tabla(run)
"""),
       md("## 5. Correrlos\nCada uno con la misma API de siempre (`DRY_RUN = True` imprime los comandos; `False` los corre en segundo plano y queda en la bitácora). "
          "Los notebooks por modelo (`notebooks/modelos/lewm.ipynb`, `vjepa2.ipynb`, `vjepa2_ac.ipynb`) tienen todas las acciones."),
       code("""
DRY_RUN = True
for modelo, opts in (("vjepa2", dict(tamano="s")), ("vjepa2_ac", dict(encoder="vjepa2_s")), ("lewm", {})):
  print("##", modelo)
  nbui.lanzar(modelo, "entrenar", dry_run=DRY_RUN, **opts)
  nbui.lanzar(modelo, "evaluar", dry_run=DRY_RUN, limite=LIMITE)
""")]
  return armar(c, "Modelos latentes")


def nb_inferencia():
  c = [md("""
# Inferencia por tecnología
Pruebas chicas de la cadena de punta a punta (cámara → encoder del codec → modelo → decoder) en lo que tenga la máquina: CUDA (RTX 3080 Ti), CPU, ROCm,
Apple Silicon (MPS), DirectML y NPU Vitis AI (laptop Ryzen AI). Todo por la CLI: `python -m src.cli <modelo> bench --cadena --device X`.
"""), code(SETUP),
       md("## 1. Backends disponibles\nLa detección sale de `src/infer/backends.py`. Un backend nuevo se registra ahí y aparece en esta tabla."),
       code("""
DISP = nbui.backends()                  # tabla de backends y lista de los disponibles
DEVICE = nbui.device_inicial(DISP)      # el que usa el repo por defecto (cuda, rocm o mps si hay GPU; si no, cpu)
DEV = nbui.device_kw(DEVICE)
print("uso:", DEVICE)
"""),
       md("## 2. Configuración\n~200 frames por defecto; `MODELO` es cualquier modelo con `bench` (con `ALEATORIO = True` mide la arquitectura sin entrenar: la latencia no "
          "depende de los pesos). En el modo prueba de `scripts/run_notebooks.py` solo imprime los comandos."),
       code("""
MODELO = "genie"          # gpt, genie, maskgit, convlstm_px, lewm, vjepa2... o codec
RUN = "genie_s"           # la corrida cuya arquitectura se mide
ALEATORIO = True          # pesos sin entrenar (mismo costo)
FRAMES = LIMITE or 200    # frames medidos (--limite)
PASOS = "4"               # solo genie y maskgit: pasos de decodificación
DRY_RUN = bool(os.environ.get("ML2_NB_LIMITE"))    # a mano corre de verdad; el modo prueba solo imprime
"""),
       md("## 3. La cadena, en bucle cerrado y a ritmo de cámara\n`--cadena` mide las tres columnas del protocolo: modelo solo, cadena cerrada y cadena a 50 ms entre "
          "frames. Si la GPU está ocupada, la acción queda en la cola."),
       code("""
extra = dict(pasos=PASOS) if MODELO in ("genie", "maskgit") else {}
t = nbui.lanzar(MODELO, "bench", dry_run=DRY_RUN, segundo_plano=False, cadena=True, ritmo=50, limite=FRAMES,
                run=RUN, aleatorio=ALEATORIO, **extra, **DEV)
"""),
       md("## 4. p50 y p99\nPor columna del protocolo y la mediana de cada etapa (encoder, modelo, decoder)."),
       code("nbui.latencia_etapas()"),
       md("El piso de ~20 ms a ritmo de cámara se diagnostica con `--diagnostico` (variantes del temporizador y de la espera):"),
       code("""
t = nbui.lanzar(MODELO, "bench", dry_run=True, cadena=True, diagnostico=True, limite=FRAMES, run=RUN, aleatorio=ALEATORIO, **extra, **DEV)
"""),
       md("## 5. Nodos que caen a la CPU con ONNX Runtime\nEn DirectML y en la NPU, los nodos que el provider no soporta se ejecutan en la CPU. `bench_ort --profile` "
          "cuenta los nodos y el tiempo por provider para el encoder y el decoder del codec."),
       code("""
for parte in ("cpu", "dml", "npu"):
  if parte in DISP:
    nbui.ort_perfil(enc=parte, dec=parte, frames=min(FRAMES, 100), dry_run=DRY_RUN)
nbui.nodos_cpu()
"""),
       md("## 6. Un mp4 de lo que predice\n`demo` con el mismo modelo y dispositivo: real a la izquierda, predicho a la derecha."),
       code("""
if not DRY_RUN:
  nbui.imaginar(RUN, aleatorio=ALEATORIO, device=DEVICE, pasos=PASOS if MODELO in ("genie", "maskgit") else None)
else:
  nbui.pendiente("el mp4 de la demo", "con DRY_RUN = False se genera y se muestra acá")
""")]
  return armar(c, "Inferencia")


def nb_colab():
  c = [md("""
# Colab (T4): el repo en una GPU prestada
Lo mismo que los notebooks locales, en la T4 de Colab: clona el repo, baja lo que falta, mide la inferencia con la CLI y copia los JSON a Drive para no perderlos
al cerrar la sesión. Elegí *Entorno de ejecución → Cambiar tipo → T4 GPU*.
"""), md("## 1. Preparación\nClona el repo si no está (en local no hace nada) e instala lo que falta."),
       code("""
import os, subprocess, sys
from pathlib import Path

REPO = "https://github.com/bpeco/ml2-gen-world-model.git"   # el repo del equipo (en Colab; en local no se usa)
if not any((d / "src" / "common.py").exists() for d in [Path.cwd(), *Path.cwd().parents]):
  if not REPO:
    raise SystemExit("completá REPO con la URL del repo, o abrí el notebook dentro de una copia del repo")
  if not Path("ml2-gen-world-model").exists():
    subprocess.run(["git", "clone", REPO], check=True)
  os.chdir("ml2-gen-world-model")
  subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
"""), code(SETUP),
       md("## 2. Dispositivo y datos\nLos checkpoints no se distribuyen: cada uno entrena lo que necesita (`python -m src.cli <modelo> entrenar`) o los trae del "
          "almacén compartido (`python -m src.store pull <corrida> --dir RUTA`)."),
       code("""
DISP = nbui.backends()
DEVICE = nbui.device_inicial(DISP)
DEV = nbui.device_kw(DEVICE)
nbui.catalogo_corridas()
"""),
       md("## 3. Latencia de punta a punta en la T4\nUna cadena por modelo, con la CLI (`bench --cadena`); sin checkpoint mide la arquitectura sin entrenar."),
       code("""
DRY_RUN = bool(os.environ.get("ML2_NB_LIMITE"))
FRAMES = LIMITE or 200
for modelo, run, extra in (("genie", "genie_s", dict(pasos="4")), ("convlstm_px", "convlstm_px", {}), ("lewm", "lewm", {})):
  nbui.lanzar(modelo, "bench", dry_run=DRY_RUN, segundo_plano=False, cadena=True, ritmo=50, limite=FRAMES, run=run, aleatorio=True, **extra, **DEV)
nbui.latencia_etapas()
"""),
       md("## 4. Calidad en línea y con la cadena\n`evaluar --cadena` mide la calidad con el codec real en el medio (necesita los checkpoints)."),
       code("""
for modelo, run in (("genie", "genie_s"), ("convlstm_px", "convlstm_px")):
  nbui.lanzar(modelo, "evaluar", dry_run=True, cadena=True, run=run, limite=LIMITE, **DEV)
nbui.tabla_video()
"""),
       md("## 5. Copiar los JSON a Drive\nTodo `results/` (sin los cachés pesados) a tu Drive, para llevarlo al repo local. En local no hace nada."),
       code("""
try:
  from google.colab import drive
  drive.mount("/content/drive")
  import shutil
  destino = Path("/content/drive/MyDrive/ml2-gen-world-model/results")
  shutil.copytree("results", destino, dirs_exist_ok=True, ignore=shutil.ignore_patterns("cache", "rollouts", "*.npy"))
  print("copiado a", destino)
except ImportError:
  print("no es Colab: results/ ya está en el repo local")
""")]
  return armar(c, "Colab")


def nb_demo_clase():
  c = [md("""
# Demo para la clase
Pensado para correrse en vivo: **configurá la celda 2 y usá *Run all***. Elegí modelos, dispositivo y de dónde sale el video; el notebook corre la acción `demo`
por `Toolkit` (la misma que `python -m src.cli <modelo> demo`, queda en la bitácora) y muestra el resultado acá mismo.

Qué se ve: el **video real a la izquierda** y lo que **predice el modelo a la derecha**; los frames de contexto (el modelo los ve) tienen borde gris y los que
imagina, borde naranja. Los modelos latentes (LeWM, V-JEPA 2, V-JEPA 2-AC) muestran su figura: decodificación, máscaras y curva de error.
"""), code(SETUP),
       md("## 1. Configuración\nTodo lo que se elige está en esta celda. Un modelo sin entrenar no rompe nada: aparece marcado y se explica cómo entrenarlo."),
       code("""
# ============== ELEGÍ ACÁ ==============
PRUEBA = bool(os.environ.get("ML2_NB_LIMITE"))     # lo pone scripts/run_notebooks.py: modelos sin entrenar y clips cortos, para probar en CPU

MODELOS = ["genie_s", "convlstm_px", "gpt_s_kd"]   # 1 a 3 corridas; las latentes: "lewm", "vjepa2_s", "vjepa2_ac"
DEVICE = nbui.device_inicial()                      # el que detecta el repo: cuda, rocm, mps o cpu (dml y npu: ver inferencia.ipynb)
FUENTE = "ejemplo"      # "ejemplo" (el video de cámara de comma), "val" (un segmento de val) o "mp4" (un video propio)
SEGMENTO, INICIO = 0, 0           # con FUENTE = "val": qué segmento de val y desde qué frame
MP4 = "mi_video.mp4"              # con FUENTE = "mp4"
PASOS = 8                         # pasos de decodificación de genie y maskgit
# =======================================
if PRUEBA:
  MODELOS, PASOS, DEVICE = ["maskgit_s", "genie_s"], 1, "cpu"
CTX, HORIZON = (4, 2) if PRUEBA else (20, 60)       # 1 s de contexto real y 3 s imaginados (a 20 fps)
CLIP = dict(fuente=FUENTE, segmento=SEGMENTO, inicio=INICIO, mp4=MP4, device=DEVICE, pasos=PASOS, aleatorio=PRUEBA)
if PRUEBA:    # el modo prueba no deja nada en results/ ni en la bitácora del repo: todo va a una carpeta temporal
  import tempfile
  from src import rutas
  rutas.RESULTS = Path(tempfile.mkdtemp(prefix="ml2_prueba_")) / "results"

print("dispositivo:", DEVICE, "| fuente:", FUENTE, "| modo prueba (pesos sin entrenar):", PRUEBA)
nbui.backends()
"""),
       md("### Modelos disponibles\n*entrenada* = hay `checkpoints/<corrida>/final.pt` o está en `results/checkpoints.json`. Las demás se saltean con el comando que las entrena."),
       code("""
nbui.catalogo_corridas(tuple(dict.fromkeys(MODELOS + ["genie_s", "gpt_s_kd", "convlstm_px", "lewm", "vjepa2_s", "vjepa2_ac"])))
"""),
       md("## 2. La demo\nCada modelo sobre el clip elegido: real contra predicho, con su latencia por frame generado."),
       code("""
for run in MODELOS[:1]:
  nbui.demo(run, ctx=CTX, horizon=HORIZON, frames=(CTX + HORIZON) * (1 if PRUEBA else 2), **CLIP)
"""),
       md("## 3. Dos o tres modelos lado a lado\nMismo clip, mismos frames de contexto; abajo, la latencia por frame de cada uno (la de la demo incluye bajar el frame a la CPU)."),
       code("""
nbui.comparar(MODELOS[:3], ctx=CTX, horizon=HORIZON, frames=(CTX + HORIZON) * (1 if PRUEBA else 2), **CLIP)
"""),
       md("## 4. Imaginar\nEl modelo **ve 1 s real e imagina 3 s**. En el video, el borde gris marca el contexto real y el naranja, lo imaginado (`imagina +k`, con k = frames a futuro)."),
       code("""
nbui.imaginar(MODELOS[0], ctx=CTX, horizon=HORIZON, **CLIP)
"""),
       md("## 5. Si falta algo\n- **Un modelo sin entrenar**: la tabla de la sección 1 dice el comando (`python -m src.cli <modelo> entrenar ...`); `nbui.lanzar(..., dry_run=False)` lo lanza desde acá, "
          "en segundo plano.\n- **Otro dispositivo**: cambiá `DEVICE`; si no está disponible, la CLI lo avisa.\n- **Un video propio**: `FUENTE = \"mp4\"` y la ruta en `MP4` (queda fuera de la distribución de entrenamiento).")]
  return armar(c, "Demo para la clase")


GENERALES = {"resultados": nb_resultados, "latentes": nb_latentes, "inferencia": nb_inferencia, "colab": nb_colab, "demo_clase": nb_demo_clase}


# ---------------------------------------------------------------- todo junto

def generar():
  """{ruta relativa a notebooks/: texto} de todo lo que genera este script."""
  from src import toolkit as T
  out = {f"modelos/{m}.ipynb": notebook_modelo(m) for m in T.modelos()}
  out.update({f"{k}.ipynb": f() for k, f in GENERALES.items()})
  return out


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--check", action="store_true", help="falla si los notebooks del disco difieren de los que se generarían")
  a = ap.parse_args(argv)
  gen = generar()
  malos = []
  for rel, texto in gen.items():
    f = NB / rel
    actual = f.read_text(encoding="utf-8") if f.exists() else None
    if actual != texto:
      malos.append(rel)
      if not a.check:
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "w", encoding="utf-8", newline="\n") as fh:
          fh.write(texto)
  sobran = sorted(str(p.relative_to(NB)).replace("\\", "/") for p in (NB / "modelos").glob("*.ipynb") if f"modelos/{p.name}" not in gen) \
    if (NB / "modelos").exists() else []
  if a.check:
    for rel in malos:
      print(f"difiere o falta: notebooks/{rel}")
    for rel in sobran:
      print(f"sobra (no es de ningún modelo registrado): notebooks/{rel}")
    if malos or sobran:
      print("regenerá con: python scripts/make_notebooks.py")
      return 1
    print(f"{len(gen)} notebooks al día")
    return 0
  print(f"{len(gen)} notebooks ({len(malos)} escritos)" + (f"; sobran sin generar: {', '.join(sobran)}" if sobran else ""))
  return 0


if __name__ == "__main__":
  sys.exit(main())
