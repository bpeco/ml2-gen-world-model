"""Contrato común de todos los modelos registrados en src/models/__init__.py (KINDS), en CPU y con configuración mínima.

Es la versión común de lo que ya prueban los tests por modelo (test_gpt, test_maskgit, test_genie, test_convlstm_px,
test_latent, test_vjepa, test_infer): un modelo nuevo entra al registro y estas pruebas lo cubren sin escribir nada más.
Cada modelo tiene que:
  1. declarar recipe() (con batch_fuente), FIDELIDAD (con fuente y categoría), ACCIONES y EVALUACION;
  2. no filtrar el futuro (donde aplica);
  3. sobreajustar un batch chico;
  4. reanudar bit a bit desde su last.pt (src.train);
  5. dar con su stream de inferencia lo mismo que con generate / forward por lotes (donde aplica);
  6. dar un plan válido para cada acción que declara (Toolkit(...).accion(..., dry_run=True)).

Cómo sabe el contrato armar cada modelo: ADAPTADORES (abajo) trae la configuración mínima y las pruebas específicas de los
modelos que existen; un modelo nuevo las declara en su módulo con CONTRATO = dict(config={...}, ...) (ver
src/models/_plantilla.py) y usa las pruebas genéricas de los modelos de tokens. Qué claves entiende CONTRATO: ver ADAPTADOR_TOKENS.
"""
import importlib
import importlib.util
import json
import sys
import types

import numpy as np
import pytest
import torch

from src import pixels, rutas
from src import recipes as R
from src.common import TPF, with_bos
from src.infer.stream import make_stream
from src.models import KINDS, build, model_class
from src.toolkit import ACCIONES_TODAS, Toolkit
from src.train import main as train_main

KEYS_FIDELIDAD = ("campo", "aquí", "categoría", "fuente")
GUIA = "(copiá src/models/_plantilla.py: tiene un ejemplo mínimo de todo lo que el contrato pide)"


# ---------------------------------------------------------------- datos de juguete

def frames(B=2, F=4, seed=1):
  return torch.randint(0, 1024, (B, F, 128), generator=torch.Generator().manual_seed(seed))


def camara(escala):
  """Cámara de juguete (sin los pesos del decoder de comma): cada token pinta un cuadrado de escala x escala."""
  def render(tok):
    pal = (torch.arange(1024)[:, None] * torch.tensor([37, 91, 53]) % 256).to(torch.uint8).to(tok.device)
    img = pal[tok.long()].view(*tok.shape[:-1], 8, 16, 3)
    return img.repeat_interleave(escala, -3).repeat_interleave(escala, -2).contiguous()
  return render


def poses(B, F, n, seed=0):
  return torch.randn(B, F, n, generator=torch.Generator().manual_seed(seed))


@pytest.fixture(scope="module")
def datos(tmp_path_factory):
  """tokens (60 segmentos x 100 frames) y poses (60 x 100 x 6, el último frame sin pose) para entrenar con src.train."""
  d = tmp_path_factory.mktemp("datos")
  rng = np.random.default_rng(0)
  np.save(d / "tok.npy", rng.integers(0, 1024, (60, 100, 128)).astype(np.int16))
  p = rng.normal(size=(60, 100, 6)).astype(np.float32)
  p[:, -1] = np.nan
  np.save(d / "pos.npy", p)
  return d / "tok.npy", d / "pos.npy"


# ---------------------------------------------------------------- pruebas genéricas de los modelos de tokens

def logits_por_frame(m, x):
  """(B, F, 129, V) logits de la convención comma sobre frames x (B, F, 128)."""
  with torch.no_grad():
    return m.logits(with_bos(x)).float().view(x.shape[0], x.shape[1], TPF, -1)


def causal_tokens(m, **_):
  """Cambiar el frame f no mueve los logits de los frames < f, y sí los de algún frame posterior."""
  x = frames(2, 4)
  base = logits_por_frame(m, x)
  for f in range(4):
    x2 = x.clone()
    x2[:, f] = (x2[:, f] + 1 + torch.randint(0, 1000, x2[:, f].shape, generator=torch.Generator().manual_seed(f))) % 1024
    out = logits_por_frame(m, x2)
    if f:
      assert (out[:, :f] - base[:, :f]).abs().max() < 1e-5, f"fuga del futuro: cambiar el frame {f} movió los logits de frames anteriores"
    if f < 3:
      assert (out[:, f + 1:] - base[:, f + 1:]).abs().max() > 1e-7, f"el frame {f} no influye en los siguientes: ¿el contexto se ignora?"


def entrada_tokens(m, B=2, F=4):
  return (frames(B, F),), {}


def stream_tokens(m, steps=None, **_):
  """El stream (observe + imagine) da los mismos tokens que generate con la misma semilla, a T = 0 y a T = 1."""
  ctx = frames(2, min(3, m.ctx_frames), seed=3)
  for T in (0.0, 1.0):
    s = make_stream(m, temperature=T, steps=steps, seed=5, graphs=False)
    for f in range(ctx.shape[1]):
      s.observe(ctx[:, f])
    via_stream = torch.stack(list(s.imagine(3)), 1)
    g = torch.Generator().manual_seed(5)
    via_generate = m.rollout(ctx, 3, temperature=T, steps=steps, generator=g)
    assert torch.equal(via_stream, via_generate), f"el stream no coincide con generate a T = {T}"


# Qué entiende el contrato de cada modelo (CONTRATO en el módulo del modelo, o ADAPTADORES abajo). Todo es opcional salvo config.
ADAPTADOR_TOKENS = dict(
  config={},                  # campos de Clase.Config con los que se arma el modelo chico (obligatorio)
  config_train=None,          # Config para src.train (si difiere: p. ej. sin n_pose, que train.py fija); default: config
  size=None,
  parche=None,                # parche(monkeypatch): cámara de juguete para los modelos que ven píxeles
  entrada=entrada_tokens,     # entrada(m) -> (args, kwargs) de train_loss
  causal=causal_tokens,       # causal(m) o None
  causal_razon="",            # por qué no aplica, si causal es None
  stream=stream_tokens,       # stream(m) o None
  stream_razon="",
  plan_opts={},               # opciones extra de Toolkit.plan por acción (p. ej. entrenar=dict(encoder=...))
  stream_kw={},               # kwargs para stream (p. ej. steps=2 en los que decodifican por pasos)
  tuning=None,                # tuning(m): ajuste de los pesos iniciales para que las pruebas discriminen (default: ruido chico)
  sobreajuste=dict(pasos=80, lr=3e-3, ratio=0.7),
  train=dict(frames=4, frameskip=None, warmup=True, poses=False),  # cómo entrenar con src.train: frames por muestra, etc.
)


def modelar(kind, ad, monkeypatch):
  if ad["parche"]:
    ad["parche"](monkeypatch)
  torch.manual_seed(0)
  m = build(kind, dict(ad["config"]), ad["size"]).eval()
  if ad["tuning"]:
    ad["tuning"](m)
  else:
    g = torch.Generator().manual_seed(7)
    for p in m.parameters():  # ruido chico: pesos iniciales (ceros, AdaLN-zero) que dejarían las pruebas sin discriminar
      if p.requires_grad:
        p.data.add_(0.05 * torch.randn(p.shape, generator=g))
  return m


# ---------------------------------------------------------------- adaptadores de los modelos que existen

def _ctx_px(m, n):
  return m.cfg.height, m.cfg.width


def _entrada_px(m, B=2, F=5):
  g = torch.Generator().manual_seed(1)
  px = torch.randint(0, 256, (B, F, m.cfg.height, m.cfg.width, 3), generator=g, dtype=torch.uint8)
  return ((px > 127).to(torch.uint8) * 255,), {}


def _stream_convlstm(m, **_):
  g = torch.Generator().manual_seed(1)
  ctx = torch.randint(0, 256, (2, 6, m.cfg.height, m.cfg.width, 3), generator=g, dtype=torch.uint8)
  s = make_stream(m)
  for t in range(6):
    s.observe_pixels(ctx[:, t])
  ref = m.rollout(ctx, 4)
  assert torch.equal(s.predict(), ref[:, 0])
  assert torch.equal(torch.stack(list(s.imagine(4)), 1), ref), "el stream no coincide con rollout"


def _causal_convlstm(m, **_):
  """El estado del encoder tras t frames no ve el frame t+1 (la fuga posible: la recurrencia mirando hacia adelante)."""
  g = torch.Generator().manual_seed(2)
  win = torch.rand(1, 3, 3, m.cfg.height, m.cfg.width, generator=g)
  win2 = win.clone()
  win2[:, 2] = torch.rand(3, m.cfg.height, m.cfg.width, generator=g)
  p = m.cfg.patch
  x, x2 = (torch.nn.functional.pixel_unshuffle(w.flatten(0, 1), p).unflatten(0, (1, 3)) for w in (win, win2))
  with torch.no_grad():
    s2, s2b, s3, s3b = m.encode(x[:, :2]), m.encode(x2[:, :2]), m.encode(x), m.encode(x2)
  assert all(torch.equal(a[0], b[0]) for a, b in zip(s2, s2b)), "fuga: el estado tras 2 frames ve el 3.º"
  assert not all(torch.equal(a[0], b[0]) for a, b in zip(s3, s3b)), "el 3.er frame no influye en el estado"


def _lewm_causal(m, **_):
  """El predictor no ve el futuro: cambiar z o la acción del paso 2 no mueve las predicciones de los pasos 0 y 1."""
  D = m.cfg.dim
  g = torch.Generator().manual_seed(3)
  z, a = torch.randn(1, 3, D, generator=g), torch.randn(1, 3, D, generator=g)
  z2, a2 = z.clone(), a.clone()
  z2[0, 2] = torch.randn(D, generator=g)
  a2[0, 2] = torch.randn(D, generator=g)
  with torch.no_grad():
    base, dz, da = m.predict(z, a), m.predict(z2, a), m.predict(z, a2)
  assert (base[:, :2] - dz[:, :2]).abs().max() == 0 and (base[:, :2] - da[:, :2]).abs().max() == 0, "fuga del futuro en el predictor"
  assert (base[:, 2] - dz[:, 2]).abs().max() > 1e-6, "el estado del paso 2 no influye en su predicción"


def _wake_lewm(m):
  g = torch.Generator().manual_seed(3)
  for n, p in m.named_parameters():
    p.data.add_(0.05 * torch.randn(p.shape, generator=g))
    if "adaLN_modulation" in n:  # saca a AdaLN-zero de la identidad (si no, la acción no influye al inicializar)
      p.data = torch.randn(p.shape, generator=g) * 0.2


def _entrada_poses(n_pose):
  def entrada(m, B=2, F=4):
    return (frames(B, F),), {"poses": poses(B, F, n_pose)}
  return entrada


def _stream_lewm(m, **_):
  """predict() del stream = el paso del modelo sobre los últimos frames codificados, con sus acciones y la acción pedida."""
  K, B = m.cfg.frameskip, 2
  tok, pose = frames(B, 3 * K, seed=4), poses(B, 3 * K, 6, seed=2)
  s = make_stream(m, graphs=False)
  for f in range(tok.shape[1]):
    s.observe(tok[:, f], pose[:, f])
  nxt = poses(B, 1, 6, seed=9)[:, 0]
  got = s.predict(pose=nxt)
  z = m.encode(tok[:, ::K])
  stack = torch.cat([torch.stack([pose[:, K * i:K * i + K].reshape(B, -1) for i in range(2)], 1), nxt.repeat(1, K)[:, None]], 1)
  assert torch.allclose(got, m._next(z, m.action_embed(stack, B, 3, "cpu")), atol=1e-6), "el stream no coincide con el forward por lotes"


def _stream_ac(m, **_):
  """predict(acción) del stream = predecir() por lotes sobre los mismos frames (uno de cada 5), estados y acciones."""
  B, F, K = 2, 3, 5
  g = torch.Generator().manual_seed(1)
  cam = torch.randint(0, 256, (B, F * K, m.cfg.height, m.cfg.width, 3), generator=g, dtype=torch.uint8).permute(1, 0, 2, 3, 4)
  P = poses(B, F * K, m.cfg.n_pose, seed=0)
  m.set_pose_stats(torch.randn(m.cfg.n_pose, generator=g) * 0.1, torch.rand(m.cfg.n_pose, generator=g) + 0.5)
  s = make_stream(m, graphs=False)
  for i in range(F * K):
    s.observe_pixels(cam[i], P[:, i])
  accion = torch.randn(B, m.cfg.action_dim, generator=g)
  zh = s.predict(accion)
  z = m.codificar(m.clip(cam[::K].permute(1, 0, 2, 3, 4)))
  st, ac = m.estados_acciones(P[:, ::K], B, F, z.device)
  ref = m.predecir(z, torch.cat([ac, accion[:, None]], 1), st)[:, -1]
  assert torch.allclose(zh, ref, atol=1e-5), "el stream no coincide con el forward por lotes"


def _causal_ac(m, **_):
  """El predictor AC no ve el futuro: cambiar el frame t (tokens, acción, estado) no mueve las salidas de frames < t."""
  A, N, D = m.cfg.action_dim, m.grid[0] * m.grid[1], m.encoder.embed_dim
  g = torch.Generator().manual_seed(1)
  z, a, s = torch.randn(2, 4, N, D, generator=g), torch.randn(2, 4, A, generator=g), torch.randn(2, 4, A, generator=g)
  with torch.no_grad():
    y = m.predecir(z, a, s)
    for t in range(4):
      z2, a2, s2 = z.clone(), a.clone(), s.clone()
      z2[:, t] += torch.randn(z2[:, t].shape, generator=g)
      a2[:, t] += 1
      s2[:, t] += 1
      y2 = m.predecir(z2, a2, s2)
      assert t == 0 or torch.allclose(y2[:, :t], y[:, :t], atol=1e-5), f"fuga del futuro: cambiar el frame {t} movió salidas anteriores"
      assert not torch.allclose(y2[:, t:], y[:, t:], atol=1e-4), f"el frame {t} no influye en sus salidas"


def _p(modulo, nombre):
  return lambda mp: mp.setattr(importlib.import_module(modulo), nombre, camara(2 if nombre == "render" else 16))


PX_CP = 2          # convlstm_px: 16 x 32 (cada token pinta 2 x 2)
PX_VJ = 4          # vjepa: 32 x 64
ADAPTADORES = {
  # src.train monitorea la CE sobre ventanas de 20 frames: el contexto de cada modelo tiene que alcanzar para 20
  "gpt": dict(config=dict(n_layer=2, n_head=2, dim=32, dropout=0.0), train=dict(frames=4, warmup=False)),
  "maskgit": dict(config=dict(n_layer=2, n_head=2, dim=32), stream_kw=dict(steps=2), train=dict(frames=4, warmup=True)),
  "genie": dict(config=dict(n_layer=2, n_head=2, dim=32, frames=20), stream_kw=dict(steps=2), train=dict(frames=4, warmup=True)),
  "convlstm_px": dict(
    config=dict(height=16, width=32, patch=4, hidden=(8, 6), kernel=3, n_in=3, n_out=2), entrada=_entrada_px, causal=_causal_convlstm,
    stream=_stream_convlstm, parche=lambda mp: mp.setattr("src.models.convlstm_px.render", camara(PX_CP)),
    sobreajuste=dict(pasos=250, lr=3e-3, ratio=0.7), train=dict(frames=5, warmup=False)),
  "lewm": dict(
    config=dict(dim=32, enc_layers=1, enc_heads=2, enc_mlp=64, proj_hidden=64, pred_layers=2, pred_heads=4, pred_dim_head=8,
                pred_mlp=64, pred_dropout=0.0, num_proj=64),
    parche=lambda mp: mp.setattr("src.models.lewm.render_tokens", camara(16)), entrada=_entrada_poses(30), causal=_lewm_causal,
    stream=_stream_lewm, tuning=_wake_lewm, sobreajuste=dict(pasos=80, lr=3e-3, ratio=0.9),
    train=dict(frames=4, warmup=True, poses=True)),
  "vjepa2": dict(
    config=dict(encoder="tiny", embed_dim=48, depth=2, num_heads=3, pred_dim=24, pred_depth=2, pred_heads=2, frames=4, height=32,
                width=64, grad_ckpt=False, bank=3),
    parche=lambda mp: mp.setattr(pixels, "render", camara(PX_VJ)),
    causal=None, causal_razon="el predictor es de máscaras 3D (bidireccional por diseño); que el target no se filtre al contexto lo "
                              "cubre test_vjepa.test_loss_solo_en_enmascarados_y_conjuntos_independientes",
    stream=None, stream_razon="V-JEPA 2 preentrenado no genera: su inferencia es solo el encoder (sin stream)",
    sobreajuste=dict(pasos=100, lr=3e-3, ratio=0.95), train=dict(frames=4, warmup=True, poses=False)),
  "vjepa2_ac": dict(
    config=dict(encoder="tiny", embed_dim=48, depth=2, num_heads=3, pred_dim=24, pred_depth=2, pred_heads=2, frames=4, height=32,
                width=64, grad_ckpt=False, n_pose=12),
    config_train=dict(encoder="tiny", embed_dim=48, depth=2, num_heads=3, pred_dim=24, pred_depth=2, pred_heads=2, frames=4,
                      height=32, width=64, grad_ckpt=False),
    plan_opts=dict(entrenar=dict(encoder="vjepa2_s"), continuar=dict(encoder="vjepa2_s"), demo=dict(encoder="vjepa2_s")),
    parche=lambda mp: mp.setattr(pixels, "render", camara(PX_VJ)), entrada=_entrada_poses(12), causal=_causal_ac, stream=_stream_ac,
    sobreajuste=dict(pasos=60, lr=1e-3, ratio=0.9), train=dict(frames=4, warmup=True, poses=True)),
}


def adaptador(kind):
  """Adaptador del modelo: CONTRATO del módulo (sobre el genérico de tokens) o el de ADAPTADORES; si no hay, falla diciendo qué hacer."""
  mod = importlib.import_module(KINDS[kind][0])
  base = dict(ADAPTADOR_TOKENS)
  if hasattr(mod, "CONTRATO"):
    return {**base, **mod.CONTRATO}
  if kind in ADAPTADORES:
    return {**base, **ADAPTADORES[kind]}
  pytest.fail(f"{kind}: no sé armar este modelo para el contrato. Declará CONTRATO = dict(config={{...}}) en {mod.__name__} "
              f"con la configuración mínima (CPU, segundos) {GUIA}", pytrace=False)


# Brechas conocidas de modelos que existen (el contrato las encontró; las arregla el dueño del modelo): xfail mientras duren, y
# falla en cuanto se arreglan para que se saque la entrada de acá.
BRECHAS = {}  # (prueba, kind) -> motivo, para brechas conocidas de modelos; vacío: todos cumplen el contrato


def con_brecha(prueba, kind, fn):
  motivo = BRECHAS.get((prueba, kind))
  if not motivo:
    return fn()
  try:
    fn()
  except AssertionError:
    pytest.xfail(motivo)
  pytest.fail(f"{kind}: la brecha conocida ya no existe ({motivo}); sacala de BRECHAS")


# ---------------------------------------------------------------- 1. declaraciones: receta, fidelidad, acciones, evaluación

def verificar_declaraciones(kind):
  """Chequea lo que un modelo tiene que declarar. Cada falla dice qué modelo, qué falta y cómo se arregla."""
  mod = importlib.import_module(KINDS[kind][0])
  cls = model_class(kind)
  falta = f"{kind} ({mod.__name__})"
  assert hasattr(cls, "recipe"), f"{falta}: la clase no define recipe(): sin receta de entrenamiento no hay corrida {GUIA}"
  for size in (None, "s"):
    r = cls.recipe(size) if size else cls.recipe()
    assert isinstance(r, R.Recipe), f"{falta}: recipe() tiene que devolver src.recipes.Recipe, devolvió {type(r).__name__}"
    assert r.name and r.fuente, f"{falta}: la receta necesita name y fuente (de dónde sale cada valor)"
    assert r.lr_segs, f"{falta}: la receta no tiene tramos de lr (lr_segs)"
    assert r.batch_fuente, f"{falta}: la receta no declara batch_fuente (el batch efectivo de la fuente; el humo escala el lr con él)"
    assert r.presupuesto > 0 and r.batch > 0 and r.frames > 0
    assert all(c in R.CATEGORIAS for _, _, c in r.forzado), f"{falta}: categorías de `forzado` válidas: {R.CATEGORIAS}"
    assert R.Recipe.from_dict(r.to_dict()).hash() == r.hash(), f"{falta}: la receta no sobrevive a to_dict/from_dict"
  assert hasattr(mod, "FIDELIDAD") and mod.FIDELIDAD, \
    f"{falta}: no define FIDELIDAD (tabla de fidelidad: una fila por decisión, con su fuente y su categoría) {GUIA}"
  for f in mod.FIDELIDAD:
    assert all(str(f.get(k, "")).strip() for k in KEYS_FIDELIDAD) and ("fuente_valor" in f or "paper" in f), \
      f"{falta}: fila de FIDELIDAD incompleta (hacen falta {KEYS_FIDELIDAD} y fuente_valor): {f}"
    assert f["categoría"] in R.FID_CATEGORIAS, f"{falta}: categoría {f['categoría']!r} inválida; válidas: {R.FID_CATEGORIAS}"
  assert hasattr(mod, "ACCIONES"), f"{falta}: no define ACCIONES (qué acciones de la CLI soporta) {GUIA}"
  assert set(mod.ACCIONES) <= set(ACCIONES_TODAS), f"{falta}: acciones desconocidas: {set(mod.ACCIONES) - set(ACCIONES_TODAS)}"
  assert {"receta", "entrenar"} <= set(mod.ACCIONES), f"{falta}: un modelo registrado entrena: ACCIONES lleva receta y entrenar"
  assert getattr(mod, "EVALUACION", None) in ("video", "latente"), f"{falta}: EVALUACION tiene que ser 'video' o 'latente'"


@pytest.mark.parametrize("kind", list(KINDS))
def test_declaraciones(kind):
  con_brecha("declaraciones", kind, lambda: verificar_declaraciones(kind))


def modulo_falso(monkeypatch, nombre, **attrs):
  mod = types.ModuleType(nombre)
  mod.__dict__.update(attrs)
  monkeypatch.setitem(sys.modules, nombre, mod)
  monkeypatch.setitem(KINDS, nombre, (nombre, "Falso"))
  return mod


def test_sin_receta_falla_con_mensaje(monkeypatch):
  mod = modulo_falso(monkeypatch, "falso_sin_receta", FIDELIDAD=[], ACCIONES=("receta",), EVALUACION="video")
  mod.Falso = type("Falso", (), {})
  with pytest.raises(AssertionError, match=r"falso_sin_receta.*no define recipe\(\)"):
    verificar_declaraciones("falso_sin_receta")


def test_sin_fidelidad_ni_fuente_falla_con_mensaje(monkeypatch):
  from src.models._plantilla import Plantilla
  mod = modulo_falso(monkeypatch, "falso_sin_fid", ACCIONES=("receta", "entrenar"), EVALUACION="video")
  mod.Falso = Plantilla
  with pytest.raises(AssertionError, match=r"no define FIDELIDAD"):
    verificar_declaraciones("falso_sin_fid")
  mod.FIDELIDAD = [dict(campo="x", fuente_valor="y", aquí="z", categoría="propio", fuente="")]
  with pytest.raises(AssertionError, match="fila de FIDELIDAD incompleta"):
    verificar_declaraciones("falso_sin_fid")
  mod.FIDELIDAD = [dict(campo="x", fuente_valor="y", aquí="z", categoría="inventada", fuente="f")]
  with pytest.raises(AssertionError, match="categoría 'inventada' inválida"):
    verificar_declaraciones("falso_sin_fid")


def test_receta_sin_batch_fuente_falla_con_mensaje(monkeypatch):
  from src.models._plantilla import FIDELIDAD, Plantilla

  class SinBatch(Plantilla):
    @classmethod
    def recipe(cls, size=None):
      return super().recipe(size).replace(batch_fuente=None)
  mod = modulo_falso(monkeypatch, "falso_sin_batch", FIDELIDAD=FIDELIDAD, ACCIONES=("receta", "entrenar"), EVALUACION="video")
  mod.Falso = SinBatch
  with pytest.raises(AssertionError, match="no declara batch_fuente"):
    verificar_declaraciones("falso_sin_batch")


def test_modelo_sin_contrato_falla_con_mensaje(monkeypatch):
  from src.models._plantilla import Plantilla
  mod = modulo_falso(monkeypatch, "falso_sin_contrato")
  mod.Falso = Plantilla
  with pytest.raises(pytest.fail.Exception, match="Declará CONTRATO"):
    adaptador("falso_sin_contrato")


# ---------------------------------------------------------------- 2. sin fuga causal

@pytest.mark.parametrize("kind", list(KINDS))
def test_sin_fuga_causal(kind, monkeypatch):
  ad = adaptador(kind)
  if ad["causal"] is None:
    pytest.skip(f"no aplica: {ad['causal_razon'] or 'el contrato no declara una prueba causal'}")
  ad["causal"](modelar(kind, ad, monkeypatch))


# ---------------------------------------------------------------- 3. sobreajusta un batch chico

@pytest.mark.parametrize("kind", list(KINDS))
def test_sobreajusta_un_batch(kind, monkeypatch):
  ad = adaptador(kind)
  m = modelar(kind, ad, monkeypatch).train()
  cfg = ad["sobreajuste"]
  args, kw = ad["entrada"](m)
  opt = (torch.optim.RMSprop(m.parameters(), lr=cfg["lr"], alpha=0.9) if kind == "convlstm_px"
         else torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=cfg["lr"]))
  hist = []
  for step in range(cfg["pasos"]):
    torch.manual_seed(step)
    loss, met = m.train_loss(*args, **kw)
    assert torch.isfinite(loss), f"{kind}: loss no finita en el paso {step}"
    opt.zero_grad()
    loss.backward()
    opt.step()
    if hasattr(m, "after_step"):
      m.after_step(step + 1, cfg["pasos"])
    hist.append(loss.item())
  ini, fin = np.mean(hist[:5]), np.mean(hist[-5:])
  assert fin < cfg["ratio"] * ini, f"{kind}: no sobreajusta un batch chico: loss {ini:.4f} -> {fin:.4f} en {cfg['pasos']} pasos"


# ---------------------------------------------------------------- 4. reanuda bit a bit desde su checkpoint

def args_train(kind, ad, tmp_path, datos, run, extra=()):
  tok, pos = datos
  t = ad["train"]
  cfg = ad["config_train"] or ad["config"]
  cls = model_class(kind)
  r = cls.recipe()
  a = ["--kind", kind, "--config", json.dumps(cfg), "--run", run, "--presupuesto", str(6 * 2 * t["frames"]), "--bs", "2",
       "--accum", "1", "--frames", str(t["frames"]), "--eval-every", "1000", "--log-every", "1", "--data", str(tok),
       "--ckpt-dir", str(tmp_path / "ck"), "--results-dir", str(tmp_path / "res"), "--seed", "3"]
  if t.get("poses"):
    a += ["--poses", str(pos)]
  if any(s.get("nombre") == "warmup" for s in r.lr_segs):
    a += ["--warmup", "2"]
  return a + list(extra)


@pytest.mark.parametrize("kind", list(KINDS))
def test_reanuda_bit_a_bit(kind, tmp_path, monkeypatch, datos):
  con_brecha("reanuda", kind, lambda: _reanuda(kind, tmp_path, monkeypatch, datos))


def _reanuda(kind, tmp_path, monkeypatch, datos):
  ad = adaptador(kind)
  monkeypatch.setenv("ML2_DEVICE", "cpu")
  if ad["parche"]:
    ad["parche"](monkeypatch)
  full = train_main(args_train(kind, ad, tmp_path, datos, "full"))
  train_main(args_train(kind, ad, tmp_path, datos, "split", ["--max-steps", "3"]))
  assert torch.load(tmp_path / "ck" / "split" / "last.pt", weights_only=False)["step"] == 3
  split = train_main(args_train(kind, ad, tmp_path, datos, "split"))  # reanuda desde last.pt (paso 3)
  assert full["steps"] == split["steps"] == 6, f"{kind}: la corrida reanudada no llegó a los 6 pasos"
  a = torch.load(tmp_path / "ck" / "full" / "final.pt", weights_only=False)["model"]
  b = torch.load(tmp_path / "ck" / "split" / "final.pt", weights_only=False)["model"]
  assert a.keys() == b.keys()
  for k in a:  # tensores con torch.equal; el estado extra (p. ej. el banco de máscaras de vjepa2) con ==
    igual = torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k]
    assert igual, f"{kind}: reanudar no es bit a bit: difiere {k}"
  assert full["final_monitor_ce"] == split["final_monitor_ce"]


# ---------------------------------------------------------------- 5. el stream de inferencia = generate / forward por lotes

@pytest.mark.parametrize("kind", list(KINDS))
def test_stream_igual_a_generate(kind, monkeypatch):
  ad = adaptador(kind)
  if ad["stream"] is None:
    pytest.skip(f"no aplica: {ad['stream_razon'] or 'el contrato no declara una prueba de stream'}")
  ad["stream"](modelar(kind, ad, monkeypatch), **ad["stream_kw"])


# ---------------------------------------------------------------- 6. cada acción declarada da un plan válido

@pytest.mark.parametrize("kind", list(KINDS))
def test_acciones_dan_un_plan_valido(kind, tmp_path, monkeypatch):
  monkeypatch.setattr(rutas, "RESULTS", tmp_path / "results")
  monkeypatch.setattr(rutas, "CKPT", tmp_path / "checkpoints")
  ad = adaptador(kind)
  mod = importlib.import_module(KINDS[kind][0])
  tk = Toolkit(kind)
  assert tuple(tk.acciones) == tuple(mod.ACCIONES) and tk.evaluacion == mod.EVALUACION
  for accion in (*mod.ACCIONES, "estado"):
    salida = []
    opts = ad["plan_opts"].get(accion, {})
    plan = tk.plan(accion, **opts)
    res = tk.accion(accion, dry_run=True, salida=salida.append, **opts)
    assert res.estado == "dry-run" and res.ok, f"{kind} {accion}: --dry-run no terminó bien: {res}"
    assert plan.comandos and salida, f"{kind} {accion}: el plan no tiene comandos"
    if accion != "continuar":  # sin corrida previa, continuar solo se puede imprimir (lo avisa)
      assert plan.error is None, f"{kind} {accion}: {plan.error}"
    for c in plan.comandos:
      argv = [str(x) for x in c.argv]
      assert argv[1] == "-m" and importlib.util.find_spec(argv[2]), f"{kind} {accion}: el módulo {argv[2:3]} no existe"
    if accion not in ("receta", "estado"):
      assert kind in " ".join(plan.cli_argv) or plan.cli_argv[0] == kind
