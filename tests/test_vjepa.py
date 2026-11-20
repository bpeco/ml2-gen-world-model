"""V-JEPA 2 y V-JEPA 2-AC: entrenamiento, EMA, AC, CEM y stream (CPU, tamaños chicos)."""
import json
import math
import random

import numpy as np
import pytest
import torch

from src import pixels
from src.infer.stream import VJEPAStream, make_stream
from src.models import load_checkpoint
from src.models.vjepa import aug
from src.models.vjepa.ac import ACPredictor
from src.models.vjepa.action import FIDELIDAD_AC, VJEPA2AC
from src.models.vjepa.action import Config as ACConfig
from src.models.vjepa.decoder import FrameDecoder
from src.models.vjepa.masks import CFG_MASCARAS, MultiSeqMasks
from src.models.vjepa.plan import cem, planificar
from src.models.vjepa.pretrain import FIDELIDAD_PRETRAIN, VJEPA2, Config
from src.train import main

CATS = {"fiel", "datos", "presupuesto", "propio"}
H, W = 32, 64
TINY = dict(encoder="tiny", embed_dim=48, depth=2, num_heads=3, pred_dim=24, pred_depth=2, pred_heads=2, frames=4,
            height=H, width=W, grad_ckpt=False, bank=3)
AC_TINY = dict(encoder="tiny", embed_dim=48, depth=2, num_heads=3, pred_dim=24, pred_depth=2, pred_heads=2, frames=4,
               height=H, width=W, grad_ckpt=False, n_pose=12)


def fake_render(tok):
  g = torch.Generator().manual_seed(int(tok.sum()) % 1000)
  return torch.randint(0, 256, (*tok.shape[:-1], H, W, 3), dtype=torch.uint8, generator=g).to(tok.device)


@pytest.fixture(autouse=True)
def sin_codec(monkeypatch):
  monkeypatch.setattr(pixels, "render", fake_render)


def px(B=4, T=4, seed=0):
  g = torch.Generator().manual_seed(seed)
  return torch.randint(0, 256, (B, T, H, W, 3), dtype=torch.uint8, generator=g)


def test_fidelidad_categorias():
  for lista in (FIDELIDAD_PRETRAIN, FIDELIDAD_AC):
    assert all(f["categoría"] in CATS and set(f) == {"campo", "paper", "aquí", "categoría", "fuente"} for f in lista)


def test_receta_vjepa2():
  r = VJEPA2.recipe()
  assert r.betas == (0.9, 0.999) and r.eps == 1e-8 and r.wd_ini == 0.04 and r.grad_clip is None and r.amp == "bf16"
  assert (r.frames, r.frameskip) == (16, 5) and r.kind_continuacion() == "wcd" and r.wd_on == "matrices"
  T = 10_000
  assert r.lr(0, T) < 2e-4 and abs(r.lr(476, T) - 5.25e-4) < 1e-6 and r.lr(5000, T) == 5.25e-4
  assert r.lr(T - 1, T) == pytest.approx(1e-6) and r.decay_start(T) == T - round(12_000 / 252_000 * T)
  assert r.lr(1, 40) < r.lr(2, 40)  # warmup con piso de 100 pasos
  ac = VJEPA2AC.recipe()
  assert ac.lr(T - 1, T) == 0.0 and ac.accum * ac.batch == 32 and ac.batch_fuente == 256 and ac.kind_continuacion() == "wcd"
  # el humo eligió el lr del paper (4,25e-4 en el tramo constante): queda declarado, sin nada "a confirmar"
  assert ac.lr(5000, T) == pytest.approx(4.25e-4) and 7.5e-5 < ac.lr(0, T) < 8e-5  # el warmup arranca en 7,5e-5
  assert any(f[0] == "batch" and "humo" in f[1] and "confirmar" not in f[1] for f in ac.forzado)
  assert all("confirmar" not in f["aquí"] for f in FIDELIDAD_AC)


# ---------------------------------------------------------------- aumentos

def test_aug_identidad_y_flip():
  x = torch.rand(3, 6, H, W)
  y = aug.recortar_y_voltear(x, (1.0, 1.0), (1.0, 1.0), flip=False)  # ratio relativo al aspecto: recorte completo
  assert torch.allclose(y, x, atol=1e-5)
  yf = aug.recortar_y_voltear(x, (1.0, 1.0), (1.0, 1.0), flip=True)
  assert all(torch.allclose(yf[i], x[i], atol=1e-5) or torch.allclose(yf[i], x[i].flip(-1), atol=1e-5) for i in range(3))
  z = aug.recortar_y_voltear(x, (0.3, 1.0), (0.75, 1.35), True, (16, 32))
  assert z.shape == (3, 6, 16, 32)
  i, j, h, w = aug.parametros_recorte(500, 128, 256, device="cpu")
  assert (h <= 128).all() and (w <= 256).all() and (i + h <= 128).all() and (j + w <= 256).all() and (h * w >= 0.3 * 128 * 256 - 300).all()
  assert 0.75 * 2 - 0.1 <= (w / h).min() and (w / h).max() <= 1.35 * 2 + 0.1  # razón respecto del aspecto 1:2


# ---------------------------------------------------------------- V-JEPA 2

def test_target_sin_gradiente_y_ema():
  torch.manual_seed(0)
  m = VJEPA2(Config(**TINY))
  assert all(not p.requires_grad for p in m.target_encoder.parameters())
  assert all(torch.equal(a, b) for a, b in zip(m.encoder.parameters(), m.target_encoder.parameters()))
  loss, met = m.train_loss(px())
  loss.backward()
  assert all(p.grad is None for p in m.target_encoder.parameters())
  assert all(p.grad is not None for p in m.encoder.parameters()) and all(p.grad is not None for p in m.predictor.parameters())
  with torch.no_grad():
    for p in m.encoder.parameters():
      p.add_(torch.randn_like(p))
  old = [p.clone() for p in m.target_encoder.parameters()]
  m.after_step(1, 10)
  mom = m.cfg.ema_momentum
  for o, t, e in zip(old, m.target_encoder.parameters(), m.encoder.parameters()):
    assert torch.allclose(t, mom * o + (1 - mom) * e, atol=1e-6)
  # el optimizador solo ve encoder y predictor
  from src.train import make_optimizer
  opt = make_optimizer(m, VJEPA2.recipe(), torch.device("cpu"), 1e-4)
  ids = {id(p) for g in opt.param_groups for p in g["params"]}
  assert not ids & {id(p) for p in m.target_encoder.parameters()}


def _fijas(m, B):
  """Máscaras fijas: dos conjuntos con contexto y objetivo conocidos (rejilla 2x2x4 = 16 tokens)."""
  me = [torch.tensor([[0, 1, 2, 3, 8, 9, 10, 11]] * B), torch.tensor([[4, 5, 6, 7, 12, 13]] * B)]
  mp = [torch.tensor([[4, 5, 6, 7, 12]] * B), torch.tensor([[0, 1, 2, 15]] * B)]
  return me, mp


def test_loss_solo_en_enmascarados_y_conjuntos_independientes(monkeypatch):
  torch.manual_seed(0)
  m = VJEPA2(Config(**TINY)).eval()
  for p in m.predictor.parameters():
    p.data.add_(0.05 * torch.randn_like(p))
  x = px()
  me, mp = _fijas(m, 4)
  monkeypatch.setattr(m, "masks", lambda B, dev: (me, mp))
  with torch.no_grad():
    base, met0 = m.train_loss(x)
    # el target cambia solo FUERA de los objetivos de los dos conjuntos -> la loss no cambia
    union = torch.zeros(16, dtype=torch.bool)
    for t in mp:
      union[t[0]] = True
    orig = m.target_encoder.forward

    def sucio(clip, masks=None):
      h = orig(clip, masks)
      h = h.clone()
      h[:, ~union] += 5.0 * torch.randn_like(h[:, ~union])  # un desplazamiento constante lo anula el layer_norm
      return h

    monkeypatch.setattr(m.target_encoder, "forward", sucio)
    fuera, _ = m.train_loss(x)
    assert torch.allclose(base, fuera, atol=1e-5)
    # y cambiarlo DENTRO de los objetivos del conjunto B no mueve la pérdida del A
    def sucio_b(clip, masks=None):
      h = orig(clip, masks).clone()
      h[:, mp[1][0]] += 3.0 * torch.randn_like(h[:, mp[1][0]])
      return h

    monkeypatch.setattr(m.target_encoder, "forward", sucio_b)
    _, met_b = m.train_loss(x)
    assert met_b["l1_B"] != met0["l1_B"] and torch.allclose(met_b["l1_A"], met0["l1_A"], atol=1e-5)
    # el conjunto A coincide con su forward suelto aunque B cambie de máscara (sin atención entre conjuntos)
    me2 = [me[0], torch.tensor([[7, 8, 9, 10, 11, 12]] * 4)]
    mp2 = [mp[0], torch.tensor([[0, 1, 2, 3]] * 4)]
    monkeypatch.setattr(m.target_encoder, "forward", orig)
    monkeypatch.setattr(m, "masks", lambda B, dev: (me2, mp2))
    _, met2 = m.train_loss(x)
    assert torch.allclose(met2["l1_A"], met0["l1_A"], atol=1e-5)


def test_banco_de_mascaras_y_forma():
  torch.manual_seed(0)
  m = VJEPA2(Config(**TINY))
  a = [m.masks(4, torch.device("cpu")) for _ in range(5)]  # bank=3: dos sorteos
  assert m._refills == 2 and len(a) == 5
  enc, pred = a[0]
  assert len(enc) == len(pred) == 2 and enc[0].shape[0] == 4
  assert not (set(enc[0][0].tolist()) & set(pred[0][0].tolist()))


def synth(n=80, t=100, seed=0):
  return np.random.default_rng(seed).integers(0, 1024, (n, t, 128)).astype(np.int16)


def run_args(tmp_path, data, run, extra=()):
  return ["--kind", "vjepa2", "--config", json.dumps(TINY), "--run", run, "--presupuesto", "320",
          "--bs", "2", "--frames", "4", "--warmup", "2", "--eval-every", "1000", "--log-every", "1",
          "--data", str(data), "--ckpt-dir", str(tmp_path / "ck"), "--results-dir", str(tmp_path / "res"), *extra]


def test_train_y_extend_from_con_decay(tmp_path, monkeypatch):
  monkeypatch.setenv("ML2_DEVICE", "cpu")
  data = tmp_path / "tok.npy"
  np.save(data, synth())
  res = main(run_args(tmp_path, data, "a"))
  assert res["steps"] == 40 and math.isfinite(res["final_monitor_ce"])
  ck = tmp_path / "ck" / "a"
  pre = torch.load(ck / "pre_decay.pt", weights_only=False)
  assert pre["step"] == pre["decay_start"] == 38 and (ck / "final.pt").exists()
  lrs = [json.loads(l)["lr"] for l in (ck / "log.jsonl").read_text().splitlines() if "lr" in l]
  assert lrs[0] < lrs[3] and lrs[-1] < 1e-5 and abs(lrs[10] - 5.25e-4) < 1e-9
  ext = main(["--extend-from", "a", "--run", "b", "--presupuesto", "640", "--ckpt-dir", str(tmp_path / "ck"),
              "--results-dir", str(tmp_path / "res"), "--data", str(data), "--eval-every", "1000", "--log-every", "1"])
  assert ext["steps"] == 80 and ext["parent"] == "a" and ext["parent_file"] == "pre_decay.pt"
  lb = {json.loads(l)["step"]: json.loads(l)["lr"] for l in (tmp_path / "ck" / "b" / "log.jsonl").read_text().splitlines() if "lr" in l}
  assert abs(lb[39] - 5.25e-4) < 1e-9 and abs(lb[70] - 5.25e-4) < 1e-9 and lb[80] < 1e-5
  m = load_checkpoint(tmp_path / "ck" / "b" / "final.pt")
  assert m.name == "b" and isinstance(m, VJEPA2)


# ---------------------------------------------------------------- V-JEPA 2-AC

def guardar_vjepa2(tmp_path):
  torch.manual_seed(0)
  m = VJEPA2(Config(**TINY))
  with torch.no_grad():
    for p in m.target_encoder.parameters():  # distinto del encoder: se tiene que cargar el TARGET
      p.add_(0.1 * torch.randn_like(p))
  path = tmp_path / "vj.pt"
  import dataclasses
  torch.save({"kind": "vjepa2", "config": dataclasses.asdict(m.cfg), "model": m.state_dict(), "name": "vj"}, path)
  return m, path


def poses(B=2, T=4, n=12, seed=0):
  g = torch.Generator().manual_seed(seed)
  return torch.randn(B, T, n, generator=g)


def test_ac_encoder_congelado_y_fuera_del_optimizador(tmp_path):
  parent, path = guardar_vjepa2(tmp_path)
  m = VJEPA2AC(ACConfig(**{**AC_TINY, "encoder_run": str(path)}))
  assert m.cfg.cargar_encoder is False
  for a, b in zip(m.encoder.parameters(), parent.target_encoder.parameters()):
    assert torch.equal(a, b)
  assert not any(torch.equal(a, b) for a, b in zip(m.encoder.parameters(), parent.encoder.parameters()))
  assert all(not p.requires_grad for p in m.encoder.parameters())
  from src.train import make_optimizer
  opt = make_optimizer(m, VJEPA2AC.recipe(), torch.device("cpu"), 1e-4)
  assert {id(p) for g in opt.param_groups for p in g["params"]} == {id(p) for p in m.predictor.parameters()}
  before = [p.clone() for p in m.encoder.parameters()]
  m.train()
  assert not m.encoder.training
  loss, met = m.train_loss(px(2, 4), poses=poses())
  loss.backward()
  opt.step()
  assert all(p.grad is None for p in m.encoder.parameters())
  assert all(torch.equal(a, b) for a, b in zip(before, m.encoder.parameters()))
  assert torch.isfinite(loss) and set(met) == {"l1_tf", "l1_ar"}
  # el checkpoint ya lleva el encoder: se recarga sin la corrida madre
  import dataclasses
  torch.save({"kind": "vjepa2_ac", "config": dataclasses.asdict(m.cfg), "model": m.state_dict(), "name": "ac"}, tmp_path / "ac.pt")
  path.unlink()
  m2 = load_checkpoint(tmp_path / "ac.pt")
  assert all(torch.equal(a, b) for a, b in zip(m2.encoder.parameters(), m.encoder.parameters()))


def test_ac_sin_fuga_entre_frames():
  torch.manual_seed(0)
  p = ACPredictor(24, 6, (2, 4), 32, 2, 2).eval()
  for q in p.parameters():
    q.data.add_(0.05 * torch.randn_like(q))
  z, a, s = torch.randn(2, 4, 8, 24), torch.randn(2, 4, 6), torch.randn(2, 4, 6)
  with torch.no_grad():
    y = p(z, a, s)
    assert y.shape == z.shape
    for t in range(4):  # cambiar el frame t (tokens, acción y estado) no mueve las salidas de frames < t
      z2, a2, s2 = z.clone(), a.clone(), s.clone()
      z2[:, t] += torch.randn_like(z2[:, t])
      a2[:, t] += 1
      s2[:, t] += 1
      y2 = p(z2, a2, s2)
      assert torch.allclose(y2[:, :t], y[:, :t], atol=1e-5)
      assert not torch.allclose(y2[:, t:], y[:, t:], atol=1e-4)
    # las salidas de los frames <= t no dependen de los posteriores: truncar la secuencia da lo mismo
    assert torch.allclose(p(z[:, :2], a[:, :2], s[:, :2]), y[:, :2], atol=1e-5)
    # la acción y el estado sí condicionan
    assert not torch.allclose(p(z, a + 1, s), y, atol=1e-4)


def test_ac_rollout_dos_pasos_igual_a_dos_llamadas():
  torch.manual_seed(0)
  m = VJEPA2AC(ACConfig(**AC_TINY)).eval()
  for q in m.predictor.parameters():
    q.data.add_(0.05 * torch.randn_like(q))
  x = m.clip(px(2, 4))
  z = m.codificar(x)
  s, a = m.estados_acciones(poses(), 2, 4, z.device)
  with torch.no_grad():
    tf, ar = m.predicciones(z, a, s)
    # a mano: dos llamadas autorregresivas del predictor, cada salida con layer_norm
    f1 = m.predecir(z[:, :1], a[:, :1], s[:, :1])  # frame 1 desde el 0 real
    z2 = m.predecir(torch.cat([z[:, :1], f1], 1), a[:, :2], s[:, :2])[:, -1:]
    assert torch.allclose(ar, torch.cat([f1, z2], 1), atol=1e-5)
    assert torch.allclose(tf[:, :1], f1, atol=1e-5)  # el frame 0 en teacher forcing ve lo mismo (causal)
  # la loss suma teacher forcing y rollout 1:1
  with torch.no_grad():
    loss, met = m.train_loss(px(2, 4), poses=poses())
  assert torch.allclose(loss, met["l1_tf"] + met["l1_ar"], atol=1e-6)


def test_ac_grad_atraviesa_un_paso_recurrente():
  torch.manual_seed(0)
  m = VJEPA2AC(ACConfig(**AC_TINY))
  loss, _ = m.train_loss(px(2, 4), poses=poses())
  loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.predictor.parameters())


def test_cem_recupera_el_minimo():
  torch.manual_seed(0)
  objetivo = torch.tensor([[0.7, -1.2, 0.3]])
  c = lambda cand: (cand - objetivo).abs().sum((-1, -2))
  g = torch.Generator().manual_seed(0)
  mean = cem(c, 1, (1, 3), generator=g)
  assert (mean - objetivo).abs().max() < 0.1

  class Stub:  # el frame siguiente = el último + la acción (en el canal D = A)
    def predecir(self, z, a, s):
      return z + a[:, :, None, :]

  z = torch.randn(1, 1, 5, 6)
  est, acc, meta = torch.zeros(1, 1, 6), torch.zeros(1, 0, 6), torch.tensor([0.5, -0.5, 0.2, 0.0, 1.0, -1.0])
  for h in (1, 2):
    goal = z[:, -1] + (meta if h == 1 else 2 * meta)[None, None, :]
    plan = planificar(Stub(), z, acc, est, goal, horizonte=h, generator=torch.Generator().manual_seed(1))
    assert plan.shape == (1, h, 6)
    if h == 1:
      assert (plan[0, 0] - meta).abs().max() < 0.15
    else:  # cualquier reparto con la misma suma llega a la meta: se compara la suma
      assert (plan.sum(1)[0] - 2 * meta).abs().max() < 0.3


def test_ac_planear_con_el_modelo():
  torch.manual_seed(0)
  m = VJEPA2AC(ACConfig(**AC_TINY)).eval()
  z = m.codificar(m.clip(px(1, 2)))
  s, a = m.estados_acciones(poses(1, 2), 1, 2, z.device)
  plan = m.planear(z[:, :1], a[:, :0], s[:, :1], z[:, 1], horizonte=2, n_muestras=16, topk=4, iters=2)
  assert plan.shape == (1, 2, 6) and torch.isfinite(plan).all()


def test_decoder_forma_y_aprende():
  torch.manual_seed(0)
  d = FrameDecoder(48, dim=32, depth=1, heads=2, grid_hw=(2, 4))
  assert d(torch.randn(3, 8, 48)).shape == (3, 3, 32, 64) and d(torch.randn(2, 3, 8, 48)).shape == (2, 3, 3, 32, 64)
  from src.models.vjepa.train_decoder import entrenar, lr_en
  assert lr_en(0, 100) < lr_en(9, 100) and lr_en(99, 100) < 1e-5
  m = VJEPA2(Config(**TINY)).eval()
  img = px(8, 1)[:, 0]
  dec, hist = entrenar(m, lambda step, bs: img[:bs], torch.device("cpu"), steps=30, bs=8, dim=32, depth=1, heads=2, log=lambda s: None)
  assert hist[-1] < hist[0]


def test_decoder_no_retiene_el_buffer_del_mse():
  # trabajo 12: el historial guardaba loss.detach(), que comparte storage con el buffer elementwise del mse_loss
  # (B*3*H*W floats); con bs 64 eran ~24 MiB por paso y el decoder se quedó sin memoria en el paso ~1000
  import gc
  from src.models.vjepa.train_decoder import entrenar
  torch.manual_seed(0)
  m = VJEPA2(Config(**TINY)).eval()
  img = px(4, 1)[:, 0]
  vivos = []

  def frames_fn(step, bs):
    gc.collect()
    vivos.append(sum(1 for o in gc.get_objects() if torch.is_tensor(o) and o.ndim == 0
                     and o.untyped_storage().nbytes() >= img[:bs].numel() * 4))
    return img[:bs]

  _, hist = entrenar(m, frames_fn, torch.device("cpu"), steps=6, bs=4, dim=32, depth=1, heads=2, log=lambda s: None)
  # solo el `loss` del paso anterior (variable del bucle); con la pérdida eran 0, 1, 2, 3, ... (uno por paso)
  assert vivos == [0] + [1] * 5 and len(hist) == 6 and all(isinstance(h, float) for h in hist)


# ---------------------------------------------------------------- stream

def _camara(B, n, seed=0):
  g = torch.Generator().manual_seed(seed)
  return torch.randint(0, 256, (n, B, H, W, 3), dtype=torch.uint8, generator=g)


def test_stream_igual_al_forward_por_lotes():
  torch.manual_seed(0)
  m = VJEPA2AC(ACConfig(**AC_TINY)).eval()
  for q in m.predictor.parameters():
    q.data.add_(0.05 * torch.randn_like(q))
  m.set_pose_stats(torch.randn(12) * 0.1, torch.rand(12) + 0.5)
  s = make_stream(m, graphs=False)
  assert isinstance(s, VJEPAStream)
  B, F = 2, 3
  cam = _camara(B, F * 5)
  P = poses(B, F * 5)
  avanzo = [s.observe_pixels(cam[i], P[:, i]) for i in range(F * 5)]
  assert avanzo == [True, False, False, False, False] * F and s.z.shape[1] == F
  accion = torch.randn(B, 6)
  zh = s.predict(accion)
  # lote: los mismos F frames (1 de cada 5), mismos estados, acciones observadas + la pedida
  frames = cam[::5].permute(1, 0, 2, 3, 4)
  x = m.clip(frames)
  z = m.codificar(x)
  st, ac = m.estados_acciones(P[:, ::5], B, F, z.device)
  ref = m.predecir(z, torch.cat([ac, accion[:, None]], 1), st)[:, -1]
  assert torch.allclose(zh, ref, atol=1e-5)
  # step: el imaginado entra al estado y el siguiente predict ve F+1 frames con estado + acción
  zh2 = s.step(accion)
  assert torch.allclose(zh2, ref, atol=1e-5) and s.z.shape[1] == F + 1
  ref2 = m.predecir(torch.cat([z, ref[:, None]], 1), torch.cat([ac, accion[:, None], torch.zeros(B, 1, 6)], 1),
                    torch.cat([st, st[:, -1:] + accion[:, None]], 1))[:, -1]
  assert torch.allclose(s.predict(), ref2, atol=1e-5)
  # la ventana se recorta a ctx
  s2 = VJEPAStream(m, graphs=False, ctx=2)
  for i in range(F * 5):
    s2.observe_pixels(cam[i], P[:, i])
  assert s2.z.shape[1] == 2 and s2.a.shape[1] == 1
  ref3 = m.predecir(z[:, -2:], torch.cat([ac[:, -1:], torch.zeros(B, 1, 6)], 1), st[:, -2:])[:, -1]
  assert torch.allclose(s2.predict(), ref3, atol=1e-5)


# ---------------------------------------------------------------- sin viajes a la CPU

def _prohibir(monkeypatch):
  def no(*a, **k):
    raise AssertionError("viaje a la CPU en el camino caliente")
  for n in ("item", "cpu", "numpy", "tolist"):
    monkeypatch.setattr(torch.Tensor, n, no)


def test_sin_viajes_a_la_cpu(monkeypatch):
  torch.manual_seed(0)
  pre = VJEPA2(Config(**TINY))
  ac = VJEPA2AC(ACConfig(**AC_TINY)).eval()
  ac.train()
  x = px()
  pre.train_loss(x)  # sortea el banco (única sincronización documentada: banco(n) lee un vector de enteros)
  stream = VJEPAStream(ac.eval(), graphs=False)
  cam, P = _camara(2, 10), poses(2, 10)
  _prohibir(monkeypatch)
  for _ in range(2):  # el banco tiene bank=3 lotes: quedan 2 sin sincronizar
    loss, met = pre.train_loss(x)
    loss.backward()
    pre.after_step(1, 10)
  ac.train()
  loss, _ = ac.train_loss(px(2, 4), poses=poses())
  loss.backward()
  ac.eval()
  for i in range(10):
    stream.observe_pixels(cam[i], P[:, i])
  stream.predict(torch.randn(2, 6))
  stream.step(torch.randn(2, 6))
  planificar(ac, stream.z, stream.a, stream.s, stream.z[:, -1], horizonte=1, n_muestras=8, topk=2, iters=2)


# ---------------------------------------------------------------- estadística de máscaras vs. el oficial literal

def oficial_literal(T, Hh, Ww, cfg, batch, rng):
  """Reimplementación literal en Python puro de _MaskGenerator.__call__ (multiseq_multiblock3d.py L145-235)."""
  smin, smax = cfg["spatial_scale"]
  amin, amax = cfg["aspect_ratio"]
  t = max(1, int(T * 1.0))
  n = int(Hh * Ww * (smin + rng.random() * (smax - smin)))
  ar = amin + rng.random() * (amax - amin)
  h, w = min(int(round(math.sqrt(n * ar))), Hh), min(int(round(math.sqrt(n / ar))), Ww)
  N = T * Hh * Ww
  masks_e, masks_p = [], []
  while len(masks_e) < batch:
    mask = [1] * N
    for _ in range(cfg["num_blocks"]):
      top, left, start = rng.randint(0, Hh - h), rng.randint(0, Ww - w), rng.randint(0, T - t)
      for tt in range(start, start + t):
        for yy in range(top, top + h):
          base = (tt * Hh + yy) * Ww
          for xx in range(left, left + w):
            mask[base + xx] = 0
    e = [i for i, v in enumerate(mask) if v]
    if e:
      masks_e.append(e)
      masks_p.append([i for i, v in enumerate(mask) if not v])
  return min(len(e) for e in masks_e), min(len(p) for p in masks_p)


@pytest.mark.parametrize("batch", [1, 24])
def test_estadistica_de_mascaras_vs_oficial_literal(batch):
  grid = (8, 16, 16)  # la rejilla oficial: 16 frames / 2, 256 / 16 por lado
  N = 8 * 16 * 16
  rng = random.Random(0)
  g = torch.Generator().manual_seed(0)
  ms = MultiSeqMasks(grid)
  mine = [[], []]
  for _ in range(40):
    for enc, pred in ms.banco(25, batch, g):
      for s in range(2):
        mine[s].append(pred[s].shape[1] / N)
  ofi = [[], []]
  for _ in range(250 if batch == 1 else 60):
    for s, cfg in enumerate(CFG_MASCARAS):
      ofi[s].append(oficial_literal(8, 16, 16, cfg, batch, rng)[1] / N)
  m_, o_ = [np.mean(x) for x in mine], [np.mean(x) for x in ofi]
  print(f"\nbatch {batch}, rejilla oficial 16x16: fracción enmascarada (tras recorte al mínimo) "
        f"nuestro A={m_[0]:.3f} B={m_[1]:.3f} | oficial literal A={o_[0]:.3f} B={o_[1]:.3f}")
  assert abs(m_[0] - o_[0]) < 0.04 and abs(m_[1] - o_[1]) < 0.04
