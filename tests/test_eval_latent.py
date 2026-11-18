"""src/eval_latent.py: la lectura común de lewm, vjepa2 y vjepa2_ac, en CPU con modelos chicos y datos sintéticos."""
import json

import numpy as np
import pytest
import torch

from src import eval_latent as EL
from src import pixels
from src.models.lewm import LeWM
from src.models.lewm import cem as cem_lewm
from src.models.vjepa.action import VJEPA2AC
from src.models.vjepa.action import Config as ACConfig
from src.models.vjepa.plan import cem as cem_vjepa
from src.models.vjepa.plan import planificar
from src.models.vjepa.pretrain import VJEPA2
from src.models.vjepa.pretrain import Config as VConfig

H, W = 32, 64
T = 140
LEWM = dict(dim=32, enc_layers=1, enc_heads=2, enc_mlp=64, proj_hidden=64, pred_layers=2, pred_heads=4, pred_dim_head=8,
            pred_mlp=64, pred_dropout=0.0, img_h=H, img_w=W)
VJ = dict(encoder="tiny", embed_dim=48, depth=2, num_heads=3, pred_dim=24, pred_depth=2, pred_heads=2, frames=4,
          height=H, width=W, grad_ckpt=False, bank=3)
AC = dict(encoder="tiny", embed_dim=48, depth=2, num_heads=3, pred_dim=24, pred_depth=2, pred_heads=2, frames=4,
          height=H, width=W, grad_ckpt=False)
CEM_CHICO = {"lewm": dict(n_samples=16, n_elites=4, iters=2), "vjepa2_ac": dict(n_muestras=16, topk=4, iters=2)}


def fake_render(tok):
  """Cámara de juguete: cada token pinta una celda de 16x16 de la imagen de H x W (sin los pesos del decoder de comma)."""
  pal = (torch.arange(1024)[:, None] * torch.tensor([37, 91, 53]) % 256).to(torch.uint8).to(tok.device)
  img = pal[tok.long()].view(*tok.shape[:-1], 8, 16, 3)[..., :H // 16, :W // 16, :]
  return img.repeat_interleave(16, -3).repeat_interleave(16, -2).contiguous()


@pytest.fixture
def datos(tmp_path, monkeypatch):
  """data/ sintético: 60 segmentos de train (50 de holdout), 12 de val, rutas distintas, y la cámara de juguete."""
  rng = np.random.default_rng(0)
  rutas = {}
  for nombre, n in (("train", 60), ("valp", 12)):
    base = np.cumsum(rng.normal(size=(n, T, 6)).astype(np.float32), 1) * 0.1  # poses con estructura
    np.save(tmp_path / f"{nombre}_tokens.npy", rng.integers(0, 1024, (n, T, 128)).astype(np.int16))
    np.save(tmp_path / f"{nombre}_poses.npy", base)
    (tmp_path / f"{nombre}_keys.json").write_text(json.dumps([f"{nombre}{i}_{i}" for i in range(n)]))
    rutas[nombre] = tuple(str(tmp_path / f"{nombre}_{x}") for x in ("tokens.npy", "poses.npy", "keys.json"))
  monkeypatch.setattr(EL, "TRAIN", rutas["train"])
  monkeypatch.setattr(EL, "VAL", rutas["valp"])
  monkeypatch.setattr(pixels, "render", fake_render)
  monkeypatch.setenv("ML2_DEVICE", "cpu")
  return tmp_path


def modelo(kind, seed=0):
  torch.manual_seed(seed)
  m = {"lewm": lambda: LeWM(LeWM.Config(**LEWM)), "vjepa2": lambda: VJEPA2(VConfig(**VJ)),
       "vjepa2_ac": lambda: VJEPA2AC(ACConfig(**AC))}[kind]()
  if kind == "lewm":  # saca a AdaLN-zero de la identidad: sin esto la acción no influye
    for n, p in m.named_parameters():
      if "adaLN_modulation" in n:
        p.data = torch.randn(p.shape, generator=torch.Generator().manual_seed(3)) * 0.2
  if kind in CEM_CHICO:
    type(m).lat_cem = CEM_CHICO[kind]
  return m.eval()


def correr(datos, monkeypatch, kind, *extra):
  m = modelo(kind)
  monkeypatch.setattr(EL, "load_model", lambda spec, dev: m)
  return EL.main(["--model", kind, "--limite", "6", "--n-plan", "2", "--n-boot", "40", "--bs", "4", "--out",
                  str(datos / "res"), *extra]), m


def check_ic(x):
  assert set(x) == {"valor", "ic95"} and x["ic95"][0] <= x["valor"] + 1e-9 <= x["ic95"][1] + 2e-9, x


@pytest.mark.parametrize("kind", ["lewm", "vjepa2", "vjepa2_ac"])
def test_corre_de_punta_a_punta_con_todas_las_referencias(datos, monkeypatch, kind):
  res, m = correr(datos, monkeypatch, kind)
  salida = datos / "res" / f"latent_{kind}.json"  # el nombre por defecto es el del modelo
  assert json.loads(salida.read_text(encoding="utf-8"))["kind"] == kind
  # probes: el modelo, el encoder sin entrenar, los píxeles y la media de train, con IC y diferencias pareadas
  tipos = ["obs", "pred"] if kind != "vjepa2" else ["obs"]
  assert [t for t in ("obs", "pred") if t in res["probes"]] == tipos
  for t in tipos:
    for dim in ("velocidad", "giro"):
      for fuente in ("modelo", "sin_entrenar", "pixeles", "media_train"):
        b = res["probes"][t][fuente][dim]
        check_ic(b["mse"])
        check_ic(b["r2"])
      for ref in ("sin_entrenar", "pixeles", "media_train"):
        check_ic(res["probes"][t][f"dif_mse_modelo_menos_{ref}"][dim])
    assert set(res["probes"]["alpha"]) == {f"{t}/{f}" for t in tipos for f in ("modelo", "sin_entrenar", "pixeles")}
  # la media de train es la referencia trivial: R² ~ 0 (<= 0 en val)
  assert res["probes"]["obs"]["media_train"]["velocidad"]["r2"]["valor"] <= 1e-6
  assert res["propio"] and all({"item", "detalle"} <= set(p) for p in res["propio"]) and res["fuente"]
  if kind == "vjepa2":
    for k in ("horizonte", "sorpresa", "cem"):
      assert res[k]["aplica"] is False and res[k]["motivo"]
    return
  # horizonte contra copiar el último latente y la acción nula
  for met in ("mse", "l1"):
    h = res["horizonte"][met]
    assert len(res["horizonte"]["pasos"]) == EL.H_MAX
    for s in ("modelo", "copiar_ultimo", "accion_nula", "razon_modelo_sobre_copiar"):
      assert len(h[s]) == EL.H_MAX
      for x in h[s]:
        check_ic(x)
  # sorpresa: modelo y referencia, pareada por segmento
  for k in ("teletransporte", "salto_2s"):
    for r in ("modelo", "referencia_copiar_ultimo"):
      s = res["sorpresa"][k][r]
      assert s["n_segmentos"] == 6 and 0 <= s["p"] <= 1
      check_ic(s["razon"])
      check_ic(s["dif"])
  # CEM: con las dos referencias
  pasos = ["h1", "h5"] if kind == "vjepa2_ac" else ["h5"]
  assert [h for h in ("h1", "h5") if h in res["cem"]] == pasos
  assert len(res["cem"]["desvio_muestreo"]) == (30 if kind == "lewm" else 6)
  for h in pasos:
    for dim in ("velocidad", "giro"):
      c = res["cem"][h][dim]
      for k in ("mae_cem", "mae_repetir_ultima", "mae_media_train", "dif_cem_menos_repetir_ultima", "dif_cem_menos_media_train"):
        check_ic(c[k])
  assert res["cem"]["n_planes"] == 2


def test_no_plan_omite_el_cem(datos, monkeypatch):
  res, _ = correr(datos, monkeypatch, "lewm", "--no-plan")
  assert res["cem"]["aplica"] is False and "horizonte" in res and res["sorpresa"]["teletransporte"]


def test_estadisticas_de_pose_sin_entrenar_vienen_de_train(datos, monkeypatch):
  res, m = correr(datos, monkeypatch, "lewm", "--no-plan")
  assert res["datos"]["estadisticas_de_pose"].startswith("de train") and m.pose_std.ne(1).all()


# ---------------------------------------------------------------- sorpresa pareada

def test_sorpresa_misma_ventana_da_efecto_cero():
  rng = np.random.default_rng(0)
  v = rng.uniform(0.5, 2.0, 40)
  seg = np.repeat(np.arange(20), 2)
  r = EL.sorpresa_pareada(v, v.copy(), seg)
  assert r["n_segmentos"] == 20
  assert r["dif"]["valor"] == 0 and r["dif"]["ic95"] == [0.0, 0.0]
  assert r["razon"]["valor"] == 1 and r["razon"]["ic95"] == [1.0, 1.0]
  assert r["t"] == 0 and r["p"] == 1


def test_sorpresa_detecta_un_efecto_y_cuenta_segmentos_no_ventanas():
  rng = np.random.default_rng(1)
  seg = np.repeat(np.arange(30), 3)
  a = rng.uniform(1, 2, 30)[seg] + rng.normal(0, 0.02, 90)  # las ventanas de un segmento se parecen mucho
  r = EL.sorpresa_pareada(a, a * 1.5, seg)
  assert r["n_segmentos"] == 30  # no 90
  assert r["razon"]["ic95"][0] > 1.4 and r["p"] < 1e-6 and r["dif"]["ic95"][0] > 0


def test_ic_por_segmento_es_mas_ancho_que_por_ventana():
  """Dos ventanas casi iguales por segmento: remuestrear segmentos no se deja engañar por las ventanas repetidas."""
  rng = np.random.default_rng(2)
  seg = np.repeat(np.arange(40), 2)
  v = rng.normal(0, 1, 40)[seg]
  cols = np.stack([v, np.ones(80)], 1)
  def ancho(ids):
    lo, hi = EL.Boot(ids, 500).ic(cols, lambda s: s[0] / s[1])["ic95"]
    return hi - lo

  assert ancho(seg) > 1.2 * ancho(np.arange(80))


# ---------------------------------------------------------------- probes

def test_ridge_recupera_una_senal_lineal():
  rng = np.random.default_rng(0)
  X = rng.normal(size=(300, 20))
  y = np.stack([X[:, 0] * 2 - X[:, 3], X[:, 5]], 1) + rng.normal(0, 0.05, (300, 2))
  pred = EL.ridge_fit(X[:200], y[:200])(X[200:], 1.0)
  assert ((pred - y[200:]) ** 2).mean() < 0.05
  boot = EL.Boot(np.arange(100), 100)
  r = EL.puntajes(pred, y[200:], np.arange(100), boot)
  assert r["velocidad"]["r2"]["valor"] > 0.95 and r["giro"]["r2"]["valor"] > 0.95


def test_el_alpha_no_depende_de_val():
  """ajustar_probe elige α con train y holdout; cambiar las features de val no lo mueve."""
  rng = np.random.default_rng(0)
  Xtr, Xho = rng.normal(size=(80, 10)), rng.normal(size=(30, 10))
  ytr = np.stack([Xtr[:, 0], Xtr[:, 1]], 1) + rng.normal(0, 0.5, (80, 2))
  yho = np.stack([Xho[:, 0], Xho[:, 1]], 1) + rng.normal(0, 0.5, (30, 2))
  a1, p1, _ = EL.ajustar_probe(Xtr, ytr, Xho, yho, rng.normal(size=(20, 10)))
  a2, p2, _ = EL.ajustar_probe(Xtr, ytr, Xho, yho, rng.normal(size=(50, 10)) * 100)
  assert a1 == a2
  assert a1 == EL.elegir_alpha(Xtr, ytr, Xho, yho)[0]
  # un α malo para el holdout (solo ruido) no se elige aunque val dijera otra cosa
  assert EL.elegir_alpha(Xtr, ytr, Xho, -yho)[0] == max(EL.ALFAS)


def test_la_corrida_elige_alpha_con_train_y_holdout(datos, monkeypatch):
  vistos = []
  orig = EL.elegir_alpha

  def espia(Ftr, ytr, Fho, yho, *a, **k):
    vistos.append((len(Ftr), len(Fho)))
    return orig(Ftr, ytr, Fho, yho, *a, **k)
  monkeypatch.setattr(EL, "elegir_alpha", espia)
  res, _ = correr(datos, monkeypatch, "vjepa2")
  d = res["datos"]["ventanas"]
  assert vistos and all(n_tr <= d["train"] and n_ho <= d["holdout"] for n_tr, n_ho in vistos)
  assert all(n_tr != d["val"] or d["train"] == d["val"] for n_tr, _ in vistos)  # nunca se le pasan las ventanas de val
  assert res["datos"]["val"] == EL.VAL[0] and EL.VAL[0] != EL.TRAIN[0]


def test_val_y_train_no_comparten_rutas_en_los_datos_reales():
  """Los datos reales (si están): val con poses sin ninguna ruta de train."""
  from pathlib import Path

  from src.data import route_of
  if not (Path("data/valp_keys.json").exists() and Path("data/train_keys.json").exists()):
    pytest.skip("sin data/")
  tr = {route_of(k) for k in json.loads(Path("data/train_keys.json").read_text())}
  va = {route_of(k) for k in json.loads(Path("data/valp_keys.json").read_text())}
  assert not tr & va


# ---------------------------------------------------------------- CEM

def test_cem_recupera_el_minimo_con_desvio_por_dimension():
  """Problema sintético con mínimo conocido y dimensiones de escalas muy distintas: el desvío de muestreo sale de los datos."""
  objetivo = torch.tensor([[[0.02, -1.5, 0.3], [-0.01, 0.8, -0.2]]])  # (B=1, horizonte=2, A=3)
  std = torch.tensor([0.02, 1.0, 0.3])
  costo = lambda c: (c - objetivo).abs().sum((-1, -2))
  for cem in (cem_lewm, cem_vjepa):
    kw = dict(n_samples=200, n_elites=20, iters=10) if cem is cem_lewm else dict(n_muestras=200, topk=20, iters=10)
    g = torch.Generator().manual_seed(0)
    m = cem(costo, 1, (2, 3), init_std=std, generator=g, **kw)
    assert ((m - objetivo).abs() / std).max() < 0.3, cem


def test_cem_con_desvio_escalar_y_tensor_da_lo_mismo():
  c = lambda x: x.square().sum((-1, -2))
  a = cem_lewm(c, 2, (1, 3), 32, 4, 2, init_std=0.5, generator=torch.Generator().manual_seed(0))
  b = cem_lewm(c, 2, (1, 3), 32, 4, 2, init_std=torch.tensor([0.5, 0.5, 0.5]), generator=torch.Generator().manual_seed(0))
  assert torch.equal(a, b)


def test_planificar_ac_con_modelo_conocido_y_en_trozos():
  """El frame siguiente = el último + la acción: el plan encuentra la acción que lleva a la meta, y `chunk` no cambia nada."""
  class Stub:
    def predecir(self, z, a, s):
      return z + a[:, :, None, :]

  z = torch.randn(1, 1, 5, 3)
  meta = torch.tensor([0.05, -0.02, 0.01])
  goal = z[:, -1] + meta[None, None]
  est, acc = torch.zeros(1, 1, 3), torch.zeros(1, 0, 3)
  std = torch.tensor([0.05, 0.05, 0.05])
  kw = dict(horizonte=1, n_muestras=400, topk=10, iters=10, init_std=std)
  p1 = planificar(Stub(), z, acc, est, goal, generator=torch.Generator().manual_seed(1), **kw)
  p2 = planificar(Stub(), z, acc, est, goal, generator=torch.Generator().manual_seed(1), chunk=100, **kw)
  assert (p1[0, 0] - meta).abs().max() < 0.01
  assert torch.allclose(p1, p2, atol=1e-6)


def test_lat_plan_ac_devuelve_pose_cruda_por_paso(datos):
  m = modelo("vjepa2_ac")
  m.set_pose_stats(np.tile([10.0, 0, 0, 0, 0, 0], 5), np.tile([2.0, 1, 1, 1, 1, 1], 5))
  img = fake_render(torch.randint(0, 1024, (2, EL.PASOS, 128)))
  poses = torch.randn(2, EL.PASOS, 30)
  out = m.lat_plan(img, poses, 3, 2, torch.full((6,), 0.1), torch.Generator().manual_seed(0))
  assert out.shape == (2, 2, 2) and torch.isfinite(out).all()


def test_lat_plan_lewm_devuelve_pose_cruda_por_paso(datos):
  m = modelo("lewm")
  img = fake_render(torch.randint(0, 1024, (2, EL.PASOS, 128)))
  out = m.lat_plan(img, torch.randn(2, EL.PASOS, 30), 3, 5, torch.ones(30), torch.Generator().manual_seed(0))
  assert out.shape == (2, 5, 2) and torch.isfinite(out).all()


def test_convenciones_de_pose_y_referencias():
  poses = np.arange(EL.PASOS * 30, dtype=np.float32).reshape(EL.PASOS, 30)
  media = (np.zeros(6), np.zeros(6))
  lewm, ac = modelo("lewm"), modelo("vjepa2_ac")
  real = EL.convencion_real(lewm, poses, 3, 5)
  assert real.shape == (5, 2)
  ult, med = EL.convencion_base(lewm, poses, 3, 5, media)
  assert np.allclose(ult[0], poses.reshape(16, 5, 6)[1].mean(0)[[0, 5]]) and (med == 0).all()  # el último bloque ya tomado
  ult, med = EL.convencion_base(ac, poses, 3, 2, media)
  s = poses[:, :6]
  assert np.allclose(ult[1], s[2, [0, 5]] + 2 * (s[2] - s[1])[[0, 5]]) and np.allclose(med, s[2, [0, 5]])
  assert np.allclose(EL.convencion_real(ac, poses, 3, 2), s[3:5, [0, 5]])


def test_cem_de_lewm_usa_las_acciones_ya_tomadas_y_la_meta_a_p_pasos():
  """El plan de lewm mira 5 pasos adelante; ni el contexto ni la meta cambian con poses futuras fuera de la ventana de contexto."""
  m = modelo("lewm")
  tk = torch.randint(0, 1024, (1, EL.PASOS, 128))
  img = fake_render(tk)
  p = torch.randn(1, EL.PASOS, 30)
  p2 = p.clone()
  p2[:, 3:] += 5  # poses de después del contexto: no las puede ver el planificador
  g = lambda: torch.Generator().manual_seed(0)
  a = m.lat_plan(img, p, 3, 2, torch.ones(30), g())
  b = m.lat_plan(img, p2, 3, 2, torch.ones(30), g())
  assert torch.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------- features y referencias

def test_el_modelo_sin_entrenar_tiene_la_misma_arquitectura_y_otros_pesos():
  for kind in ("lewm", "vjepa2", "vjepa2_ac"):
    m = modelo(kind, seed=5)  # "entrenado": otros pesos que los de la semilla 0 de la referencia
    r = EL.modelo_sin_entrenar(m)
    assert type(r) is type(m) and r.num_params() == m.num_params()
    r2 = EL.modelo_sin_entrenar(m)
    pesos = lambda mod: [v for v in mod.state_dict().values() if torch.is_tensor(v)]  # sin el estado extra (banco de máscaras)
    assert all(torch.equal(a, b) for a, b in zip(pesos(r), pesos(r2)))  # semilla fija
    assert any(not torch.equal(a, b) for a, b in zip(pesos(r), pesos(m)))


def test_features_de_cada_modelo_tienen_forma_y_son_independientes_de_las_poses():
  img = fake_render(torch.randint(0, 1024, (3, EL.PASOS, 128)))
  for kind, claves in (("lewm", {"obs", "pred"}), ("vjepa2", {"obs"}), ("vjepa2_ac", {"obs", "pred"})):
    m = modelo(kind)
    f = m.lat_features(img, EL.N_PASOS)
    assert set(f) == claves
    assert all(v.shape[0] == 3 and v.ndim == 2 and torch.isfinite(v).all() for v in f.values())
  assert f["obs"].shape[1] == 3 * 48


def test_pix_features_promedia_por_paso():
  img = torch.zeros(2, 6, 32, 64, 3, dtype=torch.uint8)
  img[:, -2:] = 255
  f = EL.pix_features(img, 2, n=2)  # 2 pasos de 2 frames: el último paso es blanco, el anterior negro
  assert f.shape == (2, 2 * 3 * 8 * 16)
  assert f[0, :384].max() == 0 and torch.allclose(f[0, 384:], torch.ones(384))


def test_horizonte_de_lewm_acciones_reales_vs_nulas_difieren():
  m = modelo("lewm")
  img = fake_render(torch.randint(0, 1024, (2, EL.PASOS, 128)))
  p = torch.randn(2, EL.PASOS, 30) * 3
  a = m.lat_horizonte(img, p, 3, 4, False)
  n = m.lat_horizonte(img, p, 3, 4, True)
  assert a["mse"].shape == (2, 4) and torch.equal(a["copia_mse"], n["copia_mse"])  # la referencia no depende de la acción
  assert not torch.allclose(a["mse"], n["mse"])


def test_horizonte_de_ac_el_primer_paso_es_el_teacher_forcing():
  """El paso 1 del rollout = la predicción de un paso con el contexto real: el error del paso 1 es el de lat_sorpresa."""
  m = modelo("vjepa2_ac")
  img = fake_render(torch.randint(0, 1024, (2, EL.PASOS, 128)))
  p = torch.randn(2, EL.PASOS, 30)
  h = m.lat_horizonte(img, p, 3, 2, False)
  s = m.lat_sorpresa(img[:, :4], p[:, :4])
  assert torch.allclose(h["mse"][:, 0], s["mse"][:, 2], atol=1e-5)  # el frame 3 predicho desde los frames 0..2
  assert torch.allclose(h["copia_mse"][:, 0], s["copia_mse"][:, 2], atol=1e-6)


def test_la_sorpresa_ve_solo_el_pasado():
  """El error del frame c no cambia si se perturban los frames desde c en adelante... salvo el propio frame c."""
  for kind in ("lewm", "vjepa2_ac"):
    m = modelo(kind)
    tk = torch.randint(0, 1024, (2, EL.PASOS, 128))
    tk2 = tk.clone()
    tk2[:, EL.CORTE + 2:] = torch.randint(0, 1024, tk2[:, EL.CORTE + 2:].shape)
    p = torch.randn(2, EL.PASOS, 30)
    a, b = m.lat_sorpresa(fake_render(tk), p), m.lat_sorpresa(fake_render(tk2), p)
    assert torch.allclose(a["mse"][:, :EL.CORTE + 1], b["mse"][:, :EL.CORTE + 1], atol=1e-5), kind


def test_tomar_alinea_poses_y_salto():
  tok = np.broadcast_to(np.arange(T)[None, :, None], (3, T, 128)).copy()
  pos = np.broadcast_to(np.arange(T, dtype=np.float32)[None, :, None], (3, T, 6)).copy()
  tk, ps = EL.tomar(tok, pos, [(1, 7, 0)])
  assert tk[0, :, 0].tolist() == [7 + 5 * i for i in range(EL.PASOS)]
  assert ps.shape == (1, EL.PASOS, 30) and ps[0, 2, ::6].tolist() == [17, 18, 19, 20, 21]
  tk, ps = EL.tomar(tok, pos, [(1, 7, 0)], EL.SALTO, EL.CORTE)
  assert tk[0, EL.CORTE - 1, 0] == 7 + 5 * (EL.CORTE - 1) and tk[0, EL.CORTE, 0] == 7 + 5 * EL.CORTE + 40
  assert ps[0, 2, 0] == 17  # las poses no se corren
  ws = EL.lista_ventanas(range(3), T, 2, 0)
  assert len(ws) == 6 and all(t + EL.SPAN + EL.SALTO * EL.SKIP <= T for _, t, _ in ws)



def test_capacidades_declaradas_junto_a_la_fidelidad():
  import importlib
  validas = {"receta", "entrenar", "continuar", "evaluar", "bench", "demo", "exportar"}
  for mod in ("src.models.lewm", "src.models.vjepa.pretrain", "src.models.vjepa.action"):
    m = importlib.import_module(mod)
    assert m.EVALUACION == "latente" and m.FIDELIDAD
    assert set(m.ACCIONES) <= validas and {"receta", "entrenar", "evaluar", "demo", "exportar"} <= set(m.ACCIONES)


def test_cada_modelo_declara_lo_que_sabe_leer():
  assert modelo("vjepa2").lat_aplica == ("probes",)
  assert set(modelo("lewm").lat_aplica) == set(modelo("vjepa2_ac").lat_aplica) == {"probes", "horizonte", "sorpresa", "plan"}
