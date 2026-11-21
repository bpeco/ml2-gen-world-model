"""Demo y export a ONNX de los latentes (K5): lewm, vjepa2 y vjepa2_ac, en CPU con modelos chicos."""
import numpy as np
import pytest
import torch

from src import export_latente
from src.infer import demo, demo_latente
from src.models.lewm_decoder import DecoderConfig, LeWMDecoder, save_decoder
from src.models.vjepa.decoder import FrameDecoder
from tests.test_eval_latent import AC, H, LEWM, VJ, W, datos, fake_render, modelo  # noqa: F401  (fixtures y modelos chicos)


@pytest.fixture
def mundo(datos, monkeypatch):  # noqa: F811
  """Datos sintéticos y cwd en tmp: los checkpoints y results/ de la prueba no tocan los del proyecto."""
  monkeypatch.chdir(datos)
  monkeypatch.setattr(demo_latente, "CKPT", datos / "checkpoints")
  monkeypatch.setattr(demo_latente, "OUT", datos / "demo")
  return datos


def _cargar(monkeypatch, m):
  monkeypatch.setattr(demo_latente, "load_model", lambda spec, dev: m)


def test_demo_lewm_sin_decoder_da_png_con_curva(mundo, monkeypatch):
  _cargar(monkeypatch, modelo("lewm"))
  info = demo_latente.main(["--model", "lewm", "--segment", "2", "--start", "10"])
  assert info["decoder"] is None and (mundo / "demo" / "lewm_latente.png").stat().st_size > 1000
  assert len(info["error_mse"]) == 8 and len(info["error_copiar_ultimo_mse"]) == 8
  assert (mundo / "demo" / "lewm_latente.json").exists()


def test_demo_lewm_con_decoder_y_mp4(mundo, monkeypatch):
  m = modelo("lewm")
  _cargar(monkeypatch, m)
  dec = LeWMDecoder(DecoderConfig(dim_in=32, dec_dim=32, layers=1, heads=2, n_kv=2, img_h=H, img_w=W))
  (mundo / "checkpoints" / "lewm").mkdir(parents=True)
  save_decoder(mundo / "checkpoints" / "lewm" / "decoder.pt", dec)
  info = demo_latente.main(["--model", "lewm", "--mp4", "--pasos", "4"])
  assert info["decoder"].endswith("decoder.pt") and info["pasos"] == 4
  assert (mundo / "demo" / "lewm_latente.png").exists() and (mundo / "demo" / "lewm_latente.mp4").stat().st_size > 100


def test_demo_vjepa2_dibuja_las_mascaras_multibloque(mundo, monkeypatch):
  m = modelo("vjepa2")
  _cargar(monkeypatch, m)
  info = demo_latente.main(["--model", "vjepa2"])
  assert (mundo / "demo" / "vjepa2_mascaras.png").stat().st_size > 1000
  assert len(info["tapado"]) == 2 and all(0 < f < 1 for f in info["tapado"])


def test_demo_vjepa2_ac_con_decoder_de_visualizacion(mundo, monkeypatch):
  m = modelo("vjepa2_ac")
  _cargar(monkeypatch, m)
  d = FrameDecoder(48, dim=32, depth=1, heads=2, grid_hw=(H // 16, W // 16))
  (mundo / "checkpoints" / "vjepa2_ac_viz").mkdir(parents=True)
  torch.save({"config": dict(embed_dim=48, dim=32, depth=1, heads=2, grid_hw=(H // 16, W // 16), patch=16),
              "model": d.state_dict()}, mundo / "checkpoints" / "vjepa2_ac_viz" / "decoder.pt")
  info = demo_latente.main(["--model", "vjepa2_ac", "--mp4", "--pasos", "3"])
  assert info["decoder"] and (mundo / "demo" / "vjepa2_ac_latente.mp4").exists()


def test_demo_vjepa2_con_decoder(mundo, monkeypatch):
  m = modelo("vjepa2")
  _cargar(monkeypatch, m)
  d = FrameDecoder(48, dim=32, depth=1, heads=2, grid_hw=(H // 16, W // 16))
  (mundo / "checkpoints" / "vjepa2_viz").mkdir(parents=True)
  torch.save({"config": dict(embed_dim=48, dim=32, depth=1, heads=2, grid_hw=(H // 16, W // 16), patch=16),
              "model": d.state_dict()}, mundo / "checkpoints" / "vjepa2_viz" / "decoder.pt")
  assert demo_latente.main(["--model", "vjepa2"])["decoder"]


def test_demo_rechaza_los_modelos_de_video(monkeypatch):
  from src.models.maskgit import MaskGIT
  monkeypatch.setattr(demo_latente, "load_model", lambda spec, dev: MaskGIT(MaskGIT.Config(n_layer=1, n_head=2, dim=16)))
  with pytest.raises(SystemExit):
    demo_latente.main(["--model", "maskgit_s"])


def test_demo_de_video_manda_los_latentes_a_demo_latente(monkeypatch):
  llamado = {}
  monkeypatch.setattr("src.models.load_model", lambda spec, dev="cpu": modelo("lewm"))
  monkeypatch.setattr(demo_latente, "main", lambda argv: llamado.setdefault("argv", argv) and "ok")
  assert demo.main(["--model", "lewm", "--pasos", "4", "--mp4"]) == "ok"
  assert llamado["argv"] == ["--model", "lewm", "--pasos", "4", "--mp4"]


def test_demo_de_video_sigue_rechazando_argumentos_desconocidos(monkeypatch):
  from src.models.maskgit import MaskGIT
  monkeypatch.setattr("src.models.load_model", lambda spec, dev="cpu": MaskGIT(MaskGIT.Config(n_layer=1, n_head=2, dim=16)))
  with pytest.raises(SystemExit):
    demo.main(["--model", "maskgit_s", "--mp4"])


def test_demo_de_video_no_tiene_caminos_de_lewm_por_tokens():
  import inspect
  src = inspect.getsource(demo)
  assert "rollout=token" not in src and "render_tokens" not in src and "--model lewm --segment" not in src


# ---------------------------------------------------------------- ONNX

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


@pytest.mark.parametrize("kind,parte", [("lewm", "encoder"), ("lewm", "predictor"), ("vjepa2", "encoder"),
                                        ("vjepa2_ac", "predictor")])
def test_onnx_ort_igual_a_pytorch(tmp_path, monkeypatch, kind, parte):
  m = modelo(kind, seed=3)
  monkeypatch.setattr(export_latente, "load_model", lambda spec, dev: m)
  res = export_latente.main(["--model", kind, "--parte", parte, "--out", str(tmp_path / f"{kind}_{parte}.onnx"),
                             "--results", str(tmp_path / "res")])
  assert res["ok"] and res["max_abs_ort_vs_torch"] < 1e-4
  assert (tmp_path / "res" / "export" / f"{kind}_{parte}.json").exists()


def test_onnx_del_predictor_de_lewm_acepta_otras_entradas(tmp_path):
  """La salida de ORT con entradas nuevas (no las del ejemplo) también coincide con PyTorch."""
  from src.models.lewm import _PredictorOnnx, exportar_onnx
  m = modelo("lewm", seed=3)
  m.set_pose_stats(np.full(30, 1.0), np.full(30, 2.0))
  exportar_onnx(m, "predictor", tmp_path / "p.onnx")
  sess = ort.InferenceSession(str(tmp_path / "p.onnx"), providers=["CPUExecutionProvider"])
  g = torch.Generator().manual_seed(9)
  z, p = torch.randn(1, 3, 32, generator=g), torch.randn(1, 3, 30, generator=g) * 4
  out = sess.run(None, {"z": z.numpy(), "poses": p.numpy()})[0]
  assert np.abs(out - _PredictorOnnx(m)(z, p).detach().numpy()).max() < 1e-4


def test_onnx_pide_la_parte_que_corresponde():
  for kind, parte in (("vjepa2", "predictor"), ("vjepa2_ac", "encoder")):
    with pytest.raises(SystemExit):
      from unittest import mock
      with mock.patch.object(export_latente, "load_model", lambda spec, dev: modelo(kind)):
        export_latente.main(["--model", kind, "--parte", parte])


def test_exportar_no_cambia_el_modo_del_modelo(tmp_path):
  from src.models.lewm import exportar_onnx
  m = modelo("lewm", seed=3)
  exportar_onnx(m, "encoder", tmp_path / "e.onnx")
  assert not m.training
