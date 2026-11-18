"""FIDELIDAD de cada modelo y recetas de los papers (IRIS para el GPT, Genie Tab. 9)."""
import math

import pytest
import torch

from src import recipes as R
from src import schedules as S
from src.models import KINDS, build, model_class
from src.train import make_optimizer

KEYS = {"campo", "aquí", "categoría", "fuente"}


@pytest.mark.parametrize("kind", R.fidelidad_kinds())
def test_fidelidad_valida(kind):
  rows = R.fidelidad(kind)
  assert rows
  for f in rows:
    assert KEYS <= set(f) and ("fuente_valor" in f or "paper" in f), f
    assert f["categoría"] in R.FID_CATEGORIAS, f
    assert all(str(f[k]).strip() for k in KEYS), f
  md = R.fidelidad_md(kind)
  assert md.count("\n") == len(rows) + 1


def test_recipe_en_modelos_entrenables():
  for kind in KINDS:
    cls = model_class(kind)
    if hasattr(cls, "recipe"):
      r = cls.recipe("s")
      assert r.fuente and r.name and all(c in R.CATEGORIAS for _, _, c in r.forzado)
  for kind in ("gpt", "genie", "maskgit"):
    assert hasattr(model_class(kind), "recipe")


def test_gpt_receta_iris():
  r = model_class("gpt").recipe("s")
  assert r.optimizer == "adamw" and tuple(r.betas) == (0.9, 0.999) and r.grad_clip == 10.0 and r.amp is None
  assert r.lr_segs == [S.constant(1e-4)] and r.wd_ini == 0.01 and r.wd_fin is None and r.wd_on == "lineales"
  assert r.batch * r.accum == 16 and r.batch_fuente == 64 and r.frames == 20
  assert all(r.lr(s, 1000) == 1e-4 for s in (0, 1, 500, 999, 5000))  # sin warmup, constante
  assert r.kind_continuacion() == "constante"
  assert model_class("gpt").Config().dropout == 0.1 and model_class("gpt").Config().pos == "abs"


def test_gpt_grupos_wd_solo_linear():
  m = build("gpt", dict(n_layer=2, n_head=2, dim=32))
  opt = make_optimizer(m, R.Recipe.from_dict(model_class("gpt").recipe("s").to_dict()), torch.device("cpu"), 1e-4)
  assert opt.defaults["betas"] == (0.9, 0.999)
  dec, nodec = opt.param_groups
  assert dec["weight_decay"] == 0.01 and nodec["weight_decay"] == 0.0
  names = {id(p): n for n, p in m.named_parameters()}
  dec_names = {names[id(p)] for p in dec["params"]}
  lin = {n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)}
  assert dec_names == {n + ".weight" for n in lin} and "net.head.weight" in dec_names
  for n in (names[id(p)] for p in nodec["params"]):
    assert n.endswith(".bias") or ".ln" in n or "wte" in n or "wpe" in n, n
  assert any("wpe" in names[id(p)] for p in nodec["params"]) and any("wte" in names[id(p)] for p in nodec["params"])


def test_gpt_pos_frame_sigue_cargando():
  m = build("gpt", dict(n_layer=1, n_head=2, dim=32, pos="frame"))
  assert hasattr(m.net, "wpe_f") and m.logits(torch.zeros(1, 129, dtype=torch.long)).shape[-1] == 1025


def test_genie_receta():
  r = model_class("genie").recipe("s")
  assert tuple(r.betas) == (0.9, 0.9) and r.wd_ini == 1e-4 and r.amp == "bf16" and r.frames == 16
  assert r.batch * r.accum == 32 and r.batch_fuente == 256 and r.paper_steps == 200_000
  warm = r.lr_segs[0]
  assert warm["nombre"] == "warmup" and warm["frac"] == 0.025 and S.lengths(r.lr_segs, 200_000)[0][1] == 5000
  assert S.lengths(r.lr_segs, 190)[0][1] == 100  # piso
  assert r.lr(5000 - 1, 200_000) == pytest.approx(3e-5) and r.lr(200_000, 200_000) == pytest.approx(3e-6)
  assert r.lr(199_999, 200_000) == pytest.approx(3e-6, rel=1e-3)


def test_maskgit_propia_y_legacy():
  r = model_class("maskgit").recipe("s")
  assert r.name == "maskgit_propia" and all(c == "propio" for _, _, c in r.forzado)
  old = R.legacy_cosine(lr=3e-3, warmup=2, min_lr_frac=0.1)
  assert R.from_legacy_args(dict(lr=3e-3, warmup=2, min_lr_frac=0.1, wd=0.1, clip=1.0, bs=16, amp=None)).lr_segs == old.lr_segs
  assert old.lr(0, 6) == pytest.approx(3e-3 / 2) and old.lr(2, 6) == pytest.approx(3e-3)
  assert old.lr(6, 6) == pytest.approx(3e-3 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi))))


def test_tabla_y_cli(capsys):
  t = R.table()
  assert all(k in t for k in ("gpt_s", "IRIS", "genie_s", "propia", "lineales"))
  R.main(["--fidelidad", "genie"])
  assert "| decisión |" in capsys.readouterr().out
