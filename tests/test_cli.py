"""CLI, Toolkit, candado y cola de GPU, bitácora (K2, K3, K6, K4). Todo en CPU y con results/ temporal."""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from src import bitacora, cli, cola, ejecutor, resultados, rutas
from src import toolkit as T
from src.ejecutor import Cmd, Plan
from src.toolkit import Toolkit, ToolkitError

ROOT = Path(__file__).resolve().parents[1]
PARES = [(m, a) for m, accs in T.catalogo().items() for a in accs]


@pytest.fixture
def entorno(tmp_path, monkeypatch):
  """results/ y checkpoints/ temporales, git limpio en el commit 'abc123' y que ese commit esté en main."""
  monkeypatch.setattr(rutas, "RESULTS", tmp_path / "results")
  monkeypatch.setattr(rutas, "CKPT", tmp_path / "checkpoints")
  monkeypatch.setattr(bitacora, "git_estado", lambda: ("abc123", True))
  monkeypatch.setattr(bitacora, "es_ancestro_de_main", lambda c: c == "abc123")
  return tmp_path


def juguete(codigo=0, escribe=None, device="cuda", cli_argv=("gpt", "entrenar"), run="toy"):
  """Plan de juguete: un python -c que sale con `codigo` y escribe el JSON `escribe` (ruta, dict)."""
  cuerpo = "import sys, json\n"
  if escribe:
    cuerpo += f"open({str(escribe[0])!r}, 'w').write(json.dumps({escribe[1]!r}))\n"
  cuerpo += f"sys.exit({codigo})"
  return Plan([Cmd([ejecutor.PY, "-c", cuerpo])], "gpt", "entrenar", run, device, None, list(cli_argv))


def lineas(capsys):
  return capsys.readouterr().out.splitlines()


# ---------------------------------------------------------------- CLI: ayuda, validación, dry-run

def test_help_lista_modelos_y_acciones(capsys):
  assert cli.main(["--help"]) == 0
  out = capsys.readouterr().out
  for m in T.modelos():
    assert m in out
  for a in ("receta", "entrenar", "continuar", "evaluar", "bench", "demo", "exportar"):
    assert a in out
  for g in ("cola", "listar", "resultados", "reporte", "bitacora"):
    assert g in out


def test_help_de_una_accion_lista_las_opciones(capsys):
  with pytest.raises(SystemExit) as e:
    cli.main(["gpt", "entrenar", "--help"])
  assert e.value.code == 0
  out = capsys.readouterr().out
  for flag in ("--run", "--tamaño", "--semilla", "--presupuesto", "--device", "--limite", "--set", "--dry-run", "--esperar",
               "--encolar", "--humo"):
    assert flag in out


@pytest.mark.parametrize("args,texto", [
  (["gpt", "entrenar", "--set", "config.zzz=1"], "zzz"),
  (["gpt", "entrenar", "--set", "receta.nada=1"], "nada"),
  (["gpt", "entrenar", "--set", "algo=1"], "receta.campo=valor"),
  (["gpt", "evaluar", "--set", "receta.betas=[0.9,0.99]"], "solo aplica"),
  (["gpt", "exportar"], "no tiene la acción"),
  (["gpt", "evaluar", "--humo"], "--humo solo aplica a entrenar"),
  (["gpt", "entrenar", "--tamaño", "xl"], "--tamaño"),
  (["gpt", "entrenar", "--presupuesto", "mucho"], "--presupuesto"),
  (["gpt", "evaluar", "--pasos", "4"], "no decodifica por pasos"),
  (["lewm", "entrenar", "--tamaño", "s"], "no tiene tamaños"),
  (["vjepa2_ac", "entrenar"], "--encoder"),
  (["codec", "entrenar"], "no tiene la acción"),
])
def test_pedidos_invalidos_fallan_con_mensaje(args, texto, capsys):
  try:
    rc = cli.main([*args, "--dry-run"])
  except SystemExit as e:  # argparse (choices)
    rc = e.code
  assert rc == 2
  err = capsys.readouterr().err
  assert texto in err


def test_set_valido_se_traduce(capsys):
  cli.main(["gpt", "entrenar", "--set", "receta.lr_peak=3e-4", "--set", "receta.extras.dropout=0.2", "--set", "config.dim=128",
            "--dry-run"])
  out = capsys.readouterr().out
  assert "--lr 3e-4" in out and "--set receta.extras.dropout=0.2" in out
  assert '--config \'{"dim":128}\'' in out


@pytest.mark.parametrize("modelo,accion", PARES)
def test_dry_run_imprime_lo_que_ejecutaria(modelo, accion, capsys):
  tk = Toolkit(modelo)
  req = (modelo, accion) == ("vjepa2_ac", "entrenar")    # lo único con una opción obligatoria: el encoder congelado
  plan = tk.plan(accion, **({"encoder": "vjepa2_s"} if req else {}))
  assert cli.main([modelo, accion, *(["--encoder", "vjepa2_s"] if req else []), "--dry-run"]) == 0
  impreso = [ln for ln in lineas(capsys) if not ln.startswith("#")]
  assert impreso == [c.texto() for c in plan.comandos]
  assert impreso and all(ln.startswith(("python -m ", "ML2_DEVICE=")) for ln in impreso)


def test_dry_run_no_corre_nada(entorno, monkeypatch):
  monkeypatch.setattr(ejecutor, "correr_comando", lambda c, salida=None: pytest.fail("corrió un comando"))
  assert cli.main(["gpt", "entrenar", "--dry-run"]) == 0
  assert not rutas.bitacora().exists()


def test_extra_pasa_tal_cual(capsys):
  cli.main(["gpt", "entrenar", "--dry-run", "--", "--bs", "8", "--compile"])
  assert "src.train --kind gpt --size s --run gpt_s --bs 8 --compile" in capsys.readouterr().out


def test_device_cpu_va_en_el_entorno(capsys):
  cli.main(["gpt", "bench", "--device", "cpu", "--dry-run"])
  assert "ML2_DEVICE=cpu python -m src.bench" in capsys.readouterr().out


def test_modulo_sin_acciones_falla_claro(monkeypatch):
  import src.models.gpt as mod
  monkeypatch.delattr(mod, "ACCIONES")
  with pytest.raises(ToolkitError, match="todavía no declara ACCIONES"):
    Toolkit("gpt").plan("entrenar")


def test_docs_envuelve_docs_gen_o_avisa(capsys, monkeypatch, tmp_path):
  monkeypatch.chdir(tmp_path)                      # sin src/docs_gen.py: "no disponible", sin traceback
  assert cli.main(["docs", "--check"]) == 2 and "no disponible" in capsys.readouterr().err
  (tmp_path / "src").mkdir()
  (tmp_path / "src" / "docs_gen.py").write_text("")
  assert cli.main(["docs", "--check", "--dry-run"]) == 0
  assert capsys.readouterr().out.strip() == "python -m src.docs_gen --check"


def test_verificar_marca_no_disponible_lo_que_falta(capsys, monkeypatch, tmp_path):
  monkeypatch.chdir(tmp_path)
  assert cli.main(["verificar", "--notebooks", "--dry-run"]) == 0
  out = capsys.readouterr().out
  assert out.count("no disponible") == 4 and "pytest" in out


# ---------------------------------------------------------------- Toolkit: contenido de los planes

def test_paridad_toolkit_cli(capsys):
  casos = [
    ("gpt", "entrenar", ["--tamaño", "m", "--semilla", "1", "--presupuesto", "2x", "--destilar"],
     dict(tamano="m", semilla=1, presupuesto="2x", destilar=True)),
    ("genie", "evaluar", ["--tamaño", "s", "--pasos", "1,4", "--limite", "5", "--temperatura", "0.5"],
     dict(tamano="s", pasos="1,4", limite=5, temperatura=0.5)),
    ("convlstm_px", "bench", ["--cadena", "--ritmo", "50", "--device", "cpu"], dict(cadena=True, ritmo=50.0, device="cpu")),
    ("lewm", "demo", ["--entrenar-decoder", "--limite", "10"], dict(entrenar_decoder=True, limite=10)),
    ("gpt", "receta", ["--set", "receta.wd_ini=0.2"], dict(set=["receta.wd_ini=0.2"])),
  ]
  for modelo, accion, flags, kw in casos:
    propio = []
    r = Toolkit(modelo).accion(accion, dry_run=True, salida=propio.append, **kw)
    assert cli.main([modelo, accion, *flags, "--dry-run"]) == 0
    de_cli = [ln for ln in lineas(capsys) if not ln.startswith("#")]
    assert [ln for ln in propio if not ln.startswith("#")] == de_cli
    assert [shlex.join(c).replace("'", "") for c in r.comandos] == [ln.replace("'", "").removeprefix("ML2_DEVICE=cpu ") for ln in de_cli]


def test_argv_canonico_reproduce_la_cli():
  av = T.argv_canonico("gpt", "entrenar", dict(tamano="s", semilla=0, presupuesto="1x", set=["receta.wd_ini=0.2"], humo=False))
  assert av == ["gpt", "entrenar", "--tamaño", "s", "--semilla", "0", "--presupuesto", "1x", "--set", "receta.wd_ini=0.2"]


def test_nombres_de_corrida():
  tk = Toolkit("gpt")
  assert tk.nombre_run({}) == "gpt_s"
  assert tk.nombre_run({"tamano": "m", "variante": "kd", "semilla": 2}) == "gpt_m_kd_s2"
  assert tk.nombre_run({"semilla": 0}) == "gpt_s"
  assert Toolkit("vjepa2").nombre_run({}) == "vjepa2_s" and Toolkit("lewm").nombre_run({}) == "lewm"
  assert Toolkit("transicion").nombre_run({}) == "transicion"


def test_evaluar_video_segun_el_modelo():
  px = [c.argv[2] for c in Toolkit("convlstm_px").plan("evaluar").comandos]
  assert px == ["src.rollouts", "src.metrics_video"]          # píxeles: sin CE
  gen = [c.argv[2] for c in Toolkit("genie").plan("evaluar", pasos="1,2").comandos]
  assert gen == ["src.evaluate", "src.rollouts", "src.metrics_video", "src.rollouts", "src.metrics_video"]
  nombres = [c.argv[c.argv.index("--name") + 1] for c in Toolkit("genie").plan("evaluar", pasos="1,25").comandos if c.argv[2] == "src.rollouts"]
  assert nombres == ["genie_s_p1", "genie_s_p25"]       # _p<pasos>: no choca con el sufijo de semilla (_s1)
  bar = Toolkit("gpt").plan("evaluar", barrer_temperatura=True, limite=5).comandos
  assert [c.argv[2:] for c in bar] == [["src.temperatura", "--model", "gpt_s", "--limite", "5"]]
  lat = Toolkit("lewm").plan("evaluar", limite=5, sin_plan=True).comandos
  assert lat[0].argv[2:] == ["src.eval_latent", "--model", "lewm", "--limite", "5", "--no-plan"]


def test_evaluar_en_cuda_usa_fp16_y_graphs(monkeypatch):
  monkeypatch.setattr(ejecutor, "resolver_device", lambda d=None: d or "cuda")
  r = [c.argv for c in Toolkit("genie").plan("evaluar", pasos="8").comandos if c.argv[2] == "src.rollouts"][0]
  assert "--dtype" in r and "fp16" in r and "--graphs" in r and r[r.index("--steps") + 1] == "8"
  r = [c.argv for c in Toolkit("convlstm_px").plan("evaluar").comandos if c.argv[2] == "src.rollouts"][0]
  assert "--graphs" not in r


def test_humo_con_batch_fuente():
  tk = Toolkit("gpt")
  from src import schedules as S
  pico = S.peak(tk.receta_efectiva({}).lr_segs)
  # batch efectivo 16, fuente 64 -> factor 1/4: raíz = pico/2, lineal = pico/4
  plan = tk.plan("entrenar", humo=True)
  trains = [c.argv for c in plan.comandos if c.argv[2] == "src.train"]
  runs = [a[a.index("--run") + 1] for a in trains]
  assert runs == ["gpt_humo_fuente", "gpt_humo_raiz", "gpt_humo_lineal", "gpt_humo_x3", "gpt_humo_x10"]
  lr = {r: float(a[a.index("--lr") + 1]) for r, a in zip(runs, trains) if "--lr" in a}
  assert "--lr" not in trains[0]
  assert lr["gpt_humo_raiz"] == pytest.approx(pico * 0.5, rel=1e-5) and lr["gpt_humo_lineal"] == pytest.approx(pico * 0.25, rel=1e-5)
  assert lr["gpt_humo_x3"] == pytest.approx(pico * 3, rel=1e-5) and lr["gpt_humo_x10"] == pytest.approx(pico * 10, rel=1e-5)
  assert all(a[a.index("--presupuesto") + 1] == str(300 * 16 * 20) for a in trains)
  h = plan.comandos[-1].argv
  assert h[2] == "src.humo" and "--batch-fuente" in h and "--colapso" not in h


def test_humo_deduplica_candidatos_con_el_mismo_lr():
  plan = Toolkit("gpt").plan("entrenar", humo=True, set=["receta.batch_fuente=16"])   # batch efectivo == fuente
  trains = [c.argv for c in plan.comandos if c.argv[2] == "src.train"]
  assert [a[a.index("--run") + 1] for a in trains] == ["gpt_humo_fuente", "gpt_humo_x3", "gpt_humo_x10"]
  h = plan.comandos[-1].argv
  assert "raiz=fuente" in h and "lineal=fuente" in h
  assert sum("el lr de" in a for a in plan.avisos) == 2


def test_humo_sin_batch_fuente_solo_fuente_x3_x10(monkeypatch):
  from src import recipes as R0
  monkeypatch.setattr(Toolkit, "receta_efectiva", lambda self, o: R0.legacy_cosine())   # una receta sin batch_fuente
  plan = Toolkit("maskgit").plan("entrenar", humo=True)
  assert [c.argv[c.argv.index("--run") + 1] for c in plan.comandos if c.argv[2] == "src.train"] ==     ["maskgit_humo_fuente", "maskgit_humo_x3", "maskgit_humo_x10"]
  assert any("batch_fuente" in a for a in plan.avisos) and "--colapso" not in plan.comandos[-1].argv
  monkeypatch.undo()
  assert "--colapso" in Toolkit("lewm").plan("entrenar", humo=True).comandos[-1].argv


def test_humo_destilado_pasa_kd():
  plan = Toolkit("gpt").plan("entrenar", humo=True, destilar=True)
  assert all("--kd" in c.argv for c in plan.comandos if c.argv[2] == "src.train")


def test_batch_fuente_no_cambia_el_hash_de_las_recetas_existentes():
  from src import recipes as R
  from src import recipes as R0
  r = R0.legacy_cosine()   # sin batch_fuente declarado
  assert "batch_fuente" not in r.to_dict() and r.batch_fuente is None
  assert r.replace(batch_fuente=8).hash() != r.hash()
  assert R.Recipe.from_dict(r.replace(batch_fuente=8).to_dict()).batch_fuente == 8


def test_humo_elige_la_menor_loss(tmp_path):
  from src import humo
  for run, ce in (("a", 3.0), ("b", 2.5), ("c", float("nan"))):
    (tmp_path / f"train_{run}.json").write_text(json.dumps({"final_monitor_ce": ce}))
  out = humo.main(["--kind", "gpt", "--cand", "fuente:a:1e-3", "--cand", "raiz:b:2e-3", "--cand", "lineal:c:4e-3", "--alias", "x3=raiz",
                   "--results-dir", str(tmp_path), "--ckpt-dir", str(tmp_path), "--batch", "64", "--batch-fuente", "16"])
  assert out["elegido"] == "raiz" and out["lr_elegido"] == 2e-3 and not out["es_la_fuente"]
  assert out["forzado"]["categoria"] == "presupuesto" and out["alias"] == {"x3": "raiz"} and out["chequeo_colapso"] == "no aplica"
  assert json.loads((tmp_path / "humo_gpt.json").read_text())["elegido"] == "raiz"


def _humo_latente(tmp_path, std, ce):
  from src import humo
  cands = []
  for nombre, lr in (("fuente", 1e-3), ("x10", 1e-2)):
    run = f"r_{nombre}"
    (tmp_path / f"train_{run}.json").write_text(json.dumps({"final_monitor_ce": ce[nombre]}))
    (tmp_path / run).mkdir(exist_ok=True)
    if std:
      (tmp_path / run / "log.jsonl").write_text(json.dumps({"step": 1, "z_std": std[nombre]}) + "\n")
    cands += ["--cand", f"{nombre}:{run}:{lr}"]
  return humo.main(["--kind", "lewm", *cands, "--colapso", "--results-dir", str(tmp_path), "--ckpt-dir", str(tmp_path)])


def test_humo_latente_descarta_colapsados(tmp_path):
  out = _humo_latente(tmp_path, {"fuente": 0.9, "x10": 0.001}, {"fuente": 2.0, "x10": 0.1})   # x10 tiene menos loss pero colapsó
  assert out["elegido"] == "fuente" and out["es_la_fuente"] and out["chequeo_colapso"] is True
  assert "colapso" in out["nota"]


def test_humo_latente_sin_medida_lo_marca(tmp_path):
  out = _humo_latente(tmp_path, None, {"fuente": 2.0, "x10": 0.1})
  assert out["elegido"] == "x10" and out["chequeo_colapso"] is False and "sin chequeo de colapso" in out["nota"]


def test_continuar_retoma_con_lo_que_guardo_la_corrida(entorno):
  import torch
  from src import recipes as R
  rec = R.legacy_cosine()
  d = rutas.ckpt() / "gpt_s"
  d.mkdir(parents=True)
  torch.save({"args": {"kind": "gpt", "size": "s", "config": '{"pos": "frame"}', "seed": 3, "presupuesto": "2x", "tokens": None,
                       "kd": True, "alpha": 0.5, "tau": 1.0, "data": "data/train_tokens.npy"},
              "recipe": rec.to_dict()}, d / "last.pt")
  tk = Toolkit("gpt")
  av = tk.plan("continuar").comandos[0].argv
  assert av[2:4] == ["src.train", "--kind"] and "--kd" in av and av[av.index("--seed") + 1] == "3"
  assert av[av.index("--presupuesto") + 1] == "2x" and "--extend-from" not in av
  ext = tk.plan("continuar", presupuesto="4x")
  av = ext.comandos[0].argv
  assert av[av.index("--extend-from") + 1] == "gpt_s" and "--restart" in av          # coseno: solo con restart
  assert any("no equivale" in a for a in ext.avisos)
  assert ext.comandos[1].argv[-1] == "gpt_s_x4"                                       # y se registra la extensión


def test_continuar_sin_checkpoint_solo_se_puede_imprimir(entorno, capsys):
  assert cli.main(["gpt", "continuar", "--run", "no_existe", "--dry-run"]) == 0
  assert cli.main(["gpt", "continuar", "--run", "no_existe"]) == 2
  assert "no existe" in capsys.readouterr().err


# ---------------------------------------------------------------- candado y cola

def test_candado_atomico_y_huerfano(entorno):
  ok, info = cola.tomar_candado("a", "entrenar")
  assert ok and info["pid"] == os.getpid() and info["run"] == "a"
  ok2, quien = cola.tomar_candado("b", "evaluar")
  assert not ok2 and quien["run"] == "a" and cola.liberar_candado()
  # candado huérfano: el pid ya no existe
  p = subprocess.Popen([sys.executable, "-c", "pass"])
  p.wait()
  rutas.candado().write_text(json.dumps({"pid": p.pid, "run": "muerta", "accion": "entrenar", "hora": "ayer"}))
  avisos = []
  ok, info = cola.tomar_candado("c", "evaluar", avisar=avisos.append)
  assert ok and info["run"] == "c" and any("huérfano" in a for a in avisos)
  cola.liberar_candado()
  assert not rutas.candado().exists()


def test_dos_procesos_solo_uno_toma_el_candado(entorno):
  codigo = ("import sys, time\nfrom pathlib import Path\nfrom src import cola, rutas\n"
            f"rutas.RESULTS = Path({str(rutas.results())!r})\nok, _ = cola.tomar_candado('p', 'entrenar')\n"
            "print('TOMADO' if ok else 'OCUPADO', flush=True)\ntime.sleep(float(sys.argv[1]))\n")
  env = {**os.environ, "PYTHONPATH": str(ROOT)}
  a = subprocess.Popen([sys.executable, "-c", codigo, "3"], stdout=subprocess.PIPE, text=True, env=env, cwd=ROOT)
  assert a.stdout.readline().strip() == "TOMADO"
  b = subprocess.run([sys.executable, "-c", codigo, "0"], capture_output=True, text=True, env=env, cwd=ROOT)
  assert "OCUPADO" in b.stdout
  a.wait()


def test_segunda_accion_de_gpu_se_encola(entorno):
  cola.tomar_candado("primera", "entrenar")
  salida = []
  r = ejecutor.ejecutar(juguete(cli_argv=("gpt", "evaluar", "--tamaño", "s")), salida=salida.append)
  assert r.estado == "encolada" and r.codigo == 0 and r.cola_id
  items = cola.pendientes()
  assert len(items) == 1 and items[0]["argv"] == ["gpt", "evaluar", "--tamaño", "s"]
  assert any("encolada" in s for s in salida)
  assert bitacora.leer()[-1]["encolada"] == r.cola_id
  assert "pendiente" in cola.cola_texto()
  assert cola.quitar(r.cola_id) and not cola.pendientes()
  cola.liberar_candado()


def test_encolar_explicito_y_cpu_no_toma_el_candado(entorno):
  r = ejecutor.ejecutar(juguete(), encolar=True, salida=lambda s: None)
  assert r.estado == "encolada" and not rutas.candado().exists()
  r = ejecutor.ejecutar(juguete(device="cpu"), salida=lambda s: None)    # CPU: no usa el candado ni la cola
  assert r.estado == "ok" and not rutas.candado().exists()


def test_accion_de_gpu_libera_el_candado_aunque_falle(entorno):
  r = ejecutor.ejecutar(juguete(codigo=3), salida=lambda s: None)
  assert r.codigo == 3 and r.estado == "fallada" and not rutas.candado().exists()


def test_esperar_toma_el_candado_cuando_se_libera(entorno):
  import threading
  import time
  cola.tomar_candado("primera", "entrenar")
  threading.Timer(0.3, cola.liberar_candado).start()
  t0 = time.time()
  r = ejecutor.ejecutar(juguete(), esperar=True, salida=lambda s: None, intervalo=0.1)
  assert r.estado == "ok" and time.time() - t0 >= 0.25 and not cola.pendientes()


def test_una_falla_frena_la_cola(entorno, monkeypatch, tmp_path):
  llamadas = []

  def falso(argv, **kw):
    llamadas.append((argv, kw))
    return 5 if argv[1] == "falla" else 0

  monkeypatch.setattr(cli, "main", falso)
  for a in ("uno", "falla", "tres"):
    cola.encolar(["m", a])
  hechos = []
  rc = cli.cola_correr(str(tmp_path / "no_hay.toml"), salida=hechos.append)
  assert rc == 5 and [c[0] for c in llamadas] == [["m", "uno"], ["m", "falla"]]
  assert all(kw["_esperar"] and kw["_desde_cola"] for _, kw in llamadas)
  est = {tuple(i["argv"]): i["estado"] for i in cola.cargar_cola()}
  assert est == {("m", "falla"): "fallada", ("m", "tres"): "pendiente"}      # el que corrió bien se sacó; el siguiente sigue esperando
  assert any("se frena" in h for h in hechos)


def test_la_cola_de_un_toml_frena_ante_una_falla(entorno, monkeypatch, tmp_path):
  q = tmp_path / "q.toml"
  q.write_text('[[trabajo]]\nid = 1\ncomandos = ["a x", "b y"]\n[[trabajo]]\nid = 2\ncomandos = ["c z"]\n')
  vistos = []
  monkeypatch.setattr(cli, "main", lambda argv, **kw: vistos.append(argv) or (1 if argv[0] == "b" else 0))
  assert cli.cola_correr(str(q), salida=lambda s: None) == 1
  assert vistos == [["a", "x"], ["b", "y"]]


def test_dry_run_de_la_cola_va_antes_del_separador(monkeypatch):
  # regresión: con `--` en el comando, el --dry-run agregado al final iba al script envuelto y la acción corría de verdad
  vistos = []
  monkeypatch.setattr(cli, "main", lambda argv, **kw: vistos.append(argv) or 0)
  cli._expandir("genie evaluar --barrer-temperatura -- --temperaturas 0.8,1.0")
  cli._expandir("gpt entrenar --tamaño s")
  assert vistos == [["genie", "evaluar", "--barrer-temperatura", "--dry-run", "--", "--temperaturas", "0.8,1.0"],
                    ["gpt", "entrenar", "--tamaño", "s", "--dry-run"]]


def test_cola_correr_dry_run_lista_toda_la_cola(capsys):
  trabajos = cli.cargar_toml(ROOT / "scripts/queue.toml")
  assert [t["id"] for t in trabajos] == list(range(len(trabajos)))
  activos = [t for t in trabajos if not t.get("opcional")]
  opcionales = len(trabajos) - len(activos)
  assert cli.main(["cola", "correr", "--dry-run", "--archivo", str(ROOT / "scripts/queue.toml")]) == 0
  out = capsys.readouterr().out
  assert "ERROR" not in out
  for t in trabajos:
    assert f"# trabajo {t['id']}:" in out
    for c in t["comandos"]:
      assert f"python -m src.cli {c}" in out
  assert f"{len(activos)} trabajos" in out.splitlines()[-1]
  assert f"{opcionales} opcionales desactivados" in out.splitlines()[-1]
  assert out.count("[OPCIONAL, desactivado") == opcionales


def test_cola_ver_y_quitar(entorno, capsys):
  assert cli.main(["cola", "ver"]) == 0
  assert "GPU: libre" in capsys.readouterr().out
  item = cola.encolar(["gpt", "entrenar"])
  assert cli.main(["cola", "quitar", item["id"]]) == 0
  assert cli.main(["cola", "quitar", item["id"]]) == 1


# ---------------------------------------------------------------- bitácora

def test_toda_accion_deja_su_linea(entorno):
  salida = []
  r = ejecutor.ejecutar(juguete(codigo=0, escribe=(rutas.results() / "eval_toy.json", {"x": 1}), cli_argv=("gpt", "entrenar", "--tamaño", "s")),
                        salida=salida.append)
  rutas.results().mkdir(parents=True, exist_ok=True)
  e = bitacora.leer()[-1]
  assert e["id"] == r.id and e["argv"] == ["gpt", "entrenar", "--tamaño", "s"] and e["salida"] == 0
  assert e["commit"] == "abc123" and e["limpio"] is True and e["device"] == "cuda" and e["duracion_s"] >= 0
  assert e["comandos"][0]["salida"] == 0 and e["fecha"]
  # una acción fallada también queda
  ejecutor.ejecutar(juguete(codigo=7), salida=salida.append)
  assert [x["salida"] for x in bitacora.leer()] == [0, 7]


def test_anota_bitacora_en_los_json_que_escribio(entorno):
  rutas.results().mkdir(parents=True)
  destino = rutas.results() / "eval_toy.json"
  r = ejecutor.ejecutar(juguete(escribe=(destino, {"x": 1})), salida=lambda s: None)
  d = json.loads(destino.read_text())
  assert d["x"] == 1 and d["_bitacora"]["id"] == r.id and d["_bitacora"]["commit"] == "abc123"
  assert "python -m src.cli gpt entrenar" in d["_bitacora"]["comando"]
  e = bitacora.leer()[-1]
  assert any(f.endswith("eval_toy.json") for f in e["escribio"])
  assert not any(f.endswith("bitacora.jsonl") or f.endswith(".gpu.lock") for f in e["escribio"])


def test_no_anota_manifiestos(entorno):
  rutas.results().mkdir(parents=True)
  m = rutas.results() / "checkpoints.json"
  ejecutor.ejecutar(juguete(escribe=(m, {"gpt_s": {}})), salida=lambda s: None)
  assert "_bitacora" not in json.loads(m.read_text())


def test_arbol_sucio_bloquea_la_gpu(entorno, monkeypatch):
  monkeypatch.setattr(bitacora, "git_estado", lambda: ("abc123", False))
  destino = rutas.results() / "eval_sucio.json"
  rutas.results().mkdir(parents=True)
  salida = []
  r = ejecutor.ejecutar(juguete(escribe=(destino, {"x": 1})), salida=salida.append)
  assert r.estado == "rechazada" and r.codigo == 2 and not destino.exists() and not rutas.candado().exists()
  assert any("sin commitear" in s for s in salida)
  e = bitacora.leer()[-1]
  assert e["rechazada"] and e["limpio"] is False
  # CPU: corre igual (queda anotado que el árbol estaba sucio)
  assert ejecutor.ejecutar(juguete(device="cpu"), salida=lambda s: None).estado == "ok"
  assert bitacora.leer()[-1]["limpio"] is False
  # --permitir-sucio: corre y queda anotado
  r = ejecutor.ejecutar(juguete(escribe=(destino, {"x": 1})), permitir_sucio=True, salida=lambda s: None)
  assert r.estado == "ok" and destino.exists()
  assert bitacora.leer()[-1]["permitir_sucio"] is True


def test_git_estado_real_detecta_cambios_en_src():
  commit, limpio = bitacora.git_estado()
  assert commit and len(commit) == 40 and isinstance(limpio, bool)


def test_repetir_dry_run_devuelve_el_argv_original(entorno, capsys):
  r = ejecutor.ejecutar(juguete(cli_argv=("convlstm_px", "bench", "--cadena", "--ritmo", "50")), salida=lambda s: None)
  assert bitacora.repetir(r.id) == ["convlstm_px", "bench", "--cadena", "--ritmo", "50"]
  assert bitacora.repetir(r.id[:12]) == bitacora.repetir(r.id)       # alcanza con un prefijo único
  assert cli.main(["bitacora", "repetir", r.id, "--dry-run"]) == 0
  assert capsys.readouterr().out.strip() == "python -m src.cli convlstm_px bench --cadena --ritmo 50"
  assert cli.main(["bitacora", "repetir", "nada", "--dry-run"]) == 2


def test_repetir_relanza_la_misma_accion(entorno, monkeypatch):
  r = ejecutor.ejecutar(juguete(cli_argv=("gpt", "estado")), salida=lambda s: None)
  llamadas = []
  monkeypatch.setattr(cli, "accion_modelo", lambda argv, **kw: llamadas.append((argv, kw)) or 0)
  assert cli.main(["bitacora", "repetir", r.id]) == 0
  assert llamadas[0][0] == ["gpt", "estado"] and llamadas[0][1]["meta"]["repetido_de"] == r.id


def test_bitacora_verificar_detecta_vigentes_sin_entrada(entorno, tmp_path):
  estado = tmp_path / "estado.json"
  resultados.marcar("results/eval_x.json", "vigente", "test", estado)
  resultados.marcar("results/eval_viejo.json", "protocolo anterior", "test", estado)
  assert bitacora.verificar(estado) == [("results/eval_x.json", "ninguna entrada de la bitácora lo escribió")]

  def entrada(**kw):
    e = {"id": "i", "fecha": "hoy", "argv": ["gpt", "evaluar"], "argv_texto": "x", "salida": 0, "limpio": True, "commit": "abc123",
         "escribio": ["results/eval_x.json"], "permitir_sucio": False}
    e.update(kw)
    bitacora.registrar(e)

  entrada()
  assert bitacora.verificar(estado) == []
  entrada(salida=1)
  assert "salida 1" in bitacora.verificar(estado)[0][1]
  entrada(limpio=False)
  assert "sucio" in bitacora.verificar(estado)[0][1]
  entrada(commit="otro")
  assert "no está en main" in bitacora.verificar(estado)[0][1]
  entrada()   # la última entrada que lo escribió bien lo deja en regla: las falladas anteriores quedan como historia
  assert bitacora.verificar(estado) == []
  assert len(bitacora.leer()) == 5


def test_bitacora_verificar_por_cli(entorno, capsys, monkeypatch, tmp_path):
  estado = tmp_path / "estado.json"
  resultados.marcar("results/eval_x.json", "vigente", "test", estado)
  monkeypatch.setattr(resultados, "ARCHIVO", estado)
  assert cli.main(["bitacora", "verificar"]) == 1
  assert "eval_x.json" in capsys.readouterr().out


def test_bitacora_vigentes_marca_solo_lo_respaldado(entorno, capsys, monkeypatch, tmp_path):
  estado = tmp_path / "estado.json"
  monkeypatch.setattr(resultados, "ARCHIVO", estado)
  res = rutas.results()
  res.mkdir(parents=True)
  for n in ("train_a.json", "train_b.json", "train_c.json", "eval_viejo.json"):
    (res / n).write_text("{}")
  resultados.marcar(res / "eval_viejo.json", "protocolo anterior", "test", estado)
  base = {"fecha": "hoy", "argv": ["gpt", "entrenar"], "argv_texto": "python -m src.cli gpt entrenar", "salida": 0,
          "limpio": True, "commit": "abc123", "permitir_sucio": False}
  bitacora.registrar({**base, "id": "ok", "escribio": [str(res / "train_a.json")]})
  bitacora.registrar({**base, "id": "sucia", "limpio": False, "escribio": [str(res / "train_b.json")]})
  assert cli.main(["bitacora", "vigentes"]) == 1          # train_b (árbol sucio) y train_c (sin entrada) no se marcan
  out = capsys.readouterr().out
  reg = resultados.cargar(estado)
  ka, kb, kc = (resultados.clave(res / n) for n in ("train_a.json", "train_b.json", "train_c.json"))
  assert reg[ka]["estado"] == "vigente" and "bitácora ok" in reg[ka]["motivo"]
  assert kb not in reg and kc not in reg and "sucio" in out and "ninguna entrada" in out
  assert reg[resultados.clave(res / "eval_viejo.json")]["estado"] == "protocolo anterior"   # lo ya marcado no se toca
  bitacora.registrar({**base, "id": "ok2", "escribio": [str(res / "train_b.json"), str(res / "train_c.json")]})
  assert cli.main(["bitacora", "vigentes"]) == 0 and resultados.listar(estado, res)["sin estado"] == []


def test_accion_real_por_la_cli_queda_en_la_bitacora(entorno, capsys):
  """`estado` es de CPU y rápida: pasa por Toolkit -> ejecutor -> subproceso de verdad."""
  assert cli.main(["gpt", "estado", "--", "--ckpt-dir", str(rutas.ckpt())]) == 0
  e = bitacora.leer()[-1]
  assert e["argv"] == ["gpt", "estado", "--", "--ckpt-dir", str(rutas.ckpt())] and e["salida"] == 0 and e["device"] == "cpu"
  assert e["comandos"][0]["argv"][:3] == ["python", "-m", "src.status"]
  assert cli.main(["bitacora", "ver"]) == 0
  assert "gpt estado" in capsys.readouterr().out


# ---------------------------------------------------------------- opciones de inferencia

def test_ctx_frames_aleatorio_e_int8():
  rol = [c.argv for c in Toolkit("gpt").plan("evaluar", ctx_frames=4).comandos if c.argv[2] == "src.rollouts"][0]
  assert rol[rol.index("--name") + 1] == "gpt_s_ctx4" and rol[rol.index("--ctx-frames") + 1] == "4"
  b = Toolkit("gpt").plan("bench", aleatorio=True, tamano="l", int8=True, device="cpu").comandos[0].argv
  assert b[b.index("--model") + 1] == "random:gpt_l" and "--int8" in b and b[b.index("--device") + 1] == "cpu"


def test_cadena_modos_y_ritmo():
  tk = Toolkit("convlstm_px")
  modos = lambda **kw: [c.argv[c.argv.index("--mode") + 1] for c in tk.plan("bench", cadena=True, **kw).comandos]
  assert modos(modo="online") == ["online"] and modos(modo="imagine", ritmo=50.0) == ["imagine"]
  assert "--protocolo" in tk.plan("bench", cadena=True).comandos[0].argv       # sin --modo: el protocolo único (D15)
  with pytest.raises(ToolkitError, match="solo con --cadena"):
    tk.plan("bench", ritmo=50.0)


# ---------------------------------------------------------------- segunda vuelta

def test_entrenar_se_niega_si_la_corrida_ya_existe(entorno, capsys):
  d = rutas.ckpt() / "gpt_s"
  d.mkdir(parents=True)
  (d / "last.pt").write_bytes(b"x")
  assert cli.main(["gpt", "entrenar", "--tamaño", "s"]) == 2
  err = capsys.readouterr().err
  assert "ya existe" in err and "continuar" in err and "--forzar" in err
  assert cli.main(["gpt", "entrenar", "--tamaño", "s", "--forzar", "--dry-run"]) == 0
  assert cli.main(["gpt", "entrenar", "--tamaño", "s", "--variante", "otra", "--dry-run"]) == 0
  (rutas.ckpt() / "gpt_humo_x3").mkdir()
  (rutas.ckpt() / "gpt_humo_x3" / "final.pt").write_bytes(b"x")
  assert cli.main(["gpt", "entrenar", "--humo"]) == 2


def test_evaluar_usa_la_temperatura_elegida_de_la_familia(entorno):
  rutas.results().mkdir(parents=True)
  tk = Toolkit("gpt")
  p = tk.plan("evaluar", variante="kd")
  assert "--temperature" not in p.comandos[1].argv and p.meta["temperatura"]["origen"] == "default del modelo"
  (rutas.results() / "temperatura_gpt_s_kd.json").write_text(json.dumps({"elegida": 0.8}))
  p = tk.plan("evaluar", variante="kd")
  roll = p.comandos[1].argv
  assert roll[roll.index("--temperature") + 1] == "0.8" and "temperatura_gpt_s_kd.json" in p.meta["temperatura"]["origen"]
  assert roll[roll.index("--name") + 1] == "gpt_s_kd"                     # la T elegida no cambia el nombre
  p = tk.plan("evaluar", variante="kd", temperatura=1.0)                   # la explícita gana
  assert p.comandos[1].argv[p.comandos[1].argv.index("--temperature") + 1] == "1.0" and p.meta["temperatura"]["origen"] == "--temperatura"
  p = tk.plan("evaluar", variante="kd", temperatura=0.5)
  assert p.comandos[1].argv[p.comandos[1].argv.index("--name") + 1] == "gpt_s_kd_t0.5"


def test_la_temperatura_queda_en_la_bitacora_y_en_el_json(entorno):
  rutas.results().mkdir(parents=True)
  plan = juguete(escribe=(rutas.results() / "eval_t.json", {"x": 1}))
  plan.meta = {"temperatura": {"valor": 0.8, "origen": "barrido"}}
  ejecutor.ejecutar(plan, salida=lambda s: None)
  assert json.loads((rutas.results() / "eval_t.json").read_text())["_bitacora"]["meta"]["temperatura"]["valor"] == 0.8
  assert bitacora.leer()[-1]["meta"]["temperatura"]["origen"] == "barrido"


def test_desde_cache_no_rehace_rollouts_que_existen(entorno):
  (rutas.results() / "rollouts").mkdir(parents=True)
  (rutas.results() / "rollouts" / "maskgit_s_s8.npy").write_bytes(b"x")
  (rutas.results() / "eval_maskgit_s.json").write_text("{}")
  p = Toolkit("maskgit").plan("evaluar", pasos="8,12", desde_cache=True)
  mods = [(c.argv[2], c.argv[c.argv.index("--name") + 1]) for c in p.comandos]
  assert mods == [("src.metrics_video", "maskgit_s_s8"), ("src.rollouts", "maskgit_s_s12"), ("src.metrics_video", "maskgit_s_s12")]
  assert "--from-cache" in p.comandos[0].argv and "--from-cache" in p.comandos[2].argv
  roll = p.comandos[1].argv
  assert roll[roll.index("--temperature") + 1] == "1.0"                    # protocolo viejo: T = 1
  todo = Toolkit("transicion").plan("evaluar", desde_cache=True)           # sin rollouts en disco temporal: los rehace
  assert [c.argv[2] for c in todo.comandos] == ["src.evaluate", "src.rollouts", "src.metrics_video"] and "copy" in todo.comandos[1].argv


def test_desde_cache_completa_no_usa_la_gpu(entorno, monkeypatch):
  monkeypatch.setattr(ejecutor, "resolver_device", lambda d=None: d or "cuda")
  (rutas.results() / "rollouts").mkdir(parents=True)
  (rutas.results() / "rollouts" / "copy.npy").write_bytes(b"x")
  (rutas.results() / "eval_copy.json").write_text("{}")
  p = Toolkit("transicion").plan("evaluar", desde_cache=True)
  assert [c.argv[2] for c in p.comandos] == ["src.metrics_video"] and p.device == "cpu"
  assert Toolkit("transicion").plan("evaluar").device == "cuda"


def test_cadena_usa_el_protocolo_y_el_diagnostico():
  tk = Toolkit("genie")
  a = tk.plan("bench", cadena=True, pasos="4", ritmo=50.0).comandos[0].argv
  assert "--protocolo" in a and a[a.index("--pace") + 1] == "50" and "--mode" not in a
  d = tk.plan("bench", cadena=True, pasos="4", ritmo=50.0, diagnostico=True).comandos[0].argv
  assert "--diagnostico" in d and "--protocolo" not in d
  with pytest.raises(ToolkitError, match="solo con --cadena"):
    tk.plan("bench", diagnostico=True)


def test_evaluar_cadena_envuelve_quality():
  p = Toolkit("genie").plan("evaluar", pasos="4", cadena=True, limite=5)
  assert [c.argv[2] for c in p.comandos] == ["src.infer.quality", "src.infer.quality", "src.metrics_video"]
  assert p.comandos[1].argv[p.comandos[1].argv.index("--name") + 1] == "genie_s_p4"
  assert p.comandos[2].argv[-1] == "genie_s_p4_stream" and "--segments" in p.comandos[0].argv


def test_real_con_clips_y_estadistica(capsys):
  p = Toolkit("real").plan("evaluar", clips="eval_b")
  assert p.comandos[0].argv[-2:] == ["--clips", "eval_b"] and "real_eval_b" in p.comandos[0].argv
  assert cli.main(["estadistica", "diferencia", "--a", "x", "--b", "y", "--dry-run"]) == 0
  assert capsys.readouterr().out.strip() == "python -m src.estadistica diferencia --a x --b y"


def test_estadistica_es_de_cpu_y_queda_en_la_bitacora(entorno, monkeypatch):
  monkeypatch.setattr(ejecutor, "correr_comando", lambda c, salida=None: 0)
  assert cli.main(["estadistica", "piso", "--reps", "2"]) == 0
  e = bitacora.leer()[-1]
  assert e["accion"] == "estadistica" and e["device"] == "cpu" and e["argv"] == ["estadistica", "piso", "--reps", "2"]


def test_verificar_arma_pytest_y_bitacora(capsys):
  assert cli.main(["verificar", "--dry-run"]) == 0
  out = capsys.readouterr().out
  assert "CUDA_VISIBLE_DEVICES=-1 python -m pytest -q" in out and "bitacora verificar" in out
  assert cli.main(["verificar", "--cuda", "--dry-run"]) == 0
  assert "CUDA_VISIBLE_DEVICES" not in capsys.readouterr().out


def test_estado_filtra_por_modelo(tmp_path):
  from src import status
  for run, kind in (("gpt_s_kd", "gpt"), ("vjepa2_s", "vjepa2"), ("vjepa2_ac", "vjepa2_ac"), ("genie_s", None)):
    d = tmp_path / run
    d.mkdir()
    (d / "final.pt").write_bytes(b"x")
    if kind:
      (d / "status.json").write_text(json.dumps({"estado": "lista", "kind": kind, "paso": 1, "total": 1, "porcentaje": 100}))
  kinds = ("gpt", "genie", "vjepa2", "vjepa2_ac")
  nombres = lambda k: [r[0] for r in status.rows(str(tmp_path), kind=k, kinds=kinds)[1:]]
  assert nombres("vjepa2") == ["vjepa2_s"] and nombres("vjepa2_ac") == ["vjepa2_ac"] and nombres("genie") == ["genie_s"]
  assert len(nombres(None)) == 4


def test_los_opcionales_se_saltean_salvo_pedido(entorno, monkeypatch, tmp_path):
  q = tmp_path / "q.toml"
  q.write_text("""[[trabajo]]
id = 1
comandos = ['a x']
[[trabajo]]
id = 2
opcional = true
comandos = ['b y']
""")
  vistos = []
  monkeypatch.setattr(cli, "main", lambda argv, **kw: vistos.append(argv[0]) or 0)
  assert cli.cola_correr(str(q), salida=lambda s: None) == 0 and vistos == ["a"]
  vistos.clear()
  cli.cola_correr(str(q), trabajos=[2], salida=lambda s: None)
  assert vistos == ["b"]
  vistos.clear()
  cli.cola_correr(str(q), incluir_opcionales=True, salida=lambda s: None)
  assert vistos == ["a", "b"]


# ---------------------------------------------------------------- solo datos actuales

def test_configs_se_derivan_de_results(tmp_path, monkeypatch):
  from src import configs
  monkeypatch.setattr(configs, "R", tmp_path)
  (tmp_path / "bench").mkdir()
  (tmp_path / "video_genie_s_p8.json").write_text("{}")
  (tmp_path / "video_genie_s_p8_stream.json").write_text("{}")
  (tmp_path / "eval_genie_s.json").write_text("{}")
  (tmp_path / "video_gpt_s_kd.json").write_text("{}")
  (tmp_path / "train_genie_s.json").write_text(json.dumps({"params": 12_000_000}))
  (tmp_path / "bench" / "genie_s_p8__fp16_graphs.json").write_text(json.dumps({"median_ms": 5.0}))
  (tmp_path / "bench" / "gpt_s__fp16_graphs.json").write_text(json.dumps({"median_ms": 7.0}))   # misma arquitectura que gpt_s_kd
  (tmp_path / "latent_lewm.json").write_text("{}")
  C = {c["label"]: c for c in configs.main([])}
  assert set(C) == {"genie_s", "genie_s_p8", "gpt_s_kd", "lewm"}
  g = C["genie_s_p8"]
  assert g["steps"] == 8 and g["size"] == "s" and g["params"] == 12_000_000 and g["eval"].endswith("eval_genie_s.json")
  assert g["bench"].endswith("genie_s_p8__fp16_graphs.json") and g["family"] == "genie"
  assert C["gpt_s_kd"]["family"] == "gpt_kd" and C["gpt_s_kd"]["bench"].endswith("gpt_s__fp16_graphs.json")
  assert json.loads((tmp_path / "configs.json").read_text())[0]["label"]


def test_un_resultado_sin_marca_es_vigente(entorno, tmp_path):
  from src import report
  f = tmp_path / "video_x.json"
  f.write_text("{}")
  assert report.estado_config({"video": str(f)}, {}) == "vigente"
  assert report.estado_config({"video": str(f)}, {resultados.clave(f): {"estado": "pendiente"}}) == "pendiente"


def test_verificar_exige_entrada_a_todo_resultado_existente(entorno, monkeypatch):
  rutas.results().mkdir(parents=True)
  (rutas.results() / "eval_suelto.json").write_text("{}")      # sin estado.json: existe, luego es vigente
  vacio = rutas.results() / "estado.json"
  vacio.write_text("{}")
  assert [k for k, _ in bitacora.verificar(vacio)] == [resultados.clave(rutas.results() / "eval_suelto.json")]


def test_humo_candidatos_acota_el_humo():
  p = Toolkit("genie").plan("entrenar", humo=True, humo_candidatos="fuente,raiz,x3")
  assert [c.argv[c.argv.index("--run") + 1] for c in p.comandos if c.argv[2] == "src.train"] == \
    ["genie_humo_fuente", "genie_humo_raiz", "genie_humo_x3"]
  with pytest.raises(ToolkitError, match="fuente siempre"):
    Toolkit("genie").plan("entrenar", humo=True, humo_candidatos="raiz")
  with pytest.raises(ToolkitError, match="solo con --humo"):
    Toolkit("genie").plan("entrenar", humo_candidatos="fuente")


def test_humo_candidatos_acepta_cualquier_multiplo():
  """Si el ganador queda en el borde de la grilla, se extiende con x<k> (la cola lo usó con x30)."""
  p = Toolkit("genie").plan("entrenar", humo=True, humo_candidatos="fuente,x10,x30")
  train = [c.argv for c in p.comandos if c.argv[2] == "src.train"]
  assert [a[a.index("--run") + 1] for a in train] == ["genie_humo_fuente", "genie_humo_x10", "genie_humo_x30"]
  from src import schedules as S
  from src.models import model_class
  pico = S.peak(model_class("genie").recipe("s").lr_segs)  # lr de la receta de Genie
  assert [float(a[a.index("--lr") + 1]) for a in train[1:]] == pytest.approx([pico * 10, pico * 30])
  for malo in ("fuente,x0", "fuente,xabc", "fuente,y3"):
    with pytest.raises(ToolkitError, match="x<k>"):
      Toolkit("genie").plan("entrenar", humo=True, humo_candidatos=malo)


def test_exportar_y_demo_de_latentes():
  assert [c.argv[2:] for c in Toolkit("lewm").plan("exportar").comandos] == [
    ["src.export_latente", "--model", "lewm", "--parte", "encoder"], ["src.export_latente", "--model", "lewm", "--parte", "predictor"]]
  assert Toolkit("vjepa2").plan("exportar").comandos[0].argv[-1] == "encoder"
  assert Toolkit("vjepa2_ac").plan("exportar").comandos[0].argv[-1] == "predictor"
  assert Toolkit("vjepa2").plan("demo").comandos[0].argv[2:] == ["src.infer.demo_latente", "--model", "vjepa2_s"]


def test_reporte_deriva_configs_primero():
  assert [c.argv[2] for c in T.plan_general("reporte").comandos] == ["src.configs", "src.report"]


def test_mps_es_una_accion_de_gpu(entorno, monkeypatch):
  p = Toolkit("gpt").plan("evaluar", device="mps")
  assert p.device == "mps" and p.gpu and p.comandos[0].env == {"ML2_DEVICE": "mps"}
  assert p.comandos[0].argv[0] == "python"                      # se lanza con sys.executable, sin rutas fijas
  cola.tomar_candado("otra", "entrenar")
  r = ejecutor.ejecutar(p, salida=lambda s: None)
  assert r.estado == "encolada"                                 # toma el candado y pasa por la cola como cuda
  cola.liberar_candado()
  with pytest.raises(ToolkitError, match="solo mide en cuda o cpu"):
    Toolkit("gpt").plan("bench", device="mps")


def test_accion_concurrente_no_se_queda_con_el_resultado(entorno, monkeypatch, tmp_path):
  """Trabajo 12: `verificar` corrió en CPU mientras terminaba `vjepa2_ac evaluar`, se atribuyó su JSON y le pisó el
  sello; `vigentes` no lo marcaba porque `verificar` salió con 1."""
  from datetime import datetime
  estado = tmp_path / "estado.json"
  monkeypatch.setattr(resultados, "ARCHIVO", estado)
  res = rutas.results()
  res.mkdir(parents=True)
  f = res / "latent_x.json"
  base = {"limpio": True, "commit": "abc123", "permitir_sucio": False}
  ev = {**base, "id": "eval", "fecha": "2026-11-18T20:03:12", "duracion_s": 4325.1, "salida": 0, "argv": ["x", "evaluar"],
        "argv_texto": "python -m src.cli x evaluar", "escribio": [str(f)]}
  ve = {**base, "id": "verif", "fecha": "2026-11-18T21:14:21", "duracion_s": 159.8, "salida": 1, "argv": ["verificar"],
        "argv_texto": "python -m src.cli verificar", "escribio": [str(f)]}
  bitacora.registrar(ev)
  bitacora.registrar(ve)
  f.write_text(json.dumps({"x": 1, "_bitacora": {"id": "verif", "comando": "python -m src.cli verificar", "commit": "abc123"}}))
  inicio_verif = datetime.fromisoformat("2026-11-18T21:14:21").timestamp()
  assert bitacora._relativa(f) in bitacora.ajenos(inicio_verif)          # al registrar: la evaluación terminó en su ventana
  assert bitacora.ajenos(inicio_verif + 3600) == set()
  marcados, problemas = bitacora.marcar_vigentes(estado)
  assert problemas == [] and [Path(m).name for m in marcados] == ["latent_x.json"]
  d = json.loads(f.read_text())
  assert d["x"] == 1 and d["_bitacora"]["id"] == "eval" and "x evaluar" in d["_bitacora"]["comando"]   # sello corregido
  assert bitacora.verificar(estado) == []
  bitacora.registrar({**ev, "id": "eval2", "fecha": "2026-11-18T22:00:00"})  # en secuencia manda la última, como siempre
  assert bitacora._escritores(bitacora.leer())[bitacora._relativa(f)]["id"] == "eval2"


def test_el_ejecutor_no_reclama_lo_de_otra_accion(entorno, monkeypatch):
  rutas.results().mkdir(parents=True)
  destino = rutas.results() / "eval_toy.json"
  monkeypatch.setattr(bitacora, "ajenos", lambda inicio, entradas=None: {bitacora._relativa(destino)})
  ejecutor.ejecutar(juguete(escribe=(destino, {"x": 1})), salida=lambda s: None)
  e = bitacora.leer()[-1]
  assert not any(f.endswith("eval_toy.json") for f in e["escribio"]) and "_bitacora" not in json.loads(destino.read_text())
