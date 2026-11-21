"""src/nbui.py: con un log sintético dibuja barra y curvas; sin ningún JSON todo muestra "pendiente" y no falla."""
import json
import os
import time

import matplotlib
import pytest

matplotlib.use("Agg")

from src import nbui, rutas  # noqa: E402
from src.recipes import legacy_cosine  # noqa: E402


@pytest.fixture
def repo(tmp_path, monkeypatch):
  """results/ y checkpoints/ vacíos en un directorio temporal."""
  monkeypatch.chdir(tmp_path)
  monkeypatch.setattr(rutas, "RESULTS", tmp_path / "results")
  monkeypatch.setattr(rutas, "CKPT", tmp_path / "checkpoints")
  (tmp_path / "results").mkdir()
  (tmp_path / "checkpoints").mkdir()
  return tmp_path


def corrida_sintetica(repo, run="demo_s", total=1000, hasta=600):
  d = repo / "checkpoints" / run
  d.mkdir(parents=True)
  r = legacy_cosine()
  with open(d / "log.jsonl", "w") as f:
    for s in range(50, hasta + 1, 50):
      f.write(json.dumps({"step": s, "tokens": s * 100, "lr": r.lr(s, total), "loss": 3.0 / (1 + s / 100), "tok_s": 1e5, "elapsed": s}) + "\n")
      if s % 200 == 0:
        f.write(json.dumps({"step": s, "tokens": s * 100, "monitor_ce": 3.0 / (1 + s / 80)}) + "\n")
    f.write('{"step": 650, "lr"')  # línea cortada: se salta
  (d / "status.json").write_text(json.dumps({
    "estado": "en curso", "paso": hasta, "total": total, "porcentaje": 100 * hasta / total, "loss": 0.4, "lr": 1e-3, "tramo": "coseno",
    "frames_s": 1500.0, "eta_s": 400, "eta_hora": "2026-11-18T12:30:00", "pid": os.getpid(), "hora": time.time(), "run": run}))
  (repo / "results" / f"train_{run}.json").write_text(json.dumps({"receta": r.to_dict(), "steps": total}))
  return d


# ---------------------------------------------------------------- entrenamiento

def test_log_y_status(repo):
  corrida_sintetica(repo)
  reg = nbui.leer_log("demo_s")
  assert len(reg) > 10 and all("step" in x for x in reg)   # la línea cortada no entra
  assert nbui.leer_status("demo_s")["paso"] == 600
  assert nbui.leer_log("no_existe") == [] and nbui.leer_status("no_existe") is None


def test_barra(repo, capsys):
  corrida_sintetica(repo)
  st = nbui.mostrar_barra("demo_s")
  out = capsys.readouterr().out
  assert "60.0 %" in out and "paso 600/1000" in out and "ETA" in out
  assert st["paso"] == 600
  h = nbui.barra_html({"run": "x", "estado": "lista", "porcentaje": 100, "paso": 5, "total": 5})
  assert "100.0 %" in h and "width:100.0%" in h


def test_barra_sin_corrida_es_pendiente(repo, capsys):
  assert nbui.mostrar_barra("nada").startswith("pendiente")
  assert "pendiente" in capsys.readouterr().out


def test_curvas_con_tramos(repo):
  corrida_sintetica(repo)
  tb = nbui.tramos("demo_s")
  assert tb and tb[0][1] == 0 and tb[-1][2] >= 1000     # el schedule completo, para sombrearlo
  assert len(tb) >= 2                                  # warmup + coseno
  fig = nbui.curvas("demo_s", mostrar=False)
  assert fig is not None and len(fig.axes) == 3
  assert len(fig.axes[1].lines[0].get_xdata()) == len(nbui.leer_log("demo_s")) - 3   # una línea de lr por registro de entrenamiento
  assert len(fig.axes[2].lines[0].get_xdata()) == 3  # tres puntos de monitoreo
  assert any(len(ax.patches) >= 2 for ax in fig.axes)  # sombreado de los tramos


def test_curvas_sin_log(repo, capsys):
  assert nbui.curvas("nada") is None
  assert "pendiente" in capsys.readouterr().out


def test_en_vivo_termina_y_pendiente(repo, capsys):
  corrida_sintetica(repo)
  st = nbui.en_vivo("demo_s", cada=0.01, max_seg=0.05)   # la corrida "en curso" no termina: vuelve por max_seg
  assert st["paso"] == 600
  assert nbui.en_vivo("nada").startswith("pendiente")


# ---------------------------------------------------------------- paneles

def test_paneles_vacios(repo, capsys):
  r = nbui.panel_cola()
  assert r["candado"] is None and r["cola"] == []
  assert nbui.panel_bitacora() == []
  out = capsys.readouterr().out
  assert "GPU libre" in out and "vacía" in out


def test_paneles_con_datos(repo, capsys):
  from src import bitacora, cola
  cola.encolar(["genie", "entrenar", "--tamaño", "s"], "genie", "entrenar", "genie_s", motivo="prueba")
  bitacora.registrar({"id": "20260101-000000-abcd", "fecha": "2026-01-01T00:00:00", "argv": ["gpt", "receta"], "argv_texto": "python -m src.cli gpt receta",
                      "salida": 0, "duracion_s": 1.5, "limpio": True})
  assert len(nbui.panel_cola()["cola"]) == 1
  assert len(nbui.panel_bitacora()) == 1
  out = capsys.readouterr().out
  assert "genie entrenar" in out and "python -m src.cli gpt receta" in out


# ---------------------------------------------------------------- resultados

def video_json(fvd, lo, hi, curva=(0.1, 0.2, 0.3)):
  return {"n_clips": 400, "lpips": {"curve": list(curva), "at": {"1": {"mean": 0.3, "lo": 0.2, "hi": 0.4}}},
          "fvd": {"1": {"value": fvd, "lo": lo, "hi": hi}, "3": {"value": fvd * 2, "lo": 0, "hi": 0}},
          "piso": {"1": {"value": 110.0}}}


def test_escanear_prefiere_la_rtx_sobre_otro_dispositivo(repo):
  b, i = repo / "results" / "bench", repo / "results" / "infer"
  b.mkdir()
  i.mkdir()
  (b / "lewm__fp32.json").write_text(json.dumps({"median_ms": 3.0}))                    # RTX: sin etiqueta
  (b / "lewm__fp32__radeon860m.json").write_text(json.dumps({"median_ms": 1.0}))        # más rápido, pero es de la laptop
  (b / "lewm__fp32_cpu__ryzen7350.json").write_text(json.dumps({"median_ms": 0.5}))     # CPU: no cuenta
  (i / "latencia_lewm__radeon860m.json").write_text("{}")
  (i / "latencia_lewm__rtx3080ti.json").write_text("{}")
  (b / "vjepa2_ac__fp32__radeon860m.json").write_text(json.dumps({"median_ms": 9.0}))   # solo en la laptop: es lo que hay
  e = nbui.escanear()
  assert e["lewm"]["bench"].name == "lewm__fp32.json" and e["lewm"]["e2e"].name == "latencia_lewm__rtx3080ti.json"
  assert e["vjepa2_ac"]["bench"].name == "vjepa2_ac__fp32__radeon860m.json"


def test_sin_json_todo_es_pendiente(repo, capsys):
  assert nbui.escanear() == {}
  assert nbui.filas_video() == [] and nbui.filas_latente() == []
  nbui.tabla_video()
  nbui.tabla_latente()
  assert nbui.pareto(mostrar=False) is None and nbui.horizonte(mostrar=False) is None
  filas = nbui.estado_modelos()
  assert filas and all(f[2] == "pendiente" and f[3] == "pendiente" for f in filas)
  assert nbui.figura(repo / "results" / "nada.png").startswith("pendiente")
  assert "pendiente" in capsys.readouterr().out


def test_tabla_video_y_pareto(repo, capsys):
  R = repo / "results"
  (R / "bench").mkdir()
  for n, fvd, ms in (("modelo_a", 200.0, 5.0), ("modelo_b", 150.0, 20.0), ("modelo_c", 300.0, 30.0)):
    (R / f"video_{n}.json").write_text(json.dumps(video_json(fvd, fvd - 10, fvd + 10)))
    (R / "bench" / f"{n}__fp16_graphs.json").write_text(json.dumps({"median_ms": ms, "p99_ms": ms * 2, "params": 1000}))
  (R / "eval_modelo_a.json").write_text(json.dumps({"ce_all": 4.2, "ce_last": 4.5}))
  fs = nbui.filas_video()
  assert [f["modelo"] for f in fs] == ["modelo_a", "modelo_b", "modelo_c"]
  a = fs[0]
  assert a["ce"] == 4.2 and a["fvd1"] == 200.0 and a["ms"] == 5.0 and a["piso1"] == 110.0 and a["estado"] == "vigente"
  nbui.tabla_video()
  out = capsys.readouterr().out
  assert "modelo_b" in out and "150 [140, 160]" in out and "piso de FVD" in out
  fig = nbui.pareto(mostrar=False)
  assert fig is not None
  assert nbui._pareto([5, 20, 30], [200, 150, 300]) == [True, True, False]   # c lo domina b
  assert nbui.horizonte(mostrar=False) is not None


def test_protocolo_anterior_no_entra_en_figuras(repo):
  R = repo / "results"
  (R / "video_viejo.json").write_text(json.dumps(video_json(100, 90, 110)))
  (R / "estado.json").write_text(json.dumps({"results/video_viejo.json": {"estado": "protocolo anterior", "motivo": "", "fecha": "2026-01-01"}}))
  # la clave del estado es relativa al cwd (que es el repo temporal)
  assert nbui.filas_video()[0]["estado"] == "protocolo anterior"
  assert nbui.horizonte(mostrar=False) is None


def test_tabla_latente(repo, capsys):
  R = repo / "results"
  d = {"params": 15e6, "horizonte": {"mse": {"razon_modelo_sobre_copiar": [{"valor": 0.9, "ic95": [0.8, 1.0]}, {"valor": 0.7, "ic95": [0.6, 0.8]}]}},
       "probes": {"obs": {"modelo": {"velocidad": {"r2": {"valor": 0.6}}}, "sin_entrenar": {"velocidad": {"r2": {"valor": 0.3}}},
                          "pixeles": {"velocidad": {"r2": {"valor": 0.2}}}}},
       "sorpresa": {"teletransporte": {"modelo": {"razon": {"valor": 3.0}}, "referencia_copiar_ultimo": {"razon": {"valor": 1.0}}}},
       "cem": {"aplica": True, "h1": {"ms_por_plan": 12.0, "velocidad": {"mae_cem": {"valor": 0.1}, "mae_repetir_ultima": {"valor": 0.2}}}}}
  (R / "latent_lewm.json").write_text(json.dumps(d))
  f = nbui.filas_latente()[0]
  assert f["razon_ultimo"] == 0.7 and f["r2"] == 0.6 and f["r2_sin_entrenar"] == 0.3 and f["cem_ref"] == 0.2 and f["sorpresa"] == 3.0
  nbui.tabla_latente()
  assert "lewm" in capsys.readouterr().out


def test_json_roto_no_falla(repo):
  R = repo / "results"
  (R / "video_roto.json").write_text("{no es json")
  fs = nbui.filas_video()
  assert fs[0]["fvd1"] is None
  nbui.tabla_video()


def test_estado_modelos_hecho(repo):
  R = repo / "results"
  (R / "latent_lewm.json").write_text("{}")
  (repo / "checkpoints" / "lewm").mkdir()
  (repo / "checkpoints" / "lewm" / "final.pt").write_bytes(b"x")
  f = {x[0]: x for x in nbui.estado_modelos()}
  assert f["lewm"] == ("lewm", "hecho", "hecho", "pendiente")
  assert f["unigram"][1] == "n/a"


# ---------------------------------------------------------------- opciones y lanzar

def test_opciones_con_comando_de_cli(repo, capsys):
  cmd = nbui.opciones("genie", "entrenar", tamano="s", semilla=1)
  assert cmd == "python -m src.cli genie entrenar --tamaño s --semilla 1"
  out = capsys.readouterr().out
  assert "--presupuesto" in out and "frames" in out   # el default sale de la receta
  assert nbui.comando_cli("lewm", "evaluar", run="lewm", limite=50, device="cpu", dry_run=True).endswith("--run lewm --device cpu --limite 50")


def test_opciones_accion_que_no_existe(repo):
  assert nbui.opciones("unigram", "entrenar").startswith("pendiente")


def test_lanzar_dry_run(repo, capsys):
  t = nbui.lanzar("gpt", "receta", dry_run=True)
  assert t.ok and t.terminado and t.estado == "dry-run"
  assert "src.recipes" in capsys.readouterr().out


def test_lanzar_pedido_invalido_no_lanza_excepcion(repo):
  t = nbui.lanzar("gpt", "entrenar", dry_run=True, tamano="z")
  assert t.error and "tamaño" in t.error and not t.ok


def test_lanzar_segundo_plano_no_bloquea(tmp_path, monkeypatch):
  monkeypatch.setattr(rutas, "RESULTS", tmp_path / "results")   # la bitácora va al directorio temporal; el cwd sigue siendo el repo
  t = nbui.lanzar("gpt", "receta", dry_run=False)      # corre `python -m src.recipes gpt` en un subproceso, en un hilo
  assert t.hilo is not None
  t.esperar(120)
  assert (tmp_path / "results" / "bitacora.jsonl").exists()   # quedó en la bitácora
  assert t.terminado and t.ok, t.lineas


def test_corridas_disponibles(repo):
  (repo / "results" / "checkpoints.json").write_text(json.dumps({"gpt_s": {}, "_rollouts": {}}))
  (repo / "checkpoints" / "genie_s").mkdir()
  (repo / "checkpoints" / "genie_s" / "final.pt").write_bytes(b"x")
  d = nbui.corridas_disponibles()
  assert d == {"gpt_s": "sin entrenar todavía", "genie_s": "entrenada"}
  assert nbui.modelo_de_corrida("gpt_s_kd") == "gpt" and nbui.modelo_de_corrida("vjepa2_ac_x") == "vjepa2_ac"


def test_backends_y_fuente():
  disp = nbui.backends(mostrar=False)
  assert "cpu" in disp and nbui.device_inicial(["cpu", "cuda"]) == "cuda" and nbui.device_inicial(["cpu"]) == "cpu"
  assert nbui.fuente_video_extra("val", 3, 10) == ["--segment", "3", "--start", "10"]
  assert nbui.fuente_video_extra("ejemplo") == ["--start", "0"]
  with pytest.raises(ValueError):
    nbui.fuente_video_extra("mp4")


def test_fidelidad_y_receta(capsys):
  assert nbui.fidelidad("genie", mostrar=False)
  assert nbui.fidelidad("modelo_que_no_existe", mostrar=False) == []
  assert nbui.receta("genie") is not None
  assert nbui.receta("unigram") is None


# ---------------------------------------------------------------- secciones propias y demo de la clase

def test_figuras_propias_sin_datos(repo, capsys):
  assert nbui.destapar_frame(pasos=4, choice_temp=4.5, mostrar=False) is not None
  assert nbui.calendario_destapado((1, 4), mostrar=False) is not None
  assert nbui.mascaras_vjepa2(mostrar=False) is not None
  for f in (nbui.curva_pasos, nbui.horizonte_latente, nbui.sorpresa_fig, nbui.probes_fig):
    assert f("x", mostrar=False).startswith("pendiente")
  nbui.cem_tabla("x")
  nbui.ce_por_posicion(mostrar=False)
  nbui.codec_resultados()
  nbui.latencia_etapas()
  nbui.nodos_cpu()
  assert nbui.nitidez_video(repo / "no.mp4", mostrar=False).startswith("pendiente")
  assert nbui.muestras().startswith("pendiente")
  assert "pendiente" in capsys.readouterr().out


def test_destapado_sigue_el_calendario_coseno(repo):
  from src.models import genie as G
  fig = nbui.destapar_frame(pasos=8, choice_temp=0.0, temperatura=0.0, mostrar=False)
  titulos = [ax.get_title() for ax in fig.axes]
  esperado = [G.n_masked(s - 1, 8) if s > 1 else G.CODES for s in range(1, 9)]
  assert [int(t.split("\n")[1].split()[0]) for t in titulos[:8]] == esperado


def test_resultados_latentes_y_pasos_con_datos(repo):
  R = repo / "results"
  for p, fvd in ((1, 300.0), (4, 200.0)):
    (R / f"video_genie_s_p{p}.json").write_text(json.dumps({"fvd": {"1": {"value": fvd, "lo": fvd - 5, "hi": fvd + 5}}}))
  assert nbui.curva_pasos("genie_s", mostrar=False) is not None
  d = {"horizonte": {"segundos": [0.25, 0.5], "mse": {k: [{"valor": v, "ic95": [v - .1, v + .1]}] * 2
                                                        for k, v in (("modelo", .5), ("copiar_ultimo", .8), ("accion_nula", .6))}},
       "probes": {"obs": {k: {"velocidad": {"r2": {"valor": v}}, "giro": {"r2": {"valor": v}}} for k, v in
                          (("modelo", .6), ("sin_entrenar", .3), ("pixeles", .2))}},
       "sorpresa": {"paso_del_corte": 3, "teletransporte": {"por_paso_original": [1, 1, 1], "por_paso_perturbado": [1, 1, 3],
                                                            "modelo": {"razon": {"valor": 2.0}}}},
       "cem": {"h1": {"ms_por_plan": 5.0, "velocidad": {"mae_cem": {"valor": .1}, "mae_repetir_ultima": {"valor": .2}}}}}
  (R / "latent_lewm.json").write_text(json.dumps(d))
  assert nbui.horizonte_latente("lewm", mostrar=False) is not None and nbui.probes_fig("lewm", mostrar=False) is not None
  assert nbui.sorpresa_fig("lewm", mostrar=False) is not None


def test_catalogo_y_demo_sin_checkpoint_no_fallan(repo, capsys):
  c = nbui.catalogo_corridas()
  assert c["genie_s"] == "sin entrenar todavía"
  assert nbui.comando_entrenar("gpt_s_kd") == "python -m src.cli gpt entrenar --tamaño s --variante kd --destilar"
  assert nbui.comando_entrenar("vjepa2_ac").endswith("--encoder vjepa2_s")
  assert nbui.demo("genie_s") is None                      # sin checkpoint: explica el comando, no falla
  res = nbui.comparar(["genie_s", "lewm"])
  assert res == [None, None]
  assert nbui.imaginar("convlstm_px") is None
  out = capsys.readouterr().out
  assert "sin entrenar todavía" in out and "python -m src.cli genie entrenar" in out and "ninguna de las corridas" in out


def test_ultimo_y_ruta_demo(repo):
  assert not nbui.ultimo("*.mp4").exists()
  d = repo / "results" / "demo"
  d.mkdir()
  (d / "a.mp4").write_bytes(b"x")
  assert nbui.ultimo("*.mp4").name == "a.mp4"
  assert nbui.ruta_demo("genie_s", pasos=8).name == "genie_s_s8_imagine.mp4"
  assert nbui.ruta_demo("lewm", latente=True).name == "lewm_latente.png"


def test_video_embebido_de_un_mp4_real(repo, capsys):
  import numpy as np
  cv2 = pytest.importorskip("cv2")
  d = repo / "results" / "demo"
  d.mkdir()
  vw = cv2.VideoWriter(str(d / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (64, 32))
  for i in range(6):
    vw.write(np.full((32, 64, 3), i * 30, np.uint8))
  vw.release()
  assert "data:image/gif;base64" in nbui.video_html(d / "v.mp4", "prueba")
  assert nbui.video_html(d / "no.mp4") == ""
  assert nbui.nitidez_video(d / "v.mp4", mostrar=False) is not None
