import json

from src import resultados


def test_marcar_pendiente_y_listar(tmp_path):
  est = tmp_path / "estado.json"
  (tmp_path / "eval_a.json").write_text("{}")
  assert resultados.inicializar(tmp_path, est) == 1
  assert resultados.inicializar(tmp_path, est) == 0  # idempotente
  resultados.pendiente("eval_nuevo", "falta medirlo", est)
  resultados.marcar("results/eval_x.json", "vigente", "ok", est)
  d = json.loads(est.read_text(encoding="utf-8"))
  assert d["eval_nuevo"]["estado"] == "pendiente" and d["results/eval_x.json"]["estado"] == "vigente"
  assert {"estado", "motivo", "fecha"} <= set(d["eval_nuevo"])
  assert resultados.estado("results/eval_x.json", d) == "vigente"
  assert resultados.estado("otro.json", d) == "sin estado"
  res = resultados.listar(est, tmp_path)
  assert res["pendiente"] == ["eval_nuevo"] and len(res["protocolo anterior"]) == 1


def test_cli_set(tmp_path, capsys):
  est = tmp_path / "estado.json"
  resultados.main(["--estado-file", str(est), "--set", "results/eval_y.json", "pendiente", "--motivo", "m"])
  assert resultados.cargar(est)["results/eval_y.json"]["motivo"] == "m"
  assert "pendiente (1)" in capsys.readouterr().out


def test_estado_repo_cubre_resultados():
  """Todo resultado existente de results/ tiene estado registrado (results/estado.json)."""
  assert resultados.listar()["sin estado"] == []


def test_los_derivados_tienen_estado_y_los_manifiestos_no(tmp_path):
  """Trabajo 16: semillas_*, diferencia_*, piso_*, temperatura_* y humo_* quedaban fuera del estado y de `bitacora
  verificar`, y check_docs no podía respaldar con ellos un número de los docs."""
  derivados = ["semillas_x.json", "diferencia_a_vs_b.json", "piso_fvd.json", "temperatura_x.json", "humo_x.json"]
  for n in derivados + ["checkpoints.json", "configs.json", "estado.json", "val_split.json", "clips_eval.json"]:
    (tmp_path / n).write_text("{}")
  assert sorted(resultados.existentes(tmp_path)) == sorted((tmp_path / n).as_posix() for n in derivados)
