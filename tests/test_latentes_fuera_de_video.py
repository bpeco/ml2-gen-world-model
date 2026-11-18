"""D12: los latentes no entran en rollouts ni evaluate; convlstm_px no tiene CE y evaluate lo dice sin error."""
import pytest

from src import evaluate, rollouts


@pytest.mark.parametrize("spec", ["random:lewm", "random:vjepa2", "random:vjepa2_ac"])
@pytest.mark.parametrize("modulo,args", [(rollouts, ["--limit", "1"]), (evaluate, ["--segments", "1"])])
def test_latentes_rechazados(spec, modulo, args):
  with pytest.raises(SystemExit) as e:
    modulo.main(["--model", spec] + args)
  assert "src.eval_latent" in str(e.value)


def test_evaluate_convlstm_px_sin_ce(capsys):
  assert evaluate.main(["--model", "random:convlstm_px"]) is None
  assert "no tiene CE" in capsys.readouterr().out


def test_genie_y_convlstm_px_no_son_latentes():
  from src.models import load_model
  for spec in ("random:genie_s", "random:convlstm_px"):
    rollouts.rechazar_latente(load_model(spec), "x")      # no levanta
