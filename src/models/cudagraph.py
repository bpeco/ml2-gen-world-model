"""Captura de un paso de inferencia en un CUDA graph (reduce el overhead de lanzar kernels)."""
from typing import Callable

import torch


class GraphedStep:
  """Captura fn(*static_inputs) -> Tensor en un CUDAGraph. __call__(*inputs) copia a los buffers
  estáticos, hace replay y devuelve la salida estática (hay que consumirla o clonarla antes de la
  próxima llamada).

  Los inputs de ejemplo definen formas y dtypes. El warm-up ejecuta fn de verdad (en un stream
  aparte), así que fn tiene que ser idempotente con esos inputs de ejemplo (p. ej. escribir en la
  caché KV la misma posición que luego se reescribe).
  """

  def __init__(self, fn: Callable[..., torch.Tensor], *example_inputs: torch.Tensor, warmup: int = 3):
    self.static_in = [t.clone() for t in example_inputs]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
      for _ in range(warmup):
        fn(*self.static_in)
    torch.cuda.current_stream().wait_stream(s)
    self.graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self.graph), torch.no_grad():
      self.static_out = fn(*self.static_in)

  def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
    for buf, t in zip(self.static_in, inputs):
      buf.copy_(t)
    self.graph.replay()
    return self.static_out
