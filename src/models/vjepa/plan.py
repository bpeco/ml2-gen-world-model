"""Método de entropía cruzada (CEM) de V-JEPA 2-AC, sin lecturas a la CPU.

Fuente: V-JEPA 2 paper §3.2 y App. (800 muestras, 10 iteraciones, top-10, horizonte 1, gaussiana inicial de media 0 y
varianza 1, energía L1 entre la representación imaginada y la del objetivo) y notebooks/utils/mpc_utils.py:
  - L17-18 `l1`: mean(|a - b|) sobre la dimensión de rasgos (se aplica a las representaciones aplanadas, L122).
  - L86-118 se sortean trayectorias de acciones y se simula con el world model paso a paso; el estado siguiente es el
    anterior más la acción (`compute_new_pose`, aquí en el espacio normalizado de estados).
  - L127-140 se eligen las k de menor energía y se actualizan media y desvío; el notebook agrega momento (media 0.25,
    desvío 0.95 en la rama general del archivo, 0.15 y 0.15 en world_model_wrapper.py L20-29). El paper no usa momento:
    por defecto `momento_media = momento_std = 0`; nuevo = elegidos*(1 - m) + viejo*m, como el notebook.
Todo es torch en el device del modelo: la elección de los elites es topk + gather, sin .item().
"""
import torch


def cem(costo, batch, forma, n_muestras=800, topk=10, iters=10, init_std=1.0, init_mean=None, device=None,
        generator=None, momento_media=0.0, momento_std=0.0, std_min=1e-4):
  """Minimiza costo(cand (S, B, *forma)) -> (S, B). Arranca en N(init_mean (o 0), init_std^2). Devuelve (B, *forma).

  init_std: escalar o tensor que se difunde sobre `forma` (el desvío de cada dimensión de la acción)."""
  mean = torch.zeros(batch, *forma, device=device)
  if init_mean is not None:
    mean = mean + init_mean.to(mean)
  std = torch.zeros_like(mean) + torch.as_tensor(init_std, dtype=mean.dtype, device=mean.device)
  for _ in range(iters):
    cand = mean + std * torch.randn(n_muestras, batch, *forma, device=device, generator=generator)
    idx = costo(cand).topk(topk, dim=0, largest=False).indices  # (k, B)
    el = cand.gather(0, idx.view(topk, batch, *([1] * len(forma))).expand(topk, batch, *forma))
    mean = el.mean(0) * (1 - momento_media) + mean * momento_media
    std = (el.std(0) * (1 - momento_std) + std * momento_std).clamp(min=std_min)
  return mean


def energia_l1(pred, objetivo):
  """mean |pred - objetivo| sobre los tokens y canales de un frame: (S, N, D) y (S, N, D) -> (S,)."""
  return (pred - objetivo).abs().flatten(1).mean(-1)


@torch.no_grad()
def planificar(modelo, z, acciones, estados, objetivo, horizonte=1, n_muestras=800, topk=10, iters=10, init_std=1.0,
               generator=None, chunk=None, **kw):
  """Plan de acciones normalizadas (B, horizonte, A) que llevan el último frame de z hacia `objetivo`.

  modelo.predecir(z, acciones, estados) -> (B, T, N, D): predicción del frame siguiente para cada frame de entrada
  (con layer_norm), como VJEPA2AC.predecir. z (B, Tc, N, D) son los frames de contexto (los reales ya codificados);
  acciones (B, Tc-1, A) las observadas entre ellos; estados (B, Tc, A); objetivo (B, N, D) ya codificado y normalizado.
  La energía es la L1 entre el frame imaginado tras `horizonte` pasos y el objetivo. Las muestras se agrupan en el batch
  (`chunk`: cuántas muestras por pasada del modelo, para acotar la memoria; no cambia el resultado).
  """
  B, Tc, N, D = z.shape
  A = estados.shape[-1]
  S = n_muestras
  rep = lambda t: t.unsqueeze(0).expand(S, *t.shape).reshape(S * B, *t.shape[1:])
  z0, a0, s0, goal = rep(z), rep(acciones), rep(estados), rep(objetivo)

  def costo(cand):  # cand (S, B, H, A)
    if chunk and chunk < S:
      return torch.cat([costo_(cand[i:i + chunk]) for i in range(0, S, chunk)])
    return costo_(cand)

  def costo_(cand):
    n = cand.shape[0]
    c = cand.reshape(n * B, horizonte, A)
    zt, at, st = z0[:n * B], a0[:n * B], s0[:n * B]
    for h in range(horizonte):
      at = torch.cat([at, c[:, h:h + 1]], 1)  # acción tomada en el último frame conocido
      nxt = modelo.predecir(zt, at, st)[:, -1:]
      zt = torch.cat([zt, nxt], 1)
      st = torch.cat([st, st[:, -1:] + c[:, h:h + 1]], 1)  # estado nuevo = anterior + acción (compute_new_pose)
    return energia_l1(zt[:, -1], goal[:n * B]).view(n, B)

  return cem(costo, B, (horizonte, A), S, topk, iters, init_std, device=z.device, generator=generator, **kw)
