import sys, json, numpy as np, torch
sys.path.insert(0, ".")
from src.models import load_model
from src.common import with_bos, TPF, BOS
import torch.nn.functional as F
dev = torch.device("cuda")
tok = np.load("data/train_tokens.npy", mmap_mode="r"); P = np.load("data/train_poses.npy", mmap_mode="r")
rng = np.random.default_rng(0); n = 48
segs = rng.integers(len(tok) - 50, len(tok), n); st = rng.integers(0, 1180, n)   # holdout de Windows
x = torch.as_tensor(np.stack([tok[s, t:t+20] for s, t in zip(segs, st)]).astype(np.int64), device=dev)
p = torch.as_tensor(np.stack([P[s, t:t+20] for s, t in zip(segs, st)]).astype(np.float32), device=dev)
out = {}
for run in ["kdP_s0", "kdP_s1"]:
  m = load_model(run, dev).eval()
  W = m.pose_proj.weight; out[run] = {"|W|": W.abs().mean().item(), "|b|": m.pose_proj.bias.abs().mean().item(),
     "|pe(pose)| / |wte|": None}
  seq = with_bos(x); xi, y = seq[:, :-1], seq[:, 1:]
  def ce(pp):
    tot = 0
    with torch.no_grad():
      for i in range(0, n, 8):
        lg = m.logits(xi[i:i+8], None if pp is None else pp[i:i+8]).float()
        tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y[i:i+8].reshape(-1), reduction="sum").item()
    return tot / y.numel()
  sh = p[torch.randperm(n, device=dev)]
  out[run].update({"CE real": ce(p), "CE sin pose": ce(None), "CE pose barajada": ce(sh)})
  with torch.no_grad():
    pe = m.pose_frames(p, 20, n, dev); pe0 = m.pose_frames(None, 20, n, dev)
    out[run]["|pe(pose)| / |wte|"] = ((pe - pe0).norm(dim=-1).mean() / m.net.wte.weight.norm(dim=-1).mean()).item()
print(json.dumps(out, indent=1))
