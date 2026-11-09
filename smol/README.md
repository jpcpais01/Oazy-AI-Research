# SmolLM2 from scratch

Hugging Face's SmolLM2 architecture with **random weights**, trained on our text in our trainer, so its speed
(tokens per second) and its loss at equal tokens can be put next to Bee5, Bee4 and Plain-100M. A plain
**next-word** model (a Llama: RMSNorm, rotary positions, grouped key-value heads, SwiGLU, tied word table).

The body is the real `transformers.LlamaForCausalLM`, built from SmolLM2's own `config.json` (read from the Hub
on the pod; `smol/model.py` keeps a copy for when the Hub is out of reach). No trained weights are used.

| | SmolLM2-135M (default) | SmolLM2-360M (`--set size=360m`) |
|---|---|---|
| width x layers | 576 x 30 | 960 x 32 |
| heads | 9 query / 3 key-value, 64 each | 15 query / 5 key-value |
| MLP | SwiGLU 1,536 | SwiGLU 2,560 |
| parameters | 134.5M | 361.8M |

## Two names, one body

- **`smol`**: their recipe, stock. AdamW (lr 3e-3, betas 0.9/0.95, weight decay 0.01), WSD (warm-up, flat,
  a straight fall over the last 20%), 2,048-word windows from the start, the full score grid in 32-bit for the
  loss (what `LlamaForCausalLM` does), no mentor.
- **`smolbee`**: our latest recipe, exactly Bee5's: Muon on the weight grids, windows 512 -> 1,024 -> 2,048,
  the batch ramp, the square-root cool-down over the last 30%, the loss in slices, the SmolLM2-360M mentor
  on a share of the passes.

Both read Sado, take 524k-token steps, run in bf16 with `torch.compile`, and clip gradients at 1.0.

## Race (0.5B tokens each, on the B200)

```
OAZY_DATA=/root/data bash runpod/train.sh smol    --preset big --tokens 0.5e9 --micro 32 --name smol-race
OAZY_DATA=/root/data bash runpod/train.sh smolbee --preset big --tokens 0.5e9 --micro 32 --name smolbee-race
```

Compare with `bee5-race`, `bee4-race` and Plain-100M's 0.5B race: tok/s in the log line, held-out loss at the
same token counts on the dashboard. `smol` vs `smolbee` measures our recipe; `smolbee` vs Bee5 measures the body.

Tests: `python3 -m smol.tests` (CPU) or `python3 -m smol.tests --device cuda`.
