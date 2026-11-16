"""MoE-150M and its dense twin, Dense-30M. NEXT-TOKEN models: they read text and guess the very next token, like every
Bee, and are scored the same way. The code is Bee6-MoE's (bee6/moe.py, bee6/experts.py) and Bee6's (bee6/model.py); this
file only holds the shapes and what they cost. The trainer builds them from MOE150 / DENSE30 in oazy/train.py.

João's ask (Nov 16): "30M active, 150M total. using the best clean speed improvements, the best really nice model MoE
Transformer Next-Token model. aiming to tackle on smollm2-135M. training on 100B tokens of Opal".

                        MoE-150M                              Dense-30M (the twin)          SmolLM2-135M
  body                  16 layers of 512, every Bee5 trick    the same                      30 layers of 576 (Llama)
  attention             8 query heads, 2 key/value heads of 64, fused q/k/v, half-truncated RoPE, gated
  each layer's MLP      1 shared + 32 routed ReLU² experts    one ReLU² MLP of 1,152        SwiGLU 1,536
                        of 192, 5 routed per word
  per word              6 x 192 = 1,152 neurons               1,152 neurons
  word table            49,152 x 512, tied (25.2M) + 2 value tables (12.6M)                 49,152 x 576 tied (28.3M)
  weights in all        152.3M                                67.1M                         134.5M
  layers at work/word   29.6M                                 29.4M                         ~106M
  at work, all counted  67.4M                                 67.1M                         134.5M

"Active" here is the layers (attention + the experts a word runs + routers), the usual way to count a mixture. The word
table and the value tables are lookups that every model pays; counted in, a word touches 67M.

The clean speed tricks only (each keeps the maths the same, docs/speed-tricks.md): torch.compile, bf16, one q/k/v
product, the loss in slices of 4,096, batched NorMuon, short windows first (512 -> 1,024 -> 2,048), plain causal
attention, the stored RoPE table, few CPU threads, fixed expert queues. Left out on purpose: FP8 (at 512 wide the products
are too small to gain, and it rounds) and Mixture of Depths (Bee6's race: -0.05 held-out loss at equal words). This run
is limited by its 100B words, not by the GPU, so quality per word comes first.

Recipe: Bee6's (RECIPE["bee6"]), with no mentor, the 3-word early lesson only for the first 5% (5B words; MTP is known to
hurt models this small over a whole run), and Opal as "opal100" (oazy/mix.py): Field1 at 2% of every phase instead of 20%,
and the cool-down's chat at 12% instead of 27% (Opal's chat is 1.1B tokens; at 27% it would be read ~9 times).

    python3 -m bee6.moe150_tests                  (CPU, ~1-2 min)
    python3 -m bee6.moe150_tests --device cuda    (on the pod: also both models' real speed at full size)
    bash runpod/train.sh moe150 --preset big --tokens 100e9 --micro 64 --data-sub opal --mentor none --name moe150-opal-100b
    bash runpod/train.sh dense30 --preset big --tokens 10e9 --micro 64 --data-sub opal --mentor none --name dense30-opal-10b
"""
