"""MoE-150M and its dense twin, Dense-30M. NEXT-TOKEN models: they read text and guess the very next token, like every
Bee, and are scored the same way. The code is Bee6-MoE's (bee6/moe.py, bee6/experts.py) and Bee6's (bee6/model.py); this
file only holds the shapes and what they cost. The trainer builds them from MOE150 / DENSE30 in oazy/train.py.

João's ask (Nov 16): "30M active, 150M total. using the best clean speed improvements, the best really nice model MoE
Transformer Next-Token model. aiming to tackle on smollm2-135M. training on 100B tokens of Opal". Then (v2, same day):
our own 8k pieces, and more speed.

                        MoE-150M (v2)                         Dense-30M (the twin)          SmolLM2-135M
  pieces                our own 8,192 (oazy/tok8k.py)         the same                      SmolLM2's 49,152
  body                  16 layers of 512, every Bee5 trick    the same                      30 layers of 576 (Llama)
  attention             8 query heads, 2 key/value heads of 64, fused q/k/v, half-truncated RoPE, gated
  each layer's MLP      1 shared + 20 routed ReLU² experts    one ReLU² MLP of 1,152        SwiGLU 1,536
                        of 384, 2 routed per word
  per word              3 x 384 = 1,152 neurons               1,152 neurons
  word table            8,192 x 512, tied (4.2M) + 2 value tables (2.1M)                    49,152 x 576 tied (28.3M)
  weights in all        149.1M                                35.7M                         134.5M
  layers at work/word   29.4M                                 29.4M                         ~106M

"Active" is the layers (attention + the experts a word runs + routers), the usual way to count a mixture.

v1 -> v2. With SmolLM2's 49,152 pieces the score head was ~40% of a step and the word + value tables 37.7M of the 152M.
With 8,192 pieces the head is a sixth, and the 31M freed weights went into the experts (20 of 384 instead of 32 of 192:
fewer, wider experts make bigger, faster products; 2 of 20 + shared still mixes). The price: a piece holds less text
(~3.7 bytes instead of ~4.4), so 100B of our pieces is ~85B SmolLM2 pieces' worth of text; bits per byte (val_bpb in the
log) is the number that compares across tokenizers.

The clean speed tricks only (each keeps the maths the same, docs/speed-tricks.md): torch.compile, bf16, one q/k/v
product, the loss in slices of 4,096, batched NorMuon, short windows first (512 -> 1,024 -> 2,048), plain causal
attention, the stored RoPE table, few CPU threads, and (v2) the experts as one dropless grouped product per grid
(torch's grouped_mm, MegaBlocks / ScatterMoE style: no queues, no spare room, no word dropped; checked at start, Bee6-MoE's
queues if the GPU or torch cannot). CUDA graphs and a micro batch of 128 are measured by the speed test, and used if
they win. Left out on purpose: FP8 (at 512 wide the products are too small to gain, and it rounds) and Mixture of Depths
(Bee6's race: -0.05 held-out loss at equal words). This run is limited by its words, so quality per word comes first.

Recipe: Bee6's (RECIPE["bee6"]), no mentor, the 3-word early lesson only for the first 5%, Opal re-cut ("opal100" mix in
oazy/mix.py): Field1 at 2% of every phase instead of 20%, and the cool-down's chat at 12% instead of 27%.

    python3 -m oazy.tok8k train && python3 -m oazy.tok8k cut     (CPU pod, once: ~20-30 min + ~1-2 h)
    python3 -m bee6.moe150_tests                  (CPU, ~5-10 min)
    python3 -m bee6.moe150_tests --device cuda    (on the pod: also both models' real speed, every switch)
    bash runpod/train.sh moe150 --preset big --tokens 100e9 --micro 128 --data-sub opal --mentor none --name moe150-opal8k-100b
    bash runpod/train.sh dense30 --preset big --tokens 10e9 --micro 128 --data-sub opal --mentor none --name dense30-opal8k-10b
"""
