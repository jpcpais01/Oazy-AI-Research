# Oazy on the phone

An Android app that runs our own models (Coral_1B, Bee_v2_0.2B, Bee3, Frog) **entirely on the phone**: no internet,
nothing sent anywhere. It looks like the landing page and the lab dashboard: the photon field, clear glass panels,
Outfit / Inter / JetBrains Mono, and each model in its own colour.

All of them are **next-word models**: the app reads the conversation the way they saw chats in training
(`User: ... / Assistant: ...`) and lets the model write the next words one at a time. **Write** mode instead lets it
carry on any text you start.

## Use it

On the pod (CPU only; it never touches a running training):

    cd /workspace/oazy && git pull
    python3 -m phone.export --run coral          # -> runs/coral/phone/Coral_1B-q8.oazy  (about 1.2 GB)
    python3 -m phone.export --run bee2           # -> Bee_v2_0.2B-q8.oazy (about 0.23 GB)
    python3 -m phone.serve                       # shares the app and the models on port 8002

Then on the phone open the pod's **8001** link (RunPod: Connect -> HTTP 8001; the template's nginx forwards it to
8002, see docs/RUNPOD.md):

1. **Install the app** (Android asks once to allow installs from the browser).
2. Tap **Open in Oazy** next to a model: the app downloads it (a cut download resumes) and opens it.

Or copy a `.oazy` file to the phone any other way and use **Add a model file**. The app itself is also in the repo at
`phone/dist/oazy.apk`.

`--quant q4` makes a smaller, faster copy (Coral about 0.7 GB): the MLP and expert grids in 4 bits, the rest in 8.
It is a little less exact; the app's **Check** shows how close a file stays to PyTorch.

## How it works

| part | file | what it does |
|---|---|---|
| exporter | `phone/export.py` | reads `ckpt.pt`, writes one `.oazy` file: the weights in 8 bits (groups of 32 with one scale each), the SmolLM2 tokenizer, and a self-check (texts with SmolLM2's own pieces for them, and PyTorch's scores after a chat prompt) |
| engine | `phone/engine/oazy.cpp` | one C++ file, no libraries: maps the file into memory and runs the model on the phone's fast CPU cores with integer dot products (ARM `sdot`, or plain NEON on older phones). Attention layers with RoPE and grouped heads, Memtrix memory layers (the delta rule, word by word), SwiGLU MLPs, Coral's router and experts, the soft-capped tied head, and SmolLM2's byte-level BPE |
| app | `phone/app/` | a WebView page (`assets/`) for the look, a small Java bridge that starts the engine as a separate process and talks to it line by line, downloads and imports models |
| share | `phone/serve.py` | the pod's page with the app and the models, with resumable downloads |
| build | `phone/build.sh` | builds the APK without Android Studio: the engine cross-compiled with Zig (static, musl), Java to dex, packed, aligned and signed |
| tests | `phone/tests.py` | small random Coral / Bee2 / hybrid models: the engine's scores match PyTorch's to 2e-6 in float32 (word by word and in chunks), 8-bit and 4-bit stay close, the tokenizer cuts 209 texts exactly like Hugging Face's, the chat protocol works |

Why the CPU and not the GPU: writing an answer reads every weight once per word, so a phone's speed is set by its
memory bandwidth, which the CPU and the GPU share. A tuned CPU path is as fast as the GPU for this, runs on every
phone, and needs no vendor drivers. Coral pays off here: only ~211M of its 1,037M parameters are read per word.

Speed, measured on a 4-core PC (no phone was available to test): Coral_1B 8-bit writes about 40 tokens/s and reads a
prompt at about 125 tokens/s; Bee_v2_0.2B about the same. A recent phone should land somewhere around 15-30 tokens/s.

## Rebuild

    bash phone/build.sh       # -> phone/dist/oazy.apk (installs the Android build tools with apt the first time)
    python3 -m phone.tests    # the checks (CPU, a minute)
    OAZY_ENGINE="qemu-aarch64-static -cpu max phone/build/lib/arm64-v8a/liboazy.so" python3 -m phone.tests   # the phone build, emulated

The signing key is not in git (it is a private key). Android only accepts an update signed by the same key, so the
project keeps it in the project files (`phone/signing/oazy.keystore`); `build.sh` finds it there or in
`$OAZY_KEYSTORE`, otherwise it makes a new one (then uninstall the old app first).

To see the page in a desktop browser, open `phone/app/assets/index.html`: without the phone it runs on a stand-in
(add `?empty` for the first-run screen).
