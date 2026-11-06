#!/usr/bin/env bash
# Code Black's first rungs, R0 to R3 (codeblack/README.md), one race after another on ONE GPU, nothing else on it.
#
#   tmux new-window -t oazy -n race "bash codeblack/ladder.sh"     # about 3.5 h on a B200 (~$28), resumable
#   tmux attach -t oazy                                            # watch (Ctrl-b then d to leave)
#   python3 -m codeblack.race --png /workspace/oazy_runs/race.png   # the report, any time (also printed after each race)
#
# The order gives the answer that matters first: Black-T 125M (the yardstick), then Code Black. After those two
# races (~65 min) the report already says whether two clocks passes rung R2. Then the phrase memory (R3), groups of 2,
# and the two bigger Black-Ts that complete the transformer line (R1).
# Stopped? Run it again: a finished race is skipped, an unfinished one resumes from its checkpoint.
#   MINUTES=30 MICRO=32 bash codeblack/ladder.sh     (the defaults; MICRO=16 if a run runs out of memory)
set -euo pipefail
cd "$(dirname "$0")/.."
MIN="${MINUTES:-30}"
MICRO="${MICRO:-32}"
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
if [ ! -f "$DATA/smol/cosmo/train.bin" ]; then
    echo "the races read Tagus; prepare it first: python3 -m oazy.tagus --run-tokens 2.5e9"
    exit 1
fi
if command -v nvidia-smi >/dev/null; then
    BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$BUSY" -gt 0 ]; then
        echo "something else is running on the GPU ($BUSY program(s)): a race needs the GPU to itself, or its minutes mean nothing."
        echo "stop it first (or run with ALLOW_SHARED=1 to race anyway, for a test only)."
        [ "${ALLOW_SHARED:-0}" = 1 ] || exit 1
    fi
fi

race() {   # race <run name> <model> [trainer options]
    local name="$1" model="$2"
    shift 2
    echo
    echo "=== race $name ($model, $MIN minutes) ==="
    python3 -m oazy.train --model "$model" --preset race --minutes "$MIN" --micro "$MICRO" --name "$name" \
        --data "$DATA" --runs "$RUNS" "$@" 2>&1 | tee -a "$RUNS/$name.log"
    python3 -m codeblack.race --dir "$RUNS" || true
}

# R0: the profiles (tokens/s and how much of the chip each uses; Black-T should reach 35% or more)
for m in blackt black; do
    [ -f "$RUNS/profile-$m.json" ] || python3 -m oazy.profile --model "$m" --preset race --micro "$MICRO" --runs "$RUNS" \
        2>&1 | tee "$RUNS/profile-$m.txt"
done

race blackt-125m blackt                                           # R1: the yardstick
race black-k4 black --set phrase_rows=0                           # R2: two clocks, groups of 4, no phrase memory
race black-k4-phrase black                                        # R3: the same with the phrase memory
race black-k2 black --set phrase_rows=0 group=2                   # R2: groups of 2
race blackt-250m blackt --set size=250m                           # R1: the transformer line
race blackt-500m blackt --set size=500m --lr 1.8e-3 --muon-lr 0.03

python3 -m codeblack.race --dir "$RUNS" --png "$RUNS/race.png" || true
echo "ladder R0-R3 done. Paste the report above in the Code Black thread."
