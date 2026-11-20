#!/usr/bin/env bash
# Opal4 (oazy/mix.py): Opal3 re-balanced and made big enough for a long run. CPUs and network only: a CPU pod.
#   1. Opal's own parts (web, books, maths, code, chat ...) built again where a 50B run needs more than is there
#   2. the everyday parts (how-to, stories, facts, science, scenes, plain web) sized to Opal4's shares
# A part that has enough is kept, one that ran out of source last time is kept, so a stopped build can be started again.
#
#   tmux new-window -t oazy -n opal4 "bash runpod/opal4.sh; bash"
#   PLAN=1 bash runpod/opal4.sh              # only show what a run needs, what is there and the disk it takes
#   TOKENS=30e9 bash runpod/opal4.sh         # for a shorter run (default 50e9)
#
# A run that reads it: --data-sub opal4 (Field1 off unless --opal-plan is given).
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"
cd "$(dirname "$0")/.."
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
INTO="$(python3 -c "from pathlib import Path; from oazy.opal3 import opal_dir; print(opal_dir(Path('$DATA')).name)")"
TOKENS="${TOKENS:-50e9}"
python3 -m pip install -q pyarrow huggingface_hub >/dev/null 2>&1 || true
if [ -n "${PLAN:-}" ]; then
    python3 -m oazy.sado --data "$DATA" --into "$INTO" --run-tokens "$TOKENS" --mix opal4 --plan
    python3 -m oazy.opal3 plan --data "$DATA" --run-tokens "$TOKENS" --mix opal4
    df -h "$DATA" | tail -1
    exit 0
fi
nice -n 10 python3 -m oazy.sado --data "$DATA" --into "$INTO" --run-tokens "$TOKENS" --mix opal4
nice -n 10 python3 -m oazy.opal3 build --data "$DATA" --run-tokens "$TOKENS" --mix opal4
python3 -m oazy.sado --data "$DATA" --into "$INTO" --run-tokens "$TOKENS" --mix opal4 --plan
