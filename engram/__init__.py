"""Old name of memtrix/ (Memtrix was called Engram until Nov 5 2026). `import engram.model` etc. still work.

New code should import memtrix directly; this shim can go once nothing uses the old name."""
import importlib
import sys

from memtrix import *  # noqa: F401,F403

for _m in ("model", "memory", "tests", "chat"):
    sys.modules[f"{__name__}.{_m}"] = importlib.import_module(f"memtrix.{_m}")
