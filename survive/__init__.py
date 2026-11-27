"""Survive: a creature in a 3D world that learns to stay alive from two eyes, two ears, an
intensity sense and its body feelings. Nothing is scripted: it only has raw muscles and a
drive that likes energy and health. See survive/README.md.
"""

import importlib as _il
import os as _os
import sys as _sys

if _os.environ.get("SURVIVE_WORLD") == "table":      # creature two: the tabletop arm replaces the world
    _sys.modules[__name__ + ".world"] = _il.import_module(__name__ + ".table")
