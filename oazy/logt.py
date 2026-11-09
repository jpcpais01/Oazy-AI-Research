"""LogT (logic and thinking), formerly LogRes: the same benchmark under its new name.

    python3 -m oazy.logt            # = python3 -m oazy.logres (both work; results keep the kind "logres")
"""
from .logres import *  # noqa: F401,F403
from .logres import main

if __name__ == "__main__":
    main()
