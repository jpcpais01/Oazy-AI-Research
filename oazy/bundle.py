"""Old name of Tagus (oazy/tagus.py): `python -m oazy.bundle` still works."""
from .tagus import *  # noqa: F401,F403
from .tagus import main, needs, prepare, stream, have, NEW  # noqa: F401

if __name__ == "__main__":
    main()
