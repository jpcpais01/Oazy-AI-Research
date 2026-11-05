"""Talk to Bee_v2_0.2B (same app as Bee0.2B: bee/chat.py).

    python3 -m bee2.chat --port 8002
"""
from bee.chat import main

if __name__ == "__main__":
    main(default_run="bee2")
