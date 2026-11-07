"""Talk to Coral Mini (same app as Bee0.2B: bee/chat.py).

    python3 -m coralmini.chat --port 8002                       # the pre-trained model (runs/coralmini)
    python3 -m coralmini.chat --run coralmini-chat --port 8002  # after the chat fine-tune (oazy/sft.py --run coralmini)
"""
from bee.chat import main

if __name__ == "__main__":
    main(default_run="coralmini")
