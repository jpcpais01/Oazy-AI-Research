"""Talk to Bee5 (same app as Bee0.2B: bee/chat.py).

    python3 -m bee5.chat --port 8002                  # the pre-trained model (runs/bee5)
    python3 -m bee5.chat --run bee5-chat --port 8002  # after the chat fine-tune (oazy/sft.py --run bee5)
"""
from bee.chat import main

if __name__ == "__main__":
    main(default_run="bee5")
