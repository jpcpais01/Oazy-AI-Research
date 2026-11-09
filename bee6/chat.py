"""Talk to Bee6 (same app as Bee0.2B: bee/chat.py).

    python3 -m bee6.chat --port 8002                  # the pre-trained model (runs/bee6)
    python3 -m bee6.chat --run bee6-chat --port 8002  # after the chat fine-tune (oazy/sft.py --run bee6)
"""
from bee.chat import main

if __name__ == "__main__":
    main(default_run="bee6")
