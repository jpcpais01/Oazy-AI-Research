"""Talk to Bee3 (same app as Bee0.2B: bee/chat.py).

    python3 -m bee3.chat --port 8002                 # the pre-trained model (runs/bee3)
    python3 -m bee3.chat --run bee3-chat --port 8002 # after the chat fine-tune (oazy/sft.py)
"""
from bee.chat import main

if __name__ == "__main__":
    main(default_run="bee3")
