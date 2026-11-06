"""Talk to Coral (same app as Bee0.2B: bee/chat.py).

    python3 -m coral.chat --port 8002                  # the pre-trained model (runs/coral)
    python3 -m coral.chat --run coral-chat --port 8002 # after the chat fine-tune (oazy/sft.py --run coral)
"""
from bee.chat import main

if __name__ == "__main__":
    main(default_run="coral")
