"""Talk to Bee4 (same app as Bee0.2B: bee/chat.py).

    python3 -m bee4.chat --port 8002                  # the pre-trained model (runs/bee4)
    python3 -m bee4.chat --run bee4-chat --port 8002  # after the chat fine-tune (oazy/sft.py --run bee4)
"""
from bee.chat import main

if __name__ == "__main__":
    main(default_run="bee4")
