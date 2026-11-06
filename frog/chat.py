"""Talk to Frog (same app as Bee0.2B: bee/chat.py).

    python3 -m frog.chat --port 8002                 # the pre-trained model (runs/frog)
    python3 -m frog.chat --run frog-chat --port 8002 # after the chat fine-tune (oazy/sft.py --run frog)
"""
from bee.chat import main

if __name__ == "__main__":
    main(default_run="frog")
