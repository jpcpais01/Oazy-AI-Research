"""Talk to Seed (same app as Bee0.2B: bee/chat.py).

    python3 -m seed.chat --port 8002                  # the pre-trained model (runs/seed)
    python3 -m seed.chat --run seed-chat --port 8002  # after the chat fine-tune (oazy/sft.py --run seed)
"""
from bee.chat import main

if __name__ == "__main__":
    main(default_run="seed")
