# Strata (working name)

A language model that is not a neural network: no weights, no gradients, no loss.

1. **Remember**: exact counts of everything read (one pass).
2. **Abstract**: words that share neighbours on both sides merge into classes; classes merge again
   into coarser layers. A move is kept only if it shortens the description of the text.
3. **Predict**: patterns over words and classes, most specific first, blended with the more abstract.

## Data

    mkdir -p strata/data && cd strata/data
    for f in train valid test; do
      curl -sSLo ptb.$f.txt https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.$f.txt
      curl -sSLo wt2.$f.txt https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/$f.txt
    done

## Run (CPU, about a minute)

    cd strata
    python3 run1.py ptb                 # pure memory (Kneser-Ney), checks the harness: 140.7 vs 141.2 published
    python3 run2.py ptb 1000,150,20     # builds the three class layers
    python3 run3.py ptb 1000,150,20     # patterns + blend

## First result, Penn Treebank test perplexity (16 Nov 2026)

| Model | Test |
|---|---|
| Pure memory, 5-gram | 141.1 |
| + patterns over three class layers | 129.3 |
| + memory of the current document (cache) | 119.6 |
| Published: KN5 141.2, KN5 + cache 125.7, RNN 124.7 (Mikolov 2011), Transformer-XL 54.5 | |

Blend weights are fitted by EM on the validation set (ten numbers).
