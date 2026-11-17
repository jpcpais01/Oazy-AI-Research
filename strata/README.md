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
    python3 run4.py ptb 800,100,16 1500,200,30 600,60,10   # more views, caches, weights by context
    python3 run5.py ptb                 # soft memory on top of run4

## First result, Penn Treebank test perplexity (16 Nov 2026)

| Model | Test |
|---|---|
| Pure memory, 5-gram | 141.1 |
| + patterns over three class layers | 129.3 |
| + memory of the current document (cache) | 119.6 |
| + four different clusterings (24 pattern views), two caches (run4.py) | 115.4 |
| + soft memory: kernel vote over all past contexts with count-made meaning vectors (run5.py, about 18 min) | 114.5 |
| + patterns over phrase units and kinds of units (run6.py): unit views alone 170-480, in the blend | 115.6 (no gain) |
| Published: KN5 141.2, KN5 + cache 125.7, RNN 124.7 (Mikolov 2011), Transformer-XL 54.5 | |

Blend weights are fitted by EM on the validation set (ten numbers).

## Reading (16 Nov 2026)

Counting finds the kinds in seconds (meaning vectors and classes are good). Predicting with them stalls near 115:
more class views, context-dependent weights, a soft kernel memory and phrase units each add 0-3%.
Rule set before the phrase test: under 100 or the counting form is at its ceiling. It is at its ceiling.
What is missing is a learned way to combine several context words into one judgement.
