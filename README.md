# SPARC: a deep, closed-form contextual encoder

**SPARC** stands for Sparse-Product And Reservoir Cloze. It is a language model and contextual encoder built **without backpropagation**. Every component is one of:

- counting;
- one SVD;
- a fixed random recurrence or product;
- a closed-form ridge solve.

It works on BPE tokens only. On 30M words of Wikipedia it trains end to end in **about 20 minutes on 2 CPU cores**.

## Architecture

```
corpus ──(one co-occurrence count, window 8)──► token meanings (768-d out / 128-d internal)  +  directional PMI tables
       ──(three passes, hashed)───────────────► pattern tables: avg meaning of the filler of each context pattern
                                                 (L, R, LR, LL, RR around a gap; fixed memory via hashed buckets)

for each of three streams                cloze: guess token t from both sides (t hidden)         "what fits here"
                                          fwd:   guess t+1 from tokens <= t (t visible)            "t in its left context"
                                          bwd:   guess t-1 from tokens >= t (t visible)            "t in its right context"
  layer 1 : [pattern lookups for the target gap ; memory states] ──ridge──► guess g1 (meaning) + scores over all tokens
  layer k+1: [everything layer k saw (residual) ; g_k ; counting attention queried by g_k ;
              2048 products g_k x (attention ; memory view)] ──ridge──► g_{k+1}, scores, hidden h_{k+1}

token vector    = [768-d meaning ; g and h of every layer of all three streams ; word x (fwd, bwd) products]
sentence vector = whitened mean of [meaning ; cloze guess]
```

- **Memory.** A fused diagonal reservoir per direction:
  - 256 fixed-decay units plus 256 token-gated (selective) units;
  - reset at every sentence and bias-corrected;
  - parallel-scannable;
  - driven by the current token only. Exact local order comes from the pattern tables.
- **Counting attention.**
  - The relevance score between a key word and the target is the corpus PMI between them, and it is directional.
  - The target is unknown, so the query is the layer's own guess, mapped into PMI space.
  - Attention is causal for the forward and backward streams.
- **Products between layers.** They are the nonlinearity that makes depth work. Without them, stacked ridge layers stay linear and stop improving after the second layer.
- **No leakage.** No stream ever sees the token it predicts. Pattern statistics use leave-one-out during training.

## Results (30M-word Wikipedia, 16k BPE vocabulary)

| | value |
|---|---|
| Linear probe, ambiguous words: POS / WordNet supersense | **83.1 / 68.8** |
| …same probe on the 768-d meaning only / no-context "most frequent tag" baseline | 78.9 / 63.0 · 80.1 / 66.6 |
| STS-B / SICK-R (Spearman ×100, unsupervised) | 64.5 / 58.9 |
| Gap-filling, cloze stream, layers 1 → 2 → 3 (MRR over 16k tokens) | 0.193 → 0.231 → 0.245 (top-1 18.2%) |
| Forward / backward LM, layer 3 (MRR) | 0.144 / 0.128 |

For context, at the same data size averaged word2vec scores 46.4 on STS-B and fastText 53.2. A pretrained RoBERTa-base mean-pooled through the same pipeline scored 41.4 (a spaCy copy, fine-tuned for parsing).

**Build time (2 CPU cores, 32M training tokens):**

| stage | time |
|---|---|
| co-occurrence count | 21 s |
| meanings + PMI factors (2 SVDs) | 113 s |
| hashed pattern tables | 21 s |
| three streams × 3 layers (incl. validation) | 6.4 + 4.9 + 4.9 min |
| whitening | 102 s |

**Encoding speed:** about 7k tokens/s per stream. The model is 1.3 GB, of which the pattern tables are 0.87 GB.

### Pattern memory budget (cloze stream, `experiments/hash_and_products.py`)

| buckets per pattern group | memory | patterns kept | MRR (layer 3) |
|---|---|---|---|
| 2,097,152 (default) | 0.87 GB | 2.53M | 0.245 |
| 524,288 | 0.22 GB | 1.36M | 0.230 |
| 131,072 | 0.06 GB | 0.39M | 0.205 |

## Usage

```bash
pip install -r requirements.txt
python scripts/prepare_corpus.py --text corpus.txt --out data/            # BPE (16k) + token ids
python scripts/train.py --tokens data/tokens.npz --tokenizer data/bpe.json --out models/sparc
python scripts/prepare_tests.py --stsb stsb-test.csv --sick sick-test.txt --out data/tests.json
python scripts/prepare_probe.py                                            # SemCor via nltk -> data/probe.json
python scripts/evaluate.py --model models/sparc --tests data/tests.json --probe data/probe.json
python scripts/demo.py --model models/sparc --tokens data/tokens.npz      # associations, fill-in, senses, similarity
```

```python
from sparc import SPARC
m = SPARC.load("models/sparc")
m.sentence_vectors(["a man is playing a guitar", "a man plays the guitar on stage"])
m.word_vector("she sat on the river bank", "bank")      # contextual vector of one word
m.fill_mask("paris is the capital of [MASK]")            # cloze stream guesses
m.next_tokens("the river flows into the")                # forward stream guesses
```

See `experiments/demo_output.md` for a sample run.

## Repository layout

- **`sparc/`, the model:**
  - `meanings.py`: co-occurrence counts, PPMI-SVD meanings, PMI attention factors.
  - `patterns.py`: hashed pattern tables (count-min admission, majority fingerprint, int8 values, numba lookups).
  - `reservoir.py`: fused fixed + selective diagonal memory.
  - `attention.py`: counting attention.
  - `products.py`: random and subspace interaction features.
  - `stream.py`: one stream. Cached, incremental layer-wise training; block-fused inference.
  - `model.py`: the full model, encoding API, demos, save/load.
- **`scripts/`:** data preparation, training, scorecard, demo.
- **`experiments/`:** the research history (rounds A–X: pooling, attention, reservoir zoo, depth, consolidation, engineering) with every scorecard log.

## Known limitations

- **No sentence-level objective yet.** Negation and contradiction are not handled.
- **Fill-in-the-blank is mostly local.** It is strong when the immediate neighbours are informative, and weak when the cue is far away ("the ___ barked").
- **Free generation from the forward stream is not coherent.** This is a contextual encoder, not a generator.
