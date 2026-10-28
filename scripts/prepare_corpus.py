"""Raw text (one paragraph per line, lower-cased, tokens separated by spaces) -> BPE tokenizer + token ids.

  python scripts/prepare_corpus.py --text corpus.txt --vocab 16000 --out data/
"""
import argparse, os
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

p = argparse.ArgumentParser(); p.add_argument("--text", required=True); p.add_argument("--vocab", type=int, default=16000); p.add_argument("--out", required=True)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
tok = Tokenizer(models.BPE(unk_token="[UNK]", continuing_subword_prefix="##")); tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
tok.train([a.text], trainers.BpeTrainer(vocab_size=a.vocab, min_frequency=3, special_tokens=["[UNK]"], continuing_subword_prefix="##"))
tok.save(os.path.join(a.out, "bpe.json"))
lines = [l.strip() for l in open(a.text) if l.strip()]
ids = np.concatenate([np.array(e.ids, np.int32) for c in range(0, len(lines), 20000) for e in tok.encode_batch(lines[c:c + 20000])])
np.savez(os.path.join(a.out, "tokens.npz"), ids=ids); print(f"{len(ids):,} tokens")
