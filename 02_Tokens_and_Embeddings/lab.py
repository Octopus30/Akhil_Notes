"""Section 2 lab: Tokens & Embeddings.

Train a tiny BPE tokenizer, turn tokens into IDs, look up embeddings, and
measure similarity with cosine, all from scratch in NumPy.

Run:  uv run python 02_Tokens_and_Embeddings/lab.py
"""

# %% 1. Setup
from collections import Counter

import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=3, suppress=True)


# %% 2. Reproduce the worked examples from notes.md
def merge_pair(tokens: list[str], pair: tuple[str, str]) -> list[str]:
    """Replace every adjacent occurrence of pair with the merged token."""
    out, i = [], 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            out.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def count_pairs(words: dict[str, list[str]], corpus: dict[str, int]) -> Counter:
    """Count adjacent token pairs, weighting each word by its frequency."""
    pairs = Counter()
    for word, tokens in words.items():
        for a, b in zip(tokens, tokens[1:]):
            pairs[(a, b)] += corpus[word]
    return pairs


def train_bpe(corpus: dict[str, int], num_merges: int, verbose: bool = False) -> list[tuple[str, str]]:
    """Learn BPE merge rules, in order, from a word -> count corpus."""
    words = {w: list(w) for w in corpus}
    merges = []
    for r in range(1, num_merges + 1):
        pairs = count_pairs(words, corpus)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        if verbose:
            top = ", ".join(f"{a}+{b}={c}" for (a, b), c in pairs.most_common(4))
            print(f"  round {r}: {top}  -> merge {best[0]} + {best[1]}")
        merges.append(best)
        words = {w: merge_pair(t, best) for w, t in words.items()}
    return merges


def tokenize(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Split a word into characters and apply the merges in the order they were learned."""
    tokens = list(word)
    for pair in merges:
        tokens = merge_pair(tokens, pair)
    return tokens


print("6.1 BPE by hand: corpus tea x4, ten x3, tent x2, neat x1")
corpus = {"tea": 4, "ten": 3, "tent": 2, "neat": 1}
merges = train_bpe(corpus, 4, verbose=True)
assert merges == [("t", "e"), ("te", "n"), ("te", "a"), ("ten", "t")]
for w, expected in [("tent", ["tent"]), ("teat", ["tea", "t"]), ("neaten", ["n", "e", "a", "ten"])]:
    got = tokenize(w, merges)
    assert got == expected
    print(f"  tokenize({w!r}) = {got}")

print("\n6.2 Embedding lookup = one-hot x matrix")
vocab = ["<pad>", "tea", "is", "hot", "cold"]
token_to_id = {t: i for i, t in enumerate(vocab)}
E = np.array([[0.00, 0.00, 0.00],
              [0.90, 0.10, 0.30],
              [-0.20, 0.40, 0.10],
              [0.30, -0.50, 0.80],
              [0.20, -0.60, -0.70]])
assert E.shape == (len(vocab), 3)            # |V| x d_model

o_hot = np.eye(len(vocab))[token_to_id["hot"]]
print("  one-hot for 'hot':", o_hot, "->", o_hot @ E)
assert np.allclose(o_hot @ E, [0.30, -0.50, 0.80])

ids = np.array([token_to_id[t] for t in "tea is hot".split()])
X = E[ids]
assert X.shape == (3, 3)                     # n x d_model
print("  'tea is hot' ids", ids, "-> X shape", X.shape, "\n", X)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity: dot product divided by both lengths."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\n6.3 Cosine similarity")
tea, coffee, brick = np.array([0.9, 0.1, 0.3]), np.array([0.8, 0.2, 0.4]), np.array([0.1, 0.9, -0.2])
print(f"  cos(tea, coffee) = {cosine(tea, coffee):.2f}   (notes: 0.98)")
print(f"  cos(tea, brick)  = {cosine(tea, brick):.2f}   (notes: 0.14)")
assert round(cosine(tea, coffee), 2) == 0.98 and round(cosine(tea, brick), 2) == 0.14

print("\n6.4 Embedding table sizes")
for name, v, d in [("GPT-2 small", 50_257, 768), ("LLaMA-2 7B", 32_000, 4_096)]:
    print(f"  {name:12s} {v:>7,} x {d:>5,} = {v * d:>12,} parameters")
assert 50_257 * 768 == 38_597_376


# %% 3. A general implementation: text -> IDs -> X
class TinyTokenizer:
    """A word-level-split + BPE tokenizer with an ID table and an <unk> fallback."""

    def __init__(self, corpus: dict[str, int], num_merges: int):
        self.merges = train_bpe(corpus, num_merges)
        pieces = sorted({p for w in corpus for p in tokenize(w, self.merges)})
        self.vocab = ["<pad>", "<unk>"] + pieces
        self.ids = {t: i for i, t in enumerate(self.vocab)}

    def encode(self, text: str) -> np.ndarray:
        """Text -> array of token IDs, shape (n,)."""
        pieces = [p for w in text.split() for p in tokenize(w, self.merges)]
        return np.array([self.ids.get(p, self.ids["<unk>"]) for p in pieces])


tok = TinyTokenizer(corpus, 4)
E_tiny = rng.normal(0, 0.1, size=(len(tok.vocab), 4))
ids = tok.encode("tea tent neat")
X = E_tiny[ids]
assert X.shape == (len(ids), 4)
print("\nTinyTokenizer vocabulary:", tok.vocab)
print("encode('tea tent neat') =", ids, "-> X shape", X.shape)


# %% 4. Experiments
STORY = {  # an original mini-corpus: word -> count
    "rain": 12, "rains": 5, "raining": 4, "train": 6, "trains": 3, "training": 2,
    "brain": 3, "drain": 2, "plain": 2, "paint": 2, "painting": 1, "sing": 5,
    "singing": 3, "ring": 4, "bring": 2, "string": 1,
}


def experiment_merges_vs_length() -> None:
    """More merges -> bigger vocabulary -> fewer tokens per word."""
    print("\n[Experiment 1] More merges = fewer tokens per word")
    total_words = sum(STORY.values())
    for k in [0, 5, 10, 20, 40]:
        m = train_bpe(STORY, k)
        n_tokens = sum(len(tokenize(w, m)) * c for w, c in STORY.items())
        print(f"  {k:>2} merges: {n_tokens / total_words:.2f} tokens per word")


def experiment_sparse_gradients() -> None:
    """Only the rows of E for tokens in the batch get a gradient."""
    print("\n[Experiment 2] Which rows of E get a gradient?")
    E_ = rng.normal(size=(6, 3))
    batch_ids = np.array([1, 4, 4, 2])
    dX = rng.normal(size=(len(batch_ids), 3))          # pretend gradient w.r.t. X
    dE = np.zeros_like(E_)
    np.add.at(dE, batch_ids, dX)                       # scatter-add back into E
    for i, row in enumerate(dE):
        print(f"  row {i}: {'updated' if np.any(row) else 'zero gradient'}")


def experiment_cosine_ignores_length() -> None:
    """Scaling a vector changes its length and distance but not its cosine."""
    print("\n[Experiment 3] Cosine ignores length")
    a = np.array([0.9, 0.1, 0.3])
    for s in [0.5, 1, 3, 10]:
        print(f"  scale {s:>4}: cosine {cosine(a, s * a):.3f}   distance {np.linalg.norm(a - s * a):.3f}")


experiment_merges_vs_length()
experiment_sparse_gradients()
experiment_cosine_ignores_length()


# %% 5. Student TODOs
def decode(tokens: list[str]) -> str:
    """Join sub-word tokens of ONE word back into the word."""
    # TODO(student): BPE tokens of a word concatenate back into the word. Hint: "".join.
    raise NotImplementedError


def most_similar(query: str, words: list[str], E: np.ndarray, ids: dict[str, int]) -> str:
    """Return the word (other than query) with the highest cosine similarity to query."""
    # TODO(student): loop over words, skip the query itself, use cosine(E[ids[q]], E[ids[w]]).
    raise NotImplementedError


def tied_savings(vocab_size: int, d_model: int) -> int:
    """Parameters saved by weight tying (reusing E^T as the output head)."""
    # TODO(student): the output head is d_model x vocab_size.
    raise NotImplementedError


toy_ids = {"tea": 0, "coffee": 1, "brick": 2}
toy_E = np.stack([tea, coffee, brick])
checks = [
    ("decode", lambda: decode(tokenize("neaten", merges)) == "neaten"),
    ("most_similar", lambda: most_similar("tea", list(toy_ids), toy_E, toy_ids) == "coffee"),
    ("tied_savings", lambda: tied_savings(50_257, 768) == 38_597_376),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
