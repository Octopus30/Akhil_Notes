"""Section 10 lab: The Decoder Stack.

A tiny decoder-only model in NumPy: causal blocks, an output head, teacher
forcing (all positions in one pass) versus generation (one pass per token),
and cross-attention for an encoder-decoder model.

Run:  uv run python 10_Decoder_Stack/lab.py
"""

# %% 1. Setup
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=2, suppress=True)


def softmax(S: np.ndarray) -> np.ndarray:
    """Stable softmax over the last axis."""
    e = np.exp(S - S.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# %% 2. Reproduce the worked examples from notes.md
p = np.array([0.5, 0.4, 0.6, 0.7])
print("6.1 Teacher forcing loss on '<bos> the bat flew <eos>':", round(float(-np.log(p).mean()), 3), "(notes: 0.619)")

BIGRAM = {"<bos>": {"the": 0.5, "a": 0.3, "bat": 0.2},
          "the": {"bat": 0.4, "ball": 0.3, "hit": 0.3},
          "a": {"bat": 0.5, "ball": 0.5},
          "bat": {"flew": 0.6, "hit": 0.25, "<eos>": 0.15},
          "ball": {"flew": 0.3, "<eos>": 0.7},
          "hit": {"the": 0.6, "<eos>": 0.4},
          "flew": {"<eos>": 0.7, "away": 0.3},
          "away": {"<eos>": 1.0}}


def greedy(table: dict, start: str = "<bos>", max_len: int = 10) -> list[str]:
    """Greedy generation from a next-token table: one 'forward pass' per new token."""
    seq = [start]
    while seq[-1] != "<eos>" and len(seq) < max_len:
        nxt = table[seq[-1]]
        seq.append(max(nxt, key=nxt.get))
        print(f"    pass {len(seq) - 1}: input {' '.join(seq[:-1])!r:28} -> {seq[-1]}")
    return seq


print("6.2 Greedy generation (4 sequential passes):")
assert greedy(BIGRAM) == ["<bos>", "the", "bat", "flew", "<eos>"]

H = np.array([[1.0, 1.0], [2.0, 0.0]])       # encoder output for "bat flew"
print("6.3 Cross-attention while translating 'bat flew':")
for word, q in [("chamgaadad (bat)", np.array([0.0, 2.0])), ("udaa (flew)", np.array([2.0, 0.0]))]:
    w = softmax(H @ q / np.sqrt(2))
    print(f"    writing {word}: weights on [bat, flew] = {w}, output {w @ H}")
assert np.allclose(np.round(softmax(H @ np.array([0.0, 2.0]) / np.sqrt(2)), 2), [0.80, 0.20])

d, d_ff, vocab = 512, 2048, 37_000
enc_layer, dec_layer = 4 * d * d + 2 * d * d_ff, 8 * d * d + 2 * d * d_ff
total = 6 * enc_layer + 6 * dec_layer + vocab * d
print(f"6.4 Original Transformer (no biases/norms): {total:,} (notes: ~63.0M)")


# %% 3. A general implementation: a tiny decoder-only language model
def layer_norm(X: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Per-token normalization."""
    return (X - X.mean(-1, keepdims=True)) / np.sqrt(X.var(-1, keepdims=True) + eps)


class TinyDecoder:
    """Embeddings + N causal pre-norm blocks + final norm + tied output head."""

    def __init__(self, vocab_size: int, d: int, N: int, max_len: int = 32):
        s = d**-0.5
        self.E = rng.normal(0, 1, (vocab_size, d))
        self.P = rng.normal(0, 0.1, (max_len, d))
        self.blocks = [{k: rng.normal(0, s, (d, d)) for k in ("Wq", "Wk", "Wv", "Wo")}
                       | {"W1": rng.normal(0, s, (d, 4 * d)), "W2": rng.normal(0, (4 * d)**-0.5, (4 * d, d))}
                       for _ in range(N)]

    def logits(self, ids: np.ndarray) -> np.ndarray:
        """Token IDs (n,) -> logits (n x vocabulary size), with a causal mask in every block."""
        n = len(ids)
        X = self.E[ids] + self.P[:n]
        causal = np.tril(np.ones((n, n), dtype=bool))
        for b in self.blocks:
            Z = layer_norm(X)
            S = (Z @ b["Wq"]) @ (Z @ b["Wk"]).T / np.sqrt(X.shape[1])
            X = X + softmax(np.where(causal, S, -np.inf)) @ (Z @ b["Wv"]) @ b["Wo"]
            X = X + np.maximum(0, layer_norm(X) @ b["W1"]) @ b["W2"]
        out = layer_norm(X) @ self.E.T              # weight tying
        assert out.shape == (n, self.E.shape[0])
        return out

    def teacher_forcing_loss(self, ids: np.ndarray) -> float:
        """One pass: input = ids[:-1], targets = ids[1:], loss averaged over every position."""
        P = softmax(self.logits(ids[:-1]))
        targets = ids[1:]
        return float(-np.log(P[np.arange(len(targets)), targets]).mean())

    def generate(self, prompt: np.ndarray, n_new: int) -> tuple[np.ndarray, int]:
        """Greedy generation; returns (sequence, number of forward passes)."""
        seq, passes = list(prompt), 0
        for _ in range(n_new):
            seq.append(int(self.logits(np.array(seq))[-1].argmax()))
            passes += 1
        return np.array(seq), passes


model = TinyDecoder(vocab_size=20, d=16, N=2)
ids = rng.integers(0, 20, size=9)
print(f"\nTinyDecoder teacher-forcing loss on random text: {model.teacher_forcing_loss(ids):.2f}"
      f"\n  (uniform guessing gives ln 20 = {np.log(20):.2f}; an untrained model with large random"
      f"\n   logits is confidently wrong, so it does worse until training fixes it)")


# %% 4. Experiments
def experiment_causal_consistency() -> None:
    """Thanks to the mask, position t's logits are identical whether we feed the prefix or the whole text."""
    print("\n[Experiment 1] Same logits from the full sequence and from each prefix?")
    full = model.logits(ids)
    same = all(np.allclose(model.logits(ids[:t + 1])[-1], full[t]) for t in range(len(ids)))
    print("  every position matches its prefix-only run:", same)
    print("  -> one teacher-forced pass gives the same predictions as", len(ids), "separate passes")


def experiment_passes() -> None:
    """Generation needs one pass per new token; training needs one pass per sequence."""
    print("\n[Experiment 2] Forward passes")
    seq, passes = model.generate(ids[:3], n_new=6)
    print(f"  generated {len(seq) - 3} tokens with {passes} passes; teacher forcing on {len(seq)} tokens: 1 pass")
    for p_len, T in [(100, 400), (1000, 1000)]:
        no_cache = T * p_len + T * (T - 1) // 2
        print(f"  prompt {p_len}, generate {T}: positions processed without cache {no_cache:,}, "
              f"with cache {p_len + T - 1:,}")


def experiment_cross_attention_alignment() -> None:
    """Cross-attention scores are n_tgt x n_src: a soft alignment between two sequences."""
    print("\n[Experiment 3] Cross-attention: 3 target tokens attending to 4 source tokens")
    H_src = rng.normal(size=(4, 8))
    Q_tgt = H_src[[2, 0, 3]] + 0.3 * rng.normal(size=(3, 8))   # each target query resembles one source key
    A = softmax(Q_tgt @ H_src.T / np.sqrt(8))
    print("  weights (rows = target, cols = source):\n", A)
    print("  aligned source positions:", A.argmax(axis=1), "(built to be [2, 0, 3])")


experiment_causal_consistency()
experiment_passes()
experiment_cross_attention_alignment()


# %% 5. Student TODOs
def shift_for_training(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Return (input shifted right, target shifted left)."""
    # TODO(student): two slices.
    raise NotImplementedError


def cross_attention(X_dec: np.ndarray, H: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray,
                    W_V: np.ndarray) -> np.ndarray:
    """Queries from the decoder, keys/values from the encoder output H; no mask. Returns n_tgt x d_v."""
    # TODO(student): like Section 5's attention, with Q = X_dec W_Q and K, V from H.
    raise NotImplementedError


def enc_dec_params(d: int, n_enc: int, n_dec: int, vocab: int) -> int:
    """Encoder layers 12 d^2, decoder layers 16 d^2 (extra cross-attention), shared embeddings."""
    # TODO(student): one line.
    raise NotImplementedError


Hx, Xd = rng.normal(size=(5, 4)), rng.normal(size=(3, 4))
Wq = Wk = Wv = np.eye(4)
checks = [
    ("shift_for_training", lambda: shift_for_training(["<bos>", "a", "b", "<eos>"])
        == (["<bos>", "a", "b"], ["a", "b", "<eos>"])),
    ("cross_attention", lambda: np.allclose(cross_attention(Xd, Hx, Wq, Wk, Wv), softmax(Xd @ Hx.T / 2) @ Hx)),
    ("enc_dec_params", lambda: enc_dec_params(768, 12, 12, 32_128) == 222_855_168),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
