"""Section 11 lab: Encoder vs Decoder vs Both.

Same block, different mask and objective. Build the four attention patterns,
count their connections, apply the three pre-training objectives to one
sentence, and see why only causal attention lets old outputs stay fixed.

Run:  uv run python 11_Encoder_vs_Decoder_vs_Both/lab.py
"""

# %% 1. Setup
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=2, suppress=True)


def softmax(S: np.ndarray) -> np.ndarray:
    """Stable softmax over the last axis."""
    e = np.exp(S - S.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def mask(kind: str, n: int, p: int | None = None) -> np.ndarray:
    """n x n boolean pattern: 'encoder' (full), 'decoder' (causal) or 'prefix' (full on first p)."""
    if kind == "encoder":
        return np.ones((n, n), dtype=bool)
    m = np.tril(np.ones((n, n), dtype=bool))
    if kind == "prefix":
        m[:p, :p] = True
    return m


def show(m: np.ndarray, labels: list[str]) -> None:
    """Print a pattern with labels: # = allowed, . = masked."""
    print("          " + " ".join(f"{l[:5]:>5}" for l in labels))
    for l, row in zip(labels, m):
        print(f"   {l[:6]:>6} " + " ".join(f"{'#' if v else '.':>5}" for v in row))


# %% 2. Reproduce the worked examples from notes.md
tokens = "the bat flew away fast".split()
print("6.1 Allowed pairs for n = 5 (prompt 'the bat flew', continuation 'away fast')")
for kind in ["encoder", "decoder", "prefix"]:
    print(f"  {kind:8s}: {int(mask(kind, 5, 3).sum())}")
assert [int(mask(k, 5, 3).sum()) for k in ("encoder", "decoder", "prefix")] == [25, 15, 18]
show(mask("prefix", 5, 3), tokens)
n_src, n_tgt = 3, 2
print(f"  encoder-decoder: {n_src**2} + {n_tgt * (n_tgt + 1) // 2} + {n_tgt * n_src}  (notes: 9 + 3 + 6)")

sentence = "students often study late at night before exams".split()
print("\n6.2 Three objectives on:", " ".join(sentence))
mlm_in = sentence.copy()
mlm_in[2] = "[MASK]"
print("  MLM    input :", " ".join(mlm_in), "| target: study  -> 1 supervised")
print("  causal input :", " ".join(["<bos>"] + sentence))
print("         target:", " ".join(sentence + ["<eos>"]), f"-> {len(sentence) + 1} supervised")


def span_corrupt(tokens: list[str], spans: list[tuple[int, int]]) -> tuple[list[str], list[str]]:
    """T5-style: replace each span with a sentinel; the target lists each sentinel and its span."""
    sentinels = ["<X>", "<Y>", "<Z>", "<W>"]
    inp, tgt, pos = [], [], 0
    for s, (a, b) in zip(sentinels, spans):
        inp += tokens[pos:a] + [s]
        tgt += [s] + tokens[a:b]
        pos = b
    return inp + tokens[pos:], tgt + [sentinels[len(spans)]]


inp, tgt = span_corrupt(sentence, [(2, 4), (6, 7)])
print("  span   input :", " ".join(inp))
print("         target:", " ".join(tgt), "-> 3 corrupted tokens")
assert " ".join(inp) == "students often <X> at night <Y> exams"
assert " ".join(tgt) == "<X> study late <Y> before <Z>"


# %% 3. A general implementation: one attention layer, three behaviours
def attention_layer(X: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Self-attention with identity projections and the given boolean pattern."""
    S = X @ X.T / np.sqrt(X.shape[1])
    return softmax(np.where(m, S, -np.inf)) @ X


X = rng.normal(size=(5, 8))
outs = {k: attention_layer(X, mask(k, 5, 3)) for k in ("encoder", "decoder", "prefix")}
print("\nSame weights, same input; only the mask differs. Output for token 1:")
for k, Y in outs.items():
    print(f"  {k:8s}: {Y[0, :4]}")


# %% 4. Experiments
def experiment_append_token() -> None:
    """Append a token: decoder outputs for old positions stay the same, encoder outputs all change."""
    print("\n[Experiment 1] Append one token. Do the earlier positions' outputs change?")
    X_new = np.vstack([X, rng.normal(size=(1, 8))])
    for kind in ["encoder", "decoder"]:
        old = attention_layer(X, mask(kind, 5))
        new = attention_layer(X_new, mask(kind, 6))[:5]
        print(f"  {kind:8s}: max change on old positions = {np.abs(old - new).max():.3f}")
    print("  -> only a causal decoder can cache old keys/values and reuse them (Section 14)")


def experiment_fractions() -> None:
    """Fraction of n^2 pairs allowed: decoder -> 1/2, prefix-LM with p = n/2 -> 5/8."""
    print("\n[Experiment 2] Fraction of all n^2 pairs allowed")
    for n in [4, 16, 128, 2048]:
        dec = n * (n + 1) / 2 / n**2
        p = n // 2
        pre = (p * p + (n * (n + 1) - p * (p + 1)) / 2) / n**2
        print(f"  n = {n:5d}: decoder {dec:.3f}   prefix-LM (p = n/2) {pre:.3f}")


def experiment_supervision() -> None:
    """Supervised predictions per sentence for each objective."""
    print("\n[Experiment 3] Supervised predictions per sentence")
    for n in [8, 20, 50, 512]:
        print(f"  {n:4d} tokens: MLM ~{0.15 * n:5.1f}   span ~{0.15 * n:5.1f}   causal {n + 1:4d} (incl. <eos>)")


experiment_append_token()
experiment_fractions()
experiment_supervision()


# %% 5. Student TODOs
def enc_dec_pairs(n_src: int, n_tgt: int) -> int:
    """Allowed pairs in an encoder-decoder: encoder self + decoder causal self + cross."""
    # TODO(student): three terms.
    raise NotImplementedError


def mlm_input(tokens: list[str], positions: list[int]) -> list[str]:
    """Replace the given positions with [MASK] (no 10/10 rule here)."""
    # TODO(student): copy the list and overwrite.
    raise NotImplementedError


def frame_prompt(review: str) -> str:
    """Decoder-only framing of sentiment classification as next-token prediction."""
    # TODO(student): return e.g. "Review: <review>. Sentiment:" (exact format below).
    raise NotImplementedError


checks = [
    ("enc_dec_pairs", lambda: enc_dec_pairs(3, 2) == 18 and enc_dec_pairs(4, 3) == 34),
    ("mlm_input", lambda: mlm_input(sentence, [2]) == mlm_in),
    ("frame_prompt", lambda: frame_prompt("the movie was great") == "Review: the movie was great. Sentiment:"),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
