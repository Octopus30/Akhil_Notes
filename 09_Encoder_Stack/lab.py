"""Section 9 lab: The Encoder Stack.

Stack simplified bidirectional blocks and watch "bat" become contextual;
corrupt text with the MLM 80/10/10 rule; count BERT-base exactly; and use
an encoder's outputs (mean pooling, [CLS]-style) for a tiny search.

Run:  uv run python 09_Encoder_Stack/lab.py
"""

# %% 1. Setup
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=2, suppress=True)


def softmax(S: np.ndarray) -> np.ndarray:
    """Stable softmax over the last axis."""
    e = np.exp(S - S.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def simple_block(X: np.ndarray) -> np.ndarray:
    """The notes' simplified encoder block: X + softmax(X X^T / sqrt(d)) X (no LN, no FFN, no mask)."""
    return X + softmax(X @ X.T / np.sqrt(X.shape[1])) @ X


# %% 2. Reproduce the worked examples from notes.md
X_a = np.array([[0.2, 0.2], [1.0, 1.0], [2.0, 0.0]])   # the bat flew
X_b = np.array([[0.2, 0.2], [1.0, 1.0], [0.0, 2.0]])   # the bat hit

print("6.1 Two layers of a tiny encoder ('the bat flew')")
print("  layer-1 attention weights:\n", softmax(X_a @ X_a.T / np.sqrt(2)))
A_layers, B_layers = [X_a], [X_b]
for _ in range(3):
    A_layers.append(simple_block(A_layers[-1]))
    B_layers.append(simple_block(B_layers[-1]))
for l in range(3):
    print(f"  layer {l}: the {A_layers[l][0]}  bat {A_layers[l][1]}  flew {A_layers[l][2]}")
assert np.allclose(np.round(A_layers[1][1], 2), [2.32, 1.46])
assert np.allclose(np.round(A_layers[2][1], 2), [5.60, 2.00])

cosines = [cosine(A_layers[l][1], B_layers[l][1]) for l in range(4)]
print("  cos(bat in A, bat in B) by layer:", np.round(cosines, 2), " (notes: 1.00 0.90 0.63 0.36)")
assert np.allclose(np.round(cosines, 2), [1.00, 0.90, 0.63, 0.36])

p = np.array([0.6, 0.2, 0.9])
print(f"\n6.2 MLM loss on 3 picked tokens: {-np.log(p).mean():.3f}  (notes: 0.742)")


def bert_params(d: int, layers: int, d_ff: int, vocab: int = 30_522, pos: int = 512, seg: int = 2) -> int:
    """BERT recipe: embeddings + LN, layers (attention + FFN + 2 LN, with biases), pooler."""
    embeddings = (vocab + pos + seg) * d + 2 * d
    per_layer = (4 * d * d + 4 * d) + (2 * d * d_ff + d_ff + d) + 4 * d
    pooler = d * d + d
    return embeddings + layers * per_layer + pooler


print(f"\n6.3 BERT-base parameters: {bert_params(768, 12, 3072):,}  (notes: 109,482,240)")
print(f"    BERT-large parameters: {bert_params(1024, 24, 4096):,}  (~335M)")
assert bert_params(768, 12, 3072) == 109_482_240


# %% 3. A general implementation: a small encoder with LayerNorm and FFN
def layer_norm(X: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Per-token normalization (gamma = 1, beta = 0)."""
    return (X - X.mean(-1, keepdims=True)) / np.sqrt(X.var(-1, keepdims=True) + eps)


def encoder(X: np.ndarray, params: list[dict], lengths: int | None = None) -> np.ndarray:
    """Pre-norm encoder: N bidirectional blocks (padding mask only) + final LayerNorm. Returns H."""
    n, d = X.shape
    real = np.arange(n) < (n if lengths is None else lengths)
    mask = np.repeat(real[None, :], n, axis=0)          # block padded key columns only
    for p in params:
        Z = layer_norm(X)
        S = (Z @ p["Wq"]) @ (Z @ p["Wk"]).T / np.sqrt(d)
        A = softmax(np.where(mask, S, -np.inf))
        X = X + A @ (Z @ p["Wv"]) @ p["Wo"]
        X = X + np.maximum(0, layer_norm(X) @ p["W1"]) @ p["W2"]
        assert X.shape == (n, d)
    return layer_norm(X)


def init_params(d: int, N: int) -> list[dict]:
    """Random weights for N blocks."""
    s = d**-0.5
    return [{"Wq": rng.normal(0, s, (d, d)), "Wk": rng.normal(0, s, (d, d)), "Wv": rng.normal(0, s, (d, d)),
             "Wo": rng.normal(0, s, (d, d)), "W1": rng.normal(0, s, (d, 4 * d)),
             "W2": rng.normal(0, (4 * d)**-0.5, (4 * d, d))} for _ in range(N)]


params = init_params(16, 4)
H = encoder(rng.normal(size=(6, 16)), params, lengths=5)
print("\nEncoder: X (6, 16) with 1 padding token -> H", H.shape)


# %% 4. Experiments
def experiment_bidirectional() -> None:
    """Changing the LAST token changes the FIRST token's output in an encoder, but not with a causal mask."""
    print("\n[Experiment 1] Does token 1 see the future?")
    X1 = rng.normal(size=(5, 16))
    X2 = X1.copy()
    X2[-1] = rng.normal(size=16) * 3                  # replace only the last token
    # (adding a constant to every entry would NOT work: LayerNorm removes it, so nothing changes)
    enc_change = np.abs(encoder(X1, params)[0] - encoder(X2, params)[0]).max()
    causal = lambda X: softmax(np.where(np.tril(np.ones((5, 5), bool)), X @ X.T, -np.inf)) @ X
    dec_change = np.abs(causal(X1)[0] - causal(X2)[0]).max()
    print(f"  encoder: first token changed by {enc_change:.3f}   causal: changed by {dec_change:.3f}")


def experiment_mlm_stats() -> None:
    """The 80/10/10 rule on 20,000 tokens."""
    print("\n[Experiment 2] MLM corruption statistics")
    ids = rng.integers(1000, 30_000, size=20_000)
    picked = rng.random(ids.shape) < 0.15
    roll = rng.random(ids.shape)
    out = ids.copy()
    out[picked & (roll < 0.8)] = 103                  # [MASK]
    rnd = picked & (roll >= 0.8) & (roll < 0.9)
    out[rnd] = rng.integers(0, 30_000, rnd.sum())
    print(f"  picked {picked.mean():.3f};  of picked: [MASK] {(out[picked] == 103).mean():.2f}, "
          f"random {((out[picked] != 103) & (out[picked] != ids[picked])).mean():.2f}, "
          f"unchanged {(out[picked] == ids[picked]).mean():.2f}")


def experiment_bi_encoder_search() -> None:
    """Mean-pooled sentence vectors + cosine similarity = a tiny semantic search (toy word vectors)."""
    print("\n[Experiment 3] Bi-encoder search with mean pooling (toy 3-D word vectors)")
    W = {"exam": [1, 0, 0], "test": [0.9, 0.1, 0], "marks": [0.8, 0, 0.2], "hostel": [0, 1, 0],
         "room": [0.1, 0.9, 0], "mess": [0, 0.8, 0.3], "food": [0, 0.3, 1], "menu": [0.1, 0.2, 0.9]}
    docs = ["exam marks", "hostel room", "mess food menu", "test marks"]
    embed = lambda s: np.mean([W[w] for w in s.split()], axis=0)
    D = np.stack([embed(d) for d in docs])
    for query in ["test", "food", "room"]:
        scores = [cosine(embed(query), d) for d in D]
        best = int(np.argmax(scores))
        print(f"  query {query!r:7}: best match {docs[best]!r} (cos {scores[best]:.2f})")


experiment_bidirectional()
experiment_mlm_stats()
experiment_bi_encoder_search()


# %% 5. Student TODOs
def mean_pool(H: np.ndarray, lengths: int) -> np.ndarray:
    """Average the first `lengths` rows of H (ignore padding), then normalize to length 1."""
    # TODO(student): H[:lengths].mean(axis=0), divided by its norm.
    raise NotImplementedError


def mlm_loss(probs_of_original: np.ndarray, picked: np.ndarray) -> float:
    """Average -log p over picked positions only."""
    # TODO(student): index with the boolean mask.
    raise NotImplementedError


def qa_span(start_scores: np.ndarray, end_scores: np.ndarray) -> tuple[int, int]:
    """Extractive QA: return (start, end) maximizing start_scores[s] + end_scores[e] with s <= e."""
    # TODO(student): try every pair s <= e (n is small), keep the best.
    raise NotImplementedError


picked_t = np.array([True, False, True, False, True])
checks = [
    ("mean_pool", lambda: np.isclose(np.linalg.norm(mean_pool(H, 5)), 1.0)
        and np.allclose(mean_pool(H, 5), H[:5].mean(0) / np.linalg.norm(H[:5].mean(0)))),
    ("mlm_loss", lambda: np.isclose(mlm_loss(np.array([0.6, 0.1, 0.2, 0.1, 0.9]), picked_t), 0.742, atol=1e-3)),
    ("qa_span", lambda: qa_span(np.array([0.1, 2.0, 0.3, 0.0]), np.array([0.0, 0.2, 0.1, 1.5])) == (1, 3)),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
