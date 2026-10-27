"""Section 5 lab: Scaled Dot-Product Attention.

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M) V, built step by step,
with causal and padding masks, plus experiments on why sqrt(d_k) matters.

Run:  uv run python 05_Scaled_Dot_Product_Attention/lab.py
"""

# %% 1. Setup
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=2, suppress=True)


# %% 2. Reproduce the worked examples from notes.md
Q = np.array([[0.2, 0.2], [1.0, 1.0], [0.0, 2.0]])   # the, bat, flew
K = np.array([[0.2, 0.2], [1.0, 1.0], [2.0, 0.0]])
V = np.array([[0.4, 0.2], [2.0, 1.0], [2.0, 0.0]])
n, d_k = Q.shape

S = Q @ K.T                       # step 1: scores
S_scaled = S / np.sqrt(d_k)       # step 2: scale
print("S / sqrt(2):\n", S_scaled)


def softmax_rows(S: np.ndarray) -> np.ndarray:
    """Stable softmax over the last axis; -inf entries get weight exactly 0."""
    S = S - S.max(axis=-1, keepdims=True)
    e = np.exp(S)
    return e / e.sum(axis=-1, keepdims=True)


A = softmax_rows(S_scaled)        # step 4 (no mask yet)
Y = A @ V                         # step 5
print("A:\n", A, "\nY:\n", Y)
assert np.allclose(np.round(A[2], 2), [0.21, 0.64, 0.16])
assert np.allclose(np.round(Y, 2), [[1.54, 0.41], [1.78, 0.46], [1.67, 0.68]])

M_causal = np.where(np.tril(np.ones((n, n))) == 1, 0.0, -np.inf)    # step 3: mask
A_causal = softmax_rows(S_scaled + M_causal)
print("\nCausal A (lower-triangular):\n", A_causal)
assert np.allclose(np.round(A_causal[1], 2), [0.24, 0.76, 0.0])
print("y_the with causal mask =", (A_causal @ V)[0], " (notes: v_the = [0.4, 0.2])")

M_pad = np.array([[0, 0, -np.inf]] * 3)   # 'flew' treated as padding: column 3 blocked
print("Padding A (column 3 blocked):\n", softmax_rows(S_scaled + M_pad))

p_unscaled, p_scaled = softmax_rows(np.array([8.0, 0.0])), softmax_rows(np.array([1.0, 0.0]))
g_u, g_s = p_unscaled[0] * (1 - p_unscaled[0]), p_scaled[0] * (1 - p_scaled[0])
print(f"\nGradient p(1-p): scores [8,0] -> {g_u:.5f}, [1,0] -> {g_s:.3f}  ({g_s / g_u:.0f}x; notes: ~590x)")


# %% 3. A general implementation
def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
              mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention. mask: booleans, True = allowed. Works with leading batch dims."""
    d_k = Q.shape[-1]
    S = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        S = np.where(mask, S, -np.inf)
    A = softmax_rows(S)
    assert np.allclose(A.sum(axis=-1), 1)
    return A @ V, A


def causal_mask(n: int) -> np.ndarray:
    """Lower-triangular boolean mask: token i may attend to j <= i."""
    return np.tril(np.ones((n, n), dtype=bool))


def padding_mask(lengths: list[int], n: int) -> np.ndarray:
    """(batch, n, n) mask that blocks padded key columns in each sequence."""
    real = np.arange(n)[None, :] < np.array(lengths)[:, None]
    return np.repeat(real[:, None, :], n, axis=1)


Y_gen, _ = attention(Q, K, V)
assert np.allclose(Y_gen, Y)

# A batch of 2 sequences of length 4 (the second has 1 padding token), d_k = 8, causal + padding.
Qb, Kb, Vb = (rng.normal(size=(2, 4, 8)) for _ in range(3))
mask = causal_mask(4)[None] & padding_mask([4, 3], 4)
Yb, Ab = attention(Qb, Kb, Vb, mask)
assert Yb.shape == (2, 4, 8) and Ab.shape == (2, 4, 4)
print("\nBatched causal+padding weights, sequence 2 (last column must be 0):\n", Ab[1])

try:  # optional: compare with PyTorch if it is installed (`uv add torch`)
    import torch
    import torch.nn.functional as F
    out = F.scaled_dot_product_attention(*(torch.tensor(x) for x in (Q, K, V)))
    print("PyTorch scaled_dot_product_attention matches:", np.allclose(out.numpy(), Y))
except ImportError:
    print("(PyTorch not installed; skipping the library comparison)")


# %% 4. Experiments
def experiment_variance() -> None:
    """Var(q . k) grows like d_k; dividing by sqrt(d_k) brings it back to ~1."""
    print("\n[Experiment 1] Variance of q.k (4,000 random pairs, seed 0; matches the notes' table)")
    r = np.random.default_rng(0)
    for d in [4, 64, 512]:
        dots = (r.standard_normal((4000, d)) * r.standard_normal((4000, d))).sum(axis=1)
        print(f"  d_k={d:4d}: var {dots.var():7.1f}  std {dots.std():6.2f}  scaled std {(dots / np.sqrt(d)).std():.2f}")


def experiment_saturation() -> None:
    """Without scaling, softmax over 10 random keys becomes almost one-hot as d_k grows."""
    print("\n[Experiment 2] Largest attention weight among 10 keys (average of 200 queries)")
    for d in [4, 64, 512]:
        q, k = rng.standard_normal((200, d)), rng.standard_normal((10, d))
        S = q @ k.T
        raw = softmax_rows(S).max(axis=1).mean()
        scaled = softmax_rows(S / np.sqrt(d)).max(axis=1).mean()
        print(f"  d_k={d:4d}: unscaled {raw:.2f}   scaled {scaled:.2f}")


def experiment_wrong_mask() -> None:
    """Masking with 0 instead of -inf leaks attention to the 'blocked' token."""
    print("\n[Experiment 3] Masking with 0 vs -inf, scores [1, 2, 3], block position 3")
    s = np.array([1.0, 2.0, 3.0])
    print("  score -> 0   :", softmax_rows(np.array([1.0, 2.0, 0.0])))
    print("  score -> -inf:", softmax_rows(np.array([1.0, 2.0, -np.inf])))
    print("  zero after softmax (rows no longer sum to 1):", softmax_rows(s) * [1, 1, 0])


experiment_variance()
experiment_saturation()
experiment_wrong_mask()


# %% 5. Student TODOs
def naive_softmax(s: np.ndarray) -> np.ndarray:
    """Softmax WITHOUT subtracting the max. Try it on [1000, 1001, 1002] and explain what you see."""
    # TODO(student): e = np.exp(s); return e / e.sum(). Then compare with softmax_rows.
    raise NotImplementedError


def sliding_window_mask(n: int, w: int) -> np.ndarray:
    """Causal mask where token i only sees the last w tokens: i-w < j <= i."""
    # TODO(student): start from causal_mask(n) and also require i - j < w.
    raise NotImplementedError


def attention_memory_bytes(n: int, heads: int, layers: int, bytes_per: int = 2) -> int:
    """Memory to store every attention matrix A at once."""
    # TODO(student): n * n numbers per head per layer.
    raise NotImplementedError


def _check_naive() -> bool:
    with np.errstate(over="ignore", invalid="ignore"):
        out = naive_softmax(np.array([1000.0, 1001.0, 1002.0]))
    return bool(np.isnan(out).any())          # the naive version overflows to inf/inf = nan


checks = [
    ("naive_softmax", _check_naive),
    ("sliding_window_mask", lambda: sliding_window_mask(4, 2).sum() == 7 and not sliding_window_mask(4, 2)[3, 1]),
    ("attention_memory_bytes", lambda: attention_memory_bytes(4096, 32, 32) == 34_359_738_368),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
