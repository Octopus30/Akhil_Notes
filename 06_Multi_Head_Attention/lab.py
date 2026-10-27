"""Section 6 lab: Multi-Head Attention.

h heads side by side: head_i = Attention(X W_Q^i, X W_K^i, X W_V^i),
MultiHead(X) = [head_1 | ... | head_h] W_O, computed with the reshape trick.
Also grouped-query attention (GQA) and the KV-cache saving it brings.

Run:  uv run python 06_Multi_Head_Attention/lab.py
"""

# %% 1. Setup
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=2, suppress=True)


def softmax(S: np.ndarray) -> np.ndarray:
    """Stable softmax over the last axis."""
    e = np.exp(S - S.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# %% 2. Reproduce the worked example from notes.md (two 1-D heads on "the bat flew")
X = np.array([[0.2, 0.2], [1.0, 1.0], [2.0, 0.0]])   # the, bat, flew
labels = ["the", "bat", "flew"]

heads, weights = [], []
for axis in [0, 1]:                       # head 1 reads animal-ness, head 2 reads sport-ness
    x = X[:, axis:axis + 1]               # q = k = v = that axis, d_k = 1
    A = softmax(x @ x.T / np.sqrt(1))
    heads.append(A @ x)
    weights.append(A)
concat = np.concatenate(heads, axis=1)    # W_O = I, so this is the output
assert concat.shape == X.shape

print("Two heads on 'the bat flew'")
for i, t in enumerate(labels):
    print(f"  {t:5s} head1 {weights[0][i]}  head2 {weights[1][i]}  output {concat[i]}")
assert np.allclose(np.round(concat[1], 2), [1.57, 0.60])
assert np.allclose(weights[1][2], 1 / 3)   # flew's sport query is 0 -> uniform

single = softmax(X @ X.T / np.sqrt(2)) @ X
print("  single head (d_k = 2) for bat:", single[1], " (notes: [1.32, 0.46])")
assert np.allclose(np.round(single[1], 2), [1.32, 0.46])

print("\nParameters per layer (4 d_model^2):")
for name, d in [("BERT-base", 768), ("LLaMA-2 7B", 4096)]:
    print(f"  {name:10s} {4 * d * d:>12,}")
print("KV cache numbers per token per layer (2 x KV heads x d_k):")
for name, kv, d_k in [("LLaMA-2 7B MHA", 32, 128), ("Mistral 7B GQA", 8, 128), ("LLaMA-2 70B GQA", 8, 128)]:
    print(f"  {name:16s} {2 * kv * d_k:>6,}")


# %% 3. A general implementation (reshape trick, with optional GQA)
def split_heads(M: np.ndarray, h: int) -> np.ndarray:
    """n x (h * d_k) -> h x n x d_k."""
    n, width = M.shape
    return M.reshape(n, h, width // h).transpose(1, 0, 2)


def multi_head_attention(X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray,
                         W_O: np.ndarray, h: int, g: int | None = None,
                         causal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """MHA (g = h), GQA (1 < g < h) or MQA (g = 1). Returns (output n x d_model, weights h x n x n)."""
    n, d_model = X.shape
    g = h if g is None else g
    d_k = d_model // h
    Q = split_heads(X @ W_Q, h)
    K, V = split_heads(X @ W_K, g), split_heads(X @ W_V, g)
    K, V = np.repeat(K, h // g, axis=0), np.repeat(V, h // g, axis=0)
    assert Q.shape == K.shape == V.shape == (h, n, d_k)
    S = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    if causal:
        S = np.where(np.tril(np.ones((n, n), dtype=bool)), S, -np.inf)
    A = softmax(S)
    assert A.shape == (h, n, n)
    concat = (A @ V).transpose(1, 0, 2).reshape(n, d_model)
    return concat @ W_O, A


# The toy example again, through the general function: W_Q = W_K = W_V = W_O = I, h = 2
Y_gen, _ = multi_head_attention(X, *(np.eye(2) for _ in range(4)), h=2)
assert np.allclose(Y_gen, concat)

# BERT-base-like shape trace: n = 128, d_model = 768, h = 12
n, d_model, h = 128, 768, 12
Xb = rng.normal(size=(n, d_model))
Ws = [rng.normal(0, d_model**-0.5, size=(d_model, d_model)) for _ in range(4)]
Yb, Ab = multi_head_attention(Xb, *Ws, h=h)
print(f"\nBERT-like layer: X {Xb.shape} -> weights {Ab.shape} -> output {Yb.shape}")


# %% 4. Experiments
def experiment_heads_differ() -> None:
    """With random weights, heads attend differently; one head forces one pattern on all dims."""
    print("\n[Experiment 1] How different are the heads' attention patterns?")
    Xs = rng.normal(size=(6, 16))
    Ws = [rng.normal(0, 0.5, size=(16, 16)) for _ in range(4)]
    _, A = multi_head_attention(Xs, *Ws, h=4)
    for i in range(4):
        print(f"  head {i + 1}: token 1 attends most to token {A[i, 0].argmax() + 1} (weight {A[i, 0].max():.2f})")


def experiment_params_constant() -> None:
    """Changing h keeps the parameters at 4 d_model^2 but multiplies the number of n x n matrices."""
    print("\n[Experiment 2] d_model = 512, n = 1,024: parameters vs attention scores")
    for h in [1, 4, 8, 16, 64]:
        print(f"  h={h:3d}  d_k={512 // h:4d}  params {4 * 512**2:>9,}  scores {h * 1024**2:>12,}")


def experiment_gqa_cache() -> None:
    """GQA output differs from MHA, but the cache shrinks by g/h."""
    print("\n[Experiment 3] KV cache for 32 layers, d_k = 128, 2-byte numbers, 8,192 tokens")
    for name, kv in [("MHA (32 KV heads)", 32), ("GQA (8 KV heads)", 8), ("MQA (1 KV head)", 1)]:
        gib = 2 * kv * 128 * 32 * 2 * 8192 / 2**30
        print(f"  {name:18s}: {gib:5.2f} GiB")


experiment_heads_differ()
experiment_params_constant()
experiment_gqa_cache()


# %% 5. Student TODOs
def mha_loop(X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray,
             W_O: np.ndarray, h: int) -> np.ndarray:
    """Slow reference: loop over heads using column slices of the weights, concatenate, apply W_O."""
    # TODO(student): head i uses columns i*d_k:(i+1)*d_k of W_Q, W_K, W_V.
    raise NotImplementedError


def output_as_sum(X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray,
                  W_O: np.ndarray, h: int) -> np.ndarray:
    """Compute MultiHead(X) as sum_i head_i @ W_O[rows of head i] (no concatenation)."""
    # TODO(student): head i's rows of W_O are i*d_k:(i+1)*d_k.
    raise NotImplementedError


def kv_cache_bytes(layers: int, kv_heads: int, d_k: int, tokens: int, bytes_per: int = 2) -> int:
    """Bytes of KV cache: 2 (K and V) x layers x kv_heads x d_k x tokens x bytes."""
    # TODO(student): one line.
    raise NotImplementedError


Xt = rng.normal(size=(5, 8))
Wt = [rng.normal(size=(8, 8)) for _ in range(4)]
Yt, _ = multi_head_attention(Xt, *Wt, h=2)
checks = [
    ("mha_loop", lambda: np.allclose(mha_loop(Xt, *Wt, h=2), Yt)),
    ("output_as_sum", lambda: np.allclose(output_as_sum(Xt, *Wt, h=2), Yt)),
    ("kv_cache_bytes", lambda: kv_cache_bytes(32, 8, 128, 8192) == 2**30),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
