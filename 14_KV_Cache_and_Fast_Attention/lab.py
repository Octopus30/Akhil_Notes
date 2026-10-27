"""Section 14 lab: KV Cache & Fast Attention.

A KV cache that gives exactly the same outputs as recomputing everything,
cache-size and decode-speed arithmetic for real models, the online softmax,
and a tiled (FlashAttention-style) attention that never builds the n x n matrix.

Run:  uv run python 14_KV_Cache_and_Fast_Attention/lab.py
"""

# %% 1. Setup
import time

import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)


def softmax(s: np.ndarray) -> np.ndarray:
    """Stable softmax over the last axis."""
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def kv_positions(p: int, T: int, cache: bool) -> int:
    """Token positions whose K/V are computed when generating T tokens after a p-token prompt."""
    return p + (T - 1) if cache else T * p + T * (T - 1) // 2


def cache_bytes(layers: int, kv_heads: int, d_head: int, tokens: int, batch: int = 1, bytes_per: int = 2) -> int:
    """2 (K and V) x layers x KV heads x d_head x bytes x tokens x batch."""
    return 2 * layers * kv_heads * d_head * bytes_per * tokens * batch


def online_softmax(s: np.ndarray, v: np.ndarray, chunk: int) -> np.ndarray:
    """Exact softmax(s) @ v for one query, chunk by chunk with running max m, sum l, output o."""
    m, l, o = -np.inf, 0.0, np.zeros(v.shape[1:])
    for a in range(0, len(s), chunk):
        sc, vc = s[a:a + chunk], v[a:a + chunk]
        m_new = max(m, sc.max())
        scale = np.exp(m - m_new)
        w = np.exp(sc - m_new)
        l, o, m = l * scale + w.sum(), o * scale + w @ vc, m_new
    return o / l


# %% 2. Reproduce the worked examples from notes.md
print("Work saved: 1,000-token prompt + 500 generated:",
      f"{kv_positions(1000, 500, False):,} vs {kv_positions(1000, 500, True):,} K/V computations")
print("            5-token prompt + 3 generated:", kv_positions(5, 3, False), "vs", kv_positions(5, 3, True))
assert (kv_positions(1000, 500, False), kv_positions(1000, 500, True)) == (624_750, 1_499)

print("\n6.2 Cache sizes (16-bit)")
for name, L, kv in [("LLaMA-2 7B (MHA)", 32, 32), ("Mistral 7B (GQA)", 32, 8), ("LLaMA-2 70B (GQA)", 80, 8)]:
    per = cache_bytes(L, kv, 128, 1)
    print(f"  {name:18s}: {per / 2**20:.3f} MiB/token, {cache_bytes(L, kv, 128, 4096) / 2**30:.2f} GiB at 4,096 tokens")
print(f"  32 users of LLaMA-2 7B at 4,096 tokens: {cache_bytes(32, 32, 128, 4096, 32) / 2**30:.0f} GiB")

print(f"\n6.3 Decode bound: 13.5 GB / 2.0 TB/s = {13.5e9 / 2.0e12 * 1000:.2f} ms/token (~{2.0e12 / 13.5e9:.0f} tokens/s)")

s, v = np.array([2.0, 1.0, 3.0, 0.0]), np.array([[1.0], [2.0], [3.0], [4.0]])
print(f"\n6.4 Online softmax, two chunks: {online_softmax(s, v, 2)[0]:.4f}   ordinary: {(softmax(s) @ v)[0]:.4f}"
      "   (notes: 2.4711)")
assert round(float(online_softmax(s, v, 2)[0]), 4) == 2.4711

print(f"6.5 Score matrix at 32,768 tokens: {32_768**2 * 2 / 1e9:.1f} GB per head per layer")


# %% 3. General implementations: cached decoding and tiled attention
class CachedCausalAttention:
    """Single-head causal attention with a KV cache: each step projects only the new token."""

    def __init__(self, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray):
        self.W_Q, self.W_K, self.W_V = W_Q, W_K, W_V
        self.K = np.zeros((0, W_K.shape[1]))
        self.V = np.zeros((0, W_V.shape[1]))

    def step(self, x_t: np.ndarray) -> np.ndarray:
        """Append k_t, v_t to the cache and attend with the single query q_t."""
        q = x_t @ self.W_Q
        self.K = np.vstack([self.K, x_t @ self.W_K])
        self.V = np.vstack([self.V, x_t @ self.W_V])
        return softmax(q @ self.K.T / np.sqrt(self.K.shape[1])) @ self.V


def causal_attention(X: np.ndarray, W_Q, W_K, W_V) -> np.ndarray:
    """Standard causal attention over the whole sequence at once (builds the n x n matrix)."""
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    n = len(X)
    S = np.where(np.tril(np.ones((n, n), bool)), Q @ K.T / np.sqrt(K.shape[1]), -np.inf)
    return softmax(S) @ V


def tiled_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, tile: int) -> np.ndarray:
    """FlashAttention-style (non-causal): loop over key/value tiles with an online softmax per query row.
    Only n x tile scores exist at any moment, never the full n x n matrix."""
    n, d = Q.shape
    m = np.full((n, 1), -np.inf)
    l = np.zeros((n, 1))
    o = np.zeros((n, V.shape[1]))
    for a in range(0, len(K), tile):
        S = Q @ K[a:a + tile].T / np.sqrt(d)                  # n x tile
        m_new = np.maximum(m, S.max(axis=1, keepdims=True))
        scale = np.exp(m - m_new)
        W = np.exp(S - m_new)
        l = l * scale + W.sum(axis=1, keepdims=True)
        o = o * scale + W @ V[a:a + tile]
        m = m_new
    return o / l


X = rng.normal(size=(12, 16))
Wq, Wk, Wv = (rng.normal(0, 0.25, (16, 8)) for _ in range(3))
attn = CachedCausalAttention(Wq, Wk, Wv)
cached_out = np.array([attn.step(x) for x in X])
print("\nCached decoding == full causal attention:", np.allclose(cached_out, causal_attention(X, Wq, Wk, Wv)))

Q, K, V = (rng.normal(size=(256, 32)) for _ in range(3))
print("Tiled attention == standard attention:",
      np.allclose(tiled_attention(Q, K, V, tile=32), softmax(Q @ K.T / np.sqrt(32)) @ V))


# %% 4. Experiments
def experiment_timing() -> None:
    """Generating with and without a cache (one layer, d = 64)."""
    print("\n[Experiment 1] Time to process 400 steps, one layer, d = 64")
    Xl = rng.normal(size=(400, 64))
    W = [rng.normal(0, 0.1, (64, 64)) for _ in range(3)]
    t0 = time.perf_counter()
    for t in range(1, len(Xl) + 1):
        causal_attention(Xl[:t], *W)[-1]
    slow = time.perf_counter() - t0
    t0 = time.perf_counter()
    c = CachedCausalAttention(*W)
    for x in Xl:
        c.step(x)
    fast = time.perf_counter() - t0
    print(f"  recompute everything: {slow:.2f} s    with KV cache: {fast:.2f} s    ({slow / fast:.0f}x)")


def experiment_memory_vs_n() -> None:
    """Peak score memory: standard n^2 vs tiled n x tile (float32)."""
    print("\n[Experiment 2] Score-matrix memory per head (float32)")
    for n in [1_024, 8_192, 32_768, 131_072]:
        print(f"  n = {n:>7,}: standard {n * n * 4 / 2**20:>10,.0f} MiB   tiled (tile 128) {n * 128 * 4 / 2**20:>6,.1f} MiB")


def experiment_users_per_gpu() -> None:
    """How many users fit in 80 GB next to 13.5 GB of weights, for MHA vs GQA, at several context lengths."""
    print("\n[Experiment 3] Users that fit on an 80 GB GPU (7B model, 13.5 GB weights)")
    free = 80e9 - 13.5e9
    for tokens in [2_048, 4_096, 16_384]:
        mha = free // cache_bytes(32, 32, 128, tokens)
        gqa = free // cache_bytes(32, 8, 128, tokens)
        print(f"  {tokens:>6,} tokens: MHA {int(mha):3d} users   GQA (8 KV heads) {int(gqa):3d} users")


experiment_timing()
experiment_memory_vs_n()
experiment_users_per_gpu()


# %% 5. Student TODOs
def sliding_window_step(K: np.ndarray, V: np.ndarray, k_t: np.ndarray, v_t: np.ndarray, W: int):
    """Append k_t, v_t and keep only the last W rows of the cache. Return (K, V)."""
    # TODO(student): np.vstack then slice [-W:].
    raise NotImplementedError


def quantize_int8(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Absmax INT8 quantization of a cache tensor: scale = max|x| / 127, q = round(x / scale)."""
    # TODO(student): return (q as np.int8, scale). The error should be at most scale / 2.
    raise NotImplementedError


def max_users(gpu_bytes: float, weight_bytes: float, per_user_cache_bytes: float) -> int:
    """How many users' caches fit next to the weights."""
    # TODO(student): integer division of the free memory.
    raise NotImplementedError


Kc, Vc = rng.normal(size=(4, 8)), rng.normal(size=(4, 8))


def _check_quant() -> bool:
    x = rng.normal(size=1000)
    q, sc = quantize_int8(x)
    return q.dtype == np.int8 and np.abs(q * sc - x).max() <= sc / 2 + 1e-12


checks = [
    ("sliding_window_step", lambda: sliding_window_step(Kc, Vc, np.ones(8), np.ones(8), 3)[0].shape == (3, 8)
        and np.allclose(sliding_window_step(Kc, Vc, np.ones(8), np.ones(8), 3)[0][-1], 1)),
    ("quantize_int8", _check_quant),
    ("max_users", lambda: max_users(80e9, 26e9, 6.25 * 2**30) == 8),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
