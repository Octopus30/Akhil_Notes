"""Section 19 lab: Interview Revision (capstone).

Part A recomputes every answer to the notes' mixed problems (P1-P10) and the final
self-test with small helper functions. Part B assembles a modern decoder-only
model from the course's pieces (RMSNorm, RoPE, grouped-query attention, SwiGLU,
causal mask, tied output head, KV cache) and checks that cached generation
matches full recomputation.

Run:  uv run python 19_Interview_Revision/lab.py
"""

# %% 1. Setup
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=3, suppress=True)


def softmax(z: np.ndarray) -> np.ndarray:
    """Stable softmax over the last axis."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# %% 2. Part A: reproduce the notes' mixed problems
def params_decoder(d: int, layers: int, vocab: int, tied: bool = True) -> int:
    """12 d^2 per block (standard FFN, no biases/norms) plus embeddings."""
    return 12 * d * d * layers + vocab * d * (1 if tied else 2)


def kv_bytes_per_token(layers: int, kv_heads: int, d_head: int, bytes_per: int = 2) -> int:
    """2 (K, V) x layers x KV heads x d_head x bytes."""
    return 2 * layers * kv_heads * d_head * bytes_per


q, K, V = np.array([1.0, 1.0]), np.array([[1.0, 0.0], [0.0, 2.0]]), np.array([[1.0, 0.0], [0.0, 1.0]])
w = softmax(K @ q / np.sqrt(2))
print("P1  attention weights", w, "output", w @ V, "  (notes: [0.33, 0.67])")

N = params_decoder(2048, 24, 32_000)
print(f"P2  parameters {N:,}  (notes: 1,273,495,552)")
kv = kv_bytes_per_token(24, 16, 128)
print(f"P3  KV cache {kv:,} B/token, {kv * 8192 / 2**30:.1f} GiB at 8,192 tokens")
print(f"P4  Chinchilla {20 * N / 1e9:.1f}B tokens, C = {6 * N * 20 * N:.2g} FLOPs")
print(f"P5  LoRA r=8 on Q,V: {24 * 2 * 8 * (2048 + 2048):,} trainable")
print(f"P6  weights {N * 2 / 1e9:.2f} GB (16-bit), {N * 0.5 / 1e9:.2f} GB (INT4); "
      f">= {N * 2 / 2e12 * 1e3:.2f} ms/token at 2 TB/s")
print("P7  T = 0.5 on [2, 1, 0]:", softmax(np.array([2.0, 1.0, 0.0]) / 0.5))
print(f"P8  perplexity e^2 = {np.exp(2.0):.2f}")
print("P9  sinusoid pos 2, d = 4:", np.array([np.sin(2), np.cos(2), np.sin(0.02), np.cos(0.02)]))
print(f"P10 448/14 -> {(448 // 14) ** 2} patches, cost {((448 // 14) ** 2 / 196) ** 2:.0f}x")
print(f"Self-test 2: d = 4096, 32 layers -> {params_decoder(4096, 32, 32_000) / 1e9:.2f}B  (notes: 6.57B)")
assert N == 1_273_495_552 and kv == 196_608


# %% 3. Part B: a modern decoder-only model, from the course's pieces
def rms_norm(X: np.ndarray, g: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Section 8: divide each row by its root-mean-square, then scale."""
    return g * X / np.sqrt((X**2).mean(-1, keepdims=True) + eps)


def rope(X: np.ndarray, pos: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Section 7: rotate pairs (2i, 2i+1) of each row by pos * base^(-2i/d). X: (..., n, d)."""
    d = X.shape[-1]
    ang = pos[:, None] * base ** (-2 * np.arange(d // 2) / d)
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(X)
    out[..., 0::2] = X[..., 0::2] * c - X[..., 1::2] * s
    out[..., 1::2] = X[..., 0::2] * s + X[..., 1::2] * c
    return out


class TinyLLM:
    """Embeddings -> N x [RMSNorm, GQA attention with RoPE, RMSNorm, SwiGLU] -> RMSNorm -> tied head."""

    def __init__(self, vocab: int = 32, d: int = 32, layers: int = 2, heads: int = 4, kv_heads: int = 2):
        self.h, self.g, self.dh = heads, kv_heads, d // heads
        d_ff = int(8 * d / 3)
        s = d**-0.5
        self.E = rng.normal(0, 0.5, (vocab, d))
        self.blocks = [dict(g1=np.ones(d), g2=np.ones(d),
                            Wq=rng.normal(0, s, (d, d)), Wk=rng.normal(0, s, (d, kv_heads * self.dh)),
                            Wv=rng.normal(0, s, (d, kv_heads * self.dh)), Wo=rng.normal(0, s, (d, d)),
                            W1=rng.normal(0, s, (d, d_ff)), W3=rng.normal(0, s, (d, d_ff)),
                            W2=rng.normal(0, d_ff**-0.5, (d_ff, d))) for _ in range(layers)]
        self.g_final = np.ones(d)
        self.cache = None

    def _attend(self, b: dict, Z: np.ndarray, pos: np.ndarray, layer: int, use_cache: bool) -> np.ndarray:
        n = len(Z)
        Q = Z @ b["Wq"]
        K = Z @ b["Wk"]
        V = Z @ b["Wv"]
        Q = rope(Q.reshape(n, self.h, self.dh).transpose(1, 0, 2), pos)          # h x n x dh
        K = rope(K.reshape(n, self.g, self.dh).transpose(1, 0, 2), pos)          # g x n x dh
        V = V.reshape(n, self.g, self.dh).transpose(1, 0, 2)
        if use_cache:                                                            # Section 14
            if self.cache[layer] is not None:
                K = np.concatenate([self.cache[layer][0], K], axis=1)
                V = np.concatenate([self.cache[layer][1], V], axis=1)
            self.cache[layer] = (K, V)
        rep = self.h // self.g                                                   # Section 6: GQA
        K, V = np.repeat(K, rep, axis=0), np.repeat(V, rep, axis=0)
        t = K.shape[1]
        S = Q @ K.transpose(0, 2, 1) / np.sqrt(self.dh)                          # Section 5
        key_pos = np.arange(pos[-1] + 1 - t, pos[-1] + 1)                        # keys end at the last query
        allowed = key_pos[None, :] <= pos[:, None]                               # causal mask
        A = softmax(np.where(allowed, S, -np.inf))
        return (A @ V).transpose(1, 0, 2).reshape(n, -1) @ b["Wo"]

    def forward(self, ids: np.ndarray, start: int = 0, use_cache: bool = False) -> np.ndarray:
        """Token IDs -> logits (n x vocabulary). With use_cache, pass only the new tokens."""
        X = self.E[ids]
        pos = np.arange(start, start + len(ids))
        for layer, b in enumerate(self.blocks):
            X = X + self._attend(b, rms_norm(X, b["g1"]), pos, layer, use_cache)
            Z = rms_norm(X, b["g2"])
            gate = Z @ b["W1"]
            X = X + ((gate / (1 + np.exp(-gate))) * (Z @ b["W3"])) @ b["W2"]      # SwiGLU
        return rms_norm(X, self.g_final) @ self.E.T                              # tied head

    def generate(self, prompt: np.ndarray, n_new: int, use_cache: bool = True) -> list[int]:
        """Greedy generation (Section 13), with or without the KV cache."""
        seq = [int(t) for t in prompt]
        if use_cache:
            self.cache = [None] * len(self.blocks)
            logits = self.forward(np.array(seq), 0, use_cache=True)              # prefill
            for _ in range(n_new):
                seq.append(int(logits[-1].argmax()))
                logits = self.forward(np.array(seq[-1:]), len(seq) - 1, use_cache=True)
        else:
            for _ in range(n_new):
                seq.append(int(self.forward(np.array(seq))[-1].argmax()))
        return seq


model = TinyLLM()
prompt = np.array([3, 14, 15, 9])
fast, slow = model.generate(prompt, 8, use_cache=True), model.generate(prompt, 8, use_cache=False)
print("\nTinyLLM (RMSNorm + RoPE + GQA + SwiGLU, 2 layers; untrained, so it soon repeats itself)")
print("  with KV cache   :", fast)
print("  full recompute  :", slow)
print("  identical:", fast == slow, "| cache per layer: K", model.cache[0][0].shape, "(KV heads x tokens x d_head)")
assert fast == slow


# %% 4. Experiments
def experiment_shift_invariance() -> None:
    """RoPE: attention scores depend only on relative positions, so shifting all positions changes nothing."""
    print("\n[Experiment 1] Shift every position by 100: are the logits unchanged?")
    ids = rng.integers(0, 32, 6)
    a = model.forward(ids, start=0)
    b = model.forward(ids, start=100)
    print("  max difference:", float(np.abs(a - b).max()))


def experiment_causality() -> None:
    """Changing a later token never changes earlier positions' logits."""
    print("\n[Experiment 2] Edit the last token: do earlier logits move?")
    ids = rng.integers(0, 32, 6)
    ids2 = ids.copy()
    ids2[-1] = (ids2[-1] + 1) % 32
    diff = np.abs(model.forward(ids) - model.forward(ids2)).max(axis=1)
    print("  per-position change:", diff)


def experiment_model_cards() -> None:
    """Quick sizes for a few configurations (no biases/norms, standard FFN, tied 32k vocabulary)."""
    print("\n[Experiment 3] Model cards")
    for d, L, h, kvh in [(2048, 24, 16, 16), (3072, 28, 24, 24), (3072, 28, 24, 8), (4096, 32, 32, 8)]:
        dh = d // h
        n = L * (2 * d * d + 2 * d * kvh * dh + 8 * d * d) + 32_000 * d
        kv = kv_bytes_per_token(L, kvh, dh)
        print(f"  d={d}, L={L}, heads={h}, KV heads={kvh}: {n / 1e9:.2f}B params, "
              f"{kv / 1024:.0f} KiB/token, {kv * 8192 / 2**30:.2f} GiB at 8k")


experiment_shift_invariance()
experiment_causality()
experiment_model_cards()


# %% 5. Student TODOs
def temperature_probs(logits: np.ndarray, T: float) -> np.ndarray:
    """softmax(logits / T)."""
    # TODO(student): one line using softmax().
    raise NotImplementedError


def chinchilla_compute(N: float) -> float:
    """Training FLOPs at the Chinchilla-optimal D = 20 N, using C = 6 N D."""
    # TODO(student): one line.
    raise NotImplementedError


def count_tinyllm_params(m: TinyLLM) -> int:
    """Total parameters of a TinyLLM instance (embeddings + every array in every block + final norm)."""
    # TODO(student): sum .size over m.E, every value in each block dict, and m.g_final.
    raise NotImplementedError


checks = [
    ("temperature_probs", lambda: np.allclose(np.round(temperature_probs(np.array([2.0, 1.0, 0.0]), 0.5), 3),
                                              [0.867, 0.117, 0.016])),
    ("chinchilla_compute", lambda: np.isclose(chinchilla_compute(1.27e9), 1.94e20, rtol=0.01)),
    ("count_tinyllm_params", lambda: count_tinyllm_params(model) == 32 * 32 + 2 * (
        2 * 32 + 32 * 32 * 2 + 2 * 32 * 16 + 3 * 32 * 85) + 32),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
