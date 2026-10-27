"""Section 8 lab: Residuals, LayerNorm & the FFN.

The parts around attention: LayerNorm / RMSNorm, the feed-forward network,
residual connections, a full pre-norm block, and exact parameter counts.

Run:  uv run python 08_Residuals_LayerNorm_FFN/lab.py
"""

# %% 1. Setup
from math import erf, sqrt

import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=2, suppress=True)


def layer_norm(X: np.ndarray, gamma=1.0, beta=0.0, eps: float = 1e-5) -> np.ndarray:
    """Normalize each row to mean 0 / variance 1, then scale by gamma and shift by beta."""
    mu = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    return gamma * (X - mu) / np.sqrt(var + eps) + beta


def rms_norm(X: np.ndarray, gamma=1.0, eps: float = 1e-5) -> np.ndarray:
    """Divide each row by its root-mean-square (no centring, no beta)."""
    return gamma * X / np.sqrt((X**2).mean(axis=-1, keepdims=True) + eps)


relu = lambda z: np.maximum(z, 0)
gelu = np.vectorize(lambda z: 0.5 * z * (1 + erf(z / sqrt(2))))


def ffn(X: np.ndarray, W1: np.ndarray, W2: np.ndarray, b1=0.0, b2=0.0, act=relu) -> np.ndarray:
    """phi(X W1 + b1) W2 + b2, applied to every row (token) on its own."""
    return act(X @ W1 + b1) @ W2 + b2


# %% 2. Reproduce the worked examples from notes.md
x = np.array([[2.0, 4.0, 6.0, 8.0]])
print("6.1 LayerNorm([2,4,6,8])            =", layer_norm(x, eps=0), "(notes: [-1.34,-0.45,0.45,1.34])")
print("    with gamma=[1,1,2,2], beta=[0,0,0,1] =", layer_norm(x, np.array([1, 1, 2, 2]), np.array([0, 0, 0, 1]), eps=0))
print("    RMSNorm([2,4,6,8])              =", rms_norm(x, eps=0), "(notes: [0.37,0.73,1.10,1.46])")
assert np.allclose(np.round(layer_norm(x, eps=0), 2), [-1.34, -0.45, 0.45, 1.34])

W1 = np.array([[1.0, -1.0, 0.5, 0.0], [0.0, 1.0, 1.0, -1.0]])
W2 = np.array([[0.5, 0.0], [0.0, 1.0], [0.0, 0.5], [1.0, 1.0]])
xt = np.array([[1.0, 0.5]])
print("\n6.2 FFN: h =", xt @ W1, " ReLU(h) =", relu(xt @ W1))
print("    residual (ReLU) =", xt + ffn(xt, W1, W2), " (notes: [1.5, 1.0])")
print("    residual (GELU) =", xt + ffn(xt, W1, W2, act=gelu), " (notes: [1.27, 0.61])")
assert np.allclose(xt + ffn(xt, W1, W2), [[1.5, 1.0]])
assert np.allclose(np.round(xt + ffn(xt, W1, W2, act=gelu), 2), [[1.27, 0.61]])

print("\n6.3 Gradient factor with F'(x) = 0.1")
for L in [3, 24]:
    print(f"    {L:2d} layers: without residuals {0.1**L:.3g}   with residuals {1.1**L:.3f}")


def gpt2_params(d: int, n_layers: int, vocab: int, ctx: int) -> int:
    """GPT-2 parameter count: blocks + biases + LayerNorms + tied embeddings + positions."""
    blocks = n_layers * 12 * d * d
    biases = n_layers * (3 * d + d + 4 * d + d)
    norms = n_layers * 2 * 2 * d + 2 * d
    return blocks + biases + norms + vocab * d + ctx * d


def llama_params(d: int, n_layers: int, d_ff: int, vocab: int) -> int:
    """LLaMA-style count: attention 4d^2 + SwiGLU 3 d d_ff per block, untied head, RMSNorm."""
    per_block = 4 * d * d + 3 * d * d_ff
    return n_layers * per_block + 2 * vocab * d + (2 * n_layers + 1) * d


print("\n6.4 Exact parameter counts")
print(f"    GPT-2 small : {gpt2_params(768, 12, 50_257, 1_024):,}   (notes: 124,439,808)")
print(f"    LLaMA-2 7B  : {llama_params(4096, 32, 11_008, 32_000):,} (notes: 6,738,415,616)")
assert gpt2_params(768, 12, 50_257, 1_024) == 124_439_808
assert llama_params(4096, 32, 11_008, 32_000) == 6_738_415_616


# %% 3. A general implementation: the pre-norm block
def attention(X: np.ndarray, W_Q, W_K, W_V, W_O) -> np.ndarray:
    """Single-head scaled dot-product self-attention plus output projection (Sections 4-5)."""
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    S = Q @ K.T / np.sqrt(Q.shape[1])
    A = np.exp(S - S.max(axis=1, keepdims=True))
    A /= A.sum(axis=1, keepdims=True)
    return A @ V @ W_O


def init_block(d: int, d_ff: int) -> dict:
    """Random weights for one block, scaled so outputs start at a sensible size."""
    return {"attn": [rng.normal(0, d**-0.5, (d, d)) for _ in range(4)],
            "W1": rng.normal(0, d**-0.5, (d, d_ff)), "W2": rng.normal(0, d_ff**-0.5, (d_ff, d))}


def pre_norm_block(X: np.ndarray, p: dict) -> np.ndarray:
    """A = X + Attn(LN(X)); X' = A + FFN(LN(A))."""
    A = X + attention(layer_norm(X), *p["attn"])
    out = A + ffn(layer_norm(A), p["W1"], p["W2"], act=gelu)
    assert out.shape == X.shape
    return out


X = rng.normal(size=(4, 8))
Y = pre_norm_block(X, init_block(8, 32))
print("\nPre-norm block: in", X.shape, "-> out", Y.shape)


# %% 4. Experiments
def experiment_gradient_depth() -> None:
    """Numerically differentiate a deep stack of small scalar layers, with and without residuals."""
    print("\n[Experiment 1] Gradient d(out)/d(in) through a stack of F(x) = 0.1 * tanh(x)")
    F = lambda z: 0.1 * np.tanh(z)
    for L in [3, 12, 24]:
        def run(x0: float, residual: bool) -> float:
            z = x0
            for _ in range(L):
                z = z + F(z) if residual else F(z)
            return z
        h = 1e-6
        g_plain = (run(0.5 + h, False) - run(0.5 - h, False)) / (2 * h)
        g_res = (run(0.5 + h, True) - run(0.5 - h, True)) / (2 * h)
        print(f"  {L:2d} layers: without residuals {g_plain:.2e}   with residuals {g_res:.2f}")


def experiment_layernorm_vs_batchnorm() -> None:
    """LayerNorm output for a token doesn't change when the rest of the batch changes; BatchNorm does."""
    print("\n[Experiment 2] Does the normalized token depend on its batch-mates?")
    token = rng.normal(size=(1, 6))
    for trial in range(2):
        batch = np.vstack([token, rng.normal(3, 2, size=(3, 6))])
        ln = layer_norm(batch)[0]
        bn = ((batch - batch.mean(axis=0)) / batch.std(axis=0))[0]
        print(f"  batch {trial + 1}: LayerNorm {ln[:3]}   BatchNorm {bn[:3]}")


def experiment_residual_stream_growth() -> None:
    """Each block adds to the stream, so row norms grow with depth; a final norm resets them."""
    print("\n[Experiment 3] Row-norm of the residual stream through 12 pre-norm blocks (d = 16)")
    Xs = rng.normal(size=(3, 16))
    for layer in range(1, 13):
        Xs = pre_norm_block(Xs, init_block(16, 64))
        if layer in (1, 3, 6, 12):
            print(f"  after block {layer:2d}: mean row norm {np.linalg.norm(Xs, axis=1).mean():6.2f}"
                  f"   after final LN {np.linalg.norm(layer_norm(Xs), axis=1).mean():.2f}")


experiment_gradient_depth()
experiment_layernorm_vs_batchnorm()
experiment_residual_stream_growth()


# %% 5. Student TODOs
def swiglu_ffn(X: np.ndarray, W1: np.ndarray, W3: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """SwiGLU FFN: (SiLU(X W1) * (X W3)) W2, where SiLU(z) = z * sigmoid(z)."""
    # TODO(student): element-wise product of the gated branch and the linear branch.
    raise NotImplementedError


def post_norm_block(X: np.ndarray, p: dict) -> np.ndarray:
    """The 2017 arrangement: A = LN(X + Attn(X)); X' = LN(A + FFN(A))."""
    # TODO(student): same pieces as pre_norm_block, different order.
    raise NotImplementedError


def block_params(d: int, d_ff: int, swiglu: bool = False) -> int:
    """Parameters in one block's attention + FFN (no biases, no norms)."""
    # TODO(student): attention 4 d^2; FFN 2 d d_ff (standard) or 3 d d_ff (SwiGLU).
    raise NotImplementedError


Xt = rng.normal(size=(3, 4))
Wa, Wb, Wc = rng.normal(size=(4, 6)), rng.normal(size=(4, 6)), rng.normal(size=(6, 4))
sig = lambda z: 1 / (1 + np.exp(-z))
pt = init_block(4, 16)
checks = [
    ("swiglu_ffn", lambda: np.allclose(swiglu_ffn(Xt, Wa, Wb, Wc), ((Xt @ Wa) * sig(Xt @ Wa) * (Xt @ Wb)) @ Wc)),
    ("post_norm_block", lambda: np.allclose(post_norm_block(Xt, pt).mean(axis=1), 0, atol=1e-6)),
    ("block_params", lambda: block_params(4096, 11_008, swiglu=True) == 202_375_168
        and block_params(768, 3072) == 7_077_888),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
