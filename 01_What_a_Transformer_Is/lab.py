"""Section 1 lab: What a Transformer Is.

Build a tiny Transformer-shaped pipeline in NumPy and watch the shapes:
tokens -> IDs -> embeddings -> N blocks -> output scores -> probabilities.

The block here is a simplified stand-in (naive self-attention + a small FFN);
Sections 3-8 replace each piece with the real thing.

Run:  uv run python 01_What_a_Transformer_Is/lab.py
"""

# %% 1. Setup
import time

import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=3, suppress=True)


# %% 2. Reproduce the worked example from notes.md ("glass is full")
vocab = {"glass": 7, "is": 2, "full": 5}
vocab_size, d_model, N = 10, 4, 2

tokens = ["glass", "is", "full"]
ids = np.array([vocab[t] for t in tokens])
assert ids.shape == (3,)

# Embedding table 10 x 4: random rows, except rows 7, 2, 5 use the numbers from the notes.
E = rng.normal(0, 0.3, size=(vocab_size, d_model))
E[7] = [0.10, 0.50, -0.20, 0.30]
E[2] = [0.40, 0.00, 0.10, -0.10]
E[5] = [-0.30, 0.20, 0.60, 0.00]

X0 = E[ids]
assert X0.shape == (3, 4)
print("Worked example: 'glass is full'")
print("token IDs:", ids, "shape", ids.shape)
print("X0 (rows 7, 2, 5 of E):\n", X0, "shape", X0.shape)


def layer_norm(X: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Rescale each row to mean 0 and variance 1 (no learned gamma/beta here)."""
    mu = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    return (X - mu) / np.sqrt(var + eps)


def softmax(S: np.ndarray) -> np.ndarray:
    """Softmax along the last axis, made stable by subtracting the row max."""
    e = np.exp(S - S.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def toy_self_attention(X: np.ndarray) -> np.ndarray:
    """Naive self-attention: every row becomes a weighted average of all rows (Section 3)."""
    A = softmax(X @ X.T)
    return A @ X


def toy_ffn(X: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """Two-layer network applied to each row on its own (Section 8)."""
    return np.maximum(0, X @ W1) @ W2


def block(X: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """Pre-norm block: A = X + Attn(LN(X)), X' = A + FFN(LN(A))."""
    A = X + toy_self_attention(layer_norm(X))
    return A + toy_ffn(layer_norm(A), W1, W2)


blocks = [(rng.normal(0, 0.3, (d_model, 4 * d_model)), rng.normal(0, 0.3, (4 * d_model, d_model)))
          for _ in range(N)]
W_out = rng.normal(0, 0.3, size=(d_model, vocab_size))

X = X0
shape_table = [("Token IDs", ids.shape), ("Embeddings X0", X.shape)]
for b, (W1, W2) in enumerate(blocks, start=1):
    X = block(X, W1, W2)
    assert X.shape == (3, 4), "a block must keep the shape n x d_model"
    shape_table.append((f"After block {b}", X.shape))
scores = X @ W_out
assert scores.shape == (3, 10)
shape_table.append(("Output scores", scores.shape))

print("\nShape summary (matches the table in notes.md):")
for stage, shape in shape_table:
    print(f"  {stage:15s} {shape}")

probs = softmax(scores)
print("Each row of probabilities sums to:", probs.sum(axis=1))


# %% 3. A general implementation: the whole pipeline as one function
def transformer_forward(ids: np.ndarray, E: np.ndarray, blocks: list, W_out: np.ndarray) -> np.ndarray:
    """Token IDs (n,) -> probabilities (n x vocabulary size)."""
    X = E[ids]
    n, d = X.shape
    for W1, W2 in blocks:
        X = block(X, W1, W2)
        assert X.shape == (n, d)
    return softmax(X @ W_out)


P = transformer_forward(ids, E, blocks, W_out)
assert np.allclose(P, probs)
print("\ntransformer_forward gives the same probabilities:", P.shape)


# %% 4. Experiments
def experiment_order_blindness() -> None:
    """Shuffle the tokens: with no position information the output rows just shuffle too."""
    perm = np.array([2, 0, 1])                      # full, glass, is
    P_shuffled = transformer_forward(ids[perm], E, blocks, W_out)
    print("\n[Experiment 1] Order blindness")
    print("  output for 'full glass is' == shuffled output for 'glass is full':",
          np.allclose(P_shuffled, P[perm]))
    print("  so the model can't tell word order apart until we add positions (Section 7).")


def experiment_quadratic_cost() -> None:
    """Time the n x n score matrix X X^T as n grows: doubling n roughly quadruples the work."""
    print("\n[Experiment 2] Attention work grows with n^2")
    print(f"  {'n':>6} {'comparisons':>14} {'time (ms)':>10}")
    for n in [500, 1000, 2000, 4000]:
        Xn = rng.normal(size=(n, 64))
        t0 = time.perf_counter()
        for _ in range(3):
            Xn @ Xn.T
        ms = (time.perf_counter() - t0) / 3 * 1000
        print(f"  {n:>6} {n * n:>14,} {ms:>10.2f}")


def experiment_path_length() -> None:
    """Steps for information to travel from token 1 to token n: RNN n-1, Transformer 1."""
    print("\n[Experiment 3] Path length from the first token to the last")
    for n in [3, 10, 100, 1000]:
        print(f"  n = {n:>5}:  RNN {n - 1:>4} steps   Transformer 1 step")


experiment_order_blindness()
experiment_quadratic_cost()
experiment_path_length()


# %% 5. Student TODOs
def count_pipeline_params(vocab_size: int, d_model: int) -> int:
    """Number of numbers in the embedding table plus the output head (no bias)."""
    # TODO(student): return the count. Hint: E is vocab_size x d_model, W_out is d_model x vocab_size.
    raise NotImplementedError


def sequential_steps(n: int, N: int, model: str) -> int:
    """Steps that must run one after another: 'rnn' (one layer) or 'transformer' (N blocks)."""
    # TODO(student): an RNN needs one step per token; a Transformer needs one step per block.
    raise NotImplementedError


def deeper_model_shapes(N: int) -> tuple[int, int]:
    """Run 'glass is full' through N blocks and return the final shape of X."""
    # TODO(student): build N blocks like `blocks` above and loop over them. Hint: the answer
    # shouldn't depend on N at all.
    raise NotImplementedError


checks = [
    ("count_pipeline_params", lambda: count_pipeline_params(10, 4) == 80),
    ("sequential_steps", lambda: sequential_steps(200, 4, "rnn") == 200
        and sequential_steps(200, 4, "transformer") == 4),
    ("deeper_model_shapes", lambda: deeper_model_shapes(12) == (3, 4)),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
