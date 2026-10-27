"""Section 4 lab: Queries, Keys & Values.

Project each token into a query, a key and a value, then attend:
Q = X W_Q, K = X W_K, V = X W_V, S = Q K^T, A = softmax_rows(S), Y = A V.
(Still no sqrt(d_k) scaling; that is Section 5.)

Run:  uv run python 04_Queries_Keys_Values/lab.py
"""

# %% 1. Setup
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=2, suppress=True)


def softmax_rows(S: np.ndarray) -> np.ndarray:
    """Softmax of each row, made stable by subtracting the row max."""
    e = np.exp(S - S.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# %% 2. Reproduce the worked example from notes.md (fixing "flew")
X = np.array([[0.2, 0.2],    # the
              [1.0, 1.0],    # bat
              [2.0, 0.0]])   # flew
W_Q = np.array([[0.0, 1.0], [1.0, 0.0]])   # swaps the axes
W_K = np.eye(2)
W_V = np.array([[1.0, 0.0], [1.0, 1.0]])

Q, K, V = X @ W_Q, X @ W_K, X @ W_V
assert Q.shape == K.shape == V.shape == (3, 2)
print("Q =\n", Q, "\nK =\n", K, "\nV =\n", V)

S = Q @ K.T
A = softmax_rows(S)
Y = A @ V
assert S.shape == A.shape == (3, 3) and Y.shape == (3, 2)

print("\nS = Q K^T:\n", S)
print("A:\n", A)
print("Y = A V:\n", Y)

naive_A = softmax_rows(X @ X.T)
print(f"\n'flew' self-weight: naive {naive_A[2, 2]:.2f} -> with Q/K {A[2, 2]:.2f}  (notes: 0.86 -> 0.10)")
print(f"'flew' attends to 'bat': {A[2, 1]:.2f}  (notes: 0.75)")
assert np.allclose(np.round(A[2], 2), [0.15, 0.75, 0.10])
assert np.allclose(np.round(Y, 2), [[1.57, 0.42], [1.85, 0.47], [1.76, 0.78]])

# 6.2 asymmetry in two lines
a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
W_Q2 = np.array([[0.0, 1.0], [0.0, 0.0]])
s_ab, s_ba = (a @ W_Q2) @ b, (b @ W_Q2) @ a
print(f"\nAsymmetry: s_ab = {s_ab:.0f}, s_ba = {s_ba:.0f}  (notes: 1 and 0)")

# 6.3 real sizes
for name, d_model in [("BERT-base", 768), ("LLaMA-2 7B", 4096)]:
    print(f"{name:10s}: Q,K,V params per layer = 3 x {d_model}^2 = {3 * d_model**2:,}")


# %% 3. A general implementation
def qkv_attention(X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray,
                  X_kv: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Attention with projections. If X_kv is given, keys/values come from it (cross-attention)."""
    X_kv = X if X_kv is None else X_kv
    Q, K, V = X @ W_Q, X_kv @ W_K, X_kv @ W_V
    assert Q.shape[1] == K.shape[1], "queries and keys must share width d_k"
    A = softmax_rows(Q @ K.T)
    Y = A @ V
    assert A.shape == (X.shape[0], X_kv.shape[0]) and Y.shape == (X.shape[0], W_V.shape[1])
    return Y, A


Y_gen, _ = qkv_attention(X, W_Q, W_K, W_V)
assert np.allclose(Y_gen, Y)

# Random, BERT-like shapes for one head: n = 128, d_model = 768, d_k = d_v = 64
n, d_model, d_k = 128, 768, 64
Xr = rng.normal(size=(n, d_model))
Wq, Wk, Wv = (rng.normal(0, d_model**-0.5, size=(d_model, d_k)) for _ in range(3))
Yr, Ar = qkv_attention(Xr, Wq, Wk, Wv)
print(f"\nBERT-like head: X {Xr.shape} -> A {Ar.shape} -> Y {Yr.shape}")


# %% 4. Experiments
def experiment_self_weight() -> None:
    """Naive attention puts lots of weight on the diagonal; random Q/K projections much less."""
    print("\n[Experiment 1] Average self-weight (diagonal of A), 8 random tokens, d_model = 16")
    Xs = rng.normal(size=(8, 16))
    naive = np.mean(np.diag(softmax_rows(Xs @ Xs.T)))
    Wq, Wk = rng.normal(0, 0.25, (16, 16)), rng.normal(0, 0.25, (16, 16))
    proj = np.mean(np.diag(softmax_rows((Xs @ Wq) @ (Xs @ Wk).T)))
    print(f"  naive X X^T: {naive:.2f}    with W_Q != W_K: {proj:.2f}    uniform would be {1 / 8:.2f}")


def experiment_symmetry() -> None:
    """S is symmetric exactly when W_Q W_K^T is symmetric."""
    print("\n[Experiment 2] When is S symmetric?")
    Xs = rng.normal(size=(5, 4))
    W = rng.normal(size=(4, 4))
    cases = {"W_Q = W_K": (W, W), "W_Q = W_K @ sym": (W @ (W.T @ W), W), "random W_Q, W_K": (W, rng.normal(size=(4, 4)))}
    for name, (wq, wk) in cases.items():
        S = (Xs @ wq) @ (Xs @ wk).T
        print(f"  {name:16s}: S symmetric? {np.allclose(S, S.T)}   M symmetric? {np.allclose(wq @ wk.T, (wq @ wk.T).T)}")


def experiment_cross_attention() -> None:
    """Queries from one sequence, keys/values from another: A is n_tgt x n_src."""
    print("\n[Experiment 3] Cross-attention shapes")
    X_tgt, X_src = rng.normal(size=(4, 8)), rng.normal(size=(6, 8))
    Wq, Wk, Wv = (rng.normal(size=(8, 3)) for _ in range(3))
    Yc, Ac = qkv_attention(X_tgt, Wq, Wk, Wv, X_kv=X_src)
    print(f"  X_tgt {X_tgt.shape}, X_src {X_src.shape} -> A {Ac.shape}, Y {Yc.shape}")


experiment_self_weight()
experiment_symmetry()
experiment_cross_attention()


# %% 5. Student TODOs
def fused_qkv(X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray) -> tuple:
    """Compute Q, K, V with ONE matrix multiply using a d_model x 3d_k matrix."""
    # TODO(student): np.concatenate the weights along axis 1, multiply once, then np.split.
    raise NotImplementedError


def qkv_params(d_model: int, h: int) -> int:
    """Parameters in W_Q, W_K, W_V for one layer with h heads of width d_k = d_model / h."""
    # TODO(student): each head has three d_model x d_k matrices.
    raise NotImplementedError


def bilinear_M(W_Q: np.ndarray, W_K: np.ndarray) -> np.ndarray:
    """Return M = W_Q W_K^T, so that s_ij = x_i M x_j^T."""
    # TODO(student): one line. Then check (X @ M @ X.T) equals S.
    raise NotImplementedError


checks = [
    ("fused_qkv", lambda: all(np.allclose(a, b) for a, b in zip(fused_qkv(X, W_Q, W_K, W_V), (Q, K, V)))),
    ("qkv_params", lambda: qkv_params(768, 12) == 1_769_472 and qkv_params(5120, 40) == 78_643_200),
    ("bilinear_M", lambda: np.allclose(X @ bilinear_M(W_Q, W_K) @ X.T, S)),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
