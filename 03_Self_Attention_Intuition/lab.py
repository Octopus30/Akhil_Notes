"""Section 3 lab: Self-Attention, the Intuition.

Naive self-attention from scratch: S = X X^T, A = softmax_rows(S), Y = A X.
No projections and no scaling yet (Sections 4 and 5 add them).

Run:  uv run python 03_Self_Attention_Intuition/lab.py
"""

# %% 1. Setup
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=2, suppress=True)


# %% 2. Reproduce the worked example from notes.md ("the bat flew" / "the bat hit")
EMB = {  # toy axes: [animal-ness, sport-ness]
    "the": [0.2, 0.2],
    "bat": [1.0, 1.0],
    "flew": [2.0, 0.0],
    "hit": [0.0, 2.0],
}


def embed(sentence: str) -> np.ndarray:
    """Stack the toy embeddings of the words as rows: n x d_model."""
    return np.array([EMB[w] for w in sentence.split()])


def softmax_rows(S: np.ndarray) -> np.ndarray:
    """Softmax of each row, made stable by subtracting the row max."""
    S = S - S.max(axis=1, keepdims=True)
    e = np.exp(S)
    return e / e.sum(axis=1, keepdims=True)


X_a = embed("the bat flew")
S = X_a @ X_a.T
A = softmax_rows(S)
Y = A @ X_a
assert S.shape == A.shape == (3, 3) and Y.shape == X_a.shape == (3, 2)
assert np.allclose(A.sum(axis=1), 1)

print("Sentence A: 'the bat flew'")
print("S = X X^T:\n", S)
print("A (rows sum to 1):\n", A)
print("y_bat =", Y[1], "  (notes: [1.38, 0.47])")
assert np.allclose(np.round(Y[1], 2), [1.38, 0.47])
print("flew attends to itself with weight", round(A[2, 2], 2), "(notes: 0.86)")

X_b = embed("the bat hit")
Y_b = softmax_rows(X_b @ X_b.T) @ X_b
print("\nSentence B: 'the bat hit'  ->  y_bat =", Y_b[1], "  (notes: [0.47, 1.38])")
assert np.allclose(np.round(Y_b[1], 2), [0.47, 1.38])


# %% 3. A general implementation
def naive_self_attention(X: np.ndarray, causal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (Y, A) for naive self-attention; causal=True hides future tokens."""
    n, d_model = X.shape
    S = X @ X.T
    if causal:
        S = np.where(np.tril(np.ones((n, n), dtype=bool)), S, -np.inf)
    A = softmax_rows(S)
    Y = A @ X
    assert A.shape == (n, n) and Y.shape == (n, d_model)
    return Y, A


Y_gen, A_gen = naive_self_attention(X_a)
assert np.allclose(Y_gen, Y)
_, A_causal = naive_self_attention(X_a, causal=True)
print("\nCausal A for 'the bat flew' (upper triangle is zero):\n", A_causal)


# %% 4. Experiments
def show_matrix(A: np.ndarray, labels: list[str]) -> None:
    """Print an attention matrix with row/column labels and a tiny bar for each weight."""
    print("        " + "".join(f"{l:>8}" for l in labels))
    for label, row in zip(labels, A):
        bars = "".join(f"{'#' * int(round(w * 5)):>8}" for w in row)
        print(f"  {label:>6}" + bars)


def experiment_context_changes_meaning() -> None:
    """The same word 'bat' drifts toward whichever axis its neighbour points to."""
    print("\n[Experiment 1] Context moves 'bat'")
    for s in ["the bat flew", "the bat hit", "the bat", "bat flew hit"]:
        X = embed(s)
        Y, _ = naive_self_attention(X)
        i = s.split().index("bat")
        print(f"  {s!r:16}: bat {X[i]} -> {Y[i]}")


def experiment_length_freezes_softmax() -> None:
    """Scaling all embeddings up makes softmax sharper until each token only sees itself."""
    print("\n[Experiment 2] Bigger vectors -> sharper attention (motivates Section 5)")
    for scale in [0.5, 1, 2, 4]:
        _, A = naive_self_attention(scale * X_a)
        print(f"  scale {scale:>3}: weights of 'bat' = {A[1]}  self-weight of 'flew' = {A[2, 2]:.2f}")


def experiment_order_blind_and_quadratic() -> None:
    """Shuffling tokens only shuffles outputs; and S has n^2 entries."""
    print("\n[Experiment 3] Order-blind and quadratic")
    perm = np.array([2, 0, 1])
    Y_perm, _ = naive_self_attention(X_a[perm])
    print("  shuffled output == shuffled original output:", np.allclose(Y_perm, Y[perm]))
    for n in [10, 100, 1_000, 10_000]:
        print(f"  n = {n:>6}: S has {n * n:>12,} entries ({n * n * 4 / 1e6:,.1f} MB in float32)")


show_matrix(A, ["the", "bat", "flew"])
experiment_context_changes_meaning()
experiment_length_freezes_softmax()
experiment_order_blind_and_quadratic()


# %% 5. Student TODOs
def attention_entropy(A: np.ndarray) -> np.ndarray:
    """Entropy of each row of A, -sum_j a_ij log a_ij: high = spread out, 0 = all on one token."""
    # TODO(student): compute it row by row. Hint: treat 0 * log 0 as 0 (use np.where).
    raise NotImplementedError


def top_attended(A: np.ndarray, labels: list[str]) -> list[str]:
    """For each token (row), return the label of the token it attends to most."""
    # TODO(student): use np.argmax along axis 1.
    raise NotImplementedError


def attention_without_self(X: np.ndarray) -> np.ndarray:
    """Naive attention where a token may NOT attend to itself (diagonal set to -inf)."""
    # TODO(student): copy naive_self_attention but block the diagonal. Return Y.
    raise NotImplementedError


checks = [
    ("attention_entropy", lambda: np.allclose(attention_entropy(np.eye(3)), 0)
        and np.isclose(attention_entropy(np.full((1, 4), 0.25))[0], np.log(4))),
    ("top_attended", lambda: top_attended(A, ["the", "bat", "flew"])[2] == "flew"),
    ("attention_without_self", lambda: np.allclose(attention_without_self(X_a)[1],
        np.array([0.2, 0.2]) * (np.exp(0.4) / (np.exp(0.4) + np.exp(2)))
        + np.array([2.0, 0.0]) * (np.exp(2) / (np.exp(0.4) + np.exp(2))))),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
