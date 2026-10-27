"""Section 7 lab: Positional Information.

Sinusoidal position vectors, adding them to embeddings so order matters,
RoPE (rotate queries and keys), and ALiBi (a distance penalty on scores).

Run:  uv run python 07_Positional_Information/lab.py
"""

# %% 1. Setup
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=3, suppress=True)


def softmax(S: np.ndarray) -> np.ndarray:
    """Stable softmax over the last axis."""
    e = np.exp(S - S.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def sinusoidal_positions(L: int, d_model: int) -> np.ndarray:
    """L x d_model table: p[pos, 2i] = sin(pos * w_i), p[pos, 2i+1] = cos(pos * w_i)."""
    pos = np.arange(L)[:, None]
    w = 10000.0 ** (-2 * np.arange(d_model // 2) / d_model)
    P = np.zeros((L, d_model))
    P[:, 0::2] = np.sin(pos * w)
    P[:, 1::2] = np.cos(pos * w)
    return P


# %% 2. Reproduce the worked examples from notes.md
print("6.1 Sinusoidal table, d_model = 4")
P4 = sinusoidal_positions(4, 4)
print(P4)
assert np.allclose(np.round(P4, 3), [[0, 1, 0, 1], [0.841, 0.540, 0.010, 1.000],
                                     [0.909, -0.416, 0.020, 1.000], [0.141, -0.990, 0.030, 1.000]])

EMB = {"the": [0.2, 0.2], "bat": [1.0, 1.0], "flew": [2.0, 0.0]}


def bat_output(sentence: str, positions: bool) -> np.ndarray:
    """Naive self-attention output for 'bat', with or without sinusoidal positions added."""
    words = sentence.split()
    X = np.array([EMB[w] for w in words])
    if positions:
        X = X + sinusoidal_positions(len(words), 2)
    return (softmax(X @ X.T) @ X)[words.index("bat")]


print("\n6.2 Order finally matters ('bat' output)")
for s in ["the bat flew", "flew bat the"]:
    print(f"  {s!r:15}: without {bat_output(s, False)}   with {bat_output(s, True)}")
assert np.allclose(np.round(bat_output("the bat flew", True), 2), [2.08, 1.04])
assert np.allclose(np.round(bat_output("flew bat the", True), 2), [1.89, 1.32])


def rotate2d(v: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 2-D vector by angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def rope_score_2d(q: np.ndarray, k: np.ndarray, m: int, n: int, theta: float) -> float:
    """Score between q at position m and k at position n after RoPE rotation."""
    return float(rotate2d(q, m * theta) @ rotate2d(k, n * theta))


print("\n6.3 RoPE depends only on the distance (theta = 30 degrees)")
th = np.deg2rad(30)
e1 = np.array([1.0, 0.0])
for m, n in [(1, 3), (4, 6), (3, 3)]:
    print(f"  q=k=[1,0], m={m}, n={n}: score {rope_score_2d(e1, e1, m, n, th):.2f}")
q, k = np.array([1.0, 0.5]), np.array([0.5, 1.0])
for m, n in [(2, 5), (7, 10)]:
    print(f"  q=[1,0.5], k=[0.5,1], m={m}, n={n}: score {rope_score_2d(q, k, m, n, th):.2f}")
assert np.isclose(rope_score_2d(q, k, 2, 5, th), -0.75)

print("\n6.4 ALiBi in one row (slope 0.5, equal content scores)")
bias = -0.5 * (3 - np.arange(4))
print("  bias", bias, " weights", softmax(1.0 + bias))
assert np.allclose(np.round(softmax(bias), 2), [0.10, 0.17, 0.28, 0.46])

print("\n6.5 Learned position parameters: GPT-2", f"{1024 * 768:,}", " BERT-base", f"{512 * 768:,}")


# %% 3. General implementations: RoPE for any even width, ALiBi bias matrix
def rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Rotate pairs (2i, 2i+1) of each row of x (n x d) by positions[row] * base^(-2i/d)."""
    n, d = x.shape
    theta = base ** (-2 * np.arange(d // 2) / d)
    ang = positions[:, None] * theta[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    out = np.empty_like(x, dtype=float)
    out[:, 0::2] = x[:, 0::2] * cos - x[:, 1::2] * sin
    out[:, 1::2] = x[:, 0::2] * sin + x[:, 1::2] * cos
    assert out.shape == x.shape
    return out


def alibi_bias(n: int, slope: float) -> np.ndarray:
    """n x n causal ALiBi bias: -slope * (i - j) for j <= i, -inf above the diagonal."""
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    return np.where(j <= i, -slope * (i - j) + 0.0, -np.inf)   # + 0.0 turns -0.0 into 0.0


Qr, Kr = rng.normal(size=(6, 8)), rng.normal(size=(6, 8))
pos = np.arange(6)
S1 = rope(Qr, pos) @ rope(Kr, pos).T
S2 = rope(Qr, pos + 50) @ rope(Kr, pos + 50).T
print("\nRoPE scores unchanged when every position shifts by 50:", np.allclose(S1, S2))
print("ALiBi bias, n = 4, slope 0.5:\n", alibi_bias(4, 0.5))


# %% 4. Experiments
def experiment_clock_speeds() -> None:
    """Fast pairs separate neighbours; slow pairs separate far-apart positions."""
    print("\n[Experiment 1] Distance between position vectors, d_model = 64")
    P = sinusoidal_positions(5001, 64)
    for a, b in [(1, 2), (10, 11), (10, 5000)]:
        fast = np.linalg.norm(P[a, :2] - P[b, :2])
        slow = np.linalg.norm(P[a, -2:] - P[b, -2:])
        print(f"  pos {a:>4} vs {b:>4}: fast pair differs by {fast:.3f}, slowest pair by {slow:.4f}")


def experiment_rope_decay() -> None:
    """With RoPE, the score of a query with an identical key tends to shrink as the distance grows."""
    print("\n[Experiment 2] RoPE: average q.k for q = k (d = 64, 200 random vectors)")
    X = rng.normal(size=(200, 64))
    for gap in [0, 1, 4, 16, 64, 256]:
        s = np.mean(np.sum(rope(X, np.full(200, gap)) * X, axis=1))
        print(f"  distance {gap:>3}: {s:6.2f}")


def experiment_alibi_slopes() -> None:
    """Different heads get different slopes: some look locally, some widely."""
    print("\n[Experiment 3] ALiBi: attention of the last of 8 tokens (equal content scores)")
    for slope in [1.0, 0.25, 0.0625]:
        print(f"  slope {slope:<6}:", softmax(alibi_bias(8, slope)[-1]))


experiment_clock_speeds()
experiment_rope_decay()
experiment_alibi_slopes()


# %% 5. Student TODOs
def learned_position_params(L_max: int, d_model: int) -> int:
    """Parameters in a learned absolute position table."""
    # TODO(student): one line.
    raise NotImplementedError


def interpolate_positions(positions: np.ndarray, trained_len: int, new_len: int) -> np.ndarray:
    """Position interpolation: squeeze positions so new_len maps onto trained_len."""
    # TODO(student): multiply by trained_len / new_len.
    raise NotImplementedError


def rope_matrix_form(x: np.ndarray, pos: int, base: float = 10000.0) -> np.ndarray:
    """RoPE for ONE row vector x (length d) by building the d x d block-diagonal rotation matrix."""
    # TODO(student): put a 2x2 rotation [[c, s], [-s, c]] (for row vectors) on each diagonal block,
    # with angle pos * base^(-2i/d), then return x @ R. Compare with rope(x[None], np.array([pos])).
    raise NotImplementedError


xv = rng.normal(size=8)
checks = [
    ("learned_position_params", lambda: learned_position_params(1024, 768) == 786_432),
    ("interpolate_positions", lambda: np.allclose(interpolate_positions(np.array([0, 8000, 16000]), 4096, 16384),
                                                   [0, 2000, 4000])),
    ("rope_matrix_form", lambda: np.allclose(rope_matrix_form(xv, 5), rope(xv[None], np.array([5]))[0])),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
