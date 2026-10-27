"""Section 13 lab: Inference & Decoding.

Turn logits into a chosen token: greedy, temperature, top-k, top-p, min-p,
penalties, beam search, and constrained decoding with a -inf mask.

Run:  uv run python 13_Inference_and_Decoding/lab.py
"""

# %% 1. Setup
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=3, suppress=True)

WORDS = ["fell", "poured", "stopped", "began", "was", "danced"]
Z = np.array([3.0, 2.5, 2.0, 1.5, 1.0, -1.0])          # "The monsoon rain ..." (notes)


def softmax_T(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Softmax of logits divided by temperature T."""
    e = np.exp((z - z.max()) / T)
    return e / e.sum()


def truncate(p: np.ndarray, top_k: int | None = None, top_p: float | None = None,
             min_p: float | None = None) -> np.ndarray:
    """Zero out tokens removed by top-k / top-p / min-p, then renormalize."""
    keep = np.ones_like(p, dtype=bool)
    order = np.argsort(-p)
    if top_k is not None:
        keep[order[top_k:]] = False
    if top_p is not None:
        cut = np.searchsorted(np.cumsum(p[order]), top_p) + 1
        keep[order[cut:]] = False
    if min_p is not None:
        keep &= p >= min_p * p.max()
    q = np.where(keep, p, 0.0)
    return q / q.sum()


# %% 2. Reproduce the worked examples from notes.md
print("6.1 Temperature")
for T in (0.5, 1.0, 2.0):
    print(f"  T = {T}: {softmax_T(Z, T)}")
assert np.allclose(np.round(softmax_T(Z, 1.0), 3), [0.425, 0.258, 0.156, 0.095, 0.058, 0.008])
assert round(softmax_T(Z, 0.5)[0], 3) == 0.636 and round(softmax_T(Z, 2.0)[-1], 3) == 0.040

p1 = softmax_T(Z)
print("6.2 Truncation at T = 1")
print("  top-k 3  :", truncate(p1, top_k=3))
print("  top-p 0.9:", truncate(p1, top_p=0.9), f"keeps {int((truncate(p1, top_p=0.9) > 0).sum())} tokens")
print("  min-p 0.1:", truncate(p1, min_p=0.1), f"keeps {int((truncate(p1, min_p=0.1) > 0).sum())} tokens")
assert (truncate(p1, top_p=0.9) > 0).sum() == 4 and (truncate(p1, min_p=0.1) > 0).sum() == 5

FIRST = {"the": 0.5, "a": 0.4, "my": 0.1}
SECOND = {"the": {"cat": 0.4, "dog": 0.3, "end": 0.3}, "a": {"cat": 0.9, "dog": 0.05, "end": 0.05},
          "my": {"cat": 0.5, "dog": 0.5}}


def beam_search(first: dict, second: dict, B: int) -> tuple[list[str], float]:
    """Two-step beam search with beam width B; B = 1 is greedy."""
    beams = sorted(first.items(), key=lambda kv: -kv[1])[:B]
    cands = sorted(([a, b], pa * pb) for a, pa in beams for b, pb in second[a].items())
    cands.sort(key=lambda c: -c[1])
    return cands[0][0], round(cands[0][1], 4)


print("6.3 Beam search: greedy", beam_search(FIRST, SECOND, 1), " B=2", beam_search(FIRST, SECOND, 2))
assert beam_search(FIRST, SECOND, 2) == (["a", "cat"], 0.36)

Zpen = Z.copy()
Zpen[0] -= 0.5
print("6.4 Presence penalty 0.5 on 'fell':", softmax_T(Zpen))
print(f"6.5 Latency: TTFT {500 / 5000:.1f} s + decode {200 * 0.03:.1f} s = {500 / 5000 + 200 * 0.03:.1f} s")


# %% 3. A general implementation: one sampling step and a generation loop
def sample_next(logits: np.ndarray, rng, temperature: float = 1.0, top_k=None, top_p=None, min_p=None,
                seen: set[int] = frozenset(), presence: float = 0.0, allowed: np.ndarray | None = None) -> int:
    """Penalties -> constraint mask -> temperature -> truncation -> sample. temperature 0 = greedy."""
    z = logits.astype(float).copy()
    for i in seen:
        z[i] -= presence
    if allowed is not None:
        z = np.where(allowed, z, -np.inf)         # constrained decoding
    if temperature == 0:
        return int(np.argmax(z))
    p = truncate(softmax_T(z, temperature), top_k, top_p, min_p)
    return int(rng.choice(len(p), p=p))


draws = np.array([sample_next(Z, rng, top_p=0.9) for _ in range(20_000)])
print("\nEmpirical top-p 0.9 frequencies:", np.bincount(draws, minlength=6) / len(draws))
print("Expected                        :", truncate(p1, top_p=0.9))


# %% 4. Experiments
def experiment_entropy_vs_temperature() -> None:
    """Entropy rises with temperature (towards ln 6 = 1.79 for uniform)."""
    print("\n[Experiment 1] Entropy of the distribution vs temperature")
    for T in [0.1, 0.5, 1, 2, 5, 50]:
        p = softmax_T(Z, T)
        H = -(p[p > 0] * np.log(p[p > 0])).sum()
        print(f"  T = {T:<4}: entropy {H:.2f}   top token {p.max():.2f}")


def experiment_top_p_adapts() -> None:
    """Top-p keeps few tokens when confident and many when unsure; top-k always keeps k."""
    print("\n[Experiment 2] How many tokens survive (vocabulary of 100)")
    confident = softmax_T(np.r_[8.0, rng.normal(0, 1, 99)])
    unsure = softmax_T(rng.normal(0, 0.3, 100))
    for name, p in [("confident", confident), ("unsure", unsure)]:
        print(f"  {name:9s}: top-p 0.9 keeps {int((truncate(p, top_p=0.9) > 0).sum()):3d}   "
              f"top-k 10 keeps {int((truncate(p, top_k=10) > 0).sum())}")


def experiment_greedy_loops() -> None:
    """A toy 'model' that likes repeating: greedy loops; a presence penalty breaks the loop."""
    print("\n[Experiment 3] Greedy loops vs presence penalty")
    vocab = ["I", "think", "that", "it", "rains", "."]
    # next-token logits depend only on the last token; 'that' -> 'I' creates a cycle
    table = {0: [0, 3, 1, 1, 0, 0], 1: [0, 0, 3, 1, 0, 0], 2: [2.5, 0, 0, 2, 0, 0],
             3: [0, 0, 0, 0, 3, 1], 4: [0, 0, 0, 0, 0, 3], 5: [1, 0, 0, 0, 0, 0]}
    for presence in [0.0, 1.5]:
        seq, seen = [0], set()
        for _ in range(9):
            seen.add(seq[-1])
            seq.append(sample_next(np.array(table[seq[-1]], float), rng, temperature=0,
                                   seen=seen, presence=presence))
        print(f"  presence {presence}: {' '.join(vocab[i] for i in seq)}")


experiment_entropy_vs_temperature()
experiment_top_p_adapts()
experiment_greedy_loops()


# %% 5. Student TODOs
def length_penalized_score(logprob: float, length: int, alpha: float = 0.7) -> float:
    """Beam-search score with a length penalty: logprob / length**alpha."""
    # TODO(student): one line.
    raise NotImplementedError


def frequency_penalty(logits: np.ndarray, counts: dict[int, int], strength: float) -> np.ndarray:
    """Subtract strength * (number of times the token appeared) from each logit."""
    # TODO(student): copy logits and subtract per token.
    raise NotImplementedError


def choose_label(logits: np.ndarray, allowed_ids: list[int]) -> int:
    """Constrained greedy choice: only tokens in allowed_ids may be picked."""
    # TODO(student): build a boolean mask and call sample_next(..., temperature=0, allowed=mask).
    raise NotImplementedError


checks = [
    ("length_penalized_score", lambda: np.isclose(length_penalized_score(-4.0, 8, 0.5), -4.0 / np.sqrt(8))),
    ("frequency_penalty", lambda: np.allclose(frequency_penalty(Z, {0: 2, 1: 1}, 0.5), Z - [1.0, 0.5, 0, 0, 0, 0])),
    ("choose_label", lambda: choose_label(Z, [2, 4, 5]) == 2),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
