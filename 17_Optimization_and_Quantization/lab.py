"""Section 17 lab: Optimization & Quantization.

Number formats (FP16 natively, BF16 emulated), absmax INT8/INT4 quantization
with per-group scales, mixed-precision pitfalls, ZeRO memory arithmetic, and a
simulation of speculative decoding.

Run:  uv run python 17_Optimization_and_Quantization/lab.py
"""

# %% 1. Setup
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)


def to_bf16(x) -> np.ndarray:
    """Round float32 values to the nearest BF16 (keep the top 16 bits, round-to-nearest-even)."""
    b = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    b = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return b.astype(np.uint32).view(np.float32)


def quantize(w: np.ndarray, bits: int, group: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric absmax quantization: one scale per group of `group` weights (or per tensor)."""
    qmax = 2 ** (bits - 1) - 1
    g = w.reshape(-1, group or w.size)
    s = np.abs(g).max(axis=1, keepdims=True) / qmax
    s[s == 0] = 1.0
    return np.clip(np.round(g / s), -qmax, qmax).astype(np.int8), s


def dequantize(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """w_hat = s * q, back to 1-D."""
    return (q * s).reshape(-1)


# %% 2. Reproduce the worked examples from notes.md
print("6.1 Formats")
print(f"  FP16 next value above 1: {float(np.nextafter(np.float16(1), np.float16(2))):.6f}"
      f"   BF16 step: {2**-7:.4f}")
print(f"  1e-8 in FP16: {float(np.float16(1e-8))}   x1024 in FP16: {float(np.float16(1e-8 * 1024)):.3e}"
      f"   in BF16: {float(to_bf16(1e-8)):.3e}")
with np.errstate(over="ignore"):
    print(f"  100,000 in FP16: {float(np.float16(100_000))}   in BF16: {float(to_bf16(100_000)):,.0f}")

w = np.array([0.52, -1.17, 0.03, 2.40, -0.71, 0.15])
for bits in (8, 4):
    q, s = quantize(w, bits)
    err = np.abs(dequantize(q, s) - w).mean()
    print(f"\n6.2 INT{bits}: scale {s.item():.4f}, codes {q.ravel()}, mean |error| {err:.3f}")
assert round(float(np.abs(dequantize(*quantize(w, 8)) - w).mean()), 3) == 0.005
assert round(float(np.abs(dequantize(*quantize(w, 4)) - w).mean()), 3) == 0.085

w_out = w.copy()
w_out[5] = 24.0
print("\n6.3 Outlier 24.0, one scale :", np.round(dequantize(*quantize(w_out, 4)), 2))
print("    groups of 3              :", np.round(dequantize(*quantize(w_out, 4, group=3)), 2))

print("\n6.4 Weight memory, GB:")
for name, N in [("7B (6.74B)", 6.74e9), ("70B", 70e9)]:
    print(f"  {name:10s}: FP16 {N * 2 / 1e9:6.1f}   INT8 {N / 1e9:6.1f}   INT4 {N * 0.5 / 1e9:6.1f}")


def zero_bytes_per_param(stage: int, G: int) -> float:
    """Bytes per parameter per GPU: weights 2, gradients 2, optimizer 12, sharded by ZeRO stage."""
    return {0: 16, 1: 4 + 12 / G, 2: 2 + 14 / G, 3: 16 / G}[stage]


print("\n6.5 ZeRO, 7B model on 64 GPUs:", [f"{zero_bytes_per_param(st, 64) * 6.74e9 / 1e9:.1f} GB" for st in range(4)])

E = (1 - 0.8**5) / 0.2
print(f"6.6 Speculative decoding k=4, alpha=0.8: E = {E:.2f}, speed-up {E / 1.2:.1f}x  (notes: 3.36, 2.8x)")


# %% 3. A general implementation: quantize a whole weight matrix and use it
def quantized_matmul(X: np.ndarray, W: np.ndarray, bits: int, group: int) -> np.ndarray:
    """X @ W where W is stored as `bits`-bit codes with one scale per `group` consecutive weights."""
    q, s = quantize(W.reshape(-1), bits, group)
    return X @ dequantize(q, s).reshape(W.shape)


Wm = rng.normal(0, 0.02, (256, 256))
Xm = rng.normal(size=(8, 256))
ref = Xm @ Wm
for bits in (8, 4):
    out = quantized_matmul(Xm, Wm, bits, group=64)
    print(f"\nINT{bits} (groups of 64) matmul: relative error {np.linalg.norm(out - ref) / np.linalg.norm(ref):.4f}")


# %% 4. Experiments
def experiment_lost_updates() -> None:
    """Without an FP32 master copy, many small updates are rounded away in BF16."""
    print("\n[Experiment 1] Add 1e-4 to a weight of 1.0, 1,000 times")
    w32, w16 = np.float32(1.0), to_bf16(1.0)
    for _ in range(1000):
        w32 = np.float32(w32 + 1e-4)
        w16 = to_bf16(w16 + np.float32(1e-4))
    print(f"  FP32 master copy: {float(w32):.4f}   BF16 only: {float(w16):.4f}   (exact: 1.1000)")


def experiment_group_size() -> None:
    """INT4 error on weights with 1% outliers, for different group sizes."""
    print("\n[Experiment 2] INT4 mean |error| vs group size (1% outliers x20)")
    wb = rng.normal(0, 0.02, 10_240)
    wb[rng.choice(wb.size, 100, replace=False)] *= 20
    for group in (None, 1024, 128, 32, 8):
        q, s = quantize(wb, 4, group)
        extra = 0 if group is None else 16 / group           # bits per weight spent on 16-bit scales
        print(f"  group {str(group):>5}: error {np.abs(dequantize(q, s) - wb).mean():.5f}"
              f"   storage {4 + extra:.2f} bits/weight")


def experiment_speculative() -> None:
    """Simulate rounds of speculative decoding and compare with the formula."""
    print("\n[Experiment 3] Speculative decoding: tokens per large-model pass")
    for k, alpha in [(4, 0.8), (4, 0.5), (8, 0.9)]:
        rounds = []
        for _ in range(50_000):
            acc = 0
            while acc < k and rng.random() < alpha:
                acc += 1
            rounds.append(acc + 1)
        formula = (1 - alpha ** (k + 1)) / (1 - alpha)
        print(f"  k={k}, alpha={alpha}: simulated {np.mean(rounds):.3f}   formula {formula:.3f}")


experiment_lost_updates()
experiment_group_size()
experiment_speculative()


# %% 5. Student TODOs
def distill_targets(teacher_logits: np.ndarray, T: float) -> np.ndarray:
    """Soft targets for knowledge distillation: softmax(teacher_logits / T)."""
    # TODO(student): stable softmax of the scaled logits.
    raise NotImplementedError


def checkpoint_compute_factor() -> float:
    """Relative compute of a step with gradient checkpointing: (1 forward + 2 backward + 1 recompute) / 3."""
    # TODO(student): return the ratio.
    raise NotImplementedError


def prune_heads(n_heads: int, importance: np.ndarray, keep: int) -> np.ndarray:
    """Structured pruning: return the sorted indices of the `keep` most important heads."""
    # TODO(student): np.argsort, take the top `keep`, then sort.
    raise NotImplementedError


checks = [
    ("distill_targets", lambda: np.allclose(distill_targets(np.array([2.0, 1.0, 0.0]), 2.0),
                                            np.exp([1, 0.5, 0]) / np.exp([1, 0.5, 0]).sum())),
    ("checkpoint_compute_factor", lambda: np.isclose(checkpoint_compute_factor(), 4 / 3)),
    ("prune_heads", lambda: list(prune_heads(6, np.array([0.1, 0.9, 0.3, 0.8, 0.05, 0.5]), 3)) == [1, 3, 5]),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
