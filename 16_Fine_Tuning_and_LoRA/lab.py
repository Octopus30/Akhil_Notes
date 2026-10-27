"""Section 16 lab: Fine-Tuning & LoRA.

LoRA from scratch: h = W0 x + (alpha / r) B A x with B = 0 at the start.
Count parameters and memory, merge adapters, and actually fine-tune a frozen
layer to a new task with a low-rank update.

Run:  uv run python 16_Fine_Tuning_and_LoRA/lab.py
"""

# %% 1. Setup
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=3, suppress=True)


def lora_params(d: int, k: int, r: int) -> int:
    """Trainable parameters of one LoRA adapter: r (d + k)."""
    return r * (d + k)


# %% 2. Reproduce the worked examples from notes.md
print("6.1 LoRA parameter counts for a 4096 x 4096 matrix")
for r in (8, 16):
    n = lora_params(4096, 4096, r)
    print(f"  r = {r:2d}: {n:,} ({n / 4096**2:.2%} of the matrix)")
model_total = 32 * 4 * lora_params(4096, 4096, 16)
print(f"  LLaMA-2 7B, r = 16 on Q,K,V,O: {model_total:,} ({model_total / 6.74e9:.2%}), file {model_total * 2 / 1e6:.0f} MB")
assert model_total == 16_777_216

W0 = np.eye(3)
A = np.array([[1.0, 0.0, 1.0]])            # r x k
B = np.array([[0.5], [0.0], [-0.5]])       # d x r
x = np.array([1.0, 2.0, 3.0])
h = W0 @ x + B @ (A @ x)
W = W0 + B @ A
print("\n6.2 LoRA forward pass: h =", h, " merged W @ x =", W @ x, " (notes: [3, 2, 1])")
print("  Delta W = B A =\n", B @ A, "\n  rank:", np.linalg.matrix_rank(B @ A))
assert np.allclose(h, [3, 2, 1]) and np.allclose(W @ x, h)

N7 = 6.74e9
print("\n6.3 Memory for a 7B model (before activations)")
print(f"  full fine-tuning : {N7 * 16 / 1e9:6.1f} GB")
print(f"  LoRA (16-bit)    : {N7 * 2 / 1e9 + model_total * 16 / 1e9:6.1f} GB")
print(f"  QLoRA (4-bit)    : {N7 * 0.5 / 1e9 + model_total * 16 / 1e9:6.1f} GB")

s = np.array([8, 4, 2, 1, 0.5])
print("\n6.4 Rank vs error for strengths 8, 4, 2, 1, 0.5:",
      [f"{np.sqrt((s[r:]**2).sum() / (s**2).sum()):.0%}" for r in range(6)])
adapter = 2 * 768 * 64 + 64 + 768
print(f"6.5 Adapters in BERT-base: {24 * adapter:,};  prompt tuning 20 x 4096 = {20 * 4096:,}")


# %% 3. A general implementation: a LoRA linear layer you can train
class LoRALinear:
    """y = X W0^T + (alpha / r) (X A^T) B^T for row inputs X (n x k); W0 frozen, A and B trainable."""

    def __init__(self, W0: np.ndarray, r: int, alpha: float, rng):
        d, k = W0.shape
        self.W0, self.scale = W0, alpha / r
        self.A = rng.normal(0, 1 / np.sqrt(k), (r, k))
        self.B = np.zeros((d, r))

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W0.T + self.scale * (X @ self.A.T) @ self.B.T

    def grads(self, X: np.ndarray, dY: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gradients of the loss w.r.t. A and B, given dY = dLoss/dY (n x d)."""
        XA = X @ self.A.T                                   # n x r
        gB = self.scale * dY.T @ XA                         # d x r
        gA = self.scale * (dY @ self.B).T @ X               # r x k
        return gA, gB

    def merge(self) -> np.ndarray:
        return self.W0 + self.scale * self.B @ self.A


def finetune(layer: LoRALinear, X: np.ndarray, Y: np.ndarray, steps: int = 500, lr: float = 0.05) -> list[float]:
    """Plain gradient descent on the mean squared error, updating only A and B."""
    losses = []
    for _ in range(steps):
        err = layer.forward(X) - Y
        losses.append(float((err**2).mean()))
        gA, gB = layer.grads(X, 2 * err / err.size)
        layer.A -= lr * gA
        layer.B -= lr * gB
    return losses


# A "pre-trained" layer W0 and a new task whose ideal weights differ by a rank-2 change.
d = k = 16
W0r = rng.normal(0, 1 / np.sqrt(k), (d, k))
delta = rng.normal(size=(d, 2)) @ rng.normal(size=(2, k)) / k
Xtr = rng.normal(size=(200, k))
Ytr = Xtr @ (W0r + delta).T
layer = LoRALinear(W0r, r=2, alpha=2, rng=rng)
assert np.allclose(layer.forward(Xtr), Xtr @ W0r.T)        # unchanged at the start
losses = finetune(layer, Xtr, Ytr, steps=1500, lr=0.5)
print(f"\nFine-tune a frozen 16x16 layer with LoRA r = 2 ({lora_params(16, 16, 2)} trainable of {16 * 16}):")
print(f"  loss {losses[0]:.4f} -> {losses[-1]:.6f};  merged == unmerged: "
      f"{np.allclose(Xtr @ layer.merge().T, layer.forward(Xtr))};  W0 untouched: {np.allclose(layer.W0, W0r)}")


# %% 4. Experiments
def experiment_rank_sweep() -> None:
    """The task needs a rank-2 change: r = 1 can't fit it, r >= 2 can."""
    print("\n[Experiment 1] Final training loss vs LoRA rank (true change has rank 2)")
    for r in [1, 2, 4, 8]:
        lay = LoRALinear(W0r, r=r, alpha=r, rng=np.random.default_rng(1))
        print(f"  r = {r}: {lora_params(16, 16, r):3d} params, final loss {finetune(lay, Xtr, Ytr, 1500, 0.5)[-1]:.6f}")


def experiment_zero_init() -> None:
    """If both A and B start at 0, nothing ever moves."""
    print("\n[Experiment 2] Initialisation matters")
    lay = LoRALinear(W0r, r=2, alpha=2, rng=rng)
    lay.A[:] = 0
    print(f"  A = B = 0 at start: loss after 300 steps {finetune(lay, Xtr, Ytr, 300, 0.5)[-1]:.4f} (stuck)")
    lay = LoRALinear(W0r, r=2, alpha=2, rng=rng)
    print(f"  A random, B = 0  : loss after 300 steps {finetune(lay, Xtr, Ytr, 300, 0.5)[-1]:.6f}")


def experiment_adapter_swap() -> None:
    """One frozen base, several small adapters, one per task."""
    print("\n[Experiment 3] Swapping adapters on one base layer")
    tasks = {name: rng.normal(size=(d, 1)) @ rng.normal(size=(1, k)) / k for name in ("task A", "task B")}
    adapters = {}
    for name, dlt in tasks.items():
        lay = LoRALinear(W0r, r=1, alpha=1, rng=rng)
        finetune(lay, Xtr, Xtr @ (W0r + dlt).T, 1500, 0.5)
        adapters[name] = (lay.A.copy(), lay.B.copy())
    for name, (Aa, Bb) in adapters.items():
        for target, dlt in tasks.items():
            pred = Xtr @ (W0r + Bb @ Aa).T
            mse = ((pred - Xtr @ (W0r + dlt).T) ** 2).mean()
            print(f"  adapter {name} on {target}: mse {mse:.5f}")


experiment_rank_sweep()
experiment_zero_init()
experiment_adapter_swap()


# %% 5. Student TODOs
def model_lora_params(layers: int, d_model: int, r: int, matrices: int) -> int:
    """LoRA parameters for `matrices` square d_model x d_model projections in every layer."""
    # TODO(student): layers * matrices * r * (d + d).
    raise NotImplementedError


def best_rank_r(M: np.ndarray, r: int) -> np.ndarray:
    """Best rank-r approximation of M using the SVD (keep the r largest singular values)."""
    # TODO(student): U, S, Vt = np.linalg.svd(M); keep the first r of each.
    raise NotImplementedError


def qlora_memory_gb(n_params: float, n_lora: float) -> float:
    """Frozen 4-bit base (0.5 bytes/param) + LoRA trained with 16 bytes/param, in GB."""
    # TODO(student): one line.
    raise NotImplementedError


Mt = rng.normal(size=(8, 8))
checks = [
    ("model_lora_params", lambda: model_lora_params(32, 4096, 16, 4) == 16_777_216),
    ("best_rank_r", lambda: np.linalg.matrix_rank(best_rank_r(Mt, 3)) == 3
        and np.isclose(np.linalg.norm(Mt - best_rank_r(Mt, 3)), np.sqrt((np.linalg.svd(Mt)[1][3:]**2).sum()))),
    ("qlora_memory_gb", lambda: np.isclose(qlora_memory_gb(6.74e9, 16_777_216), 3.64, atol=0.01)),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
