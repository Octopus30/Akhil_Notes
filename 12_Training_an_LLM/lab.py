"""Section 12 lab: Training an LLM.

Cross-entropy and perplexity, AdamW written by hand, the warm-up + cosine
schedule, and a real (tiny) training run: a bigram language model learned
with AdamW on a small corpus. Plus compute/memory estimates, SFT loss masking
and the DPO loss.

Run:  uv run python 12_Training_an_LLM/lab.py
"""

# %% 1. Setup
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=3, suppress=True)


def adamw_step(theta, g, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
    """One AdamW update (works on scalars or arrays); returns (theta, m, v)."""
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat, v_hat = m / (1 - b1**t), v / (1 - b2**t)
    return theta - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * theta), m, v


def lr_schedule(step: int, peak: float, warmup: int, total: int, min_lr: float) -> float:
    """Linear warm-up to peak, then cosine decay to min_lr."""
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (peak - min_lr) * (1 + np.cos(np.pi * progress))


# %% 2. Reproduce the worked examples from notes.md
p = np.array([0.5, 0.25, 0.8, 0.1])
loss = -np.log(p).mean()
print(f"6.1 loss {loss:.3f}, perplexity {np.exp(loss):.2f}   (notes: 1.151, 3.16)")
print(f"    uniform over 50,257 tokens: loss {np.log(50_257):.2f}   (notes: 10.82)")

theta, _, _ = adamw_step(1.0, 0.5, 0.0, 0.0, 1, lr=0.1, wd=0.01)
print(f"6.2 one AdamW step: theta 1.0 -> {theta:.3f}   (notes: 0.899)")
assert round(theta, 3) == 0.899

print("6.3 schedule:", [f"{lr_schedule(s, 3e-4, 2000, 100_000, 3e-5):.3g}" for s in (1000, 2000, 51_000, 100_000)],
      "  (notes: 1.5e-4, 3e-4, 1.65e-4, 3e-5)")

N, D = 6.74e9, 2e12
C = 6 * N * D
print(f"6.4 LLaMA-2 7B: C = {C:.2g} FLOPs, {C / 1.25e14 / 3600:,.0f} GPU-hours at 1.25e14 FLOP/s"
      f" (reported 184,320); memory 16N = {16 * N / 1e9:.0f} GB; Chinchilla {20 * N / 1e9:.0f}B tokens")

sig = lambda z: 1 / (1 + np.exp(-z))
print(f"6.6 DPO loss: start {-np.log(sig(0)):.3f}, after (+2, -1) with beta 0.1: {-np.log(sig(0.1 * 3)):.3f}"
      "   (notes: 0.693, 0.554)")


# %% 3. A general implementation: train a bigram language model with AdamW
CORPUS = """the rain falls on the hills
the rain stops in the evening
the sun rises over the hills
the sun sets in the evening
children play in the rain
children sleep in the evening""".split("\n")

words = sorted({w for line in CORPUS for w in line.split()} | {"<bos>", "<eos>"})
idx = {w: i for i, w in enumerate(words)}
V = len(words)
pairs = np.array([(idx[a], idx[b]) for line in CORPUS
                  for a, b in zip(["<bos>"] + line.split(), line.split() + ["<eos>"])])
prev, nxt = pairs[:, 0], pairs[:, 1]


def bigram_loss_and_grad(W: np.ndarray) -> tuple[float, np.ndarray]:
    """W is V x V logits (row = previous token). Mean cross-entropy over all pairs and its gradient."""
    logits = W[prev]                                          # T x V
    P = np.exp(logits - logits.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    loss = -np.log(P[np.arange(len(nxt)), nxt]).mean()
    dlogits = P.copy()
    dlogits[np.arange(len(nxt)), nxt] -= 1                    # softmax + cross-entropy gradient
    dW = np.zeros_like(W)
    np.add.at(dW, prev, dlogits / len(nxt))
    return float(loss), dW


def train(steps: int, schedule, seed: int = 0) -> list[float]:
    """Full-batch AdamW training of the bigram model; returns the loss curve."""
    W = np.random.default_rng(seed).normal(0, 0.01, (V, V))
    m, v, losses = np.zeros_like(W), np.zeros_like(W), []
    for t in range(1, steps + 1):
        loss, g = bigram_loss_and_grad(W)
        W, m, v = adamw_step(W, g, m, v, t, lr=schedule(t), wd=0.0)
        losses.append(loss)
    return losses


def optimal_loss() -> float:
    """Best possible bigram loss: use the empirical next-token frequencies."""
    counts = np.zeros((V, V))
    np.add.at(counts, (prev, nxt), 1)
    totals = counts.sum(1, keepdims=True)
    P = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    return float(-np.log(P[prev, nxt]).mean())


losses = train(300, lambda t: lr_schedule(t, 0.1, 20, 300, 0.01))
print(f"\nBigram LM on {len(pairs)} next-token pairs, vocabulary {V}")
for s in (1, 10, 50, 100, 300):
    print(f"  step {s:3d}: loss {losses[s - 1]:.3f}  perplexity {np.exp(losses[s - 1]):5.2f}")
print(f"  start = ln {V} = {np.log(V):.3f};  best possible bigram loss = {optimal_loss():.3f}")


# %% 4. Experiments
def experiment_schedules() -> None:
    """How the learning rate's size affects progress, and an honest note about warm-up."""
    print("\n[Experiment 1] Loss after 30 and 150 steps for different learning rates")
    for name, sched in [("constant 0.01", lambda t: 0.01), ("constant 0.05", lambda t: 0.05),
                        ("constant 0.3", lambda t: 0.3),
                        ("warm-up + cosine (peak 0.3)", lambda t: lr_schedule(t, 0.3, 15, 150, 0.03))]:
        L = train(150, sched)
        print(f"  {name:28s}: step 30 {L[29]:.3f}   step 150 {L[-1]:.3f}")
    print("  This model is one softmax table, so its loss is convex and warm-up isn't needed here.")
    print("  Warm-up matters in deep networks, where random weights and fresh Adam statistics")
    print("  make large early steps unstable.")


def experiment_confident_mistake() -> None:
    """One confident mistake costs far more loss than several mild ones."""
    print("\n[Experiment 2] Loss contribution -ln p")
    for pr in [0.9, 0.5, 0.1, 0.01, 0.001]:
        print(f"  p = {pr:<6}: {-np.log(pr):.2f}")


def experiment_sft_masking() -> None:
    """SFT: the same per-token losses, averaged with and without masking the prompt."""
    print("\n[Experiment 3] SFT loss masking (5 prompt tokens, 7 response tokens)")
    per_token = rng.uniform(0.2, 3.0, size=12)
    is_response = np.array([False] * 5 + [True] * 7)
    print(f"  loss over all 12 positions: {per_token.mean():.3f}")
    print(f"  loss over the 7 response positions only (what SFT uses): {per_token[is_response].mean():.3f}")


experiment_schedules()
experiment_confident_mistake()
experiment_sft_masking()


# %% 5. Student TODOs
def perplexity(probs_of_correct: np.ndarray) -> float:
    """exp of the mean negative log-probability."""
    # TODO(student): one line.
    raise NotImplementedError


def training_days(N: float, D: float, flops_per_gpu: float, n_gpus: int) -> float:
    """Days of training from C = 6ND."""
    # TODO(student): convert seconds to days.
    raise NotImplementedError


def dpo_loss(delta_w: float, delta_l: float, beta: float) -> float:
    """-log sigmoid(beta * (delta_w - delta_l)), where deltas are log-ratio changes vs the reference."""
    # TODO(student): use sig() defined above.
    raise NotImplementedError


checks = [
    ("perplexity", lambda: np.isclose(perplexity(np.array([0.5, 0.25, 0.8, 0.1])), 3.16, atol=0.01)),
    ("training_days", lambda: np.isclose(training_days(13e9, 1e12, 4e14, 1024), 2.204, atol=0.01)),
    ("dpo_loss", lambda: np.isclose(dpo_loss(1.5, -0.5, 0.2), 0.513, atol=1e-3)),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
