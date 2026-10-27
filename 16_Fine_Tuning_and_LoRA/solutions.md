# Section 16 — Solutions: Fine-Tuning & LoRA

---

## A. Warm-up

**A1.**
- **Prompting** is cheapest, but the documents can't all fit in the prompt. *(1 mark)*
- **RAG** retrieves the relevant passages at question time. It stays current as documents change, and the answers can cite their sources. *(1 mark)*
- **LoRA fine-tuning** teaches style, format or behaviour well, but it is a poor way to store large, changing facts: you would need to retrain whenever the documents change, and it can hallucinate. *(1 mark)*

**Try RAG first**, possibly plus LoRA for tone and format.

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **True.** With $B = 0$, $BA = 0$, so $h = W_0x$.
2. **False.** Merging gives one matrix $W = W_0 + \frac{\alpha}{r}BA$ with the original shape, so there is no extra cost.
3. **False.** QLoRA stores the *frozen base* in 4-bit (NF4); the LoRA matrices are trained in 16-bit.

**A3.** *(1 mark each)*
1. $B$: `d × r`, $A$: `r × k`
2. 4
3. $10^{-4}$ (to $2 \times 10^{-4}$)

---

## B. Multiple choice

**B1. (b).**
- (a) Wrong: if *both* were 0, or if $A = 0$ and $B$ were random, no gradient would reach $B$ at the start (see F2).
- (c) Wrong: the choice has nothing to do with memory.
- (d) Wrong: size isn't the reason either.

**B2. (b).**
- (a) Wrong: pre-trained weights are typically full-rank.
- (c) Wrong: rank does limit what an update can express.
- (d) Wrong: GPUs multiply matrices of any rank.

**B3. (b).**
- (a) and (d) can be merged.
- (c) Full fine-tuning simply replaces the weights, adding nothing at inference.

**B4. (b).**
- (a) Exposure bias is a train/generate mismatch.
- (c) Softmax saturation is about large attention scores.
- (d) Quantization error is about precision.

**B5. (b).**
- (a) Wrong: too few GPUs is a capacity problem, not a silent quality failure.
- (c) Wrong: AdamW is the standard optimizer.
- (d) Wrong: one epoch is often fine.

---

## C. Calculation

**C1.**
1. $r = 8$: $8 \times (5120 + 5120) = 81{,}920$. $r = 32$: $327{,}680$. The $r = 8$ share is $81{,}920/26{,}214{,}400 = 0.31\%$. *(2 marks)*
2. $40 \times 2 \times 8 \times 10{,}240 = 6{,}553{,}600$ parameters. At 2 bytes each the file is $\approx 13.1$ MB. *(2 marks)*
3. $16 \times (4096 + 11{,}008) = 241{,}664$, against $45{,}088{,}768$ for the full matrix, a share of $0.54\%$. *(1 mark)*

**C2.**
1. *(3 marks)*
   - $W_0x = [2, -2, 1]^{\top}$
   - $Ax = 2 - 1 + 0 = 1$
   - $\frac{\alpha}{r}B(Ax) = 2 \times [1, 2, 0]^{\top} = [2, 4, 0]^{\top}$
   - $h = [4, 2, 1]^{\top}$
2. *(1.5 marks)*

$$2BA = \begin{bmatrix} 2 & 2 & 0 \\ 4 & 4 & 0 \\ 0 & 0 & 0 \end{bmatrix}, \qquad W = \begin{bmatrix} 3 & 2 & 0 \\ 4 & 6 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

   $Wx = [6 - 2,\; 8 - 6,\; 1] = [4, 2, 1]^{\top}$, which matches.
3. At initialisation $B = 0$, so $h = W_0x = [2, -2, 1]^{\top}$. *(0.5 marks)*

**C3.** *(1 mark each)*
- Full fine-tuning: $13\text{B} \times 16 = 208$ GB.
- LoRA: base $13\text{B} \times 2 = 26$ GB plus $6.55\text{M} \times 16 = 0.10$ GB, about **26.1 GB**.
- QLoRA: base $13\text{B} \times 0.5 = 6.5$ GB plus $0.10$ GB, about **6.6 GB**.

**C4.** The sum of squared strengths is $36 + 9 + 4 + 1 = 50$. *(3 marks)*

| $r$ | Parameters | Error |
|---|---|---|
| 0 | 0 | 100% |
| 1 | 16 | $\sqrt{14/50} = 53\%$ |
| 2 | 32 | $\sqrt{5/50} = 32\%$ |
| 3 | 48 | $\sqrt{1/50} = 14\%$ |
| 4 | 64 | 0% |

**C5.** *(1 mark each)*
1. Each adapter has $2 \times 1024 \times 32 + 32 + 1024 = 66{,}592$ parameters, so $48 \times 66{,}592 = 3{,}196{,}416$.
2. $50 \times 5120 = 256{,}000$.

---

## D. Conceptual

**D1.**
- **Full fine-tuning:** 20 copies of about 13.5 GB is 270 GB of storage. Serving them means loading each copy separately, or swapping 13.5 GB models in and out of the GPU. *(1 mark)*
- **LoRA:** one base model plus 20 adapters of a few tens of MB each, under 1 GB in total. *(1 mark)*
- One GPU can keep the base model loaded and apply each customer's adapter per request, even mixing customers in one batch. *(1 mark)*

**D2.** Only a tiny set of parameters moves, and $B$ starts at zero, so with a small learning rate the update would grow very slowly. *(1 mark)* The frozen base also protects the model from damage, so larger steps on the few trainable parameters are safe. Full fine-tuning moves billions of weights at once, where large steps would wreck what the model has learned. *(1 mark)*

**D3.** *(1 mark each, any two)*
- Mix in general-purpose data.
- Train for few epochs, with early stopping on a general-skills evaluation set.
- Use a lower learning rate, or a parameter-efficient method such as LoRA with a small rank, since the frozen base keeps the original knowledge.
- Merge the result with the original weights (interpolation).

---

## E. Applied / coding

**E1.** *(5 marks: 2 forward, 1 merge, 2 checks)*

```python
import numpy as np

class LoRALinear:
    """Frozen W0 (d x k) plus a trainable low-rank update (alpha / r) * B @ A."""

    def __init__(self, W0: np.ndarray, r: int, alpha: float, rng):
        self.W0 = W0
        d, k = W0.shape
        self.A = rng.normal(0, 0.01, (r, k))
        self.B = np.zeros((d, r))
        self.scale = alpha / r

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: n x k (one input per row) -> n x d."""
        return X @ self.W0.T + self.scale * (X @ self.A.T) @ self.B.T

    def merge(self) -> np.ndarray:
        """Single d x k matrix with the update folded in."""
        return self.W0 + self.scale * self.B @ self.A

rng = np.random.default_rng(0)
layer = LoRALinear(np.diag([1.0, 2.0, 1.0]), r=1, alpha=2, rng=rng)
x = np.array([[2.0, -1.0, 1.0]])
assert np.allclose(layer.forward(x), x @ layer.W0.T)            # unchanged at initialisation
layer.A, layer.B = np.array([[1.0, 1.0, 0.0]]), np.array([[1.0], [2.0], [0.0]])
print(layer.forward(x))                                         # [[4. 2. 1.]]
print(layer.merge())
assert np.allclose(x @ layer.merge().T, layer.forward(x))
```

**E2.** *(3 marks)*

```python
def fit_lora(W_target, r, steps=3000, lr=0.05, seed=0):
    """Fit B @ A to W_target by gradient descent on the mean squared error (W0 = 0, scale 1)."""
    g = np.random.default_rng(seed)
    d, k = W_target.shape
    A, B = g.normal(0, 0.1, (r, k)), np.zeros((d, r))
    err0 = np.linalg.norm(W_target)
    for _ in range(steps):
        R = B @ A - W_target                     # residual
        gB, gA = R @ A.T, B.T @ R                # gradients of 0.5 * ||R||^2
        B -= lr * gB
        A -= lr * gA
    return err0, np.linalg.norm(B @ A - W_target)

U = rng.normal(size=(6, 2))
V = rng.normal(size=(2, 6))
W_star = U @ V / 3                               # exactly rank 2
for r in (2, 1):
    before, after = fit_lora(W_star, r)
    print(f"r = {r}: error {before:.3f} -> {after:.4f}")
```

With $r = 2$ the error falls to almost 0. With $r = 1$ it stalls at a positive value: a rank-1 update can't represent a rank-2 target. The best it can do is keep the larger singular direction, so the remaining error equals the smaller singular value.

---

## F. ★ Challenge

1. Write $\Delta W = \sum_i \sigma_i u_i v_i^{\top}$ with orthonormal $u_i$, $v_i$ and $\sigma_1 \ge \sigma_2 \ge \cdots$. In the Frobenius norm $\lVert\Delta W\rVert^2 = \sum_i \sigma_i^2$, because the pieces are orthogonal. Keeping any $r$ of the pieces leaves an error of $\sum_{\text{dropped}} \sigma_i^2$, which is smallest when we drop the smallest ones. (This is the Eckart–Young theorem, which also shows no other rank-$r$ matrix does better.) The relative error is then $\sqrt{\sum_{i>r}\sigma_i^2/\sum_i\sigma_i^2}$, the C4 formula. *(2 marks)*
2. $h = W_0x + sBAx$. The gradients are $\partial L/\partial B = s\,(\partial L/\partial h)(Ax)^{\top}$ and $\partial L/\partial A = s\,B^{\top}(\partial L/\partial h)x^{\top}$. If $A = B = 0$, then $Ax = 0$, so the gradient for $B$ is 0, and $B^{\top} = 0$, so the gradient for $A$ is 0. Both stay at 0 forever. With $A$ random and $B = 0$, the gradient for $B$ is non-zero, so learning starts. *(1 mark)*
3. With $A$ random, the size of $BAx$ at a given point in training grows with $r$, since it sums $r$ terms. Dividing by $r$ (times a fixed $\alpha$) keeps the update's scale roughly the same for every $r$, so the same learning rate keeps working when you try $r = 8$ or $r = 64$, without retuning. *(1 mark)*
