# Section 5 — Solutions: Scaled Dot-Product Attention

---

## A. Warm-up

**A1.** The steps are (1) scores $S = QK^{\top}$, (2) scale by $1/\sqrt{d_k}$, (3) add the mask $M$, (4) softmax each row, (5) multiply by $V$. *(2 marks: 1 for all five steps, 1 for the right order)*

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **False.** Softmax is unchanged when the same constant is subtracted from every score in a row, so the weights are *exactly* the same. Only overflow is avoided.
2. **False.** A score of 0 still gets weight $e^0 = 1$. Masked positions need $-\infty$, so that their weight is exactly 0.
3. **True.** $q \cdot k$ is a sum of $d_k$ independent terms $q_ik_i$, each with variance 1, and variances of independent terms add.

**A3.** *(1 mark each)* `10 × 10`, `10 × 10`, `10 × 48`.

---

## B. Multiple choice

**B1. (b).**
- (a) Wrong: both are cheap, and the cost is irrelevant here.
- (c) Wrong: dividing by a positive number never changes signs.
- (d) Wrong: the paper motivates it with exactly this variance argument.

**B2. (b).** Weights near $[1, 0, 0]$ have gradient $p(1-p) \approx 0$, so learning stalls.
- (a) Wrong: confidence doesn't help when the gradients vanish.
- (c) Wrong: 30 doesn't overflow, since $e^{30} \approx 10^{13}$ fits easily in a float, and stable softmax removes the risk anyway.
- (d) Wrong: softmax is invariant to *adding* a constant, not to *multiplying* by one.

**B3. (c).** $1 + 2 + 3 + 4 = 10 = n(n+1)/2$.
- (a) 4 is the diagonal only.
- (b) 6 is the strictly lower triangle, which leaves out the diagonal.
- (d) 16 means no mask at all.

**B4. (b).** Nobody may attend *to* the padding token, so its **column** is blocked.
- (a) Wrong: the padding token's own row is computed but never used. Blocking that row would leave it with no allowed entry, giving NaNs.
- (c) Wrong: the other tokens would still attend to the padding token.
- (d) Wrong: blocking row 4 too is the same mistake as (a). Its output is ignored anyway.

**B5. (b).** After setting some weights to 0, the remaining weights sum to less than 1, so you would have to renormalise. Adding $-\infty$ *before* the softmax does this automatically.
- (a) Wrong: speed isn't the issue.
- (c) Wrong: softmax outputs can be changed. Doing so just breaks the "rows sum to 1" property.
- (d) Wrong: which entries are zeroed depends on the mask, not on when it is applied.

---

## C. Calculation

**C1.**
1. Divide Section 4's $S$ by $\sqrt2 = 1.414$. *(2 marks)*

| $S/\sqrt2$ | a | light | bag |
|---|---|---|---|
| **a** | 0.02 | 0.21 | 0.11 |
| **light** | 0.21 | 2.12 | 1.06 |
| **bag** | 0.21 | 2.12 | 0.00 |

2. "bag" row: $e^{0.21} = 1.24$, $e^{2.12} = 8.34$, $e^{0} = 1$. The sum is $10.58$, so the weights are $[0.12, 0.79, 0.09]$. *(2 marks)* Compared with the unscaled $[0.06, 0.90, 0.04]$, the distribution is **softer**: light's share dropped from 0.90 to 0.79.
3. $y_{\text{bag}} = 0.12[0.1, 0.2] + 0.79[1, 2] + 0.09[1.5, 1.5] = [0.94, 1.74]$. *(1 mark)* (Unrounded: $[0.942, 1.742]$.)

**C2.** *(4 marks: 2 for the matrix, 1 for each output)*

| $A_{\text{causal}}$ | a | light | bag |
|---|---|---|---|
| **a** | 1 | 0 | 0 |
| **light** | 0.13 | 0.87 | 0 |
| **bag** | 0.12 | 0.79 | 0.09 |

- "light" row: $e^{0.21} = 1.24$ and $e^{2.12} = 8.34$, so the weights are $1.24/9.58 = 0.13$ and $0.87$.
- $y_{\text{a}} = v_{\text{a}} = [0.1, 0.2]$.
- $y_{\text{light}} = 0.13[0.1, 0.2] + 0.87[1, 2] = [0.88, 1.77]$.
- The "bag" row is unchanged, because the last token already sees everything.

**C3.** "bag" is padding, so its **column** is blocked: *(1 mark)*

$$M_{\text{pad}} = \begin{bmatrix} 0 & 0 & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & -\infty \end{bmatrix}$$

- "a" row: the scaled scores are $[0.02, 0.21]$. $e^{0.02} = 1.02$ and $e^{0.21} = 1.24$, so the weights are $[0.45, 0.55, 0]$. *(1.5 marks)*
- "light" row: the scores are $[0.21, 2.12]$, so the weights are $[0.13, 0.87, 0]$. This matches the causal row, because both masks hide only "bag" from light. *(1.5 marks)*

**C4.**
1. $\sqrt{16} = 4$, so the scaled score is $[1, 0]$. *(1 mark)*
2. *(1 mark)*
   - Unscaled: $p = e^4/(e^4 + 1) = 0.982$, so $p(1-p) = 0.0177$.
   - Scaled: $p = e/(e+1) = 0.731$, so $p(1-p) = 0.197$.
3. $0.197 / 0.0177 \approx$ **11×** bigger. *(1 mark)*

**C5.**
1. Subtract 1002 to get $[-2, -1, 0]$. Then $e^{-2} = 0.135$, $e^{-1} = 0.368$, $e^{0} = 1$, with sum $1.503$, so the weights are $[0.09, 0.24, 0.67]$. *(1 mark)*
2. *(2 marks)*
   - Setting score 3 to 0: softmax$([1, 2, 0]) = [0.24, 0.67, 0.09]$. Position 3 still gets 9%, so the mask fails.
   - Setting it to $-\infty$: softmax$([1, 2]) = [0.27, 0.73, 0]$. Position 3 is fully blocked, and the row still sums to 1.

---

## D. Conceptual

**D1.** With unit-variance entries, $\text{Var}(q \cdot k) = 512$, so a typical score is about $\sqrt{512} \approx 22.6$ in size. *(1 mark)* Differences of 20–40 between scores make softmax almost one-hot: one token gets nearly all the weight. *(1 mark)* The gradient $p(1-p)$ is then nearly zero for every weight, so the attention can barely learn. Initial training is very slow or unstable. *(1 mark)*

**D2.** With teacher forcing, the input is the whole sentence, and at position $i$ the target is token $i+1$. *(1 mark)* The causal mask stops position $i$ from attending to positions $> i$, so it can't see the token it must predict, or any later one. *(1 mark)* Every position therefore makes an honest prediction from its own past, and all $n$ predictions, with their losses, come out of the same forward pass. *(1 mark)*

**D3.** One head needs $4{,}096^2 \times 2 = 33{,}554{,}432$ bytes, about **33.6 MB**. *(1 mark)* For 32 × 32 = 1,024 heads that is about **34.4 GB**, more than many GPUs hold, and it grows 4× each time $n$ doubles. *(1 mark)* FlashAttention computes the same output tile by tile in fast on-chip memory. It never stores the full $n \times n$ matrix, so the memory grows like $n$ rather than $n^2$ (Section 14). *(1 mark)*

---

## E. Applied / coding

**E1.** *(5 marks: 2 attention with scaling, 1 masking with $-\infty$ and stable softmax, 1 causal mask, 1 padding mask)*

```python
import numpy as np

def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
              mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention. mask: n x n booleans, True = allowed."""
    d_k = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        S = np.where(mask, S, -np.inf)
    S = S - S.max(axis=-1, keepdims=True)
    A = np.exp(S)
    A = A / A.sum(axis=-1, keepdims=True)
    return A @ V, A

def causal_mask(n: int) -> np.ndarray:
    """True on and below the diagonal: token i may see j <= i."""
    return np.tril(np.ones((n, n), dtype=bool))

def padding_mask(lengths: list[int], n: int) -> np.ndarray:
    """Shape (batch, n, n): column j is allowed only if j < that sequence's real length."""
    real = np.arange(n)[None, :] < np.array(lengths)[:, None]   # (batch, n)
    return np.repeat(real[:, None, :], n, axis=1)

Q = np.array([[0.1, 0.2], [1.0, 2.0], [0.0, 3.0]])
K = np.array([[0.1, 0.1], [1.0, 1.0], [1.5, 0.0]])
V = np.array([[0.1, 0.2], [1.0, 2.0], [1.5, 1.5]])

Y, A = attention(Q, K, V)
print("C1 bag weights", np.round(A[2], 2), " y_bag", np.round(Y[2], 2))
Yc, Ac = attention(Q, K, V, causal_mask(3))
print("C2 causal A\n", np.round(Ac, 2), "\n   y_a", Yc[0], " y_light", np.round(Yc[1], 2))
Yp, Ap = attention(Q, K, V, padding_mask([2], 3)[0])
print("C3 padding A\n", np.round(Ap, 2))
```

Output:

```
C1 bag weights [0.12 0.79 0.09]  y_bag [0.94 1.74]
C2 causal A
 [[1.   0.   0.  ]
 [0.13 0.87 0.  ]
 [0.12 0.79 0.09]]
   y_a [0.1 0.2]  y_light [0.88 1.77]
C3 padding A
 [[0.45 0.55 0.  ]
 [0.13 0.87 0.  ]
 [0.13 0.87 0.  ]]
```

**E2.** *(3 marks)*

```python
rng = np.random.default_rng(0)
for d_k in [8, 128, 1024]:
    q = rng.standard_normal((5000, d_k))
    k = rng.standard_normal((5000, d_k))
    dots = (q * k).sum(axis=1)
    print(f"d_k={d_k:5d}: var {dots.var():8.1f}   scaled var {(dots / np.sqrt(d_k)).var():.2f}")
```

The variance comes out close to $d_k$ (about 8, 128 and 1,024), and close to 1 after scaling.

---

## F. ★ Challenge

1. $\frac{e^{s_j + c}}{\sum_k e^{s_k + c}} = \frac{e^c e^{s_j}}{e^c \sum_k e^{s_k}} = \frac{e^{s_j}}{\sum_k e^{s_k}}$. *(1 mark)*
2. $p = \dfrac{e^z}{e^z + (n-1)}$. *(1 mark)*
3. Set $p = \tfrac12$: $e^z = n - 1 = 99$, so $z = \ln 99 \approx$ **4.6**. *(1 mark)*
4. The competition from the $n - 1$ other tokens grows with $n$, so a score gap that dominates a short context, e.g. $z = 2$ with $n = 3$ gives $p = 0.79$, is diluted in a long one: with $n = 100$, $p = 0.07$. To stay focused, a model needs score gaps that grow like $\ln n$. Some long-context methods scale the scores by a factor that grows with $\log n$ for exactly this reason. *(1 mark)*
