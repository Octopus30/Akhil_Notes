# Section 5 — Exercises: Scaled Dot-Product Attention

**Total: 49 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 55 minutes**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right) V$$

Round to 2 decimal places unless told otherwise.

---

## A. Warm-up (recall)

**A1.** List the five steps of scaled dot-product attention in order. *(2 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. Subtracting the row maximum before softmax changes the attention weights slightly, but avoids overflow.
2. A masked position should get the score 0.
3. If $q$ and $k$ have independent entries with mean 0 and variance 1, then $\text{Var}(q \cdot k) = d_k$.

**A3.** Fill in the shapes. $n = 10$, $d_k = 32$, $d_v = 48$. *(3 marks)*

1. $QK^{\top}$ is `__ × __`.
2. The mask $M$ is `__ × __`.
3. The output is `__ × __`.

---

## B. Multiple choice (1 mark each)

**B1.** Why divide by $\sqrt{d_k}$ rather than by $d_k$?
- (a) $\sqrt{d_k}$ is cheaper to compute.
- (b) The standard deviation of $q \cdot k$ grows like $\sqrt{d_k}$, so dividing by it keeps the scores at size about 1; dividing by $d_k$ would over-shrink them.
- (c) Dividing by $d_k$ would make the scores negative.
- (d) The original paper chose it arbitrarily.

**B2.** What happens to learning if the scores are very large (e.g. $[30, 0, 0]$)?
- (a) Learning speeds up, because the model is confident.
- (b) Softmax saturates: the weights are almost one-hot and the gradients are almost zero.
- (c) The scores overflow and training stops.
- (d) Nothing; softmax is scale-invariant.

**B3.** In a causal mask for $n = 4$, how many entries are allowed (not $-\infty$)?
- (a) 4
- (b) 6
- (c) 10
- (d) 16

**B4.** Token 4 of a batch is padding. Which entries of $M$ are set to $-\infty$?
- (a) Row 4
- (b) Column 4
- (c) Only the diagonal entry $(4, 4)$
- (d) Row 4 and column 4

**B5.** Why not zero out the masked weights **after** the softmax instead?
- (a) It is too slow.
- (b) The remaining weights in the row would no longer sum to 1.
- (c) Softmax outputs can't be changed.
- (d) It would make the matrix upper-triangular.

---

## C. Calculation

Continue with the Section 4 exercise sentence *"a light bag"* ($d_k = 2$):

| Token | $q$ | $k$ | $v$ |
|---|---|---|---|
| a | [0.1, 0.2] | [0.1, 0.1] | [0.1, 0.2] |
| light | [1, 2] | [1, 1] | [1, 2] |
| bag | [0, 3] | [1.5, 0] | [1.5, 1.5] |

**C1. Scaling.** *(5 marks)*

1. Compute the scaled score matrix $S/\sqrt{2}$.
2. Compute the attention weights for the "bag" row and compare them with the unscaled weights $[0.06, 0.90, 0.04]$ from Section 4. Did the distribution get sharper or softer?
3. Compute $y_{\text{bag}}$.

**C2. Causal mask.** Apply the causal mask to the scaled scores. Give the full $A_{\text{causal}}$ and the outputs $y_{\text{a}}$ and $y_{\text{light}}$. *(4 marks)*

**C3. Padding mask.** Suppose "bag" is actually a padding token. Write $M_{\text{pad}}$ and compute the new weights for the "a" row and the "light" row. *(4 marks)*

**C4. Saturation.** For $d_k = 16$, an unscaled two-way score is $[4, 0]$. *(3 marks)*

1. What is the scaled score?
2. Compute the softmax weight $p$ and the gradient $p(1-p)$ for both the unscaled and the scaled version.
3. By what factor is the gradient bigger after scaling?

**C5. Stable softmax and the wrong mask.** *(3 marks)*

1. Compute softmax$([1000, 1001, 1002])$ using the max-subtraction trick.
2. Scores are $[1, 2, 3]$ and we want to block position 3. Compare the weights you get by setting score 3 to 0 versus to $-\infty$.

---

## D. Conceptual

**D1.** Explain, using variance, what goes wrong if we use $d_k = 512$ and **forget** the scaling. Estimate the typical size of a score. *(3 marks)*

**D2.** How does the causal mask let a decoder learn to predict every next token of a sentence in **one** forward pass, without cheating? *(3 marks)*

**D3.** For $n = 4{,}096$ tokens with 2-byte numbers, how much memory does one head's attention matrix $A$ need? For 32 heads in each of 32 layers? Why is this a problem, and what does FlashAttention change? *(3 marks)*

---

## E. Applied / coding (NumPy)

**E1.** Implement *(5 marks)*

```python
def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
              mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention. mask: n x n booleans, True = allowed."""
```

Then write `causal_mask(n)` and `padding_mask(lengths, n)`. The second returns, for each sequence in a batch, an `n × n` boolean mask that blocks the padded columns. Check your answers to C1–C3.

**E2. Simulate the variance.** For $d_k \in \{8, 128, 1024\}$, draw 5,000 random $q, k$ pairs with `rng.standard_normal`, and print the variance of $q \cdot k$ before and after dividing by $\sqrt{d_k}$. *(3 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

1. Show that softmax is unchanged by adding the same constant $c$ to every score in a row.
2. A row of scaled scores is $[z, 0, 0, \ldots, 0]$ with $n$ entries. Find the weight $p$ on the first token as a function of $z$ and $n$.
3. For $n = 100$, how large must $z$ be for the first token to get half of the attention?
4. Use this to explain why long contexts make attention more "diffuse" for the same score sizes.
