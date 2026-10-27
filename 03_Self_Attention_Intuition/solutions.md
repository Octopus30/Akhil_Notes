# Section 3 — Solutions: Self-Attention, the Intuition

---

## A. Warm-up

**A1.** *(1 mark each)*
1. **Score:** $s_{ij} = x_i \cdot x_j$
2. **Normalize:** $a_{ij} = e^{s_{ij}} / \sum_k e^{s_{ik}}$
3. **Weighted average:** $y_i = \sum_j a_{ij} x_j$

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **False.** Each **row** sums to 1, because softmax is applied to each row. Columns can sum to anything.
2. **False.** The weights are computed fresh from the current input $X$ every time. Section 3's naive version has no learned parameters at all.
3. **True.** Causal attention lets token $i$ see positions $1..i$ only, so the future token 4 is hidden.

**A3.** *(1 mark each)* `7 × 7`, `7 × 7`, `7 × 16`.

---

## B. Multiple choice

**B1. (b).** $x \cdot x = \lVert x\rVert^2$. A long vector usually scores itself highest.
- (a) Wrong: softmax doesn't favour the diagonal; it only exponentiates the scores. The diagonal wins only when its score is largest.
- (c) Wrong: naive attention allows every pair.
- (d) Wrong: softmax never removes anything.

**B2. (b).** All the scores are equal, so the weights are uniform ($1/n$). The average of $n$ identical vectors is that same vector.
- (a) Wrong: an average of non-zero vectors isn't zero.
- (c) Wrong: there is no position information in naive attention.
- (d) Wrong: equal scores are fine and simply give uniform weights.

**B3. (c).**
- (a) "Global" means any token can reach any other token in one step.
- (b) "Parallel" means all outputs are computed at once.
- (d) "Quadratic cost" is about how the work grows with $n$.

**B4. (c).** $(900/300)^2 = 9$.
- (a) 3× would be linear growth.
- (b) Wrong: 6× doesn't come from any rule here.
- (d) Wrong: 27× = $3^3$ would be cubic growth.

**B5. (b).** Cross-attention lets the tokens of one sequence (the target being written) attend to those of another sequence (the source).
- (a), (c) and (d) each work within a single sequence, so they use self-attention: bidirectional in (a) and (d), causal in (c).

---

## C. Calculation

**C1.**
1. Scores for "light" = [1, 1]: *(1 mark)*
   - light · a $= 0.1 + 0.1 = 0.2$
   - light · light $= 1 + 1 = 2$
   - light · bag $= 1.5 + 0 = 1.5$
2. $e^{0.2} = 1.22$, $e^{2} = 7.39$, $e^{1.5} = 4.48$. The sum is $13.09$. *(2 marks)*

| attends to | a | light | bag |
|---|---|---|---|
| weight | 1.22 / 13.09 = **0.09** | 7.39 / 13.09 = **0.56** | 4.48 / 13.09 = **0.34** |

3. $y_{\text{light}} = 0.09[0.1, 0.1] + 0.56[1, 1] + 0.34[1.5, 0] = [1.09, 0.57]$. *(1 mark)* (Unrounded weights 0.0933, 0.5644, 0.3423 give $[1.087, 0.574]$.)
4. "light" moved toward the **weight** axis: its first entry rose from 1 to 1.09 and its second fell from 1 to 0.57. That fits *"a light bag"*, where light means *not heavy*. *(1 mark)*

**Marking:** 1 for the scores; 1 for the exponentials; 1 for weights that sum to 1; 1 for $y$; 1 for the interpretation.

**C2.** light · lamp $= 0 + 1.5 = 1.5$, so the scores $[0.2, 2, 1.5]$ and the weights $[0.09, 0.56, 0.34]$ are unchanged. *(1 mark)*

$$y_{\text{light}} = 0.09[0.1, 0.1] + 0.56[1, 1] + 0.34[0, 1.5] = [0.57, 1.09]$$

Now "light" leans toward **brightness**. *(1 mark)* The weights match because bag and lamp have the same dot product with [1, 1], but the vectors being averaged are different, so the outputs differ. *(1 mark)*

**C3.**
1. *(2 marks)*

| $S$ | a | light | bag |
|---|---|---|---|
| **a** | 0.02 | 0.20 | 0.15 |
| **light** | 0.20 | 2.00 | 1.50 |
| **bag** | 0.15 | 1.50 | 2.25 |

2. *(2 marks)*

| $A$ | a | light | bag | row sum |
|---|---|---|---|---|
| **a** | 0.30 | 0.36 | 0.34 | 1 |
| **light** | 0.09 | 0.56 | 0.34 | 1 (0.99 after rounding) |
| **bag** | 0.08 | 0.30 | **0.63** | 1 (1.01 after rounding) |

   Row "a": $e^{0.02} = 1.02$, $e^{0.20} = 1.22$, $e^{0.15} = 1.16$; sum $3.40$.
   Row "bag": $e^{0.15} = 1.16$, $e^{1.5} = 4.48$, $e^{2.25} = 9.49$; sum $15.13$.
3. **bag** attends 63% to itself. Its self-score $\lVert\text{bag}\rVert^2 = 2.25$ is the largest entry in its row. This is the self-attention problem from the notes, and Section 4 fixes it. *(1 mark)*

**C4.** With the causal mask, the scores above the diagonal become $-\infty$. *(3 marks: 1 for the matrix, 1 for both outputs, 1 for the explanation)*

| $A$ (causal) | a | light | bag |
|---|---|---|---|
| **a** | 1 | 0 | 0 |
| **light** | 0.14 | 0.86 | 0 |
| **bag** | 0.08 | 0.30 | 0.63 |

- "a" can only see itself, so $y_{\text{a}} = [0.1, 0.1]$.
- For "light": $e^{0.2} = 1.22$ and $e^{2} = 7.39$, so the weights are $1.22/8.61 = 0.14$ and $0.86$, and $y_{\text{light}} = 0.14[0.1, 0.1] + 0.86[1, 1] = [0.87, 0.87]$.

The **bag** row is unchanged: it is the last token, so it could already see everything before it and nothing is hidden from it.

**C5.** $2{,}000^2 = 4{,}000{,}000$ scores. *(1 mark)* At 4 bytes each that is $16{,}000{,}000$ bytes = **16 MB**. *(1 mark)* For 32 × 32 = 1,024 heads, it is $16 \times 1{,}024 = 16{,}384$ MB ≈ **16.4 GB**. *(1 mark)* (Section 14 shows how FlashAttention avoids storing these matrices.)

---

## D. Conceptual

**D1.** All the scores are dot products, so they grow by $10^2 = 100$. For "light" the scores become $[20, 200, 150]$. The softmax becomes almost **one-hot**: light's weights go to about $[0, 1, 0]$, all on itself, because $e^{200}$ swamps everything else. *(2 marks)* This hints that scores need to be kept in a sensible range; otherwise attention "freezes" onto a single token. Section 5 fixes this by dividing by $\sqrt{d_k}$. *(1 mark)*

**D2.** **Right:** each output is a weighted average of vectors, with non-negative weights that sum to 1, which is a kind of blending. *(1 mark)* **Wrong:** a blur filter has fixed weights and looks only at nearby pixels. Attention weights are **computed from the input**, so they are different for every sentence and every token, and they can reach any position, however far away. *(2 marks)*

**D3.** Each output $y_i$ depends only on the input $X$, which is fully known at the start. No output needs another output first, so $S$, $A$ and $Y$ are just matrix products over the whole sentence. *(1 mark)* An RNN's $h_t$ needs $h_{t-1}$, so it must compute its states in order. *(1 mark)*

---

## E. Applied / coding

**E1.** *(6 marks: 1 scores, 2 stable softmax, 2 causal mask, 1 shape and row-sum checks)*

```python
import numpy as np

def naive_self_attention(X: np.ndarray, causal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (Y, A). X is n x d_model; A is n x n; Y is n x d_model."""
    n, d_model = X.shape
    S = X @ X.T
    if causal:
        allowed = np.tril(np.ones((n, n), dtype=bool))   # j <= i
        S = np.where(allowed, S, -np.inf)
    S = S - S.max(axis=1, keepdims=True)                 # stable softmax
    A = np.exp(S)
    A = A / A.sum(axis=1, keepdims=True)
    Y = A @ X
    assert A.shape == (n, n) and Y.shape == (n, d_model)
    assert np.allclose(A.sum(axis=1), 1.0)
    return Y, A

X = np.array([[0.1, 0.1], [1.0, 1.0], [1.5, 0.0]])       # a, light, bag
Y, A = naive_self_attention(X)
print(np.round(A, 2))
print("y_light =", np.round(Y[1], 3))                    # [1.087 0.574]
Yc, Ac = naive_self_attention(X, causal=True)
print(np.round(Ac, 2))
print("causal y_a =", Yc[0], " y_light =", np.round(Yc[1], 2))
```

Output:

```
[[0.3  0.36 0.34]
 [0.09 0.56 0.34]
 [0.08 0.3  0.63]]
y_light = [1.087 0.574]
[[1.   0.   0.  ]
 [0.14 0.86 0.  ]
 [0.08 0.3  0.63]]
causal y_a = [0.1 0.1]  y_light = [0.87 0.87]
```

**E2.** *(2 marks)*

```python
perm = np.array([2, 0, 1])
Y_shuffled, _ = naive_self_attention(X[perm])
print(np.allclose(Y_shuffled, Y[perm]))   # True
```

---

## F. ★ Challenge

1. $(PX)(PX)^{\top} = P X X^{\top} P^{\top} = P S P^{\top}$. *(1 mark)*
2. Multiplying by $P$ on the left reorders the rows, and by $P^{\top}$ on the right reorders the columns the same way. Softmax acts on each row separately, and on each row it doesn't care about the order of the entries, since $e^{s_j}/\sum_k e^{s_k}$ only needs the set of values. So reordering the rows and columns before the softmax gives the same matrix as reordering after it. *(1 mark)*
3. $Y' = (P A P^{\top})(P X) = P A (P^{\top} P) X = P A X = P Y$. *(1 mark)*
4. **No.** The causal mask is tied to positions: it allows $j \le i$. After shuffling, a token sees a different set of tokens, because which ones come "before" it has changed, so its output changes. In general $P M P^{\top} \ne M$ for the mask $M$. Causal attention therefore carries a weak sense of order. It knows how many tokens come before, but not their exact positions. *(1 mark)*
