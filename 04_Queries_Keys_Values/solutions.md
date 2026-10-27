# Section 4 — Solutions: Queries, Keys & Values

---

## A. Warm-up

**A1.** *(1 mark each)*
- **Query:** what the token is looking for, like a recruiter's search.
- **Key:** what the token offers to be matched against, like a profile headline.
- **Value:** the information the token hands over when it is attended to, like the full CV.

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **False.** $Q$ and $K$ must share the width $d_k$ so that $q \cdot k$ exists. $V$ has width $d_v$, which can differ (although usually $d_v = d_k$).
2. **False.** One $W_Q$ is shared by every position, so the layer works for any $n$.
3. **True.** $s_{ij} = x_i W_Q W_K^{\top} x_j^{\top}$, and this is symmetric in $i, j$ only if $W_Q W_K^{\top}$ is a symmetric matrix.

**A3.** *(1 mark each)* `1024 × 128`, `512 × 128`, `512 × 128`.

---

## B. Multiple choice

**B1. (b).**
- (a) Wrong: Q/K/V adds work, it doesn't save any.
- (c) Wrong: there is no such limit.
- (d) Wrong: naive attention has no parameters at all.

**B2. (b).** $W_Q$, $W_K$, $W_V$ are learned once; everything computed from $X$ is new for every input.
- (a) and (d) list parameters, not computed values.
- (c) Wrong: $Q$, $K$, $V$, $S$, $A$ are recomputed too.

**B3. (b).** $s_{ij} = x_i M x_j^{\top}$ with $M = W_Q W_K^{\top}$, and $s_{ij} = s_{ji}$ for all inputs exactly when $M = M^{\top}$.
- (a) Wrong: $W_V$ doesn't take part in the scores.
- (c) Wrong: the widths have nothing to do with symmetry.
- (d) Wrong: $X$ is $n \times d_{model}$ and need not be square. A square $X$ wouldn't make $S$ symmetric anyway.

**B4. (b).** What a token advertises for matching (its key) can differ from what it hands over (its value).
- (a) Wrong: keys have the same width as queries and could be mixed.
- (c) Wrong: the softmax uses only the scores.
- (d) Wrong: a separate value projection *adds* $d_{model} \times d_v$ parameters.

**B5. (b).** $3 \times 768^2 = 1{,}769{,}472$.
- (a) 589,824 = $768^2$, a single matrix.
- (c) 2,359,296 = $4 \times 768^2$, which also counts $W_O$ (Section 6).
- (d) 147,456 is for one head only.

---

## C. Calculation

**C1.** *(1 mark each)*
- $Q = XW_Q$. $W_Q$ maps $[x_1, x_2] \mapsto [x_2, 2x_1]$, so the rows are $[0.1, 0.2]$, $[1, 2]$, $[0, 3]$.
- $K = XW_K = X$, so the rows are $[0.1, 0.1]$, $[1, 1]$, $[1.5, 0]$.
- $V = XW_V$. $W_V$ maps $[x_1, x_2] \mapsto [x_1, x_1 + x_2]$, so the rows are $[0.1, 0.2]$, $[1, 2]$, $[1.5, 1.5]$.

**C2.** $q_{\text{bag}} = [0, 3]$.
1. Scores: $\cdot k_{\text{a}} = 0.3$, $\cdot k_{\text{light}} = 3$, $\cdot k_{\text{bag}} = 0$. *(1 mark)*
2. $e^{0.3} = 1.35$, $e^{3} = 20.09$, $e^{0} = 1$. The sum is $22.44$, so the weights are $[0.06, 0.90, 0.04]$. *(2 marks)*
3. $y_{\text{bag}} = 0.06[0.1, 0.2] + 0.90[1, 2] + 0.04[1.5, 1.5] = [0.97, 1.87]$. *(1 mark)* (Unrounded weights 0.0602, 0.8953, 0.0446 give $[0.968, 1.869]$.)
4. The self-weight fell from 0.63 to **0.04**. bag now attends **90% to light**, the word that tells it what kind of bag this is. *(1 mark)*

**C3.** *(4 marks: 2 for $S$, 1 for the pair, 1 for the explanation)*

| $S$ | a | light | bag |
|---|---|---|---|
| **a** | 0.03 | 0.30 | 0.15 |
| **light** | 0.30 | 3.00 | 1.50 |
| **bag** | 0.30 | 3.00 | 0.00 |

For example $s_{\text{light,bag}} = 1.5$ but $s_{\text{bag,light}} = 3.0$. Likewise $s_{\text{a,bag}} = 0.15$ but $s_{\text{bag,a}} = 0.30$.

$W_Q W_K^{\top} = W_Q I = \begin{bmatrix} 0 & 2 \\ 1 & 0 \end{bmatrix}$. This isn't symmetric ($2 \ne 1$), so $S$ isn't either. Compare the notes' example, where $W_Q W_K^{\top}$ *was* symmetric and $S$ happened to stay symmetric.

**C4.** $q_p = [1, 1] W_Q = [1 + 1, 0] = [2, 0]$ and $q_r = [1, -1] W_Q = [1 - 1, 0] = [0, 0]$. With $W_K = I$, $k_p = [1, 1]$ and $k_r = [1, -1]$. *(1 mark)*
- $s_{pr} = q_p \cdot k_r = [2, 0] \cdot [1, -1] = 2$. *(1 mark)*
- $s_{rp} = q_r \cdot k_p = 0$.

So $p$ is looking for $r$, but $r$ isn't looking for $p$. *(1 mark)*

**C5.**
1. $d_k = 5120 / 40 = 128$. *(1 mark)*
2. $3 \times 5120 \times 128 = 1{,}966{,}080$. *(1 mark)*
3. $3 \times 5120^2 = 78{,}643{,}200$, which also equals $40 \times 1{,}966{,}080$. *(1 mark)*

---

## D. Conceptual

**D1.** If $W_Q = W_K = W$, then $s_{ii} = (x_i W)\cdot(x_i W) = \lVert x_i W\rVert^2 \ge 0$, a squared length again. *(1 mark)* $M = WW^{\top}$ is symmetric, so every score matrix is symmetric too. *(1 mark)* The Section 3 problems partly return: a token with a long projected vector tends to attend to itself, and relevance always goes both ways. It is only "partly", because the projection $W$ can still shrink or rotate the vectors. *(1 mark)*

**D2.** The weights act on **one token at a time**: $x_i W_Q$ is a $1 \times d_{model}$ row times a $d_{model} \times d_k$ matrix, whatever $n$ is. The same matrix is simply applied to every row. *(2 marks)* $n$ affects the sizes of $Q, K, V$ ($n \times d_k$) and of $S, A$ ($n \times n$), but not any parameter. *(1 mark)*

**D3.** $Q$ is $n_{tgt} \times d_k$ and $K$ is $n_{src} \times d_k$, so $S = QK^{\top}$ is $n_{tgt} \times n_{src}$. $V$ is $n_{src} \times d_v$, so $Y = AV$ is $n_{tgt} \times d_v$: one output per decoder token. *(2 marks)* $A$'s column $j$ refers to key $j$, and that weight is used to pick value $j$. So each key and its value must belong to the same token of the same sequence. *(1 mark)*

---

## E. Applied / coding

**E1.** *(5 marks: 1 projections, 1 scores, 1 softmax, 1 output, 1 shape checks)*

```python
import numpy as np

def qkv_attention(X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray,
                  W_V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (Y, A). X: n x d_model; W_Q, W_K: d_model x d_k; W_V: d_model x d_v."""
    n, d_model = X.shape
    d_k, d_v = W_Q.shape[1], W_V.shape[1]
    assert W_Q.shape == W_K.shape == (d_model, d_k) and W_V.shape == (d_model, d_v)

    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    assert Q.shape == K.shape == (n, d_k) and V.shape == (n, d_v)

    S = Q @ K.T
    assert S.shape == (n, n)
    A = np.exp(S - S.max(axis=1, keepdims=True))
    A = A / A.sum(axis=1, keepdims=True)
    assert np.allclose(A.sum(axis=1), 1)

    Y = A @ V
    assert Y.shape == (n, d_v)
    return Y, A

X = np.array([[0.1, 0.1], [1.0, 1.0], [1.5, 0.0]])
W_Q = np.array([[0.0, 2.0], [1.0, 0.0]])
W_K = np.eye(2)
W_V = np.array([[1.0, 1.0], [0.0, 1.0]])

Y, A = qkv_attention(X, W_Q, W_K, W_V)
print("S =\n", (X @ W_Q) @ (X @ W_K).T)
print("A =\n", np.round(A, 2))
print("y_bag =", np.round(Y[2], 3))
```

Output:

```
S =
 [[0.03 0.3  0.15]
 [0.3  3.   1.5 ]
 [0.3  3.   0.  ]]
A =
 [[0.29 0.38 0.33]
 [0.05 0.77 0.17]
 [0.06 0.9  0.04]]
y_bag = [0.968 1.869]
```

**E2.** *(3 marks: 1 concatenation, 1 split, 1 check)*

```python
W_QKV = np.concatenate([W_Q, W_K, W_V], axis=1)      # d_model x 3 d_k
assert W_QKV.shape == (2, 6)
Q2, K2, V2 = np.split(X @ W_QKV, 3, axis=1)
assert np.allclose(Q2, X @ W_Q) and np.allclose(K2, X @ W_K) and np.allclose(V2, X @ W_V)
print("fused projection matches the three separate ones")
```

---

## F. ★ Challenge

1. $s_{ij} = q_i \cdot k_j = (x_i W_Q)(x_j W_K)^{\top} = x_i W_Q W_K^{\top} x_j^{\top} = x_i M x_j^{\top}$. $M$ is $(d_{model} \times d_k)(d_k \times d_{model}) = d_{model} \times d_{model}$. *(1 mark)*
2. $\text{rank}(M) \le d_k$, because the product passes through a $d_k$-wide bottleneck. *(1 mark)* A single head can only compare tokens along $d_k$ directions of the $d_{model}$-dimensional space. It can't express every possible bilinear score pattern, which is one reason to use several heads that look along different directions (Section 6).
3. $s_{ij} - s_{ji} = x_i M x_j^{\top} - x_j M x_i^{\top}$. A scalar equals its own transpose, so $x_j M x_i^{\top} = x_i M^{\top} x_j^{\top}$. Hence $s_{ij} - s_{ji} = x_i (M - M^{\top}) x_j^{\top} = 2\, x_i M_{anti} x_j^{\top}$. *(1 mark)* If $M_{anti} = 0$, all scores are symmetric, and the head can't express one-way relationships such as "the pronoun looks for the noun, but the noun doesn't look for the pronoun". *(1 mark)*
