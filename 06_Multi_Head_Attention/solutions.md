# Section 6 — Solutions: Multi-Head Attention

---

## A. Warm-up

**A1.** *(1 mark each)*
- **A single head** runs scaled dot-product attention with its own $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$, giving an $n \times d_k$ output.
- **The concatenation** places the $h$ head outputs side by side, giving $n \times d_{model}$.
- **$W_O$** mixes them, so every output dimension can use information from every head.

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **False.** Each head becomes narrower ($d_k = d_{model}/h$), so the total stays $3d_{model}^2$.
2. **True.** That is the definition of GQA; it shrinks the KV cache by $g/h$.
3. **True.** The concatenation puts head 1's output in dimensions $1..d_k$. Only $W_O$ mixes the heads.

**A3.** *(1 mark each)*
1. $1024/16 = 64$
2. `16 × 50 × 64`
3. `50 × 50`

---

## B. Multiple choice

**B1. (b).**
- (a) Wrong: the parameter count is the same.
- (c) Wrong: every head still has an $n \times n$ matrix.
- (d) Wrong: heads don't change the mask.

**B2. (b).**
- (a) Wrong: $3d_{model}^2$ forgets $W_O$.
- (c) Wrong: $4hd_{model}^2$ wrongly assumes every head is full width.
- (d) Wrong: $hd_k^2$ isn't a parameter count of anything in the layer.

**B3. (c).** $h$ heads give $h$ score matrices, $h \cdot n^2$ scores in total.
- (a) Wrong: the parameters stay at $4d_{model}^2$.
- (b) Wrong: the output width stays $d_{model}$.
- (d) Wrong: $W_O$ stays $d_{model} \times d_{model}$.

**B4. (b).** $32/8 = 4$.
- (a) 2 would mean 16 KV heads.
- (c) 8 is the number of KV heads, not the group size.
- (d) 32 describes MQA, with a single KV head.

**B5. (b).** All its scores are $0 \cdot k_j = 0$, and softmax of equal scores is uniform.
- (a) Wrong: softmax weights always sum to 1, so they can't all be zero.
- (c) Wrong: the token's own score is also 0, so it isn't favoured.
- (d) Wrong: equal scores are perfectly well defined.

---

## C. Calculation

**C1.** Head 1 for "light": $q = 1$ and the keys are $[0.1, 1, 1.5]$, so the scores are $[0.1, 1, 1.5]$. *(1 mark)*
- $e^{0.1} = 1.11$, $e^{1} = 2.72$, $e^{1.5} = 4.48$. The sum is $8.31$, so the weights are $[0.13, 0.33, 0.54]$. *(1 mark)*
- The output is $0.13(0.1) + 0.33(1) + 0.54(1.5) = 1.15$. *(1 mark)*

**C2.** Head 2 for "light": $q = 1$ and the keys are $[0.1, 1, 0]$, so the scores are $[0.1, 1, 0]$. *(1 mark)*
- $e^{0.1} = 1.11$, $e^{1} = 2.72$, $e^{0} = 1$. The sum is $4.82$, so the weights are $[0.23, 0.56, 0.21]$. *(1 mark)*
- The output is $0.23(0.1) + 0.56(1) + 0.21(0) = 0.59$.
- Head 1 focuses on **bag** (0.54); head 2 focuses on **light** itself (0.56). *(1 mark)*

**C3.**
1. $[1.15, 0.59]$ *(1 mark)*
2. $[1.15, 0.59]\begin{bmatrix} 1 & 1 \\ 0 & 1\end{bmatrix} = [1.15,\; 1.15 + 0.59] = [1.15, 1.74]$ *(1 mark)*
3. Dimension 2 is the sum of both heads. Dimension 1 still comes from head 1 alone, because the first column of $W_O$ is $[1, 0]$. *(1 mark)*

**C4.** All the scores are $0 \times k_j = 0$, so the weights are uniform: $[\tfrac13, \tfrac13, \tfrac13]$. *(1 mark)* The output is the mean of the head-2 values: $(0.1 + 1 + 0)/3 = 0.37$. *(1 mark)* This is a head with "nothing to look for".

**C5.**
1. $4 \times 1024^2 = 4{,}194{,}304$ *(1 mark)*
2. Per token per layer: $2 \times 8 \times 128 = 2{,}048$ numbers. Over 32 layers at 2 bytes: $2{,}048 \times 32 \times 2 = 131{,}072$ bytes = **128 KiB** per token. *(2 marks)*
3. $131{,}072 \times 8{,}192 = 2^{30}$ bytes = **1 GiB**. *(1 mark)* With MHA (32 KV heads) it would be 4× larger: **4 GiB**. *(1 mark)*

---

## D. Conceptual

**D1.** Each head would compare tokens along a single learned direction. A 1-D query times a 1-D key is just a product of two numbers, so each head could only express "large × large = attend". *(1 mark)* Such heads are too weak to match on combinations of features. *(1 mark)* With $d_k$ = 64–128, each head can still form a rich comparison, and there are enough heads (12–64) for different patterns. *(1 mark)*

**D2.** **Same:** the parameters stay at $4d_{model}^2$, the output shape stays $n \times d_{model}$, and the matrix-multiply cost stays roughly the same, since $h$ heads of width $d_{model}/h$ cost as much as one wide head. *(2 marks)* **Changes:** there are $h$ attention matrices, so $h \cdot n^2$ scores (more memory for $A$ and more softmax work), and each head becomes narrower. *(1 mark)*

**D3.** **Why redundant:** gradient descent has no penalty for duplication. Heads start from random weights and often converge to similar useful patterns, and redundancy makes the network robust. *(1.5 marks)* **Why useful during training:** more heads give more parallel "search paths", so it is more likely that some head finds a useful pattern early, and the optimisation landscape is easier. The extra capacity can then be pruned afterwards. *(1.5 marks)*

---

## E. Applied / coding

**E1.** *(6 marks: 1 projection, 2 reshapes, 1 batched attention, 1 $W_O$, 1 matches the loop)*

```python
import numpy as np

def softmax(S: np.ndarray) -> np.ndarray:
    """Stable softmax over the last axis."""
    e = np.exp(S - S.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def split_heads(M: np.ndarray, h: int) -> np.ndarray:
    """n x (h d_k) -> h x n x d_k."""
    n, width = M.shape
    return M.reshape(n, h, width // h).transpose(1, 0, 2)

def multi_head_attention(X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray,
                         W_O: np.ndarray, h: int) -> np.ndarray:
    """X: n x d_model; all W: d_model x d_model. Returns n x d_model."""
    n, d_model = X.shape
    d_k = d_model // h
    Q, K, V = (split_heads(X @ W, h) for W in (W_Q, W_K, W_V))     # h x n x d_k
    A = softmax(Q @ K.transpose(0, 2, 1) / np.sqrt(d_k))           # h x n x n
    heads = A @ V                                                  # h x n x d_k
    concat = heads.transpose(1, 0, 2).reshape(n, d_model)          # n x d_model
    return concat @ W_O

def mha_loop(X, W_Q, W_K, W_V, W_O, h):
    """Slow reference: one head at a time using column slices of the weights."""
    d_k = X.shape[1] // h
    outs = []
    for i in range(h):
        cols = slice(i * d_k, (i + 1) * d_k)
        Q, K, V = X @ W_Q[:, cols], X @ W_K[:, cols], X @ W_V[:, cols]
        outs.append(softmax(Q @ K.T / np.sqrt(d_k)) @ V)
    return np.concatenate(outs, axis=1) @ W_O

rng = np.random.default_rng(0)
n, d_model, h = 5, 8, 2
X = rng.normal(size=(n, d_model))
W_Q, W_K, W_V, W_O = (rng.normal(size=(d_model, d_model)) for _ in range(4))
Y = multi_head_attention(X, W_Q, W_K, W_V, W_O, h)
assert Y.shape == (n, d_model)
assert np.allclose(Y, mha_loop(X, W_Q, W_K, W_V, W_O, h))
print("reshape version matches the per-head loop:", Y.shape)
```

**E2.** *(3 marks)*

```python
def gqa_attention(X, W_Q, W_K, W_V, W_O, h, g):
    """Grouped-query attention: h query heads, g shared key/value heads."""
    n, d_model = X.shape
    d_k = d_model // h
    Q = split_heads(X @ W_Q, h)                                    # h x n x d_k
    K, V = split_heads(X @ W_K, g), split_heads(X @ W_V, g)        # g x n x d_k
    K, V = np.repeat(K, h // g, axis=0), np.repeat(V, h // g, axis=0)
    A = softmax(Q @ K.transpose(0, 2, 1) / np.sqrt(d_k))
    concat = (A @ V).transpose(1, 0, 2).reshape(n, d_model)
    return concat @ W_O

assert np.allclose(gqa_attention(X, W_Q, W_K, W_V, W_O, h=2, g=2), Y)
W_K1, W_V1 = rng.normal(size=(d_model, 4)), rng.normal(size=(d_model, 4))   # g = 1 (MQA)
print("MQA output shape:", gqa_attention(X, W_Q, W_K1, W_V1, W_O, h=2, g=1).shape)
```

---

## F. ★ Challenge

1. Write the concatenation as a row of blocks $C = [H_1 \mid \cdots \mid H_h]$ and $W_O$ as a column of blocks $W_O^{(1)}, \ldots, W_O^{(h)}$. Block matrix multiplication gives $C W_O = \sum_i H_i W_O^{(i)}$, just as a dot product sums its terms. *(2 marks)*
2. Each head computes its own result, projects it into the full $d_{model}$-wide space with its own $W_O^{(i)}$, and **adds** it to the output. The heads are independent "writers" whose contributions are summed, and in a full block the sum is added to the residual stream (Section 8). *(1 mark)*
3. One wide head has **one** attention matrix $A$: every value dimension is mixed with the same weights. In the sum form, $h$ heads have $h$ different matrices $A_i$, and each term $A_i V_i W_O^{(i)}$ can attend to different tokens. One head can't reproduce this, because a single softmax can't give different weight patterns to different parts of its output. The parameter counts are equal, but the functions they can represent differ. *(1 mark)*
