# Section 4 — Exercises: Queries, Keys & Values

**Total: 49 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 55 minutes**

Use $Q = XW_Q$, $K = XW_K$, $V = XW_V$, $S = QK^{\top}$, $A = \text{softmax}_{\text{rows}}(S)$, $Y = AV$. There is **no** $\sqrt{d_k}$ scaling yet. Round to 2 decimal places unless told otherwise.

---

## A. Warm-up (recall)

**A1.** In the job-portal picture, say in one sentence each what the **query**, the **key** and the **value** of a token stand for. *(3 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. $Q$ and $V$ must have the same width.
2. A different $W_Q$ is learned for each position in the sentence.
3. With queries and keys, the score $s_{ij}$ need not equal $s_{ji}$.

**A3.** Fill in the shapes. $n = 512$, $d_{model} = 1024$, $d_k = d_v = 128$. *(3 marks)*

1. $W_K$ is `__ × __`.
2. $K$ is `__ × __`.
3. $Y$ is `__ × __`.

---

## B. Multiple choice (1 mark each)

**B1.** What is the main problem with naive self-attention ($S = XX^{\top}$) that queries and keys fix?
- (a) It is too slow.
- (b) A token's score with itself is $\lVert x\rVert^2$, so long vectors mostly attend to themselves; and every score is symmetric.
- (c) It can't handle sentences longer than $d_{model}$.
- (d) It has too many parameters.

**B2.** What is computed fresh for every new input sentence?
- (a) $W_Q$, $W_K$, $W_V$
- (b) $Q$, $K$, $V$, $S$, $A$ and $Y$
- (c) Only $Y$
- (d) $W_Q$ only

**B3.** The score matrix $S = QK^{\top}$ is symmetric for every input $X$ exactly when:
- (a) $W_Q = W_V$
- (b) $W_Q W_K^{\top}$ is a symmetric matrix
- (c) $d_k = d_v$
- (d) $X$ is square

**B4.** Why do we need a separate value projection, instead of mixing the keys?
- (a) Keys are too short to be mixed.
- (b) What a token advertises for matching can differ from the information it hands over.
- (c) Values are needed to make the softmax work.
- (d) It halves the number of parameters.

**B5.** How many parameters are in $W_Q$, $W_K$ and $W_V$ together for one full layer with $d_{model} = 768$ (all heads together, no biases)?
- (a) 589,824
- (b) 1,769,472
- (c) 2,359,296
- (d) 147,456

---

## C. Calculation

Use the Section 3 exercise sentence *"a light bag"* with $d_{model} = d_k = d_v = 2$:

$$X = \begin{bmatrix} 0.1 & 0.1 \\ 1 & 1 \\ 1.5 & 0 \end{bmatrix}\begin{matrix}\leftarrow \text{a} \\ \leftarrow \text{light} \\ \leftarrow \text{bag}\end{matrix}, \quad
W_Q = \begin{bmatrix} 0 & 2 \\ 1 & 0 \end{bmatrix}, \quad
W_K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
W_V = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$$

**C1. Project.** Compute $Q$, $K$ and $V$. *(3 marks)*

**C2. One row by hand.** For "bag": *(5 marks)*

1. Compute its scores against every key.
2. Apply softmax.
3. Compute $y_{\text{bag}}$.
4. In naive attention (Section 3 exercises), bag attended 0.63 to itself. What is its self-weight now, and which token does it attend to most?

**C3. The full score matrix.** Compute $S = QK^{\top}$. Find a pair $(i, j)$ with $s_{ij} \ne s_{ji}$. Then compute $W_Q W_K^{\top}$ and explain why $S$ is not symmetric here. *(4 marks)*

**C4. Asymmetry from scratch.** Two tokens $p = [1, 1]$ and $r = [1, -1]$, with $W_K = I$ and $W_Q = \begin{bmatrix} 1 & 0 \\ 1 & 0 \end{bmatrix}$. Compute $s_{pr}$ and $s_{rp}$. Who is "looking for" whom? *(3 marks)*

**C5. Real sizes.** LLaMA-2 13B has $d_{model} = 5120$ and 40 heads. *(3 marks)*

1. What is $d_k$?
2. How many parameters are in one head's $W_Q$, $W_K$, $W_V$?
3. How many in $W_Q$, $W_K$, $W_V$ for the whole layer (all heads)?

---

## D. Conceptual

**D1.** Suppose training ends with $W_Q = W_K$. What can you say about the scores $s_{ii}$, and what problem from Section 3 partly returns? *(3 marks)*

**D2.** Why does the same set of weights $W_Q, W_K, W_V$ work for sentences of any length $n$? What part of the computation does depend on $n$? *(3 marks)*

**D3.** In a translation model's **cross-attention**, $Q$ comes from the decoder and $K, V$ come from the encoder. With $n_{tgt}$ decoder tokens and $n_{src}$ encoder tokens, what are the shapes of $S$ and $Y$? Why must $K$ and $V$ come from the same sequence? *(3 marks)*

---

## E. Applied / coding (NumPy)

**E1.** Implement *(5 marks)*

```python
def qkv_attention(X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray,
                  W_V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (Y, A). X: n x d_model; W_Q, W_K: d_model x d_k; W_V: d_model x d_v."""
```

Assert every intermediate shape, and that the rows of `A` sum to 1. Check your answers to C1–C3.

**E2. One fused matrix.** Real implementations store `W_QKV` of shape `d_model × 3d_k` and do one multiplication. Write code that builds `W_QKV = np.concatenate([W_Q, W_K, W_V], axis=1)`, computes `X @ W_QKV`, splits the result into $Q, K, V$ with `np.split`, and checks they match the separate products. *(3 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

1. Show that $s_{ij} = x_i\, M\, x_j^{\top}$ with $M = W_Q W_K^{\top}$, a $d_{model} \times d_{model}$ matrix.
2. $W_Q$ and $W_K$ are each $d_{model} \times d_k$ with $d_k < d_{model}$. What is the largest possible rank of $M$? What does that say about which kinds of score patterns a single head can express?
3. Any square matrix can be split as $M = M_{sym} + M_{anti}$ with $M_{sym} = \frac{1}{2}(M + M^{\top})$ and $M_{anti} = \frac{1}{2}(M - M^{\top})$. Show that $s_{ij} - s_{ji} = 2\,x_i M_{anti} x_j^{\top}$. What does a head with $M_{anti} = 0$ fail to express?
