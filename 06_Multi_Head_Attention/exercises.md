# Section 6 — Exercises: Multi-Head Attention

**Total: 48 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 55 minutes**

$$\text{head}_i = \text{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)}), \qquad \text{MultiHead}(X) = [\text{head}_1 \mid \cdots \mid \text{head}_h]\, W_O$$

Round to 2 decimal places unless told otherwise.

---

## A. Warm-up (recall)

**A1.** In one sentence each: what does a single head compute, what does the concatenation do, and what does $W_O$ do? *(3 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. Doubling the number of heads (keeping $d_{model}$ fixed) doubles the parameters in $W_Q$, $W_K$, $W_V$.
2. In grouped-query attention, several query heads share one key head and one value head.
3. Without $W_O$, output dimension 1 could only contain information from head 1.

**A3.** Fill in the shapes. $n = 50$, $d_{model} = 1024$, $h = 16$. *(3 marks)*

1. $d_k = $ `__`.
2. After splitting $Q$ into heads: `__ × __ × __`.
3. One head's attention matrix: `__ × __`.

---

## B. Multiple choice (1 mark each)

**B1.** What is the main benefit of several heads over one head of the same total width?
- (a) Fewer parameters.
- (b) Each head computes its own attention weights, so different heads can follow different relationships at once.
- (c) It removes the $n^2$ cost.
- (d) Heads let the model see future tokens.

**B2.** How many parameters does one multi-head attention layer have, including $W_O$, with no biases?
- (a) $3 d_{model}^2$
- (b) $4 d_{model}^2$
- (c) $4 h\, d_{model}^2$
- (d) $h\, d_k^2$

**B3.** What does grow with $h$ (for fixed $d_{model}$)?
- (a) The number of parameters.
- (b) The width of the output.
- (c) The number of $n \times n$ attention matrices.
- (d) The size of $W_O$.

**B4.** A model has 32 query heads and 8 KV heads. How many query heads share each KV head?
- (a) 2
- (b) 4
- (c) 8
- (d) 32

**B5.** A head's query is all zeros for some token. Its attention weights for that token will be:
- (a) All zero.
- (b) Uniform over all allowed tokens.
- (c) All on the token itself.
- (d) Undefined.

---

## C. Calculation

Use the sentence *"a light bag"* again, with axis 1 = *weight-ness* and axis 2 = *brightness*:

| Token | $x$ |
|---|---|
| a | [0.1, 0.1] |
| light | [1, 1] |
| bag | [1.5, 0] |

$h = 2$ heads with $d_k = d_v = 1$. Head 1 reads axis 1 ($q = k = v = x_1$) and head 2 reads axis 2 ($q = k = v = x_2$). Because $d_k = 1$, the scaling factor is 1.

**C1. Head 1 for "light".** Compute its scores, weights and output. *(3 marks)*

**C2. Head 2 for "light".** Compute its scores, weights and output. Which token does each head focus on most? *(3 marks)*

**C3. Combine.** *(3 marks)*

1. Write the concatenation for "light".
2. With $W_O = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$, compute the final output for "light".
3. Which output dimension now mixes both heads?

**C4. Head 2 for "bag".** bag's query in head 2 is $0$. What are its weights and its output in head 2, without a full calculation? *(2 marks)*

**C5. Parameters and caches.** *(5 marks)*

1. How many attention parameters (with $W_O$) does one layer have when $d_{model} = 1024$?
2. A model has 32 layers, 32 query heads, **8** KV heads and $d_k = 128$, and stores its cache in 2-byte numbers. How many bytes of KV cache does it need per token, over all layers?
3. How many GiB for an 8,192-token context? How many with full MHA (32 KV heads) instead?

---

## D. Conceptual

**D1.** Take $d_{model} = 768$ to the extreme with $h = 768$ heads of $d_k = 1$. What would each head be able to compare? Why do real models keep $d_k$ at 64 or 128 instead? *(3 marks)*

**D2.** "Multiple heads are almost free." Explain what stays the same and what changes when $h$ grows with $d_{model}$ fixed. *(3 marks)*

**D3.** Researchers find many heads can be removed after training with little loss. Give one explanation for why a model would train redundant heads, and one reason it is still useful to have them during training. *(3 marks)*

---

## E. Applied / coding (NumPy)

**E1.** Implement multi-head attention with the reshape trick: *(6 marks)*

```python
def multi_head_attention(X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray,
                         W_O: np.ndarray, h: int) -> np.ndarray:
    """X: n x d_model; all W: d_model x d_model. Returns n x d_model."""
```

Steps: project once; reshape to `h × n × d_k`; batched scaled attention; reshape back; multiply by `W_O`. Then check it against a slow version that loops over heads with sliced weight columns, using random inputs ($n = 5$, $d_{model} = 8$, $h = 2$).

**E2. GQA.** Modify your function to take `W_K` and `W_V` of shape `d_model × (g · d_k)` for `g` KV heads, and repeat each KV head `h // g` times before attention. Check that with `g = h` you get the same result as E1. *(3 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

1. Split $W_O$ into $h$ blocks of rows, $W_O^{(1)}, \ldots, W_O^{(h)}$, each $d_k \times d_{model}$. Show that

$$[\text{head}_1 \mid \cdots \mid \text{head}_h]\, W_O = \sum_{i=1}^{h} \text{head}_i\, W_O^{(i)}.$$

2. Interpret this result: what does each head "write" into the output?
3. Using this form, explain why a single head with $d_k = d_{model}$ is **not** the same as $h$ heads, even though both have $4d_{model}^2$ parameters.
