# Section 3 — Exercises: Self-Attention, the Intuition

**Total: 49 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 55 minutes**

Use the naive self-attention of Section 3: $S = XX^{\top}$, $A = \text{softmax}_{\text{rows}}(S)$, $Y = AX$. There are no projections and no scaling yet. Round to 2 decimal places unless told otherwise.

---

## A. Warm-up (recall)

**A1.** Name the three steps of self-attention, and write the formula for each one for a single token $i$. *(3 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. Each **column** of the attention matrix $A$ sums to 1.
2. The attention weights are learned parameters, stored in the model after training.
3. In causal self-attention, token 3 may attend to tokens 1, 2 and 3 but not token 4.

**A3.** Fill in the shapes, for a sentence of $n = 7$ tokens with $d_{model} = 16$. *(3 marks)*

1. $S = XX^{\top}$ is `__ × __`.
2. $A$ is `__ × __`.
3. $Y = AX$ is `__ × __`.

---

## B. Multiple choice (1 mark each)

**B1.** In naive self-attention, why does a token with a long vector tend to attend mostly to itself?
- (a) Softmax always gives the diagonal the largest weight.
- (b) Its dot product with itself is $\lVert x \rVert^2$, which is usually its largest score.
- (c) Tokens are not allowed to attend to others in naive attention.
- (d) Long vectors are removed by the softmax.

**B2.** Every token in a sentence has exactly the same embedding. What does self-attention output?
- (a) All zeros.
- (b) The same vector for every token, equal to that shared embedding.
- (c) Different vectors depending on position.
- (d) It can't be computed, because the scores are all equal.

**B3.** Which property means the attention weights change when the sentence changes?
- (a) Global
- (b) Parallel
- (c) Input-dependent
- (d) Quadratic cost

**B4.** A sentence grows from 300 to 900 tokens. The score matrix $S$ grows by a factor of:
- (a) 3
- (b) 6
- (c) 9
- (d) 27

**B5.** In which situation would you use **cross-attention** rather than self-attention?
- (a) A sentiment classifier reading one review.
- (b) A translation decoder looking at the source sentence.
- (c) A GPT model predicting the next word of its own text.
- (d) BERT filling in a masked word.

---

## C. Calculation

Toy 2-D embeddings with axes we chose to mean *weight-ness* and *brightness*:

| Token | Vector |
|---|---|
| a | [0.1, 0.1] |
| light | [1, 1] (ambiguous: "not heavy" or "lamp light") |
| bag | [1.5, 0] |
| lamp | [0, 1.5] |

**C1. One token by hand.** Sentence: *"a light bag"*. *(5 marks)*

1. Compute the scores of "light" against a, light and bag.
2. Apply softmax to get the three attention weights.
3. Compute $y_{\text{light}}$.
4. Which axis did "light" move toward? Does that match the meaning of "light" in this sentence?

**C2. The other sentence.** Sentence: *"a light lamp"*. Compute $y_{\text{light}}$ again, and explain in one sentence why the attention weights are the same as in C1 even though the result is different. *(3 marks)*

**C3. The full matrices.** For *"a light bag"*: *(5 marks)*

1. Compute the full $3 \times 3$ score matrix $S = XX^{\top}$.
2. Compute $A$ (round to 2 d.p.) and check that each row sums to 1.
3. Which token attends most strongly to itself, and why?

**C4. Causal version.** Repeat C3 for *"a light bag"* but with **causal** attention, so token $i$ only sees tokens $1..i$. Give the new $A$ and the new $y_{\text{a}}$ and $y_{\text{light}}$. Which row is unchanged, and why? *(3 marks)*

**C5. Cost.** For an input of 2,000 tokens, one attention head computes how many scores? At 4 bytes per score, how many MB does $S$ take? A model has 32 heads in each of 32 layers. If every $S$ were stored at once, how many GB would that be? *(3 marks)*

---

## D. Conceptual

**D1.** Multiply every embedding in C1 by 10. Without full calculation, describe what happens to the attention weights of "light", and why. What problem does this hint at? *(3 marks)*

**D2.** A classmate says: "Self-attention is just a fixed weighted average, like a blur filter in image processing." Explain what is right and what is wrong about this. *(3 marks)*

**D3.** Why can all $n$ outputs of self-attention be computed at the same time, while an RNN must compute its hidden states one by one? *(2 marks)*

---

## E. Applied / coding (NumPy)

**E1.** Implement *(6 marks)*

```python
def naive_self_attention(X: np.ndarray, causal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (Y, A). X is n x d_model; A is n x n; Y is n x d_model."""
```

Requirements:
- softmax over rows, made stable by subtracting each row's max;
- if `causal=True`, token $i$ may only attend to tokens $j \le i$ (set the other scores to $-\infty$ before softmax);
- assert the shapes and that every row of `A` sums to 1.

Test it on *"a light bag"* and check your answers to C3 and C4.

**E2.** Using your function, show numerically that shuffling the rows of $X$ shuffles the rows of $Y$ in the same way (for non-causal attention). *(2 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

Let $P$ be an $n \times n$ permutation matrix, so $PX$ shuffles the rows of $X$, and $P^{\top}P = I$.

1. Show that the score matrix for $PX$ is $PSP^{\top}$.
2. Explain why $\text{softmax}_{\text{rows}}(PSP^{\top}) = P\,\text{softmax}_{\text{rows}}(S)\,P^{\top}$.
3. Conclude that the output for $PX$ is $PY$.
4. Does the same proof work for causal attention? Explain briefly.
