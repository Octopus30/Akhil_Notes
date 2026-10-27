# Section 1 — Exercises: What a Transformer Is

**Total: 46 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 50 minutes**

Use only ideas from Section 1. Write every shape as `rows × columns`. Calculators are allowed.

---

## A. Warm-up (recall)

**A1.** In one sentence each, say what the two sub-layers of a Transformer block do. *(2 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. A Transformer with no position information gives a different output vector for the word "bites" in "dog bites man" and in "man bites dog".
2. In a decoder-only model such as GPT, every block has the same input shape and output shape.
3. Most of a typical Transformer's parameters sit in its self-attention sub-layers.

**A3.** Fill in the shapes. A model has $d_{model} = 64$, vocabulary size 1,000 and $N = 6$ blocks. It reads a sentence of $n = 12$ tokens. *(3 marks)*

1. The embedding table has shape `__ × __`.
2. The input to block 4 has shape `__ × __`.
3. The output scores (before softmax) have shape `__ × __`.

---

## B. Multiple choice (1 mark each)

**B1.** Why do RNNs train slowly on GPUs?
- (a) They have more parameters than Transformers.
- (b) Step $t$ needs the hidden state from step $t-1$, so the steps can't run in parallel.
- (c) They can't use matrix multiplication.
- (d) Their hidden state is larger than $d_{model}$.

**B2.** Two tokens are 40 positions apart. How many steps must a signal take between them in one Transformer layer?
- (a) 40
- (b) 39
- (c) 1
- (d) $\log_2 40 \approx 5.3$

**B3.** What is the main idea of the 2017 Transformer paper?
- (a) Add attention on top of an RNN.
- (b) Keep attention and remove the recurrence.
- (c) Replace attention with convolutions.
- (d) Make RNNs bidirectional.

**B4.** Which model is encoder-only?
- (a) GPT-2
- (b) LLaMA-2
- (c) BERT
- (d) T5

**B5.** A context grows from 2,000 to 6,000 tokens. Roughly how much more work does the attention step do?
- (a) 3×
- (b) 6×
- (c) 9×
- (d) 27×

---

## C. Calculation

**C1. Shape trace.** A toy model has $d_{model} = 8$, vocabulary size 50 and $N = 3$ blocks. It reads *"the cat sat on the mat"*, one token per word. *(4 marks)*

1. What is $n$?
2. Give the shape at each stage: token IDs, $X^{(0)}$, after each block, output scores.
3. How many numbers are in the embedding table?
4. The output head is a weight matrix with no bias. What is its shape, and how many numbers does it hold?

**C2. Quadratic cost.** In self-attention, every token is compared with every token, including itself. *(3 marks)*

1. How many token-to-token comparisons does one attention step make for $n = 512$? For $n = 2048$?
2. By what factor does the work grow?
3. A model handles 1,024 tokens comfortably. Without changing anything else, how long a context could it handle if you allowed 4× the attention work?

**C3. Path lengths and sequential steps.** *(3 marks)*

1. In a 12-token sentence, how many steps does information need to travel from token 2 to token 9 in an RNN? In one Transformer layer?
2. A one-layer RNN reads 200 tokens. A Transformer with $N = 4$ blocks reads the same 200 tokens. How many steps must run *one after another* in each case?

**C4. Output head by hand.** After the last block, two tokens have the vectors below ($d_{model} = 3$). The output head $W_{out}$ maps them to a toy vocabulary of 4 words: `[cup, plate, is, full]`. *(5 marks)*

$$
X^{(N)} = \begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & -1 \end{bmatrix}, \qquad
W_{out} = \begin{bmatrix} 1 & 0 & 0 & 1 \\ 0 & 2 & 1 & 0 \\ 1 & -1 & 0 & 0 \end{bmatrix}
$$

1. Check that the shapes allow the product $X^{(N)} W_{out}$, and give the shape of the result.
2. Compute the score matrix $S = X^{(N)} W_{out}$.
3. Apply softmax to each row, $p_i = e^{s_i} / \sum_j e^{s_j}$, rounding to 3 decimal places.
4. Which word is most likely for each token?

---

## D. Conceptual

**D1.** The residual connection computes $A = X + \text{SelfAttention}(\text{LN}(X))$. Explain why this forces every sub-layer, and so every block, to keep the shape $n \times d_{model}$. What else would break if a block changed the width? *(3 marks)*

**D2.** A friend says: "Transformers have beaten RNNs on everything, so nobody should ever think about recurrence again." Give one concrete situation where an RNN-style model has an advantage, and explain why. *(3 marks)*

**D3.** A model is given *"Riya poured water into the glass until it was full"* with **no** position information. Then the same words are given in reverse order. What is the same and what is different about the two outputs? Why is this a problem for language? *(3 marks)*

---

## E. Applied / coding (NumPy)

**E1. Shape tracer.** Write a function

```python
def trace_shapes(ids: np.ndarray, d_model: int, vocab_size: int, N: int,
                 seed: int = 0) -> list[tuple[str, tuple[int, ...]]]:
```

that builds a toy pipeline with random weights from `np.random.default_rng(seed)`:

- an embedding table `E` of shape `vocab_size × d_model`, and `X = E[ids]`;
- `N` toy blocks, each computing `X = X + np.tanh(X @ W)` with its own `W` of shape `d_model × d_model`;
- an output head `W_out` of shape `d_model × vocab_size`, giving scores `X @ W_out`.

Return a list of `(stage name, shape)` pairs, starting with `("ids", ids.shape)` and ending with `("scores", ...)`. Test it on `ids = np.array([3, 14, 15, 9, 2])` with `d_model = 8`, `vocab_size = 20`, `N = 3`. *(5 marks)*

**E2. Order blindness.** Write a toy "mixing" block

```python
def mix_block(X: np.ndarray) -> np.ndarray:
    """Each row becomes itself plus the mean of all rows."""
```

Show in code that if you shuffle the rows of `X` with a permutation `perm`, then `mix_block(X[perm])` equals `mix_block(X)[perm]`. Explain in one sentence what this means for word order. *(4 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

Count the multiply-adds per layer in a simplified way:

- An RNN layer updates its hidden state with one $d_{model} \times d_{model}$ matrix per token, so it costs about $n \cdot d_{model}^2$.
- The comparison step of self-attention computes one $d_{model}$-long dot product for every pair of tokens, so it costs about $n^2 \cdot d_{model}$.

1. For what $n$ are the two costs equal?
2. For $d_{model} = 4096$, which is cheaper at $n = 1{,}000$, and by what factor? Which at $n = 32{,}000$?
3. Even when attention does *more* total work, it usually trains faster than an RNN. Why?
