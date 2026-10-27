# Section 1 — Solutions: What a Transformer Is

---

## A. Warm-up

**A1.** *(1 mark each)*
- **Self-attention:** each token gathers information from every other token in the sequence ("tokens talk to each other").
- **Feed-forward network:** each token's vector is processed on its own by a small two-layer network ("each token thinks individually").

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **False.** Without position information the model only sees the set of words. Both sentences contain the same words, so "bites" gets the same output vector in each; the other rows are just in a different order.
2. **True.** Every block maps $n \times d_{model}$ to $n \times d_{model}$. That is what lets blocks be stacked, and the residual addition needs it.
3. **False.** Roughly two-thirds of the parameters sit in the feed-forward networks. Attention moves information between tokens, but most of the weights are in the FFNs.

**A3.** *(1 mark each)*
1. `1000 × 64`: one row per vocabulary entry.
2. `12 × 64`: every block's input and output is $n \times d_{model}$.
3. `12 × 1000`: one score per vocabulary word for each of the 12 tokens.

---

## B. Multiple choice

**B1. (b).** Each hidden state depends on the previous one, so the 1,000 steps of a 1,000-token sentence must run one after another.
- (a) Wrong: the parameter count has nothing to do with it. A small RNN is still sequential.
- (c) Wrong: RNNs do use matrix multiplication at every step. The problem is that the steps are in a chain.
- (d) Wrong: the hidden-state size changes the cost of each step, not whether the steps can run in parallel.

**B2. (c).** Attention connects any two tokens directly, in one step.
- (a) and (b) are RNN-style counts: the gap is 40 positions, so an RNN needs about 40 steps.
- (d) is the depth of a tree-shaped network. Nothing like that appears in this section.

**B3. (b).** "Keep the attention, throw away the recurrence."
- (a) Wrong: attention on top of an RNN was Bahdanau et al., 2014. The Transformer removed the RNN.
- (c) Wrong: the Transformer doesn't use convolutions.
- (d) Wrong: bidirectional RNNs existed earlier and are still recurrent.

**B4. (c) BERT.**
- (a) GPT-2 and (b) LLaMA-2 are decoder-only.
- (d) T5 is encoder–decoder.

**B5. (c) 9×.** The work grows with $n^2$, and $(6000/2000)^2 = 3^2 = 9$.
- (a) 3× would be true if the cost grew linearly, as an RNN's does.
- (b) Wrong: 6× comes from confusing 6,000 with the growth factor.
- (d) Wrong: 27× = $3^3$ would be cubic growth.

---

## C. Calculation

**C1.**
1. $n = 6$ (the, cat, sat, on, the, mat). *(1 mark)* The word "the" appears twice, but it is still two tokens.
2. *(1 mark)*

| Stage | Shape |
|---|---|
| Token IDs | $6$ |
| $X^{(0)}$ | $6 \times 8$ |
| After block 1, 2, 3 | $6 \times 8$ each |
| Output scores | $6 \times 50$ |

3. $50 \times 8 = 400$ numbers. *(1 mark)*
4. $W_{out}$ has shape $8 \times 50$, because it must turn width 8 into 50 scores. It holds $8 \times 50 = 400$ numbers. *(1 mark)*

**Marking:** 1 mark for each part. Part 2 needs every stage right, including the observation that the width stays at 8.

**C2.**
1. $n^2$ comparisons: $512^2 = 262{,}144$ and $2048^2 = 4{,}194{,}304$. *(1 mark)*
2. $4{,}194{,}304 / 262{,}144 = 16$. Equivalently $(2048/512)^2 = 4^2 = 16$. *(1 mark)*
3. 4× the work means $n^2$ grows 4×, so $n$ grows $\sqrt{4} = 2$×, giving **2,048 tokens**. *(1 mark)*

**C3.**
1. RNN: from position 2 to position 9 is $9 - 2 = 7$ steps. Transformer: **1** step. *(1 mark)*
2. The RNN needs **200** sequential steps, one per token. *(1 mark)* The Transformer needs **4** sequential steps, one per block. Inside each block all 200 tokens are processed in parallel. *(1 mark)*

**C4.**
1. $X^{(N)}$ is $2 \times 3$ and $W_{out}$ is $3 \times 4$. The inner sizes match (3 = 3), so the result is $2 \times 4$. *(1 mark)*
2. Each entry is a row of $X^{(N)}$ dotted with a column of $W_{out}$. *(2 marks)*
   - Row 1, $[1, 0, 2]$: cup $1\cdot1 + 0\cdot0 + 2\cdot1 = 3$; plate $0 + 0 + 2\cdot(-1) = -2$; is $0 + 0 + 0 = 0$; full $1 + 0 + 0 = 1$.
   - Row 2, $[0, 1, -1]$: cup $0 + 0 + (-1)\cdot1 = -1$; plate $0 + 2 + (-1)(-1) = 3$; is $0 + 1 + 0 = 1$; full $0 + 0 + 0 = 0$.

$$S = \begin{bmatrix} 3 & -2 & 0 & 1 \\ -1 & 3 & 1 & 0 \end{bmatrix}$$

3. Softmax for each row. *(1 mark)*
   - Row 1: $e^{3} = 20.086$, $e^{-2} = 0.135$, $e^{0} = 1$, $e^{1} = 2.718$. The sum is $23.939$, so $p = [0.839, 0.006, 0.042, 0.114]$.
   - Row 2: $e^{-1} = 0.368$, $e^{3} = 20.086$, $e^{1} = 2.718$, $e^{0} = 1$. The sum is $24.172$, so $p = [0.015, 0.831, 0.112, 0.041]$.
   - Check: each row sums to 1 (up to rounding).
4. Token 1 → **cup** (0.839). Token 2 → **plate** (0.831). *(1 mark)*

**Marking:** 1 for the shape check; 2 for $S$ (one per row); 1 for both softmax rows summing to 1; 1 for the predictions.

---

## D. Conceptual

**D1.** You can only add two matrices of the same shape. $X$ is $n \times d_{model}$, so SelfAttention(LN($X$)) must also be $n \times d_{model}$ for $X + \ldots$ to make sense; the same holds for the FFN. *(1 mark)* A block's output then feeds the next block as its input, so if one block changed the width, the next block's weights would no longer fit and the blocks couldn't be stacked freely. *(1 mark)* The output head is also built for width $d_{model}$, so it would break too. *(1 mark)*

**D2.** Very long sequences: a book, hours of audio, a long sensor recording. *(1 mark)* Attention's work grows with $n^2$, but an RNN's grows with $n$, and an RNN keeps a fixed-size hidden state rather than looking back at every earlier token. *(1 mark)* So for large $n$, RNN-like models (e.g. state-space models) can be much cheaper, which is why they are being revisited. *(1 mark)*

**D3.** **Same:** each word gets exactly the same output vector in both runs. *(1 mark)* **Different:** only the order of the output rows, which is reversed. *(1 mark)* **Problem:** meaning depends on order. "Dog bites man" and "man bites dog" contain the same words but mean different things, and the model couldn't tell them apart. Position information (Section 7) fixes this. *(1 mark)*

---

## E. Applied / coding

**E1.** *(5 marks: 1 embedding lookup, 2 blocks with residual, 1 output head, 1 correct list of shapes)*

```python
import numpy as np

def trace_shapes(ids: np.ndarray, d_model: int, vocab_size: int, N: int,
                 seed: int = 0) -> list[tuple[str, tuple[int, ...]]]:
    """Run a toy Transformer pipeline and record the shape after every stage."""
    rng = np.random.default_rng(seed)
    shapes = [("ids", ids.shape)]

    E = rng.normal(0, 0.1, size=(vocab_size, d_model))
    X = E[ids]                                  # pick one row per token
    shapes.append(("X0", X.shape))

    for b in range(1, N + 1):
        W = rng.normal(0, 0.1, size=(d_model, d_model))
        X = X + np.tanh(X @ W)                  # residual: shape must not change
        shapes.append((f"block {b}", X.shape))

    W_out = rng.normal(0, 0.1, size=(d_model, vocab_size))
    shapes.append(("scores", (X @ W_out).shape))
    return shapes

for name, shape in trace_shapes(np.array([3, 14, 15, 9, 2]), d_model=8, vocab_size=20, N=3):
    print(f"{name:8s} {shape}")
```

Output:

```
ids      (5,)
X0       (5, 8)
block 1  (5, 8)
block 2  (5, 8)
block 3  (5, 8)
scores   (5, 20)
```

**E2.** *(4 marks: 1 correct block, 2 a working permutation check, 1 explanation)*

```python
import numpy as np

def mix_block(X: np.ndarray) -> np.ndarray:
    """Each row becomes itself plus the mean of all rows."""
    return X + X.mean(axis=0, keepdims=True)

rng = np.random.default_rng(0)
X = rng.normal(size=(5, 4))
perm = rng.permutation(5)

print(np.allclose(mix_block(X[perm]), mix_block(X)[perm]))   # True
```

The mean of all rows doesn't depend on their order, so shuffling the input just shuffles the output rows in the same way. Self-attention behaves like this too: without position information, the model can't see word order.

---

## F. ★ Challenge

1. $n \cdot d_{model}^2 = n^2 \cdot d_{model}$. Divide both sides by $n \cdot d_{model}$ to get $n = d_{model}$. *(1 mark)*
2. At $n = 1{,}000$ with $d_{model} = 4096$, the ratio attention / RNN is $n / d_{model} = 1000/4096 \approx 0.24$. So attention is about **4.1× cheaper**. *(1 mark)* At $n = 32{,}000$ the ratio is $32000/4096 \approx 7.8$, so attention is about **7.8× more expensive**. *(1 mark)*
3. The RNN's work is split into $n$ steps that must run one after another, while the Transformer's work in each layer is one large matrix computation that runs in parallel across all tokens. GPUs are built for exactly that, so the Transformer usually finishes sooner even when it does more multiply-adds. *(1 mark)*
