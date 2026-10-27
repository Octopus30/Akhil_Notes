# Section 7 — Exercises: Positional Information

**Total: 48 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 55 minutes**

Positions count from 0. Use radians unless the question uses degrees. Round to 3 decimal places for position vectors and to 2 elsewhere.

---

## A. Warm-up (recall)

**A1.** Why does self-attention need position information at all? Answer in two sentences, using the result from Section 3. *(2 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. A model with learned absolute positions and $L_{max} = 1024$ can process a 1,500-token input without changes.
2. With RoPE, the score between a query at position $m$ and a key at position $n$ depends only on $q$, $k$ and $m - n$.
3. ALiBi adds a learned vector to every token embedding.

**A3.** Fill in. *(4 marks)*

1. The learned position table of a model with $L_{max} = 2048$ and $d_{model} = 1024$ has shape `__ × __` and `__` parameters.
2. For sinusoidal encodings with $d_{model} = 8$, there are `__` (sin, cos) pairs.
3. RoPE is applied to `__` and `__`, but not to values.

---

## B. Multiple choice (1 mark each)

**B1.** In sinusoidal encodings, which dimension pair changes fastest from one position to the next?
- (a) The last pair.
- (b) The first pair ($i = 0$).
- (c) All pairs change at the same speed.
- (d) The middle pair.

**B2.** What does moving forward $k$ positions do to each (sin, cos) pair of a sinusoidal encoding?
- (a) Multiplies it by $k$.
- (b) Rotates it by the angle $k\omega_i$, the same for every starting position.
- (c) Adds $k$ to both entries.
- (d) Nothing.

**B3.** Which scheme do most open LLMs today (LLaMA, Mistral, Qwen) use?
- (a) Learned absolute positions.
- (b) Sinusoidal positions.
- (c) RoPE.
- (d) No positions; the causal mask is enough.

**B4.** LLaMA-3 raised the RoPE base from 10,000 to 500,000. What is the effect?
- (a) The "clocks" turn faster, so nearby positions are easier to tell apart.
- (b) The clocks turn more slowly, so angles stay within the trained range for longer contexts.
- (c) It adds more parameters.
- (d) It turns RoPE into an absolute scheme.

**B5.** In ALiBi with slope $m > 0$, which tokens does a head prefer when the content scores are all equal?
- (a) The first token in the sequence.
- (b) Nearby tokens.
- (c) Far-away tokens.
- (d) All equally.

---

## C. Calculation

**C1. Sinusoidal table.** For $d_{model} = 6$: *(5 marks)*

1. Compute $\omega_0, \omega_1, \omega_2$.
2. Compute the full 6-dimensional encoding for positions 0, 1 and 2.
3. Which pairs would you rely on to tell position 1 from position 2? To tell position 10 from position 5,000?

**C2. Order now matters.** Use the toy embeddings a = [0.1, 0.1], light = [1, 1], bag = [1.5, 0] and $d_{model} = 2$, so $p_{pos} = [\sin pos, \cos pos]$. *(5 marks)*

1. Write $x'$ for each token in *"a light bag"* and in *"bag light a"*.
2. Without positions, naive attention gives light the output $[1.09, 0.57]$ in both orders. Explain why, without recomputing.
3. A computer gives light's outputs with positions as $[1.88, 1.30]$ and $[1.77, 1.42]$ for the two orders. What does this show?

**C3. RoPE in 2-D.** Rotate by $\theta = 45°$ per position. Let $q = [1, 0]$ and $k = [0, 1]$. *(5 marks)*

1. Compute the score for (query position, key position) = (1, 2) and (5, 6).
2. Compute the score for (2, 1) and (3, 3).
3. What do your answers show about RoPE? Is the score symmetric in $m - n$?

**C4. ALiBi.** The token at position 2 (causal: it sees 0, 1, 2) has content scores $[2, 1, 0]$ for $j = 0, 1, 2$. Slope $m = 0.25$. *(3 marks)*

1. Compute the biased scores and the attention weights.
2. Compare with the weights without ALiBi, and say which token lost the most.

---

## D. Conceptual

**D1.** A BERT model ($L_{max} = 512$) must summarise a 2,000-token document. Give two practical options, and explain why simply "extending the table" doesn't work out of the box. *(3 marks)*

**D2.** Explain why RoPE rotates only the queries and keys, and not the values. *(2 marks)*

**D3.** In a decoder with a causal mask and **no** position scheme, could the model still work out how far into the text a token is? Explain how, and why real models still add a position scheme. *(3 marks)*

---

## E. Applied / coding (NumPy)

**E1.** Implement *(4 marks)*

```python
def sinusoidal_positions(L: int, d_model: int) -> np.ndarray:
    """Return an L x d_model table; even columns sin, odd columns cos."""
```

Check it against the notes' $d_{model} = 4$ table, and against your answer to C1.

**E2.** Implement *(4 marks)*

```python
def rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Rotate each pair (2i, 2i+1) of every row of x (n x d) by positions[row] * base^(-2i/d)."""
```

Show numerically, with random $q$ and $k$ of width 8, that $\text{rope}(q, m) \cdot \text{rope}(k, n)$ is unchanged when both $m$ and $n$ are shifted by the same amount.

---

## F. ★ Challenge (bonus, 4 marks)

1. Let $R(\alpha)$ be the 2-D rotation matrix by angle $\alpha$. Using $R(\alpha)^{\top} = R(-\alpha)$ and $R(\alpha)R(\beta) = R(\alpha + \beta)$, show that $(R(m\theta)q^{\top})^{\top}(R(n\theta)k^{\top}) = q\, R((n - m)\theta)\, k^{\top}$.
2. A model was trained with RoPE on contexts up to 4,096 tokens, base 10,000 and $d_k = 128$. For the **slowest** pair ($i = 63$), what is the largest angle it saw during training (in radians)? Use $\theta_{63} = 10000^{-126/128}$.
3. With position interpolation by a factor of 4, what angle does position 16,000 get on that pair? Why does this help?
