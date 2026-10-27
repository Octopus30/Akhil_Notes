# Section 7 — Solutions: Positional Information

---

## A. Warm-up

**A1.** Self-attention is **order-blind**: shuffling the tokens only shuffles the outputs, so "dog bites man" and "man bites dog" get the same vectors for each word. *(1 mark)* Word order carries meaning, so the model needs some signal that tells positions apart. *(1 mark)*

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **False.** There is no row for positions 1,024 and beyond. The input must be cut down, or the table must be extended and fine-tuned.
2. **True.** Rotating $q$ by $m\theta$ and $k$ by $n\theta$ changes the angle between them by $(m - n)\theta$ only.
3. **False.** ALiBi adds a fixed, distance-based penalty $-m(i - j)$ to the attention **scores**. Nothing is added to the embeddings, and nothing is learned.

**A3.** *(1 mark each)*
1. `2048 × 1024`, with $2{,}097{,}152$ parameters. *(2 marks)*
2. 4 pairs.
3. Queries and keys.

---

## B. Multiple choice

**B1. (b).** $\omega_0 = 1$ radian per position; later pairs turn more and more slowly.
- (a) Wrong: the last pair is the slowest.
- (c) Wrong: the speeds $\omega_i$ fall geometrically with $i$.
- (d) Wrong: the middle pair has a middle speed, not the fastest.

**B2. (b).** $\sin(\omega(pos + k))$ and $\cos(\omega(pos + k))$ is the pair at $pos$ rotated by $k\omega$.
- (a) Wrong: the values stay between −1 and 1.
- (c) Wrong: for the same reason, nothing is added.
- (d) Wrong: the encoding does change with position.

**B3. (c).**
- (a) Learned absolute positions are BERT/GPT-2 style.
- (b) Sinusoidal positions were used by the original 2017 model.
- (d) Wrong: every production model adds a position scheme.

**B4. (b).** A larger base makes $\theta_i = \text{base}^{-2i/d}$ smaller, so the clocks turn more slowly.
- (a) Wrong: this is the opposite of what happens.
- (c) Wrong: RoPE has no parameters.
- (d) Wrong: the scores still depend only on $m - n$.

**B5. (b).** The penalty grows with distance.
- (a) Wrong: the first token is the farthest, so it gets the largest penalty.
- (c) Wrong: far tokens are penalised most.
- (d) Wrong: that would only be true with slope 0.

---

## C. Calculation

**C1.**
1. $\omega_i = 10000^{-2i/6}$: $\omega_0 = 1$, $\omega_1 = 10000^{-1/3} = 0.0464$, $\omega_2 = 10000^{-2/3} = 0.00215$. *(1 mark)*
2. *(3 marks, one per position)*

| pos | $\sin\omega_0 p$ | $\cos\omega_0 p$ | $\sin\omega_1 p$ | $\cos\omega_1 p$ | $\sin\omega_2 p$ | $\cos\omega_2 p$ |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| 1 | 0.841 | 0.540 | 0.046 | 0.999 | 0.002 | 1.000 |
| 2 | 0.909 | −0.416 | 0.093 | 0.996 | 0.004 | 1.000 |

3. Positions 1 and 2 are told apart by the **fast** pair 0, where the values change a lot. For 10 versus 5,000, the fast pair repeats every $2\pi \approx 6.3$ positions, so it is ambiguous, but the **slow** pair 2 has turned $0.00215 \times 5000 \approx 10.8$ radians versus $0.02$, so it separates far-apart positions. *(1 mark)*

**C2.**
1. $p_0 = [0, 1]$, $p_1 = [0.841, 0.540]$, $p_2 = [0.909, -0.416]$. *(2 marks)*

| *"a light bag"* | $x'$ | *"bag light a"* | $x'$ |
|---|---|---|---|
| a (0) | [0.10, 1.10] | bag (0) | [1.50, 1.00] |
| light (1) | [1.84, 1.54] | light (1) | [1.84, 1.54] |
| bag (2) | [2.41, −0.42] | a (2) | [1.01, −0.32] |

2. Without positions both sentences contain the same set of vectors, and light's output is a weighted average over that set, with weights that depend only on the vectors. The order doesn't enter anywhere, so the output is identical. *(1.5 marks)*
3. The outputs now differ, so the model can tell *"a light bag"* from *"bag light a"*. Even light itself, sitting at the same position 1 in both, is affected, because its neighbours' vectors changed. *(1.5 marks)*

**C3.** A score is $\lVert q\rVert\lVert k\rVert\cos(\text{angle between them})$. $q = [1, 0]$ points at 0° and $k = [0, 1]$ at 90°. After rotation the angle from $q$ to $k$ is $90° + n\theta - m\theta$.
1. *(2 marks)*
   - (1, 2): $90 + 90 - 45 = 135°$, so the score is $\cos 135° = -0.71$.
   - (5, 6): $90 + 270 - 225 = 135°$, so the score is again $-0.71$.
2. *(2 marks)*
   - (2, 1): $90 + 45 - 90 = 45°$, so the score is $+0.71$.
   - (3, 3): $90°$, so the score is $0$.
3. The score depends only on the gap $m - n$: the gaps $-1$ in (1, 2) and (5, 6) give the same score. It is **not symmetric** in the gap: a gap of $+1$ gives $+0.71$ and a gap of $-1$ gives $-0.71$. So RoPE can distinguish "the key is just before me" from "just after me". *(1 mark)*

**C4.**
1. The distances are $[2, 1, 0]$ and the biases are $[-0.5, -0.25, 0]$, so the biased scores are $[1.5, 0.75, 0]$. *(1 mark)* $e^{1.5} = 4.48$, $e^{0.75} = 2.12$, $e^{0} = 1$, with sum $7.60$, so the weights are $[0.59, 0.28, 0.13]$. *(1 mark)*
2. Without ALiBi, softmax$([2, 1, 0]) = [0.67, 0.24, 0.09]$. The far token at $j = 0$ lost the most (0.67 → 0.59); the near tokens gained. *(1 mark)*

---

## D. Conceptual

**D1.** *(1 mark each, any two options)*
- **Split into chunks** of at most 512 tokens (with some overlap), encode each one, and combine the chunk vectors, e.g. by averaging or with a second model.
- **Truncate**: keep the most relevant 512 tokens, for example the beginning and the end.
- **Switch** to a long-context encoder that uses RoPE or ALiBi.

**Why not just extend the table:** *(1 mark)* the new rows (512 and beyond) were never trained. They would be random vectors that the model has never seen, so their positions would carry nonsense. At the very least you would need to fine-tune on long inputs, or initialise the new rows sensibly, for example by copying or interpolating the old ones.

**D2.** Position matters for **deciding who attends to whom**, which happens only in the scores $q \cdot k$. *(1 mark)* The values are the content that gets mixed. Rotating them would scramble the information passed on and gain nothing, since the relative-position effect is already in the weights. *(1 mark)*

**D3.** **Yes, partly.** With a causal mask, token $i$ sees exactly $i + 1$ tokens. For example, if it attends uniformly, a constant feature gets averaged over $i + 1$ tokens, so the model can infer roughly how many tokens are visible, and hence its position. *(2 marks)* This signal is weak and imprecise: it is hard to tell positions 900 and 901 apart, and it gives no clean relative distance. An explicit scheme such as RoPE gives precise, relative information and helps a lot in practice. *(1 mark)*

---

## E. Applied / coding

**E1.** *(4 marks: 2 correct frequencies, 1 sin/cos interleaving, 1 checks)*

```python
import numpy as np

def sinusoidal_positions(L: int, d_model: int) -> np.ndarray:
    """Return an L x d_model table; even columns sin, odd columns cos."""
    pos = np.arange(L)[:, None]                          # L x 1
    i = np.arange(d_model // 2)[None, :]                 # 1 x d/2
    angles = pos / 10000 ** (2 * i / d_model)            # L x d/2
    P = np.zeros((L, d_model))
    P[:, 0::2] = np.sin(angles)
    P[:, 1::2] = np.cos(angles)
    return P

P4 = sinusoidal_positions(4, 4)
assert np.allclose(np.round(P4[3], 3), [0.141, -0.990, 0.030, 1.000])
print(np.round(sinusoidal_positions(3, 6), 3))
```

**E2.** *(4 marks: 2 rotation of the pairs, 1 frequencies, 1 shift check)*

```python
def rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Rotate each pair (2i, 2i+1) of every row of x (n x d) by positions[row] * base^(-2i/d)."""
    n, d = x.shape
    theta = base ** (-2 * np.arange(d // 2) / d)          # d/2
    ang = positions[:, None] * theta[None, :]            # n x d/2
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x, dtype=float)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos
    return out

rng = np.random.default_rng(0)
q, k = rng.normal(size=(1, 8)), rng.normal(size=(1, 8))
s1 = rope(q, np.array([3])) @ rope(k, np.array([7])).T
s2 = rope(q, np.array([103])) @ rope(k, np.array([107])).T
print(s1.item(), s2.item())
assert np.isclose(s1, s2)
```

---

## F. ★ Challenge

1. $(R(m\theta)q^{\top})^{\top}(R(n\theta)k^{\top}) = q\,R(m\theta)^{\top}R(n\theta)\,k^{\top} = q\,R(-m\theta)R(n\theta)\,k^{\top} = q\,R((n - m)\theta)\,k^{\top}$. This depends on the positions only through $n - m$. *(2 marks)*
2. $\theta_{63} = 10000^{-126/128} \approx 1.155 \times 10^{-4}$. The largest position seen was 4,095, so the largest angle was $4095 \times 1.155\times10^{-4} \approx$ **0.47 rad**. *(1 mark)*
3. Interpolated, position 16,000 acts like $16000/4 = 4000$, giving $4000 \times 1.155\times10^{-4} \approx$ **0.46 rad**, inside the trained range. Without interpolation it would be $\approx 1.85$ rad, an angle the slow pair never saw, which confuses the model. Interpolation keeps every pair's angles within the range seen in training, at the cost of packing nearby positions closer together. That is why a short fine-tune is still needed. *(1 mark)*
