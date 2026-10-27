# Section 17 — Solutions: Optimization & Quantization

---

## A. Warm-up

**A1.** FP16 is 1 / 5 / 10 bits, and BF16 is 1 / 8 / 7. *(1 mark)* BF16 has the same 8 exponent bits as FP32, so it has the **same range**, and gradients and activations rarely overflow or underflow. *(1 mark)* It gives up precision, which training tolerates, and it doesn't need loss scaling. *(1 mark)*

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **True.** It stores activations only at checkpoints and recomputes the rest during the backward pass.
2. **False.** The verification step makes the output distribution **exactly** the large model's.
3. **True.** Each GPU keeps only $16/G$ bytes per parameter.

**A3.** *(1 mark each)* 7; $s/2$; 16.

---

## B. Multiple choice

**B1. (b).** FP16's smallest positive value is about $6 \times 10^{-8}$, so $2 \times 10^{-8}$ underflows to 0.
- (a) Wrong: the value can't be stored.
- (c) Wrong: it underflows, it doesn't overflow.
- (d) Wrong: it gives 0, not NaN.

**B2. (b).**
- (a) Wrong: extra scales make the model slightly *larger*.
- (c) Wrong: rounding is still needed.
- (d) Wrong: group scales have nothing to do with FP32 arithmetic.

**B3. (b).**
- (a) Wrong: PTQ needs no retraining.
- (c) Wrong: QAT involves training, so it costs more.
- (d) Wrong: they are different methods.

**B4. (b).**
- (a) Wrong: the softened probabilities carry the "dark knowledge" that labels alone don't.
- (c) Wrong: the student has different weights and usually a different shape.
- (d) Wrong: noise teaches nothing.

**B5. (c).**
- (a) Data parallelism copies the whole model to every GPU.
- (b) Pipeline parallelism splits by layers.
- (d) Gradient accumulation is a single-GPU memory trick.

---

## C. Calculation

**C1.** *(1 mark each)*
1. The FP16 step near 1 is $2^{-10} = 0.000977$, so the nearest value is $1 + 3 \times 2^{-10} = 1.00293$. BF16's step is $2^{-7} = 0.0078$, so 1.003 rounds to **1.0**.
2. FP16 **overflows** to $\infty$, since its maximum is 65,504. BF16 stores it fine, rounded to the nearest representable value, 70,144.
3. $2 \times 10^{-8} \times 512 = 1.024 \times 10^{-5}$, which is representable: FP16 stores $\approx 1.025 \times 10^{-5}$. It is divided by 512 after the backward pass, in FP32.

**C2.**
1. INT8: $s = 1.8/127 = 0.01417$. *(3 marks)*

| $w$ | $w/s$ | $q$ | $\hat w$ | error |
|---|---|---|---|---|
| 0.95 | 67.03 | 67 | 0.9496 | 0.0004 |
| −0.35 | −24.69 | −25 | −0.3543 | 0.0043 |
| 0.12 | 8.47 | 8 | 0.1134 | 0.0066 |
| −1.8 | −127 | −127 | −1.8 | 0 |
| 0.6 | 42.33 | 42 | 0.5953 | 0.0047 |
| 0.05 | 3.53 | 4 | 0.0567 | 0.0067 |

   Mean absolute error **0.0038**.

2. INT4: $s = 1.8/7 = 0.2571$. *(2 marks)*

| $w$ | $w/s$ | $q$ | $\hat w$ | error |
|---|---|---|---|---|
| 0.95 | 3.69 | 4 | 1.0286 | 0.0786 |
| −0.35 | −1.36 | −1 | −0.2571 | 0.0929 |
| 0.12 | 0.47 | 0 | 0 | 0.12 |
| −1.8 | −7 | −7 | −1.8 | 0 |
| 0.6 | 2.33 | 2 | 0.5143 | 0.0857 |
| 0.05 | 0.19 | 0 | 0 | 0.05 |

   Mean absolute error **0.0712**.

3. The INT4 error is $0.0712/0.0038 \approx$ **19×** larger. 0.12 and 0.05 became 0. *(1 mark)*

**C3.**
1. With one scale, $s = 12/7 = 1.714$. The codes are $[1, 0, 0, -1, 0, 7]$, so the dequantized values are $[1.71, 0, 0, -1.71, 0, 12]$. **Five of six** are badly wrong: three became 0, and 0.95 became 1.71. Only the outlier is exact. *(2 marks)*
2. *(2 marks)*
   - Group 1 $[0.95, -0.35, 0.12]$: $s = 0.95/7 = 0.1357$, codes $[7, -3, 1]$, values $[0.95, -0.41, 0.14]$. This group is good again.
   - Group 2 $[-1.8, 0.6, 12]$: $s = 1.714$, codes $[-1, 0, 7]$, values $[-1.71, 0, 12]$. It is still damaged by the outlier, but the damage stays in that group.

**C4.**
1. FP16: 26 GB. INT8: 13 GB. INT4: 6.5 GB. *(1 mark)*
2. *(3 marks)*

| Setup | Bytes per parameter per GPU | Per GPU |
|---|---|---|
| Plain data parallel | 16 | 208 GB |
| ZeRO-1 | $4 + 12/32 = 4.375$ | 56.9 GB |
| ZeRO-2 | $2 + 14/32 = 2.4375$ | 31.7 GB |
| ZeRO-3 | $16/32 = 0.5$ | 6.5 GB |

**C5.** $E = (1 - 0.7^6)/0.3 = (1 - 0.1176)/0.3 = 2.94$ tokens per round. *(1 mark)* The cost per round is $1 + 5 \times 0.1 = 1.5$ large-model steps. *(1 mark)* The speed-up is $2.94/1.5 \approx$ **1.96×**. *(1 mark)*

---

## D. Conceptual

**D1.** Decoding is **memory-bandwidth bound**: each token requires reading every weight once (Section 14). *(1 mark)* 4-bit weights are a quarter of the bytes of 16-bit weights, so each step reads about 4× less data. Allowing for dequantization overhead, generation is typically 2–3× faster. *(1 mark)*

**D2.** The updates $\eta \cdot \text{step}$ are often far smaller than the weight itself. In a low-precision format they would round away to nothing. *(1 mark)* **Example:** a weight of 1.0 in BF16 with an update of $10^{-4}$. BF16's step near 1 is 0.0078, so $1.0 + 0.0001$ rounds back to 1.0, and the update is lost, at every step, forever. *(1 mark)* An FP32 master copy (step about $10^{-7}$) accumulates these small updates; a BF16 copy is made from it for the fast matrix multiplications. *(1 mark)*

**D3.** A training step costs about 1 forward pass plus a backward pass worth about 2 forwards, so 3 units. *(1 mark)* Checkpointing recomputes the forward pass inside each segment during the backward pass, roughly one extra forward. That makes $4/3$ of the original cost, i.e. about 33% more. *(1 mark)*

---

## E. Applied / coding

**E1.** *(5 marks: 2 quantize, 1 dequantize, 1 checks, 1 group-size experiment)*

```python
import numpy as np

def quantize(w: np.ndarray, bits: int, group: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric absmax quantization. Returns (integer codes q, scales); one scale per group."""
    qmax = 2 ** (bits - 1) - 1
    g = w.reshape(-1, group if group else w.size)               # rows = groups
    scales = np.abs(g).max(axis=1, keepdims=True) / qmax
    scales[scales == 0] = 1.0                                   # all-zero group: any scale works
    q = np.clip(np.round(g / scales), -qmax, qmax).astype(np.int8)
    return q, scales

def dequantize(q: np.ndarray, scales: np.ndarray, group: int | None = None) -> np.ndarray:
    """w_hat = s * q, flattened back to the original 1-D order."""
    return (q * scales).reshape(-1)

w = np.array([0.95, -0.35, 0.12, -1.8, 0.6, 0.05])
for bits in (8, 4):
    q, s = quantize(w, bits)
    print(bits, q.ravel(), np.round(dequantize(q, s), 4), round(float(np.abs(dequantize(q, s) - w).mean()), 4))

w_out = w.copy()
w_out[5] = 12.0
print(np.round(dequantize(*quantize(w_out, 4)), 2))           # one scale
print(np.round(dequantize(*quantize(w_out, 4, group=3)), 2))  # groups of 3

rng = np.random.default_rng(0)
big = rng.normal(0, 0.02, 10_240)
big[rng.choice(big.size, 100, replace=False)] *= 20             # 1% outliers
for group in (None, 128, 32, 8):
    q, s = quantize(big, 4, group)
    print(f"group {str(group):>4}: mean |error| {np.abs(dequantize(q, s) - big).mean():.5f}")
```

The error shrinks steadily as the groups get smaller. (The weight count, 10,240, is chosen to be divisible by every group size.)

**E2.** *(3 marks)*

```python
def tokens_per_round(k: int, alpha: float, rng) -> int:
    """Accepted drafted tokens before the first rejection, plus one token from the large model."""
    accepted = 0
    while accepted < k and rng.random() < alpha:
        accepted += 1
    return accepted + 1

sims = [tokens_per_round(4, 0.8, rng) for _ in range(100_000)]
print(f"simulated {np.mean(sims):.3f}   formula {(1 - 0.8**5) / 0.2:.3f}")   # both about 3.36
```

---

## F. ★ Challenge

1. Let $X$ be the number of drafted tokens accepted before the first rejection, capped at $k$. Then $P(X \ge j) = \alpha^j$ for $j = 1..k$. The tokens per round are $1 + X$, so *(2 marks)*
$$E = 1 + \sum_{j=1}^{k} P(X \ge j) = \sum_{j=0}^{k} \alpha^j = \frac{1 - \alpha^{k+1}}{1 - \alpha}.$$
2. *(1 mark)*

| $k$ | $E$ | Speed-up $E/(1 + 0.05k)$ |
|---|---|---|
| 1 | 1.80 | 1.71 |
| 2 | 2.44 | 2.22 |
| 4 | 3.36 | 2.80 |
| 8 | 4.33 | **3.09** |
| 16 | 4.89 | 2.72 |

   $k = 8$ is the best of these.
3. Each extra drafted token helps only if every earlier one was accepted, which has probability $\alpha^j$. With a high $\alpha$, later drafts still have a good chance of being used, so their cost $c$ is worth paying for longer. With a low $\alpha$, the chain usually breaks early, and extra drafts are wasted work. *(1 mark)*
