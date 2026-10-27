# Section 17 — Exercises: Optimization & Quantization

**Total: 49 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 60 minutes**

Absmax quantization: $s = \max|w| / q_{\max}$ with $q_{\max} = 2^{b-1} - 1$, $q = \text{round}(w/s)$, $\hat w = sq$. Training memory per parameter: 2 (weights) + 2 (gradients) + 12 (FP32 master, $m$, $v$) bytes. Speculative decoding: $E = \frac{1 - \alpha^{k+1}}{1 - \alpha}$.

---

## A. Warm-up (recall)

**A1.** Give the sign, exponent and mantissa bits of FP16 and of BF16, and explain in one sentence each why BF16 is preferred for training. *(3 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. Gradient checkpointing reduces activation memory at the cost of extra computation.
2. Speculative decoding changes the output distribution slightly in exchange for speed.
3. ZeRO-3 shards the weights, the gradients and the optimizer states across GPUs.

**A3.** Fill in. *(3 marks)*

1. For INT4 absmax quantization, $q_{\max} = $ `__`.
2. The maximum rounding error per weight is `__` (in terms of $s$).
3. Plain data-parallel training with AdamW needs about `__` bytes per parameter on every GPU.

---

## B. Multiple choice (1 mark each)

**B1.** A gradient of $2 \times 10^{-8}$ is stored in FP16. What happens, and what is the standard fix?
- (a) It is stored exactly; no fix is needed.
- (b) It underflows to 0; multiply the loss by a scale factor before the backward pass (loss scaling).
- (c) It overflows; clip the gradients.
- (d) It becomes NaN; use INT8.

**B2.** Why do group-wise scales (e.g. one scale per 128 weights) help 4-bit quantization?
- (a) They make the model smaller than per-tensor scales.
- (b) An outlier only inflates the scale of its own group, so the other weights keep their precision.
- (c) They remove the need for rounding.
- (d) They allow FP32 arithmetic.

**B3.** Which is true of post-training quantization (PTQ) versus quantization-aware training (QAT)?
- (a) PTQ needs full retraining.
- (b) PTQ quantizes a finished model (e.g. GPTQ, AWQ); QAT simulates quantization during training and holds up better at very low bit widths.
- (c) QAT is always cheaper.
- (d) They are the same.

**B4.** Knowledge distillation trains the student to match:
- (a) Only the correct labels.
- (b) The teacher's (temperature-softened) output probabilities.
- (c) The teacher's weights directly.
- (d) Random noise.

**B5.** Which parallelism splits individual matrix multiplications across GPUs and needs very fast links?
- (a) Data parallelism
- (b) Pipeline parallelism
- (c) Tensor parallelism
- (d) Gradient accumulation

---

## C. Calculation

**C1. Formats.** *(3 marks)*

1. Which FP16 value is closest to 1.003? What does BF16 round it to?
2. An activation of 70,000: what happens in FP16 and in BF16?
3. A gradient of $2 \times 10^{-8}$ with a loss scale of 512: what value is stored in FP16?

**C2. INT8 and INT4.** Weights $w = [0.95, -0.35, 0.12, -1.8, 0.6, 0.05]$. *(6 marks)*

1. INT8: compute $s$, every $q$ and $\hat w$, and the mean absolute error.
2. INT4: the same.
3. How many times larger is the INT4 error? Which weights became 0?

**C3. An outlier.** Replace 0.05 by **12.0**. *(4 marks)*

1. Quantize all six to INT4 with one scale. How many weights are badly wrong?
2. Use groups of 3 instead. Give the scales and the dequantized values of each group.

**C4. Memory.** *(4 marks)*

1. Weights of a 13B model in FP16, INT8 and INT4 (ignore group-scale overhead).
2. Training that model on $G = 32$ GPUs: the per-GPU memory, before activations, for plain data parallel, ZeRO-1, ZeRO-2 and ZeRO-3.

**C5. Speculative decoding.** $k = 5$ drafted tokens, acceptance $\alpha = 0.7$, and one draft step costs 10% of a large-model step. Compute $E$, the cost per round, and the speed-up. *(3 marks)*

---

## D. Conceptual

**D1.** Explain why INT4 weights make decoding faster, not just smaller. *(2 marks)*

**D2.** Why does mixed-precision training keep an FP32 master copy of the weights? Give a numerical example. *(3 marks)*

**D3.** Gradient checkpointing costs about 33% more compute. Explain where that number comes from. *(2 marks)*

---

## E. Applied / coding (NumPy)

**E1.** Implement *(5 marks)*

```python
def quantize(w: np.ndarray, bits: int, group: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric absmax quantization. Returns (integer codes q, scales); one scale per group."""

def dequantize(q: np.ndarray, scales: np.ndarray, group: int | None = None) -> np.ndarray:
```

Check them against C2 and C3. Then quantize 10,000 Gaussian weights with 1% outliers (×20) to INT4, using one scale and then group sizes 128, 32 and 8, and print the mean absolute error for each.

**E2.** Simulate speculative decoding: for $k = 4$ and $\alpha = 0.8$, draw acceptances at random (stop at the first rejection, and the large model always adds one token of its own), and check that the average number of tokens per round is close to $E$. *(3 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

1. Derive $E = \frac{1 - \alpha^{k+1}}{1 - \alpha}$. (Hint: the number of tokens produced in a round is $1 +$ the number of drafted tokens accepted before the first rejection.)
2. With a draft cost $c$ per step (relative to the large model), the speed-up is $\frac{E}{1 + kc}$. For $\alpha = 0.8$ and $c = 0.05$, compute it for $k = 1, 2, 4, 8, 16$. Which $k$ is best?
3. Explain why the best $k$ is larger when $\alpha$ is higher.
