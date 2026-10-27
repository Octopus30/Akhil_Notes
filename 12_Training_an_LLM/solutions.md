# Section 12 — Solutions: Training an LLM

---

## A. Warm-up

**A1.** *(1 mark each)*
- **Pre-training:** next-token prediction on trillions of tokens of raw text (web, books, code).
- **SFT:** instruction–response conversations, with the loss on the responses only.
- **Preference tuning (RLHF or DPO):** prompts with a preferred and a rejected response.

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **False.** The prompt tokens are context only and are masked out of the loss.
2. **True.** A perplexity of 1 means a loss of 0, so $P = 1$ for every correct token.
3. **False.** DPO optimises directly on the preference pairs against a frozen reference model. There is no reward model and no RL loop.

**A3.** *(1 mark each)*
1. $\ln 32{,}000 = 10.37$, so the perplexity is 32,000.
2. $4 + 4 = 8$ bytes.
3. $2N$ forward and $4N$ backward.

---

## B. Multiple choice

**B1. (b).**
- (a) Wrong: warm-up doesn't change memory use.
- (c) Wrong: the loss is *large* at the start, not small.
- (d) Wrong: pre-training uses warm-up too.

**B2. (a).** Dividing by $1 - \beta^t$ rescales the averages at early $t$.
- (b) Wrong: large gradients are handled by gradient clipping.
- (c) Wrong: weight decay is a separate term.
- (d) Wrong: the learning rate is never negative.

**B3. (b).** Inference cost is paid on every request, so a smaller model trained for longer pays off.
- (a) Wrong: Chinchilla is about compute-optimal *training*, which is a different goal.
- (c) Wrong: more tokens don't change memory use.
- (d) Wrong: the tokenizer has nothing to do with it.

**B4. (b).**
- (a) Wrong: the penalty doesn't speed anything up.
- (c) Wrong: the vocabulary is unchanged.
- (d) Wrong: RLHF still needs the reward model.

**B5. (b).**
- (a) Wrong: a higher learning rate would make spikes worse.
- (c) and (d) are architecture changes, not a response to a spike in a run that is already going.

---

## C. Calculation

**C1.**
1. $-\ln p = [0.511, 1.204, 0.105, 2.996, 0.693]$. The sum is $5.509$, so $\mathcal{L} = 5.509/5 = 1.102$ and the perplexity is $e^{1.102} = 3.01$. *(3 marks)*
2. Position 4 ($p = 0.05$) contributes $2.996/5.509 = 54\%$ of the total. One confident mistake dominates. *(1 mark)*

**C2.**
1. Step 1: *(3 marks)*
   - $m_1 = 0.1(-0.2) = -0.02$ and $v_1 = 0.001(0.04) = 0.00004$.
   - $\hat m_1 = -0.02/0.1 = -0.2$ and $\hat v_1 = 0.00004/0.001 = 0.04$.
   - The Adam step is $0.01 \times (-0.2/0.2) = -0.01$, and the decay is $0.01 \times 0.1 \times 2.0 = 0.002$.
   - So $\theta_1 = 2.0 - (-0.01) - 0.002 = 2.008$.
2. Step 2: *(1 mark)*
   - $m_2 = 0.9(-0.02) + 0.1(-0.2) = -0.038$, so $\hat m_2 = -0.038/(1 - 0.81) = -0.2$.
   - $v_2 = 0.999(0.00004) + 0.001(0.04) = 0.00007996$, so $\hat v_2 = 0.00007996/(1 - 0.998001) = 0.04$.
   - The Adam step is $-0.01$ again, and the decay is $0.01 \times 0.1 \times 2.008 = 0.002008$.
   - So $\theta_2 = 2.008 + 0.01 - 0.002008 = 2.0160$.
3. With a constant gradient, the bias-corrected averages equal the gradient and its square exactly, so $\hat m/\sqrt{\hat v} = g/|g| = -1$. The Adam step is always $\eta$ in size, whatever the size of $g$. *(1 mark)*

**C3.** *(4 marks)*

| Step | Calculation | Learning rate |
|---|---|---|
| 500 | warm-up: $6\times10^{-4} \times 500/1000$ | $3.0 \times 10^{-4}$ |
| 1,000 | peak | $6.0 \times 10^{-4}$ |
| 25,500 | halfway through the decay, $\cos(\pi/2) = 0$, so $\frac{6\times10^{-4} + 6\times10^{-5}}{2}$ | $3.3 \times 10^{-4}$ |
| 37,750 | ¾ of the decay: $6\times10^{-5} + \tfrac12(5.4\times10^{-4})(1 + \cos\tfrac{3\pi}{4})$ | $1.39 \times 10^{-4}$ |
| 50,000 | end | $6.0 \times 10^{-5}$ |

**C4.** *(1 mark each)*
1. $C = 6 \times 13\times10^9 \times 10^{12} = 7.8 \times 10^{22}$ FLOPs.
2. $7.8\times10^{22} / 4\times10^{14} = 1.95 \times 10^8$ s, which is about **54,200 GPU-hours**. On 1,024 GPUs that is about 52.9 hours, or **2.2 days**.
3. $20 \times 13\times10^9 = 260$ B tokens. This run uses about 3.8× more.
4. $16 \times 13\times10^9 = 208$ GB, so at least **3** GPUs of 80 GB, before activations.

**C5.** The margin is $1.5 - (-0.5) = 2.0$, and $\beta \times 2.0 = 0.4$. Then $\sigma(0.4) = 0.599$, so $\mathcal{L} = -\ln 0.599 = 0.513$. *(1 mark)* That is lower than the starting value of $\ln 2 = 0.693$. *(1 mark)*

---

## D. Conceptual

**D1.** The prompt must stay in the input, because the response depends on it: the model needs it as context. *(1 mark)* We want the model to learn to *write good answers*, not to *imitate users' questions*. *(1 mark)* Training on the prompt tokens would spend capacity on predicting user text, and could make the model continue a question rather than answer it. *(1 mark)*

**D2.** Adam divides each parameter's step by $\sqrt{\hat v}$, which is about the typical size of *that parameter's* gradient. *(1 mark)* A layer with gradients 1,000× larger also has $\sqrt{\hat v}$ 1,000× larger, so its steps come out at the same scale, about $\eta$. *(1 mark)*

**D3.**
- **RLHF:** train a reward model on the preference pairs, then optimise the LLM with RL (PPO) against it, with a KL penalty to the SFT model. It needs the policy, a reference copy, the reward model and often a value model, all in memory. *(1 mark)*
- **DPO:** needs only the policy, a frozen reference and the preference pairs, with a simple classification-style loss. *(1 mark)*
- **Advantage of DPO:** it is much simpler and cheaper, with no RL loop and no reward model, so it is more stable and easier to tune. *(1 mark)*

---

## E. Applied / coding

**E1.** *(5 marks: 1 moments, 1 bias correction, 1 decoupled decay, 1 checks, 1 minimisation)*

```python
import numpy as np

def adamw_step(theta, g, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
    """One AdamW update; returns (theta, m, v)."""
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    theta = theta - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * theta)
    return theta, m, v

th, m, v = adamw_step(1.0, 0.5, 0.0, 0.0, 1, lr=0.1, wd=0.01)
print(round(th, 3))                                   # 0.899 (notes)

th, m, v = 2.0, 0.0, 0.0
for t in (1, 2):
    th, m, v = adamw_step(th, -0.2, m, v, t, lr=0.01, wd=0.1)
    print(t, round(th, 4))                            # 2.008, 2.016

th, m, v = 0.0, 0.0, 0.0
for t in range(1, 201):
    th, m, v = adamw_step(th, 2 * (th - 3), m, v, t, lr=0.1)
print("minimum found at", round(th, 3))               # close to 3
```

**E2.** *(3 marks)*

```python
def lr_schedule(step, peak, warmup, total, min_lr):
    """Linear warm-up, then cosine decay from peak to min_lr."""
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (peak - min_lr) * (1 + np.cos(np.pi * progress))

for s in (500, 1000, 25500, 37750, 50000):
    print(s, f"{lr_schedule(s, 6e-4, 1000, 50000, 6e-5):.3g}")
```

---

## F. ★ Challenge

1. $C = 6N(20N) = 120N^2$, so $N = \sqrt{C/120}$ and $D = 20\sqrt{C/120}$. *(1 mark)*
2. $N = \sqrt{10^{24}/120} \approx 9.1 \times 10^{10}$, about **91B** parameters, and $D \approx 1.8 \times 10^{12}$, about **1.8T** tokens. *(1 mark)*
3. With $N' = N/4 \approx 22.8$B, $D' = C/(6N') = 4D \approx 7.3$T tokens, which is $7.3\text{T}/22.8\text{B} \approx$ **320 tokens per parameter**. *(1 mark)* The reason: the final loss is a little worse than the compute-optimal model's, but the model is 4× cheaper and faster to serve. When it will answer billions of requests, the inference savings outweigh the extra training cost. *(1 mark)*
