# Section 13 — Solutions: Inference & Decoding

---

## A. Warm-up

**A1.** **Prefill** processes the whole prompt in one parallel pass and stores its keys and values; it determines the **TTFT**. *(1.5 marks)* **Decode** then produces one token per forward pass (logits, then the decoding strategy, then append) until a stop condition. *(1.5 marks)*

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **False.** Dividing every logit by the same $T > 0$ keeps their order, so the ranking never changes.
2. **False.** Top-p keeps the smallest set that reaches cumulative probability $p$: few tokens when the model is confident, many when it is unsure.
3. **False.** Beam search tends to give bland, repetitive text. Chat uses sampling, while beam search suits translation and speech-to-text.

**A3.** *(1 mark each)* greedy; $0.1 \times 0.6 = 0.06$; $-\infty$.

---

## B. Multiple choice

**B1. (b).**
- (a) Wrong: truncation saves essentially no compute.
- (c) Wrong: sampling stays random.
- (d) Wrong: the cut is made fresh at each step and changes nothing permanently.

**B2. (b).**
- (a) Wrong: the KV cache has nothing to do with the score.
- (c) Wrong: beams are pruned by score, not by length.
- (d) Wrong: `<eos>` has no guaranteed high probability.

**B3. (b).** There is one correct answer and the format must be valid.
- (a) and (d) add randomness, which is harmful here.
- (c) Wrong: a huge beam is slow and unnecessary.

**B4. (b).** Code legitimately repeats identifiers.
- (a) Wrong: the effect is the opposite.
- (c) Wrong: the penalty does affect code.
- (d) Wrong: penalties change the logits, not whether the choice is random.

**B5. (b).**
- (a) Wrong: the network isn't the bottleneck.
- (c) Wrong: an answer has many tokens.
- (d) Wrong: streaming doesn't change memory use.

---

## C. Calculation

**C1.**
1. The exponentials are $e^{2} = 7.389$, $e^{1} = 2.718$, $e^{0.5} = 1.649$, $e^{0} = 1$, $e^{-0.5} = 0.607$, $e^{-3} = 0.050$, with sum $13.412$. *(2 marks)*

| relieved | tired | happy | nervous | hungry | purple |
|---|---|---|---|---|---|
| 0.551 | 0.203 | 0.123 | 0.075 | 0.045 | 0.004 |

2. *(2 marks)*
   - $T = 0.5$: the logits double to $[4, 2, 1, 0, -1, -6]$. The exponentials are $54.60, 7.39, 2.72, 1, 0.37, 0.002$, with sum $66.08$, so "relieved" gets $54.60/66.08 = 0.826$.
   - $T = 2$: the logits halve to $[1, 0.5, 0.25, 0, -0.25, -1.5]$. The exponentials sum to $7.653$, so "purple" gets $e^{-1.5}/7.653 = 0.223/7.653 = 0.029$, seven times its share at $T = 1$.
3. **No.** Dividing by $T > 0$ keeps the order of the logits, so the ranking is the same at every temperature. *(1 mark)*

**C2.**
1. Top-2 keeps relieved and tired, which total $0.754$. Renormalized: $0.551/0.754 = 0.731$ and $0.203/0.754 = 0.269$. *(1.5 marks)*
2. The cumulative sums are $0.551$, $0.754$, then $0.877 \ge 0.8$, so **3** tokens are kept. Dividing by $0.877$ gives relieved $0.629$, tired $0.231$, happy $0.140$. *(2 marks)*
3. The threshold is $0.2 \times 0.551 = 0.110$. relieved, tired and happy (0.123) are kept; nervous (0.075), hungry and purple are dropped. Here it gives the same set as top-p. *(1.5 marks)*

**C3.**
1. Greedy picks go (0.6), then now (0.4): "go now" with probability $0.24$. *(1 mark)*
2. Step 1 keeps go (0.6) and stay (0.4). Step 2 extends them: go home 0.18, go out 0.18, go now 0.24, stay home **0.32**, stay up 0.08. The best two are "stay home" (0.32) and "go now" (0.24). **"stay home"** wins with probability 0.32. *(2 marks)*
3. $\ln 0.32 = -1.14$ beats $\ln 0.24 = -1.43$. *(1 mark)*

**C4.** The new logits are $[1.0, 1.0, 0.5, 0, -0.5, -3]$. The exponentials are $2.718, 2.718, 1.649, 1, 0.607, 0.050$, with sum $8.742$. The probabilities are relieved $0.311$, tired $0.311$, happy $0.189$, nervous $0.114$, hungry $0.069$, purple $0.006$. *(2 marks)*

**C5.** TTFT $= 2000/8000 = 0.25$ s. Decode $= 300 \times 0.02 = 6.0$ s. Total about **6.25 s**. *(2 marks)*

---

## D. Conceptual

**D1.** Top-p keeps however many tokens it takes to reach $p$. *(1 mark)*
- **Confident:** with $[0.95, 0.03, 0.01, \ldots]$ and $p = 0.9$, it keeps **1** token, effectively greedy. *(1 mark)*
- **Unsure:** with many tokens near 0.05 each, it keeps about 18 of them to reach 0.9, allowing variety exactly where several continuations are reasonable. *(1 mark)*

A fixed top-k would keep too many tokens in the first case and too few in the second.

**D2.** Once a phrase has appeared, it becomes very likely to appear again: the context now contains it, and models learn that text often repeats. Greedy always takes the top token, so it can lock into a cycle. *(1 mark)* *(1 mark each, any two)*
- Sample with temperature and top-p instead of greedy.
- Add presence, frequency or repetition penalties.
- Use stop sequences or maximum-length limits.

**D3.** In translation there is roughly **one right answer**, the most probable sequence, and beam search finds sequences of higher probability than greedy does. *(1 mark)* Good stories need surprise and variety. The most probable continuation is bland and generic, and beam search pushes exactly towards it. *(1 mark)*

---

## E. Applied / coding

**E1.** *(6 marks: 1 penalty, 1 temperature and greedy, 1 top-k, 1 top-p, 1 min-p, 1 empirical check)*

```python
import numpy as np

def sample_next(logits: np.ndarray, rng, temperature: float = 1.0, top_k: int | None = None,
                top_p: float | None = None, min_p: float | None = None,
                seen: set[int] = frozenset(), presence: float = 0.0) -> int:
    """Penalties -> temperature -> truncation -> renormalize -> sample."""
    z = logits.astype(float).copy()
    for i in seen:
        z[i] -= presence
    if temperature == 0:
        return int(np.argmax(z))
    z = z / temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    keep = np.ones_like(p, dtype=bool)
    order = np.argsort(-p)
    if top_k is not None:
        keep[order[top_k:]] = False
    if top_p is not None:
        cum = np.cumsum(p[order])
        cut = np.searchsorted(cum, top_p) + 1          # smallest prefix reaching top_p
        keep[order[cut:]] = False
    if min_p is not None:
        keep &= p >= min_p * p.max()
    p = np.where(keep, p, 0.0)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))

words = ["relieved", "tired", "happy", "nervous", "hungry", "purple"]
z = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -3.0])
rng = np.random.default_rng(0)
draws = np.array([sample_next(z, rng, top_p=0.8) for _ in range(10_000)])
print({w: round(float((draws == i).mean()), 3) for i, w in enumerate(words)})
```

The frequencies come out close to 0.629, 0.231 and 0.140, with 0 for the other three words.

**E2.** *(3 marks)*

```python
def beam_search(first: dict, second: dict, B: int) -> tuple[list[str], float]:
    """Two-step beam search over probability tables; returns the best sequence and its probability."""
    beams = sorted(first.items(), key=lambda kv: -kv[1])[:B]
    candidates = [([w1, w2], p1 * p2) for w1, p1 in beams for w2, p2 in second[w1].items()]
    candidates.sort(key=lambda c: -c[1])
    best, prob = candidates[0]
    return best, round(prob, 4)

first = {"go": 0.6, "stay": 0.4}
second = {"go": {"home": 0.3, "out": 0.3, "now": 0.4}, "stay": {"home": 0.8, "up": 0.2}}
print(beam_search(first, second, B=1))   # (['go', 'now'], 0.24)  = greedy
print(beam_search(first, second, B=2))   # (['stay', 'home'], 0.32)
```

---

## F. ★ Challenge

1. $\frac{p_i(T)}{p_j(T)} = \frac{e^{z_i/T}}{e^{z_j/T}} = e^{(z_i - z_j)/T}$, because the denominators cancel. *(1 mark)*
   - As $T \to 0$, the exponent goes to $\pm\infty$ for any gap, so the top token takes all the mass, which is greedy.
   - As $T \to \infty$, the exponent goes to 0, so every ratio goes to 1 and the distribution becomes uniform.
2. *(2 marks)*
   - $T = 0.5$: $H \approx 0.63$ nats.
   - $T = 1$: $H \approx 1.27$.
   - $T = 2$: $H \approx 1.60$.
   - The entropy **rises** with $T$, because the distribution spreads out. The maximum possible is $\ln 6 = 1.79$, the uniform distribution.
3. A higher temperature lifts the middle tokens, which gives creativity, but it also lifts the junk tail ("purple"). Min-p cuts the tail relative to the top token's probability, so the system gets more variety among sensible words without nonsense. *(1 mark)*
