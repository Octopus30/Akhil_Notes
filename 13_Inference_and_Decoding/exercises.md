# Section 13 — Exercises: Inference & Decoding

**Total: 49 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 55 minutes**

Order of operations: logits → penalties → temperature → truncation → renormalize → sample. Round probabilities to 3 decimal places.

---

## A. Warm-up (recall)

**A1.** Describe the prefill and decode phases, and say which one determines the time to first token (TTFT). *(3 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. Raising the temperature can change which token is the most likely.
2. Top-p keeps the same number of tokens at every step.
3. Beam search is the usual choice for open-ended chat.

**A3.** Fill in. *(3 marks)*

1. Temperature $T \to 0$ is equivalent to `__` decoding.
2. Min-p 0.1 with $p_{\max} = 0.6$ keeps tokens with $p \ge$ `__`.
3. To force valid JSON, set the logits of invalid tokens to `__`.

---

## B. Multiple choice (1 mark each)

**B1.** Why truncate the distribution (top-k, top-p, min-p) before sampling?
- (a) To make the computation faster.
- (b) To avoid occasionally sampling a very unlikely token from the long tail that derails the answer.
- (c) To make the output deterministic.
- (d) To reduce the vocabulary size permanently.

**B2.** Why does plain beam search favour short outputs?
- (a) Short outputs have fewer tokens to cache.
- (b) Every extra token adds a negative log-probability, so longer sequences have lower total scores.
- (c) Beams are pruned by length.
- (d) `<eos>` always has high probability.

**B3.** Which setting suits extracting dates from invoices into a fixed format?
- (a) Temperature 1.2, top-p 0.95.
- (b) Temperature 0 to 0.2, ideally with constrained decoding.
- (c) Beam search with $B = 50$.
- (d) Temperature 2.0 with min-p.

**B4.** A strong repetition penalty in a coding assistant might:
- (a) Make it write more variable names correctly.
- (b) Stop it from reusing variable names it legitimately needs.
- (c) Have no effect on code.
- (d) Make greedy decoding random.

**B5.** Why are chatbot answers streamed token by token?
- (a) Because the network is slow.
- (b) Decoding takes most of the total time, so streaming lets users start reading right after the first token arrives.
- (c) Because the model can only output one token per answer.
- (d) To save GPU memory.

---

## C. Calculation

Logits for the next word after *"After the exam, I felt…"*:

| relieved | tired | happy | nervous | hungry | purple |
|---|---|---|---|---|---|
| 2.0 | 1.0 | 0.5 | 0.0 | −0.5 | −3.0 |

**C1. Temperature.** *(5 marks)*

1. Compute the probabilities at $T = 1$, showing the exponentials and their sum.
2. Compute the probability of "relieved" at $T = 0.5$, and of "purple" at $T = 2$.
3. Does the ranking change at any temperature? Why or why not?

**C2. Truncation at $T = 1$.** *(5 marks)*

1. Top-k with $k = 2$: which tokens remain, and what are their renormalized probabilities?
2. Top-p with $p = 0.8$: how many tokens are kept? Give their renormalized probabilities.
3. Min-p 0.2: what is the threshold, and which tokens are kept?

**C3. Beam search.** First word: go 0.6, stay 0.4. The second word depends on the first: *(4 marks)*

| After… | home | out | now | up |
|---|---|---|---|---|
| go | 0.3 | 0.3 | 0.4 | — |
| stay | 0.8 | — | — | 0.2 |

1. What does greedy decoding produce, and with what probability?
2. Run beam search with $B = 2$. Which sequence wins, and with what probability?
3. Give both answers as log-probabilities.

**C4. Presence penalty.** "relieved" has already appeared, and the presence penalty is 1.0. Compute the new $T = 1$ probabilities. *(2 marks)*

**C5. Latency.** A 2,000-token prompt, a 300-token answer; the server prefills 8,000 tokens/s and decodes at 20 ms per token. Compute the TTFT, the decode time and the total. *(2 marks)*

---

## D. Conceptual

**D1.** Explain why top-p adapts to the model's confidence, using one "confident" and one "unsure" distribution as examples. *(3 marks)*

**D2.** Greedy decoding sometimes falls into loops ("I think that I think that…"). Why, and name two ways to reduce it. *(3 marks)*

**D3.** Why is beam search good for translation but poor for storytelling? *(2 marks)*

---

## E. Applied / coding (NumPy)

**E1.** Implement *(6 marks)*

```python
def sample_next(logits: np.ndarray, rng, temperature: float = 1.0, top_k: int | None = None,
                top_p: float | None = None, min_p: float | None = None,
                seen: set[int] = frozenset(), presence: float = 0.0) -> int:
```

Apply the operations in the order given at the top of this sheet. `temperature = 0` means greedy. Then sample 10,000 times from the C1 logits with `top_p = 0.8`, and check that the empirical frequencies match your C2 answer.

**E2.** Implement `beam_search(first, second, B)` for two-step tables like C3 (dictionaries of probabilities). Return the best sequence and its probability. Check it against C3. *(3 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

1. Show that for any two tokens $i, j$, $\frac{p_i(T)}{p_j(T)} = e^{(z_i - z_j)/T}$. Use it to explain what $T \to 0$ and $T \to \infty$ do.
2. The **entropy** of a distribution is $H = -\sum_i p_i \ln p_i$. Compute $H$ for the C1 distribution at $T = 0.5$, $1$ and $2$ (using your rounded probabilities). How does $H$ change with $T$?
3. Why might a system raise the temperature but also apply min-p, rather than raise the temperature alone?
