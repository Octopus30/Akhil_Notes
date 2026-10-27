# Section 14 — Exercises: KV Cache & Fast Attention

**Total: 48 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 55 minutes**

Cache bytes $= 2 \times \text{layers} \times \text{KV heads} \times d_{\text{head}} \times \text{bytes per number} \times \text{tokens} \times \text{batch}$. Use 1 GiB $= 2^{30}$ bytes and 1 MiB $= 2^{20}$ bytes. Counting convention: without a cache, decode pass $k$ processes $p + k - 1$ tokens.

---

## A. Warm-up (recall)

**A1.** Why can keys and values be cached during generation, but queries are not? *(3 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. With a KV cache, the output tokens differ slightly from those of recomputing everything.
2. FlashAttention is an approximation of attention that drops small scores.
3. At batch size 1, decoding speed is limited mainly by memory bandwidth, not arithmetic.

**A3.** Fill in. *(3 marks)*

1. Each decode step computes new $q$, $k$, $v$ for `__` token(s).
2. The KV cache grows `__` with context length (linearly / quadratically).
3. FlashAttention's memory use grows like `__` instead of $n^2$.

---

## B. Multiple choice (1 mark each)

**B1.** Which technique lets several query heads share one key/value head?
- (a) PagedAttention
- (b) GQA / MQA
- (c) Prefix caching
- (d) FlashAttention

**B2.** What does PagedAttention mainly improve?
- (a) The accuracy of attention.
- (b) Memory fragmentation, so more users' caches fit on one GPU.
- (c) The speed of the softmax.
- (d) The size of the model weights.

**B3.** In the online softmax, when a new chunk has a larger maximum $m' > m$, the old totals are:
- (a) Discarded.
- (b) Multiplied by $e^{m - m'}$.
- (c) Divided by $m'$.
- (d) Left unchanged.

**B4.** Why does batching many users raise throughput during decoding?
- (a) It reduces the size of each user's cache.
- (b) Each read of the weights from memory serves many users' tokens at once.
- (c) It makes the model smaller.
- (d) It removes the need for a KV cache.

**B5.** A chatbot sends the same 2,000-token system prompt with every request. Which technique saves the most prefill work?
- (a) Sliding-window attention
- (b) Prefix caching
- (c) A higher temperature
- (d) MQA

---

## C. Calculation

**C1. Work saved.** A prompt of 8 tokens; the model generates 4 tokens. *(3 marks)*

1. How many K/V computations (token positions processed) are needed without a cache? With one?
2. Give the general formulas for a prompt of length $p$ and $T$ generated tokens.

**C2. Cache size.** A model has 40 layers, 40 KV heads (MHA), $d_{\text{head}} = 128$ and 16-bit numbers. *(5 marks)*

1. Compute the cache per token, in bytes and in MiB.
2. Compute it for one user at 8,192 tokens, in GiB.
3. Compute it for 16 users at 8,192 tokens.
4. A GQA version has 8 KV heads. Compute the cache for one user at 8,192 tokens.

**C3. Decode speed bound.** The same model has 13B parameters in 16-bit, on a GPU with 3.35 TB/s of memory bandwidth. *(3 marks)*

1. Find the minimum time per token at batch size 1 (ignore the cache) and the maximum tokens per second.
2. What would 8-bit weights give?

**C4. Online softmax by hand.** One query, scores $s = [1, 2, 4, 3]$, scalar values $v = [2, 0, 1, 5]$. Process them in two chunks of two. *(5 marks)*

1. Chunk 1: compute $m$, $\ell$ and $o$.
2. Chunk 2: rescale, add the new terms, and give the final output $o/\ell$.
3. Check your answer with an ordinary softmax over all four scores.

**C5. Score-matrix memory.** For $n = 16{,}384$ and 16-bit numbers, how much memory does one head's score matrix need? How much do 32 heads need? *(2 marks)*

---

## D. Conceptual

**D1.** Explain why decoding at batch size 1 is memory-bound while prefill is compute-bound. *(3 marks)*

**D2.** Sliding-window attention caps the cache at $W$ tokens. What does a model give up, and how can stacking layers partly recover it? *(3 marks)*

**D3.** Why does FlashAttention run faster than standard attention even though it does the same arithmetic (or even slightly more)? *(2 marks)*

---

## E. Applied / coding (NumPy)

**E1.** Implement a single-head causal attention layer in two ways: *(5 marks)*

- `full_recompute(X, W_Q, W_K, W_V)`: for each step $t$, recompute attention over `X[:t+1]` and keep the last row;
- a class `KVCacheAttention` with a method `step(x_t)` that projects only the new token, appends $k_t$ and $v_t$ to the cache, and attends with one query.

Assert that both give the same outputs for random $X$ of shape $10 \times 8$.

**E2.** Implement `online_softmax_attention(s, v, chunk)` for one query row, using the running $m$, $\ell$, $o$. Check it against C4, and against an ordinary softmax on random inputs with various chunk sizes. *(3 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

1. Prove that the online softmax gives exactly $\frac{\sum_j e^{s_j} v_j}{\sum_j e^{s_j}}$, by showing that after every chunk $\ell = \sum_{j \text{ seen}} e^{s_j - m}$ and $o = \sum_{j \text{ seen}} e^{s_j - m} v_j$, where $m$ is the current running maximum.
2. A GPU has 80 GB of memory. The model weights take 26 GB, and each user's cache needs 6.25 GiB (from C2). How many users fit at once (ignore activations)? How many with the GQA version?
3. Explain why the answer to part 2 limits throughput more than the GPU's arithmetic speed does.
