# Section 14 — Solutions: KV Cache & Fast Attention

---

## A. Warm-up

**A1.** In a causal decoder, an earlier token never attends to later tokens. So its key and value, in every layer, are **final** once computed and are needed again at every later step. *(2 marks)* A query is used only once, by its own token at its own step, to read from the cache. Later steps never need old queries, so there is nothing to gain from storing them. *(1 mark)*

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **False.** The cached $K$ and $V$ are exactly the values a full recompute would produce, so the outputs are identical, apart from floating-point rounding.
2. **False.** FlashAttention computes exact attention. It only changes the order of the computation (tiling plus the online softmax).
3. **True.** Every weight is read once to produce one token, about 2 FLOPs per weight read, which is far too little arithmetic to keep the GPU busy.

**A3.** *(1 mark each)* one (the newest); linearly; $n$.

---

## B. Multiple choice

**B1. (b).**
- (a) PagedAttention is about memory layout.
- (c) Prefix caching reuses the cache of a shared prompt.
- (d) FlashAttention computes exact attention in tiles.

**B2. (b).**
- (a) Wrong: attention is still exact.
- (c) Wrong: it doesn't change the softmax.
- (d) Wrong: it doesn't touch the weights.

**B3. (b).** This rescales the old terms from $e^{s - m}$ to $e^{s - m'}$.
- (a) Wrong: discarding them would lose information.
- (c) Wrong: dividing by $m'$ doesn't rescale exponentials.
- (d) Wrong: leaving them unchanged mixes two different reference points.

**B4. (b).**
- (a) Wrong: each user's cache is the same size.
- (c) Wrong: the model is unchanged.
- (d) Wrong: the cache is still needed.

**B5. (b).** Its K/V is computed once and reused.
- (a) Wrong: sliding windows limit what is attended to, not prefill work.
- (c) Wrong: temperature only changes sampling.
- (d) Wrong: MQA shrinks the cache but still needs the prefill.

---

## C. Calculation

**C1.**
1. Without a cache, the four passes process $8 + 9 + 10 + 11 = 38$ positions. With a cache: a prefill of 8, then 1 per step, $8 + 1 + 1 + 1 = 11$. *(2 marks)*
2. Without a cache: $\sum_{k=1}^{T}(p + k - 1) = Tp + \frac{T(T-1)}{2}$. With a cache: $p + (T - 1)$. *(1 mark)*

**C2.**
1. $2 \times 40 \times 40 \times 128 \times 2 = 819{,}200$ bytes $= 0.78$ MiB per token. *(2 marks)*
2. $819{,}200 \times 8{,}192 = 6.71 \times 10^9$ bytes $= 6.25$ GiB. *(1 mark)*
3. $16 \times 6.25 = 100$ GiB, more than an 80 GB GPU holds. *(1 mark)*
4. GQA with 8 KV heads is $40/8 = 5$× smaller: $1.25$ GiB. *(1 mark)*

**C3.**
1. The weights are $13\times10^9 \times 2 = 26$ GB, and $26\times10^9/3.35\times10^{12} = 7.8$ ms per token, so at most about **129 tokens/s**. *(2 marks)*
2. 8-bit weights are 13 GB, about 3.9 ms per token, or roughly **258 tokens/s**: twice as fast. *(1 mark)*

**C4.**
1. Chunk 1, scores $[1, 2]$: *(2 marks)*
   - $m = 2$
   - $\ell = e^{-1} + e^{0} = 0.3679 + 1 = 1.3679$
   - $o = 0.3679(2) + 1(0) = 0.7358$
2. Chunk 2, scores $[4, 3]$: the new maximum is $m' = 4$, so rescale by $e^{2-4} = 0.1353$. *(2 marks)*
   - Rescaled: $\ell = 0.1851$ and $o = 0.0996$.
   - The new terms are $e^{0} = 1$ (value 1) and $e^{-1} = 0.3679$ (value 5).
   - $\ell = 0.1851 + 1 + 0.3679 = 1.5530$
   - $o = 0.0996 + 1 + 1.8394 = 2.9390$
   - Output: $2.9390/1.5530 = \mathbf{1.8924}$.
3. The ordinary softmax of $[1, 2, 4, 3]$ is $[0.0321, 0.0871, 0.6439, 0.2369]$, so the output is $0.0641 + 0 + 0.6439 + 1.1844 = 1.8924$, the same. *(1 mark)*

**C5.** One head: $16{,}384^2 \times 2 = 537$ MB. 32 heads: about **17.2 GB**, for one layer. *(2 marks)*

---

## D. Conceptual

**D1.** In decode at batch size 1, each step does about 2 FLOPs per weight but must read every weight, plus the cache, from memory. The GPU waits on memory, so the step is **memory-bound**. *(1.5 marks)* In prefill, each weight read is reused across all $n$ prompt tokens at once, in one big matrix multiplication with about $2n$ FLOPs per weight read. That is enough arithmetic to keep the compute units busy, so prefill is **compute-bound**. *(1.5 marks)*

**D2.** A token can no longer attend **directly** to anything more than $W$ positions back, so long-range details can be lost. *(1 mark)* Stacking helps: layer 1 lets position $i$ see $i - W$; at layer 2, position $i$ sees tokens that themselves saw $W$ further back. Information can travel about $N \times W$ positions through $N$ layers, though indirectly and with some loss. *(2 marks)*

**D3.** Standard attention is limited by **memory traffic**: it writes and reads the $n \times n$ score and weight matrices in slow GPU main memory. *(1 mark)* FlashAttention keeps tiles in fast on-chip SRAM and never writes the full matrix, so it moves far less data. A little extra arithmetic, the rescaling, is cheap by comparison. *(1 mark)*

---

## E. Applied / coding

**E1.** *(5 marks: 2 full recompute, 2 cache class, 1 equality check)*

```python
import numpy as np

def softmax(s):
    """Stable softmax over the last axis."""
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def full_recompute(X, W_Q, W_K, W_V):
    """For every step t, redo causal attention over X[:t+1] and keep the last row."""
    outs = []
    for t in range(len(X)):
        Q, K, V = X[:t + 1] @ W_Q, X[:t + 1] @ W_K, X[:t + 1] @ W_V
        A = softmax(Q[-1] @ K.T / np.sqrt(K.shape[1]))
        outs.append(A @ V)
    return np.array(outs)

class KVCacheAttention:
    """Single-head causal attention that projects only the newest token."""

    def __init__(self, W_Q, W_K, W_V):
        self.W_Q, self.W_K, self.W_V = W_Q, W_K, W_V
        self.K = np.zeros((0, W_K.shape[1]))
        self.V = np.zeros((0, W_V.shape[1]))

    def step(self, x_t):
        q, k, v = x_t @ self.W_Q, x_t @ self.W_K, x_t @ self.W_V
        self.K = np.vstack([self.K, k])
        self.V = np.vstack([self.V, v])
        return softmax(q @ self.K.T / np.sqrt(self.K.shape[1])) @ self.V

rng = np.random.default_rng(0)
X = rng.normal(size=(10, 8))
W_Q, W_K, W_V = (rng.normal(size=(8, 4)) for _ in range(3))
cache = KVCacheAttention(W_Q, W_K, W_V)
cached = np.array([cache.step(x) for x in X])
assert np.allclose(cached, full_recompute(X, W_Q, W_K, W_V))
print("KV-cache outputs identical to full recompute; cache shape", cache.K.shape)
```

**E2.** *(3 marks)*

```python
def online_softmax_attention(s, v, chunk):
    """Exact softmax(s) @ v, processing scores chunk by chunk with running m, l, o."""
    m, l, o = -np.inf, 0.0, np.zeros_like(v[0], dtype=float)
    for start in range(0, len(s), chunk):
        sc, vc = s[start:start + chunk], v[start:start + chunk]
        m_new = max(m, sc.max())
        scale = np.exp(m - m_new)                  # rescale old totals (0 on the first chunk)
        w = np.exp(sc - m_new)
        l = l * scale + w.sum()
        o = o * scale + w @ vc
        m = m_new
    return o / l

print(round(float(online_softmax_attention(np.array([1., 2, 4, 3]), np.array([2., 0, 1, 5]), 2)), 4))  # 1.8924
for chunk in (1, 3, 7, 64):
    s, v = rng.normal(size=50) * 3, rng.normal(size=(50, 4))
    assert np.allclose(online_softmax_attention(s, v, chunk), softmax(s) @ v)
print("online softmax matches ordinary softmax for every chunk size")
```

---

## F. ★ Challenge

1. Use induction over the chunks. *(2 marks)*
   - **Start:** after the first chunk, $\ell = \sum e^{s_j - m}$ and $o = \sum e^{s_j - m} v_j$ over that chunk, with $m$ its maximum.
   - **Step:** suppose the invariant holds with maximum $m$, and a new chunk arrives, so the maximum becomes $m' = \max(m, \text{chunk max})$. Multiplying the old totals by $e^{m - m'}$ turns each old term $e^{s_j - m}$ into $e^{s_j - m'}$. Adding the new chunk's terms $e^{s_j - m'}$ (and $e^{s_j - m'} v_j$) restores the invariant with the new maximum.
   - **End:** after the last chunk, $\frac{o}{\ell} = \frac{\sum_j e^{s_j - m} v_j}{\sum_j e^{s_j - m}} = \frac{\sum_j e^{s_j} v_j}{\sum_j e^{s_j}}$, because the common factor $e^{-m}$ cancels.
2. The free memory is $80 - 26 = 54$ GB, and $54\times10^9 / (6.25 \times 2^{30}) = 8.05$, so **8 users** fit. With GQA: $54\times10^9/(1.25\times2^{30}) = 40.2$, so **40 users**. *(1 mark)*
3. Throughput in decode comes from batching: each weight read serves every user in the batch. With only 8 users there is too little arithmetic per weight read to use the GPU's compute, so memory capacity for caches, not FLOPs, sets the ceiling. GQA's 5× smaller cache allows 5× more users, and so roughly 5× the throughput. *(1 mark)*
