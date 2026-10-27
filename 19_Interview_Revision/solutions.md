# Section 19 — Solutions: Mock Exam (Interview Revision)

---

## A. Warm-up

**A1.** $\text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)V$. Here $Q$ is $n \times d_k$, $K$ is $n \times d_k$, $V$ is $n \times d_v$, the mask $M$ is $n \times n$, and the output is $n \times d_v$. (Section 5)

**A2.** $A = X + \text{MHA}(\text{LN}(X))$ and $X' = A + \text{FFN}(\text{LN}(A))$. (Section 8)

**A3.** Pre-training (next-token prediction), supervised fine-tuning (loss on responses only), and preference tuning (RLHF or DPO). (Section 12)

**A4.** Any four of: a KV cache; GQA/MQA; weight quantization (INT8/INT4); KV-cache quantization; FlashAttention; continuous batching; PagedAttention; prefix caching; speculative decoding; distillation; pruning. (Sections 14, 17, 18)

**A5.** **False.** Each head gets narrower ($d_k = d_{model}/h$), so the total stays $4d_{model}^2$. (Section 6)

**A6.** **True.** Dividing every logit by $T > 0$ keeps their order but sharpens the distribution when $T < 1$. (Section 13)

---

## B. Multiple choice

**B1. (b).** With a cache, only the new token is projected and passed through the FFN.
- (a) Wrong: attention still reads all $t$ cached entries, so it still grows.
- (c) Wrong: the softmax runs over a row of $t$ scores, which also grows.
- (d) Wrong: the projections and FFN no longer grow.

**B2. (b).** The cache shrinks by $h/g$.
- (a) Wrong: $d_{ff}$ doesn't affect the cache.
- (c) Wrong: more layers *increase* the cache.
- (d) Wrong: the vocabulary is unrelated to the cache.

**B3. (b).**
- (a) Wrong: MLM gives *fewer* training signals.
- (c) Wrong: the vocabulary has nothing to do with it.
- (d) Wrong: speed is not the reason.

**B4. (b).**
- (a), (c) and (d) don't address ranking.

**B5. (b).** (Section 14)
- (a) Wrong: the GPU's compute sits mostly idle at batch size 1.
- (c) Wrong: tokenizing is negligible.
- (d) Wrong: the network isn't the bottleneck for one GPU.

**B6. (b).**
- (a) Wrong: a zero start doesn't speed training up.
- (c) Wrong: the rank is fixed by $r$.
- (d) Wrong: quantization is unrelated.

---

## C. Calculation

**C1.** *(3 marks: 1 scores, 1 weights, 1 output)*
- Scores: $q \cdot k / \sqrt2 = [2, 0, 4]/1.414 = [1.414, 0, 2.828]$.
- $e^{1.414} = 4.11$, $e^{0} = 1$, $e^{2.828} = 16.91$, with sum $22.02$.
- Weights: $[0.19, 0.05, 0.77]$.
- Output: $0.19(1) + 0.05(2) + 0.77(3) = 2.58$.

**C2.** *(3 marks)*
- Blocks: $12 \times 3072^2 \times 28 = 3{,}170{,}893{,}824$.
- Embeddings: $50{,}000 \times 3072 = 153{,}600{,}000$.
- Total: **3,324,493,824 ≈ 3.32B**.

**C3.** *(4 marks)*
- MHA: $2 \times 28 \times 24 \times 128 \times 2 = 344{,}064$ bytes per token. Times 16,384, that is $5.64 \times 10^9$ bytes = **5.25 GiB**.
- GQA with 8 KV heads: $114{,}688$ bytes per token, which is **1.75 GiB** at 16,384 tokens (3× smaller).

**C4.** *(2 marks per part)*
1. $D = 20N \approx 66.5$B tokens, and $C = 6 \times 3.32\times10^9 \times 6.65\times10^{10} \approx 1.3 \times 10^{21}$ FLOPs.
2. $16 \times 3.32\times10^9 \approx 53$ GB.

**C5.**
1. 16-bit: **6.6 GB**. INT4: **1.7 GB**. *(1 mark)*
2. $6.65\times10^9 / 3.35\times10^{12} = 1.98$ ms per token, so at most about 500 tokens/s. *(2 marks)*

**C6.** $28 \times 2 \times 16 \times (3072 + 3072) = 5{,}505{,}024$ parameters, which is about **0.17%** of Model M. *(2 marks)*

**C7.** *(3 marks: 2 probabilities, 1 top-p)*
- $T = 2$ turns the logits into $[1.5, 0.5, 0, -0.5]$.
- The exponentials are $4.48, 1.65, 1, 0.61$, with sum $7.74$, so the probabilities are $[0.579, 0.213, 0.129, 0.078]$.
- For top-p 0.8, the cumulative sums are $0.579$, $0.792$, then $0.922$. **3 tokens** survive, renormalized to $[0.63, 0.23, 0.14]$.

**C8.** $e^{1.5} = 4.48$ *(1 mark)*

**C9.** $[\sin 5, \cos 5, \sin 0.05, \cos 0.05] = [-0.959, 0.284, 0.050, 0.999]$ *(2 marks)*

**C10.** $(512/16)^2 = 1{,}024$ patches, and the cost ratio is $(1024/196)^2 \approx 27$×. *(2 marks)*

**C11.** $E = (1 - 0.75^4)/0.25 = 2.73$ tokens per pass. *(1 mark)*

**C12.** *(3 marks)*
- $s = 1.4/7 = 0.2$, so $w/s = [2, -1, 7, -3.25]$.
- Codes: $[2, -1, 7, -3]$.
- Dequantized: $[0.4, -0.2, 1.4, -0.6]$.
- Errors: $[0, 0, 0, 0.05]$, so the mean absolute error is **0.0125**.

---

## D. Interview questions *(2 marks each: 1 for the key idea, 1 for the precise detail)*

**D1.** With unit-variance entries, $q \cdot k$ has variance $d_k$, so its typical size grows like $\sqrt{d_k}$. Unscaled, the scores at $d_k$ = 64–128 would push the softmax towards one-hot, where its gradients $p(1-p)$ almost vanish and learning stalls. Dividing by $\sqrt{d_k}$ keeps the scores at size about 1.

**D2.** Each sub-layer reads the running vector and **adds** its result to it, so information and gradients have an identity path through the whole depth: $\partial y/\partial x = 1 + F'$. Pre-norm normalizes only the sub-layer's *input*, leaving that highway untouched. Post-norm rescales the stream after every layer, which makes deep stacks fragile and needs careful warm-up.

**D3.** One simple objective, next-token prediction, uses every token of any text, and every task can be posed as "continue this text", so prompting and in-context learning come for free. The single causal stack also scales simply, and it supports KV caching, which makes chat generation cheap.

**D4.** FlashAttention computes exactly the same attention, but in tiles that fit in the GPU's fast on-chip memory. An online softmax (running max and sum, with rescaling) means it never has to write the $n \times n$ score matrix to slow memory. That cuts memory from $O(n^2)$ to $O(n)$ and makes attention several times faster, because memory traffic was the bottleneck.

**D5.** Use **RAG** when the model must answer from specific, changing or private documents and should cite them: updating an index is cheap, and grounding reduces hallucination. Use **LoRA** to change *behaviour*: style, format, a narrow skill, or domain jargon. The two can be combined.

**D6.** BF16 keeps FP32's 8 exponent bits, so it has the same huge range: small gradients don't underflow and large activations don't overflow, and no loss scaling is needed. FP16 has more precision but a tiny range (max 65,504, smallest positive value about $6 \times 10^{-8}$). Training tolerates BF16's lower precision well.

---

## E. System design

**E1.** *(6 marks: 1 each for model, quantization, ingestion and chunking, retrieval, prompting and safety, evaluation. Each needs a concrete number.)*

- **Model:** a multilingual instruction-tuned model of about 7–8B parameters that handles Telugu reasonably well. In INT4, the weights take about **4–5 GB**.
- **Memory budget on 24 GB:** about 5 GB of weights, a GQA model's KV cache of about 0.125 MB per token, and a serving engine with PagedAttention and continuous batching. That leaves room for roughly 100k cached tokens, e.g. about 25 concurrent users with 4k-token contexts.
- **Ingestion:** extract the text from the 2,000 PDFs (OCR for scanned ones), then chunk at about **400–500 tokens with 50 tokens of overlap**. Store metadata: circular number, date, language and page. That is about 2,000 × 5 chunks = 10k chunks, a tiny index.
- **Retrieval:** a multilingual embedding model (e.g. 768-dimensional vectors, about 30 MB for 10k chunks). Hybrid search with **BM25** for circular numbers and dates, merged with **RRF** ($k = 60$). Retrieve the top 20, re-rank with a cross-encoder, and keep the top **4–5**.
- **Prompt:** "Answer only from the context; cite circular numbers; if the answer is not found, say so". Answer in the question's language, at temperature about **0.2**.
- **Evaluation:** a test set of about 200 real questions with known source circulars. Measure **recall@5** (target ≥ 0.9), MRR, and faithfulness and correctness on a sample checked by hand or by an LLM judge. Also track TTFT (target under 1 s) and p99 latency.

---

## F. ★ Challenge

```python
def model_card(d, layers, heads, kv_heads, vocab, d_ff, tied=True):
    """Print the key sizes of a decoder-only Transformer (no biases or norms; standard 2-matrix FFN)."""
    d_head = d // heads
    attn = 2 * d * d + 2 * d * (kv_heads * d_head)           # W_Q, W_O full; W_K, W_V for kv_heads
    ffn = 2 * d * d_ff
    emb = vocab * d * (1 if tied else 2)
    N = layers * (attn + ffn) + emb
    kv = 2 * layers * kv_heads * d_head * 2                   # bytes per token, 16-bit
    D = 20 * N
    print(f"params {N:,} ({N / 1e9:.2f}B) | KV {kv:,} B/token | train mem {16 * N / 1e9:.0f} GB"
          f" | Chinchilla {D / 1e9:.1f}B tokens | C = {6 * N * D:.2e} FLOPs")
    return N, kv

assert model_card(3072, 28, 24, 24, 50_000, 4 * 3072) == (3_324_493_824, 344_064)          # Model M
assert model_card(2048, 24, 16, 16, 32_000, 4 * 2048) == (1_273_495_552, 196_608)          # notes P2/P3
model_card(3072, 28, 24, 8, 50_000, 4 * 3072)                                              # GQA variant
```

*(4 marks: 2 for correct GQA-aware parameters, 1 for the KV cache, 1 for the checks)*

For the GQA variant, the parameters drop by $28 \times 2 \times 3072 \times (3072 - 1024) = 352{,}321{,}536$, and the cache is 3× smaller.
