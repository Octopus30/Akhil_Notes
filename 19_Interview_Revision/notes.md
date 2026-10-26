# Section 19 — Interview Revision

> **In one line:** this section pulls the whole course together — a one-page formula sheet, the numbers worth remembering, a question bank with model answers, mixed numerical problems, and two system-design questions — so you can revise for exams and interviews in one place.

---

## 1. Learning objectives

By the end of this section you should be able to:

1. **Recall** the core formulas of the Transformer from memory, with shapes.
2. **Answer** common conceptual interview questions clearly in two or three sentences.
3. **Solve** mixed numerical problems: attention, parameter counts, KV cache, compute, LoRA, quantization, decoding.
4. **Sketch** an end-to-end design for an LLM application (a RAG assistant, a serving setup) and justify the choices.
5. **Identify** your weak topics and know which section to revisit.

---

## 2. How to use this section

1. **Cover the answers.** Try each question before reading the model answer.
2. **Say answers out loud.** Interviews reward clear, short explanations: *definition → why it matters → one example or number*.
3. **Do the numerical problems on paper.** Then check them with the calculators in the HTML version.
4. **Mark weak spots** and go back to the section listed next to each question.

**A good answer has three parts:**
- *What* it is (one sentence).
- *Why* it exists — the problem it solves.
- *A concrete detail* — a formula, a number, or a real model that uses it.

---

## 3. One-page formula sheet

| Idea | Formula | Section |
|---|---|---|
| Embedding lookup | $X = E[\text{ids}]$, $E$: $\lvert\mathcal{V}\rvert \times d_{model}$ | 2 |
| Cosine similarity | $\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert \lVert b \rVert}$ | 2 |
| Projections | $Q = XW_Q,\; K = XW_K,\; V = XW_V$ | 4 |
| Scaled dot-product attention | $\text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)V$ | 5 |
| Multi-head attention | $[\text{head}_1 \mid \cdots \mid \text{head}_h]\,W_O$, $d_k = d_{model}/h$ | 6 |
| Sinusoidal positions | $p_{pos,2i} = \sin\!\left(pos/10000^{2i/d}\right)$, $p_{pos,2i+1} = \cos(\cdot)$ | 7 |
| Pre-norm block | $A = X + \text{MHA}(\text{LN}(X))$, $X' = A + \text{FFN}(\text{LN}(A))$ | 8 |
| LayerNorm | $\gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ | 8 |
| FFN | $\phi(xW_1 + b_1)W_2 + b_2$, $d_{ff} \approx 4d$ | 8 |
| Parameters per block | $\approx 12\, d_{model}^2$ (attention $4d^2$ + FFN $8d^2$) | 8 |
| Next-token loss | $\mathcal{L} = -\frac{1}{T}\sum_t \log P(x_{t+1} \mid x_{\le t})$; perplexity $e^{\mathcal{L}}$ | 10, 12 |
| Training compute | $C \approx 6ND$; Chinchilla $D \approx 20N$ | 12 |
| Training memory (AdamW) | ≈ 16 bytes / parameter | 12 |
| Temperature | $p_i \propto e^{z_i / T}$ | 13 |
| KV cache | $2 \times \text{layers} \times \text{KV heads} \times d_{head} \times \text{bytes} \times \text{tokens}$ | 14 |
| Decode time bound | $\ge (\text{weights} + \text{cache}) / \text{bandwidth}$ | 14 |
| ViT patches | $N = (H/P)(W/P)$, each $P^2 C$ numbers | 15 |
| LoRA | $h = W_0x + \frac{\alpha}{r}BAx$; params $r(d + k)$ | 16 |
| Quantization | $s = \max\lvert w\rvert / q_{\max}$, $q = \text{round}(w/s)$ | 17 |
| Speculative decoding | $E = (1 - \alpha^{k+1})/(1 - \alpha)$ | 17 |
| RRF | $\sum 1/(k + \text{rank})$, $k \approx 60$ | 18 |

---

## 4. Numbers worth remembering

| Fact | Value |
|---|---|
| Original Transformer (base) | 6+6 layers, $d = 512$, 8 heads, ≈ 65M params |
| BERT-base | 12 layers, $d = 768$, 12 heads, **109,482,240** params |
| GPT-2 small | 12 layers, $d = 768$, **124,439,808** params |
| LLaMA-2 7B | 32 layers, $d = 4096$, 32 heads, $d_{ff}$ = 11,008, **6.74B** params, trained on 2T tokens |
| ViT-B/16 | 196 patches + `[CLS]`, **86.6M** params |
| Typical $d_{head}$ | 64 or 128 |
| English rule of thumb | 1 token ≈ ¾ word ≈ 4 characters |
| KV cache, LLaMA-2 7B | 0.5 MB per token, 2 GiB at 4,096 tokens |
| Weights, 7B model | 13.5 GB (16-bit), 6.7 GB (INT8), 3.4 GB (INT4) |
| LLaMA-2 7B training | ≈ 8×10²² FLOPs, ≈ 184k A100 GPU-hours |
| Chinchilla | ≈ 20 training tokens per parameter |
| BERT MLM | 15% picked; 80% `[MASK]` / 10% random / 10% unchanged |
| MAE | 75% of patches masked |

---

## 5. Question bank with model answers

### Foundations (Sections 1–3)
1. **What problem did the Transformer solve compared with RNNs?** RNNs process tokens one after another, so they're slow to train and struggle to carry information across long distances. The Transformer uses self-attention, letting every token see every other token in one parallel step — fast on GPUs, with a one-hop path between any two tokens.
2. **Why sub-word tokenization?** Word-level vocabularies are huge and can't handle unseen words; character-level sequences are too long. Sub-words (BPE, WordPiece) keep common words whole and split rare ones into reusable pieces, so nothing is ever unknown.
3. **What does self-attention compute, in one sentence?** Each token's new vector is a weighted average of all tokens' value vectors, with weights from a softmax over query–key similarity.
4. **Is self-attention aware of word order?** No — without positional information it is permutation-equivariant: shuffling the input just shuffles the outputs.

### Attention (Sections 4–7)
5. **Why separate Q, K and V?** With one vector per token, tokens over-attend to themselves and relevance is forced to be symmetric. Separate projections let "what I look for", "how I'm found" and "what I pass on" differ.
6. **Why divide by $\sqrt{d_k}$?** Dot products of random vectors have variance ≈ $d_k$; large scores saturate softmax and kill gradients. Dividing by $\sqrt{d_k}$ keeps scores at variance ≈ 1.
7. **What does the causal mask do and why is it $-\infty$?** It blocks attention to future positions so a decoder can't see the answer. $-\infty$ gives weight exactly 0 while rows still sum to 1; a 0 would leave weight $e^0 = 1$.
8. **Why multiple heads?** One softmax gives one weighting; several heads can track different relationships (syntax, coreference, nearby words) at the same cost, since $d_k = d_{model}/h$.
9. **What is GQA?** Grouped-query attention keeps all query heads but shares key/value heads among groups, shrinking the KV cache by $h/g$ with little quality loss.
10. **Why is RoPE popular?** It rotates queries and keys by position-dependent angles so attention scores depend only on relative distance, has no parameters, and can be stretched to longer contexts.

### The block and architectures (Sections 8–11)
11. **What do residual connections do?** $y = x + F(x)$ gives gradients a direct path ($\partial y / \partial x = 1 + F'$), making deep stacks trainable, and turns each layer into a small edit of a shared "residual stream".
12. **LayerNorm vs BatchNorm?** LayerNorm normalizes each token's features independently; BatchNorm normalizes each feature across the batch. LayerNorm doesn't depend on batch size or sequence length.
13. **What does the FFN do, and how many parameters does it have?** It processes each token separately: expand to $d_{ff}$, non-linearity, project back. With $d_{ff} = 4d$ it has $8d^2$ parameters — about two-thirds of a block.
14. **Pre-norm vs post-norm?** Pre-norm, $x + F(\text{LN}(x))$, keeps a clean residual path and trains stably when deep; most modern LLMs use it.
15. **Encoder vs decoder?** Same block, different mask and objective: encoders are bidirectional and trained with masked LM (good for understanding and embeddings); decoders are causal and trained with next-token prediction (good for generation).
16. **Why are most LLMs decoder-only?** One simple objective works on any text, every position gives a training signal, every task can be phrased as text continuation (prompting, in-context learning), and generation is efficient with a KV cache.
17. **What is teacher forcing?** During training the decoder is fed the true previous tokens (input shifted right) and predicts every next token in one parallel pass.

### Training and inference (Sections 12–14)
18. **What are the training stages of a chat model?** Pre-training (next-token prediction on trillions of tokens) → supervised fine-tuning on instruction–response pairs (loss on responses only) → preference tuning (RLHF or DPO).
19. **What does AdamW add over plain SGD?** Per-parameter step sizes from running averages of the gradient and its square (with bias correction), plus weight decay applied directly to the weights.
20. **What is the Chinchilla result?** For a fixed compute budget, parameters and data should scale together — about 20 tokens per parameter. Models are now often trained much longer so that they're cheaper to run.
21. **Temperature vs top-p?** Temperature reshapes the whole distribution (sharper below 1, flatter above 1) without changing the ranking; top-p keeps the smallest set of tokens covering probability $p$ and renormalizes.
22. **When would you use beam search?** When there's essentially one correct output, as in translation or speech recognition. For open-ended chat it produces bland, repetitive text.
23. **What is the KV cache and why is decoding memory-bound?** Keys and values of past tokens never change, so they are stored rather than recomputed. Each decode step must still read all weights and the cache to produce one token, so its speed is limited by memory bandwidth.
24. **What does FlashAttention change?** It computes exact attention in tiles using an online softmax, never storing the $n \times n$ score matrix — less memory traffic, $O(n)$ memory, faster.

### Applications (Sections 15–18)
25. **How does a ViT turn an image into tokens?** Split it into $P \times P$ patches, flatten each, project it linearly to $d_{model}$, add a `[CLS]` token and position embeddings, and run a standard encoder.
26. **How does CLIP do zero-shot classification?** It embeds the image and text prompts ("a photo of a dog") in a shared space and picks the prompt with the highest cosine similarity.
27. **Explain LoRA.** Freeze $W_0$ and learn a low-rank update $BA$ ($B$ starts at 0); this trains $r(d + k)$ instead of $dk$ parameters per matrix and can be merged afterwards with no inference cost.
28. **Why does one outlier hurt quantization?** The scale is set by the largest value in a group, so one outlier stretches it and rounds ordinary small weights to zero. Group-wise scales and outlier handling fix this.
29. **RAG or fine-tuning?** RAG for knowledge (changing facts, private documents, citations); fine-tuning for behaviour (format, tone, a narrow skill). They are often combined.
30. **Why does continuous batching help?** Finished requests leave and waiting ones join every decode step, so batch slots stay busy instead of idling until the longest request ends.

---

## 6. Mixed numerical problems

**P1 — Attention (Section 5).** One query $q = [1, 1]$, keys $k_1 = [1, 0]$, $k_2 = [0, 2]$, values $v_1 = [1, 0]$, $v_2 = [0, 1]$, $d_k = 2$. Find the weights and the output.

> Scores: $1/\sqrt{2} = 0.707$ and $2/\sqrt{2} = 1.414$. Softmax: $e^{0.707} = 2.03$, $e^{1.414} = 4.11$ → weights **[0.33, 0.67]**. Output $= 0.33[1,0] + 0.67[0,1] = $ **[0.33, 0.67]**.

**P2 — Parameter count (Section 8).** A decoder has $d = 2048$, 24 layers, standard FFN ($d_{ff} = 4d$), a tied vocabulary of 32,000 tokens. Ignoring biases, norms and positions, how many parameters?

> Blocks: $12 \times 2048^2 \times 24 = 1{,}207{,}959{,}552$. Embeddings: $32{,}000 \times 2048 = 65{,}536{,}000$. Total **1,273,495,552 ≈ 1.27B**.

**P3 — KV cache (Section 14).** The same model uses 16 heads of $d_{head} = 128$ (standard multi-head attention), 16-bit cache. Cache per token and for an 8,192-token context?

> $2 \times 24 \times 16 \times 128 \times 2 = $ **196,608 bytes** per token; × 8,192 = 1,610,612,736 bytes = **1.5 GiB**.

**P4 — Compute (Section 12).** How many tokens would be "Chinchilla-optimal" for this model, and how much training compute?

> $D \approx 20N = 20 \times 1.27 \times 10^9 \approx$ **25.5B tokens**. $C \approx 6ND = 6 \times 1.27 \times 10^9 \times 2.55 \times 10^{10} \approx$ **1.9 × 10²⁰ FLOPs**.

**P5 — LoRA (Section 16).** LoRA with $r = 8$ on $W_Q$ and $W_V$ ($2048 \times 2048$ each) in all 24 layers. Trainable parameters?

> $24 \times 2 \times 8 \times (2048 + 2048) = $ **1,572,864** ≈ 0.12% of the model.

**P6 — Quantization (Section 17).** Weight memory of this model in 16-bit and in INT4? Lower bound on decode time per token at batch 1 on a 2 TB/s GPU (16-bit)?

> 16-bit: $1.27 \times 10^9 \times 2 = $ **2.5 GB**; INT4: **0.64 GB**. Time ≥ $2.55 \times 10^9 / 2 \times 10^{12} \approx$ **1.3 ms/token** (≈ 790 tokens/s at most).

**P7 — Decoding (Section 13).** Logits [2, 1, 0] at temperature 0.5?

> Divide by 0.5 → [4, 2, 0]; $e^4 = 54.6$, $e^2 = 7.39$, $e^0 = 1$ → **[0.867, 0.117, 0.016]**.

**P8 — Loss (Section 12).** A model's average cross-entropy on a test set is 2.0 nats per token. Perplexity?

> $e^{2.0} = $ **7.39** — as unsure as choosing among about 7 tokens.

**P9 — Positions (Section 7).** Sinusoidal encoding for position 2 with $d_{model} = 4$?

> $[\sin 2, \cos 2, \sin 0.02, \cos 0.02] = $ **[0.909, −0.416, 0.020, 1.000]**.

**P10 — ViT (Section 15).** A 448 × 448 image with 14 × 14 patches: how many patches, and how does attention cost compare with 196 patches?

> $(448/14)^2 = 32^2 = $ **1,024** patches; cost ratio $(1024/196)^2 \approx$ **27×**.

---

## 7. System-design questions

### 7.1 "Design a question-answering assistant over our college's circulars."

A strong answer covers:
1. **Requirements:** questions in English (and maybe Telugu/Hindi), answers must cite the circular, data changes weekly, students may only see documents meant for them.
2. **Indexing:** extract text from PDFs → chunk (~300–500 tokens, ~10% overlap) → embed with a multilingual embedding model → store vectors + text + metadata (date, department, audience) in a vector database; re-index new circulars automatically.
3. **Retrieval:** hybrid search (BM25 for exact terms like course codes + vectors for meaning), filter by the student's access rights and date, re-rank the top ~20 with a cross-encoder, keep the top ~5.
4. **Generation:** a prompt that says "answer only from the context, cite [n], say you don't know otherwise"; a mid-sized instruction-tuned LLM.
5. **Evaluation:** a test set of real student questions with known answers; track recall@5, faithfulness and answer correctness; monitor "I don't know" rates.
6. **Risks:** outdated circulars (prefer the newest by date), missing documents, prompt injection inside documents (treat retrieved text as data, not instructions).

### 7.2 "Serve a 7B chat model to 1,000 concurrent users."

A strong answer covers:
1. **Memory:** 13.5 GB weights in 16-bit (or ~4 GB in 4-bit); KV cache 0.5 MB/token for LLaMA-2-style MHA — GQA models need far less. 1,000 users × 2,000 tokens × 0.5 MB = 1 TB of cache in the worst case, so you need GQA, an 8-bit cache, and several GPUs or replicas.
2. **Engine:** vLLM/TGI-style continuous batching with PagedAttention and prefix caching for the shared system prompt.
3. **Latency targets:** TTFT under ~1 s, TPOT ~30–50 ms; stream tokens; autoscale replicas on queue length.
4. **Cost:** measure tokens/s per GPU and compute cost per million tokens; consider quantization and speculative decoding.
5. **Safety and reliability:** rate limits, input/output filtering, logging, fallbacks when a replica fails.

---

## 8. Revision checklist

Tick each when you can do it **without notes**:

- [ ] Write the scaled dot-product attention formula and give every shape.
- [ ] Explain Q, K, V with an analogy and say why they're separate.
- [ ] Explain $\sqrt{d_k}$ with the variance argument.
- [ ] Draw a pre-norm decoder block and label residuals, norms, attention and FFN.
- [ ] Compute a 2-token attention example by hand.
- [ ] Count the parameters of a model from $d$, layers and vocabulary.
- [ ] Compute a KV-cache size for a given model and context.
- [ ] Explain MLM vs causal LM vs span corruption.
- [ ] Explain pre-training → SFT → RLHF/DPO.
- [ ] Apply temperature, top-k and top-p to a small distribution.
- [ ] Explain the KV cache and why decoding is memory-bound.
- [ ] Explain LoRA's $BA$, its initialisation and merging.
- [ ] Quantize a few numbers to INT4 and explain outliers.
- [ ] Draw the RAG pipeline and name two evaluation metrics.

---

## 9. Common interview mistakes

- **Reciting buzzwords without the "why".** Always say what problem a technique solves.
- **Mixing up training and inference.** Training is parallel (teacher forcing); generation is sequential (one pass per token, with a KV cache).
- **Forgetting shapes.** Interviewers often ask "what's the shape of $QK^{\top}$?" — answer $n \times n$, and say why that's the $O(n^2)$ cost.
- **Over-claiming.** Say "reduces" hallucination (RAG), "usually small" accuracy loss (4-bit), "roughly" 20 tokens per parameter.
- **Ignoring cost and latency** in design questions. Mention memory, throughput and cost per token.

---

## 10. Summary

- The whole course rests on a few formulas: embeddings, $\text{softmax}(QK^{\top}/\sqrt{d_k} + M)V$, multi-head concatenation, residual + norm + FFN blocks, and next-token cross-entropy.
- A handful of numbers ($12d^2$ per block, 16 bytes/param for training, $6ND$, 0.5 MB/token KV cache for a 7B MHA model, ~20 tokens/param) let you sanity-check almost any claim.
- Good answers follow *what → why → a concrete detail*.
- Design answers should cover requirements, data/memory, the pipeline, evaluation, cost and risks.

---

## 11. Final self-test

1. In one sentence each, explain: the KV cache, LoRA, and RAG.
2. A model has $d = 4096$, 32 layers, tied vocabulary 32,000, standard FFN. Estimate its parameters (ignore biases and norms).
3. Why can't you train a bidirectional encoder with next-token prediction?

<details>
<summary>Answers</summary>

1. **KV cache:** store each past token's keys and values so each generation step computes projections only for the new token. **LoRA:** freeze the weights and train a small low-rank update $BA$ that can be merged afterwards. **RAG:** retrieve relevant documents at question time and put them in the prompt so the model answers from them with citations.
2. $12 \times 4096^2 \times 32 = 6{,}442{,}450{,}944$ plus $32{,}000 \times 4096 = 131{,}072{,}000$ → **≈ 6.57B**. (LLaMA-2 7B is 6.74B because it uses a SwiGLU FFN with $d_{ff} = 11{,}008$ and a separate output head.)
3. Without a causal mask, each position can attend directly to the next token — the one it's supposed to predict — so the model would just copy it and learn nothing useful. Encoders use masked LM instead.
</details>

---

## 12. Further reading

The papers below are the "spine" of the course; each earlier section lists more.

1. Vaswani, A. et al. (2017). **"Attention Is All You Need."** arXiv:1706.03762.
2. Devlin, J. et al. (2019). **"BERT."** arXiv:1810.04805.
3. Brown, T. et al. (2020). **"Language Models are Few-Shot Learners."** arXiv:2005.14165.
4. Hoffmann, J. et al. (2022). **"Training Compute-Optimal Large Language Models."** arXiv:2203.15556.
5. Touvron, H. et al. (2023). **"Llama 2: Open Foundation and Fine-Tuned Chat Models."** arXiv:2307.09288.
6. Jurafsky, D., & Martin, J. H. **_Speech and Language Processing_** (3rd edition draft) — the Transformer and LLM chapters, for a textbook-length treatment.
