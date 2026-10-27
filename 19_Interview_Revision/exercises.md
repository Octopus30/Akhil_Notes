# Section 19 — Exercises: Mock Exam (Interview Revision)

**Total: 61 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 75 minutes**

This is a mixed paper covering the whole course. The numbers are new, so don't look up the worked answers in Section 19's notes. Use the formula sheet in the notes if you need it.

Several questions use **Model M**: a decoder-only Transformer with $d_{model} = 3072$, 28 layers, 24 heads of $d_{head} = 128$ (standard MHA), a standard FFN ($d_{ff} = 4d$), and a tied vocabulary of 50,000 tokens.

---

## A. Warm-up (recall, 1 mark each)

**A1.** Write the scaled dot-product attention formula, with the shape of every matrix.

**A2.** Write the pre-norm Transformer block in two lines.

**A3.** What are the three stages of training an assistant LLM?

**A4.** Name four parameter-free or cheap ways to make LLM inference faster or smaller.

**A5.** True or false, with a reason: "Doubling the number of heads at fixed $d_{model}$ doubles a layer's attention parameters."

**A6.** True or false, with a reason: "Temperature 0.5 makes the most likely token even more likely, but it never changes the ranking."

---

## B. Multiple choice (1 mark each)

**B1.** A decoder generates 100 tokens with a KV cache. Per step, which work no longer grows with the context length?
- (a) The attention over cached keys and values.
- (b) The Q/K/V projections and the FFN.
- (c) The softmax.
- (d) Nothing; every part grows.

**B2.** Which choice most reduces KV-cache memory without changing the number of query heads?
- (a) A larger $d_{ff}$
- (b) Grouped-query attention
- (c) More layers
- (d) A larger vocabulary

**B3.** BERT is pre-trained with masked LM rather than next-token prediction because:
- (a) MLM gives more training signals.
- (b) A bidirectional model would see the next token and simply copy it.
- (c) Next-token prediction needs a larger vocabulary.
- (d) MLM is faster per token.

**B4.** For a RAG system whose right chunk is retrieved but ranked low, the best fix is:
- (a) Increase the temperature.
- (b) Add a cross-encoder re-ranker.
- (c) Fine-tune the tokenizer.
- (d) Remove the system prompt.

**B5.** At batch size 1, generating with a 7B model on one GPU is mostly limited by:
- (a) GPU FLOPs.
- (b) Memory bandwidth (reading the weights for each token).
- (c) Tokenizer speed.
- (d) Network latency.

**B6.** LoRA's $B$ is initialised to zero so that:
- (a) Training is faster.
- (b) The adapted model starts out exactly equal to the base model.
- (c) The rank is minimised.
- (d) It can be quantized.

---

## C. Calculation

**C1. Attention.** One query $q = [2, 0]$; keys $k_1 = [1, 1]$, $k_2 = [0, 2]$, $k_3 = [2, 1]$; scalar values $v = [1, 2, 3]$; $d_k = 2$. Compute the weights and the output. *(3 marks)*

**C2. Model M parameters.** Ignoring biases, norms and positions, compute the parameters of Model M. *(3 marks)*

**C3. Model M KV cache.** 16-bit cache. Compute the bytes per token and the GiB for a 16,384-token context. Then redo it for a GQA version with 8 KV heads. *(4 marks)*

**C4. Model M training.** *(4 marks)*

1. How many tokens are Chinchilla-optimal, and how much training compute ($C \approx 6ND$) would they take?
2. Estimate the memory for weights, gradients and AdamW states.

**C5. Model M inference.** *(3 marks)*

1. Weight memory in 16-bit and in INT4.
2. A lower bound on decode time per token at batch 1, in 16-bit, on a GPU with 3.35 TB/s of memory bandwidth.

**C6. LoRA.** $r = 16$ on $W_Q$ and $W_V$ (each $3072 \times 3072$) in all 28 layers. How many trainable parameters, and what share of Model M is that? *(2 marks)*

**C7. Decoding.** Logits $[3, 1, 0, -1]$ at temperature 2. Give the probabilities, and say which tokens survive top-p 0.8. *(3 marks)*

**C8. Loss.** A test-set loss of 1.5 nats per token. What is the perplexity? *(1 mark)*

**C9. Positions.** The sinusoidal encoding of position 5 with $d_{model} = 4$. *(2 marks)*

**C10. ViT.** A $512 \times 512$ image with $P = 16$: how many patch tokens (without `[CLS]`), and what is the attention cost relative to 196 patches? *(2 marks)*

**C11. Speculative decoding.** $k = 3$, $\alpha = 0.75$. What is the expected number of tokens per large-model pass? *(1 mark)*

**C12. Quantization.** Quantize $[0.4, -0.2, 1.4, -0.65]$ to INT4 with absmax. Give the codes, the dequantized values, and the mean absolute error. *(3 marks)*

---

## D. Interview questions (2–3 sentences each)

**D1.** Why divide attention scores by $\sqrt{d_k}$? *(2 marks)*

**D2.** What does the residual stream do, and why does pre-norm train more stably than post-norm? *(2 marks)*

**D3.** Why did decoder-only models become the default for LLMs? *(2 marks)*

**D4.** Explain FlashAttention to a fellow student in two sentences. *(2 marks)*

**D5.** When would you choose RAG, and when LoRA fine-tuning? *(2 marks)*

**D6.** Why is BF16 preferred to FP16 for training? *(2 marks)*

---

## E. System design (short)

**E1.** A college wants a chatbot that answers questions about its 2,000 PDF circulars in English and Telugu, runs on one 24 GB GPU, and must not invent facts. Sketch the design: model choice and size, quantization, retrieval pipeline (chunking, embedding, hybrid search, re-ranking), prompting, and evaluation. Give at least one number for each part. *(6 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

Write a function `model_card(d, layers, heads, kv_heads, vocab, d_ff, tied=True)` that prints: the total parameters (with attention adjusted for GQA, a standard 2-matrix FFN, and no biases or norms); the KV cache per token in 16-bit; the training memory at 16 bytes per parameter; the Chinchilla tokens; and $C = 6ND$ at that $D$. Check it on Model M (MHA) and on the notes' P2/P3 model ($d = 2048$, 24 layers, 16 heads, vocabulary 32,000).
