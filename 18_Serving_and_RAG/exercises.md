# Section 18 — Exercises: Serving & RAG

**Total: 52 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 55 minutes**

Total time ≈ TTFT + $n$ × TPOT. Static-batch utilisation $= \sum_i \text{len}_i / (B \cdot \max_i \text{len}_i)$. RRF$(d) = \sum_{\text{lists}} 1/(60 + \text{rank})$.

---

## A. Warm-up (recall)

**A1.** List the two phases of a RAG system (offline and online), with the steps of each. *(4 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. Continuous batching waits for every request in a batch to finish before starting new ones.
2. BM25 is better than embeddings at matching exact codes and rare names.
3. recall@5 measures whether the answer the model generated is faithful to the context.

**A3.** Define TTFT, TPOT and p99 latency, one sentence each. *(3 marks)*

---

## B. Multiple choice (1 mark each)

**B1.** Why do documents get chunked with some overlap?
- (a) To make the index bigger.
- (b) So a sentence cut at a chunk boundary still appears whole in one chunk.
- (c) Because embedding models need repeated text.
- (d) To speed up BM25.

**B2.** Why is an approximate-nearest-neighbour index (HNSW, IVF) used?
- (a) It is more accurate than exact search.
- (b) Exact search over hundreds of millions of vectors is too slow; ANN finds almost the same top results in milliseconds.
- (c) It removes the need for embeddings.
- (d) It compresses the LLM.

**B3.** Retrieval finds the right chunk, but at rank 9 of 10, and the model ignores it. The best next step is:
- (a) Increase $k$ to 50.
- (b) Add a cross-encoder re-ranker and use a smaller $k$.
- (c) Remove the citations.
- (d) Raise the temperature.

**B4.** What mainly sets the cost per million tokens?
- (a) The prompt length only.
- (b) GPU price divided by throughput.
- (c) The number of users' names.
- (d) The tokenizer's vocabulary size.

**B5.** Which is a good reason to prefer RAG over fine-tuning for a company handbook that changes monthly?
- (a) RAG trains the model faster.
- (b) Updating the index is cheap and immediate, and answers can cite their sources.
- (c) RAG doesn't need an embedding model.
- (d) Fine-tuning can't change a model's behaviour.

---

## C. Calculation

**C1. Batching.** Five requests in one static batch need 30, 5, 25, 60 and 40 output tokens. *(3 marks)*

1. Compute the utilisation.
2. How many steps does the 5-token request wait after finishing, and how many does a newly arrived request wait?

**C2. Cost and latency.** *(3 marks)*

1. A GPU at \$3.50 per hour sustains 3,000 output tokens/s. What is the cost per million tokens?
2. TTFT 0.3 s, TPOT 20 ms, a 400-token answer: what is the total time? When does the user start reading?

**C3. Chunking and storage.** *(4 marks)*

1. A 25,000-token document, chunk size 800, overlap 100. How many chunks, and where does the last one start?
2. Two million chunks with 1,024-dimensional float32 embeddings: how much vector storage? With float16?

**C4. Retrieval.** Toy embeddings of college notices: *(5 marks)*

| Chunk | Text | Embedding |
|---|---|---|
| c1 | "The campus bus leaves the main gate at 8 am." | [0.1, 0.9, 0.2] |
| c2 | "The library opens at 9 am." | [0.9, 0.1, 0.1] |
| c3 | "On Fridays a late bus leaves at 10 pm." | [0.2, 0.8, 0.5] |
| c4 | "Semester fees are due on 30 June." | [0.1, 0.1, 0.95] |

Question: *"What time does the bus leave?"*, embedding $q = [0.1, 0.95, 0.3]$.

1. Compute the cosine similarity with every chunk, and rank them.
2. Which chunks are retrieved with $k = 2$? Write the prompt you would send, with citations.
3. Write a good, faithful answer.

**C5. Metrics.** Four test questions; the first relevant chunk appears at ranks 2, 1, not retrieved, and 4. Compute recall@1, recall@3 and MRR. *(3 marks)*

**C6. Hybrid search.** BM25 returns [B, C, A]; vector search returns [A, D, E, B, C]. Compute the RRF scores ($k = 60$) and the final order. *(3 marks)*

---

## D. Conceptual

**D1.** Why does continuous batching give much higher throughput than static batching? Why does KV-cache memory limit how big the batch can get? *(3 marks)*

**D2.** A RAG bot answers *"The library is open 24 hours"*, which no retrieved chunk says. Name the failure, and give two fixes. *(3 marks)*

**D3.** A user searches for *"form 27B-X"* and gets nothing useful from a pure embedding search. Explain why, and how hybrid search helps. *(2 marks)*

---

## E. Applied / coding (NumPy)

**E1.** Build a mini RAG retriever: *(5 marks)*

- `chunk(tokens, size, overlap)` returns a list of token lists that covers the whole input;
- `embed(text)` is a simple bag-of-words vector over a fixed vocabulary (use your own 6–8 notices), normalised to length 1;
- `retrieve(question, k)` returns the top-$k$ notices by cosine similarity;
- `build_prompt(question, chunks)` produces numbered context with citation instructions.

Check `chunk` on C3 (36 chunks), and show `retrieve` returning sensible notices for two questions.

**E2.** Implement `recall_at_k(ranks, k)`, `mrr(ranks)` and `rrf(lists, k=60)`, and check them against C5 and C6. *(3 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

You must serve a 7B model to 1,000 students during exam week. Peak load is about 100 simultaneous requests, with an average prompt of 1,500 tokens (system prompt + retrieved chunks) and an average answer of 300 tokens.

1. Estimate the KV-cache memory at peak for an MHA model with 0.5 MB/token, and for a GQA model with 0.125 MB/token (Section 14).
2. Each 80 GB GPU holds 13.5 GB of weights. How many GPUs do you need for the caches in each case?
3. Suggest two further ways to cut memory or cost, and one way to keep TTFT low when everyone shares the same 1,000-token system prompt.
