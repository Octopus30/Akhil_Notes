# Section 18 — Solutions: Serving & RAG

---

## A. Warm-up

**A1.** *(2 marks per phase)*
- **Offline (indexing):** load the documents, chunk them with overlap, embed each chunk, and store the vectors (with text and metadata) in a vector index.
- **Online (answering):** embed the question, retrieve the top-$k$ chunks, optionally re-rank them, build a prompt with the numbered chunks and the question, and generate an answer with citations.

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **False.** That describes static batching. Continuous batching rebuilds the batch every step, so finished requests leave and new ones join at once.
2. **True.** BM25 matches exact words, while embeddings can blur rare strings that the embedding model never learned well.
3. **False.** recall@5 is a retrieval metric (is a relevant chunk in the top 5?). Faithfulness is judged on the generated answer.

**A3.** *(1 mark each)*
- **TTFT:** the time from sending the request until the first token arrives, set by queueing and prefill.
- **TPOT:** the time between consecutive output tokens, set by decoding speed.
- **p99 latency:** the latency that 99% of requests beat, which measures the slow tail.

---

## B. Multiple choice

**B1. (b).**
- (a) Wrong: the bigger index is a side effect, not the goal.
- (c) Wrong: embedding models don't need repeated text.
- (d) Wrong: overlap doesn't affect BM25 speed.

**B2. (b).**
- (a) Wrong: ANN is slightly *less* accurate than exact search.
- (c) Wrong: ANN indexes the embeddings, so they are still needed.
- (d) Wrong: it has nothing to do with the LLM.

**B3. (b).** Re-ranking moves the right chunk to the top, and a smaller $k$ cuts the noise.
- (a) Wrong: a larger $k$ buries the chunk further.
- (c) Wrong: removing citations makes the answers harder to check.
- (d) Wrong: a higher temperature adds randomness and doesn't help.

**B4. (b).**
- (a), (c) and (d) don't set the cost.

**B5. (b).**
- (a) Wrong: RAG involves no training.
- (c) Wrong: RAG does use an embedding model.
- (d) Wrong: fine-tuning does change behaviour; it is just a poor way to store facts that change.

---

## C. Calculation

**C1.**
1. The sum of the lengths is $30 + 5 + 25 + 60 + 40 = 160$. The batch runs $\max = 60$ steps, so the utilisation is $160/(5 \times 60) = 53\%$. *(2 marks)*
2. The 5-token request finishes at step 5 and its slot sits idle for 55 steps. A new request waits for the whole batch: up to 60 steps. *(1 mark)*

**C2.**
1. $3{,}000 \times 3{,}600 = 10.8$M tokens per hour, so the cost is $\$3.50/10.8 \approx \$0.32$ per million. *(2 marks)*
2. $0.3 + 400 \times 0.02 = 8.3$ s in total. Streaming means the user starts reading after 0.3 s. *(1 mark)*

**C3.**
1. Each chunk starts $800 - 100 = 700$ tokens after the previous one, at 0, 700, …, 24,500. The chunk starting at 24,500 is the first that reaches the end ($24{,}500 + 800 \ge 25{,}000$), so there are $24{,}500/700 + 1 = $ **36 chunks**, the last starting at **24,500**. *(2 marks)*
2. In float32: $2\text{M} \times 1024 \times 4 = 8.2$ GB. In float16: 4.1 GB. *(2 marks)*

**C4.**
1. $\lVert q\rVert = \sqrt{0.01 + 0.9025 + 0.09} = 1.001$. *(3 marks)*

| Chunk | $q \cdot c$ | $\lVert c\rVert$ | Cosine | Rank |
|---|---|---|---|---|
| c1 | $0.01 + 0.855 + 0.06 = 0.925$ | 0.927 | **0.996** | 1 |
| c3 | $0.02 + 0.76 + 0.15 = 0.930$ | 0.964 | **0.963** | 2 |
| c4 | $0.01 + 0.095 + 0.285 = 0.390$ | 0.960 | 0.406 | 3 |
| c2 | $0.09 + 0.095 + 0.03 = 0.215$ | 0.911 | 0.236 | 4 |

   c3 has the larger dot product, but c1 points more exactly the same way as $q$, so cosine ranks c1 first.
2. With $k = 2$, c1 and c3 are retrieved. The prompt: *(1 mark)*
   > Answer using only the context below. Cite sources as [1], [2]. If the answer is not in the context, say so.
   > [1] The campus bus leaves the main gate at 8 am.
   > [2] On Fridays a late bus leaves at 10 pm.
   > Question: What time does the bus leave?
3. *"The campus bus leaves the main gate at 8 am [1]. On Fridays there is also a late bus at 10 pm [2]."* *(1 mark)*

**C5.** *(1 mark each)*
- recall@1 $= 1/4 = 0.25$ (only question 2)
- recall@3 $= 2/4 = 0.50$ (questions 1 and 2)
- MRR $= \frac14(\frac12 + 1 + 0 + \frac14) = 0.4375$

**C6.** *(3 marks)*

| Doc | BM25 rank | Vector rank | RRF |
|---|---|---|---|
| A | 3 | 1 | $1/63 + 1/61 = 0.03227$ |
| B | 1 | 4 | $1/61 + 1/64 = 0.03202$ |
| C | 2 | 5 | $1/62 + 1/65 = 0.03151$ |
| D | — | 2 | $1/62 = 0.01613$ |
| E | — | 3 | $1/63 = 0.01587$ |

The final order is **A, B, C, D, E**. Documents found by both methods rank above documents found by only one.

---

## D. Conceptual

**D1.** In static batching, finished requests keep their slots until the longest one ends, and new requests can't join. Continuous batching refills each slot as soon as it frees up, so the GPU processes a nearly full batch at every step, and each weight read serves more tokens. *(2 marks)* Every request in the batch needs its own KV cache, which grows with its context length. The batch size is capped by how many caches fit in the GPU's memory next to the weights. *(1 mark)*

**D2.** It is a **faithfulness failure**, or hallucination: the claim is not supported by the context. *(1 mark)* *(1 mark each, any two fixes)*
- Instruct the model more strictly ("use only the context; say *I don't know* otherwise") and require a citation for every claim.
- Check the retrieval: maybe the right chunk wasn't retrieved, so fix it with hybrid search or better chunking.
- Use a stronger model, or verify the answer against the retrieved text automatically (a faithfulness check).
- Lower the temperature.

**D3.** The string "27B-X" is rare. The embedding model has probably never learned a meaningful vector for it, so it is blurred into "some form" and the specific document doesn't score highest. *(1 mark)* BM25 matches the exact token "27B-X", so the right document ranks first in the keyword list, and RRF lifts it to the top of the merged list. *(1 mark)*

---

## E. Applied / coding

**E1.** *(5 marks: 1 chunking, 1 embedding, 1 retrieval, 1 prompt, 1 checks)*

```python
import numpy as np

def chunk(tokens: list, size: int, overlap: int) -> list[list]:
    """Fixed-size chunks with overlap; the last chunk reaches the end of the input."""
    step, out, start = size - overlap, [], 0
    while True:
        out.append(tokens[start:start + size])
        if start + size >= len(tokens):
            return out
        start += step

pieces = chunk(list(range(25_000)), 800, 100)
print(len(pieces), pieces[-1][0])                  # 36 24500

NOTICES = [
    "the campus bus leaves the main gate at 8 am",
    "the library opens at 9 am and closes at 9 pm",
    "on fridays a late bus leaves at 10 pm",
    "semester fees are due on 30 june",
    "the hostel mess serves dinner from 7 pm",
    "lab record submission is on monday",
    "the library is closed on public holidays",
]
STOP = {"the", "a", "at", "on", "is", "and", "from", "are", "when", "does", "what", "time", "do"}

def words(text: str) -> list[str]:
    """Lower-case words without stopwords, with a crude plural/verb 's' stripped (leaves -> leave)."""
    out = []
    for w in text.lower().split():
        if w not in STOP:
            out.append(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out

VOCAB = sorted({w for n in NOTICES for w in words(n)})

def embed(text: str) -> np.ndarray:
    """Bag-of-words vector over VOCAB, normalised to length 1."""
    ws = words(text)
    v = np.array([ws.count(w) for w in VOCAB], float)
    return v / (np.linalg.norm(v) or 1.0)

INDEX = np.stack([embed(n) for n in NOTICES])

def retrieve(question: str, k: int = 2) -> list[str]:
    """Top-k notices by cosine similarity."""
    scores = INDEX @ embed(question)
    return [NOTICES[i] for i in np.argsort(-scores)[:k]]

def build_prompt(question: str, chunks: list[str]) -> str:
    """Numbered context plus citation instructions."""
    ctx = "\n".join(f"[{i}] {c}" for i, c in enumerate(chunks, 1))
    return ("Answer using only the context. Cite sources as [1], [2]. "
            "If the answer is not there, say you don't know.\n" + ctx + f"\nQuestion: {question}")

for q in ["when does the bus leave", "when is the library open"]:
    print(build_prompt(q, retrieve(q)), "\n")
```

**E2.** *(3 marks)*

```python
def recall_at_k(ranks: list, k: int) -> float:
    """Share of questions whose first relevant chunk is within the top k (None = not retrieved)."""
    return sum(r is not None and r <= k for r in ranks) / len(ranks)

def mrr(ranks: list) -> float:
    """Mean of 1/rank (0 when not retrieved)."""
    return float(np.mean([1 / r if r else 0.0 for r in ranks]))

def rrf(lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion of several ranked lists."""
    scores = {}
    for ranked in lists:
        for rank, doc in enumerate(ranked, 1):
            scores[doc] = scores.get(doc, 0.0) + 1 / (k + rank)
    return sorted(((d, round(s, 5)) for d, s in scores.items()), key=lambda x: -x[1])

ranks = [2, 1, None, 4]
print(recall_at_k(ranks, 1), recall_at_k(ranks, 3), mrr(ranks))     # 0.25 0.5 0.4375
print(rrf([["B", "C", "A"], ["A", "D", "E", "B", "C"]]))
```

---

## F. ★ Challenge

1. At peak, $100 \times (1{,}500 + 300) = 180{,}000$ tokens are in flight. MHA: $180{,}000 \times 0.5$ MB ≈ **90 GB**. GQA: ≈ **22.5 GB**. *(1 mark)*
2. Each GPU has $80 - 13.5 = 66.5$ GB free. MHA needs $90/66.5 \to$ **2 GPUs** (one replica's weights on each, with caches spread across them). GQA fits on **1 GPU**. *(1 mark)*
3. *(2 marks)*
   - **Cutting memory or cost:** quantize the weights (INT4 frees about 10 GB per GPU) or the KV cache (8-bit halves it); use PagedAttention so fragmented memory isn't wasted; fetch fewer but better chunks (re-ranking), which shortens the prompts; route easy questions to a smaller model.
   - **Shared system prompt:** turn on **prefix caching**. The K/V of the common 1,000-token prefix is computed once and reused by every request, so each prefill only processes the new tokens, which cuts TTFT and compute.
