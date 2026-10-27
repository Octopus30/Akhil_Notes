"""Section 18 lab: Serving & RAG.

Serving arithmetic (batching utilisation, cost per token, latency), a
simulation of static vs continuous batching, and a small but complete RAG
retriever: chunking, TF-IDF embeddings, cosine search, BM25, hybrid search
with reciprocal rank fusion, and retrieval metrics.

Run:  uv run python 18_Serving_and_RAG/lab.py
"""

# %% 1. Setup
import math
from collections import Counter

import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=3, suppress=True)


def chunk(tokens: list, size: int, overlap: int) -> list[list]:
    """Fixed-size chunks with overlap; the last chunk reaches the end."""
    out, start = [], 0
    while True:
        out.append(tokens[start:start + size])
        if start + size >= len(tokens):
            return out
        start += size - overlap


def rrf(lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: sum of 1 / (k + rank) over the ranked lists."""
    scores: dict[str, float] = {}
    for ranked in lists:
        for r, d in enumerate(ranked, 1):
            scores[d] = scores.get(d, 0.0) + 1 / (k + r)
    return sorted(scores.items(), key=lambda kv: -kv[1])


# %% 2. Reproduce the worked examples from notes.md
lengths = [10, 50, 20, 40]
print(f"6.1 Static batch utilisation: {sum(lengths)} / {len(lengths) * max(lengths)} = "
      f"{sum(lengths) / (len(lengths) * max(lengths)):.0%}")
print(f"6.2 Cost: $2 / ({2000 * 3600 / 1e6:.1f}M tokens) = ${2 / 7.2:.2f} per million;"
      f" latency 0.2 + 300 x 0.025 = {0.2 + 300 * 0.025:.1f} s")
chunks = chunk(list(range(10_000)), 500, 50)
print(f"6.3 Chunks of a 10,000-token document (500, overlap 50): {len(chunks)};"
      f" 1M x 768 float32 vectors = {1e6 * 768 * 4 / 1e9:.1f} GB")
assert len(chunks) == 23

C = {"c1": [0.9, 0.1, 0.2], "c2": [0.2, 0.9, 0.1], "c3": [0.6, 0.3, 0.6], "c4": [0.1, 0.2, 0.9]}
q = np.array([0.9, 0.2, 0.2])
cos = {c: float(q @ np.array(v) / (np.linalg.norm(q) * np.linalg.norm(v))) for c, v in C.items()}
print("6.4 Cosine:", {c: round(s, 3) for c, s in sorted(cos.items(), key=lambda kv: -kv[1])},
      " (notes: c1 0.994, c3 0.848)")

ranks = [1, 3, None]
rec = lambda k: sum(r is not None and r <= k for r in ranks) / len(ranks)
print(f"6.5 recall@1 {rec(1):.2f}, recall@3 {rec(3):.2f}, MRR {np.mean([1 / r if r else 0 for r in ranks]):.2f}")
print("6.6 RRF:", [(d, round(s, 5)) for d, s in rrf([["A", "B"], ["B", "C", "A"]])], " (notes: B, A, C)")
print(f"6.7 Context budget: 200 + 5 x 500 + 50 + 500 = {200 + 5 * 500 + 50 + 500} of 4,096")


# %% 3. A general implementation: a small RAG retriever
NOTICES = [
    "The central library is open from 8 am to 10 pm on weekdays.",
    "On Sundays the central library closes early, at 6 pm.",
    "The hostel mess serves dinner between 7 pm and 9:30 pm.",
    "Semester exam fees must be paid online before 15 March.",
    "Late fee of 500 rupees applies to exam fee payments after the deadline.",
    "The campus bus to the railway station leaves the main gate at 8 am and 5 pm.",
    "Lab record books must be submitted to the department office by Friday.",
    "The sports complex gym is open to students from 6 am to 8 am.",
    "Wi-Fi passwords for the new hostel block are available at the warden's office.",
    "Form 27B-X is needed to apply for a hostel room change.",
]
STOP = {"the", "a", "an", "to", "of", "is", "are", "at", "on", "for", "and", "be", "by", "from",
        "must", "in", "when", "what", "do", "does", "i", "how", "can", "my", "after", "before"}


def words(text: str) -> list[str]:
    """Lower-case words, punctuation stripped, stopwords removed, a trailing plural 's' stripped."""
    out = []
    for w in text.lower().replace(",", " ").replace(".", " ").replace("?", " ").split():
        if w not in STOP:
            out.append(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


DOCS = [words(n) for n in NOTICES]
VOCAB = sorted({w for d in DOCS for w in d})
DF = Counter(w for d in DOCS for w in set(d))
IDF = {w: math.log(len(DOCS) / DF[w]) + 1 for w in VOCAB}


def tfidf(ws: list[str]) -> np.ndarray:
    """TF-IDF vector over VOCAB, normalised to length 1 (a stand-in for an embedding model)."""
    c = Counter(ws)
    v = np.array([c[w] * IDF[w] for w in VOCAB])
    return v / (np.linalg.norm(v) or 1.0)


INDEX = np.stack([tfidf(d) for d in DOCS])           # one row per notice
assert INDEX.shape == (len(NOTICES), len(VOCAB))


def vector_search(question: str, k: int = 3) -> list[int]:
    """Indices of the top-k notices by cosine similarity."""
    return [int(i) for i in np.argsort(-(INDEX @ tfidf(words(question))))[:k]]


def bm25_search(question: str, k: int = 3, k1: float = 1.5, b: float = 0.75) -> list[int]:
    """Indices of the top-k notices by BM25 keyword score."""
    avg = np.mean([len(d) for d in DOCS])
    qw = words(question)
    scores = []
    for d in DOCS:
        c, s = Counter(d), 0.0
        for w in qw:
            if w in c:
                idf = math.log(1 + (len(DOCS) - DF[w] + 0.5) / (DF[w] + 0.5))
                s += idf * c[w] * (k1 + 1) / (c[w] + k1 * (1 - b + b * len(d) / avg))
        scores.append(s)
    return [int(i) for i in np.argsort(-np.array(scores))[:k] if scores[i] > 0]


def build_prompt(question: str, ids: list[int]) -> str:
    """Numbered context with citation instructions."""
    ctx = "\n".join(f"[{n}] {NOTICES[i]}" for n, i in enumerate(ids, 1))
    return f"Answer using only the context; cite sources as [n].\n{ctx}\nQuestion: {question}"


print("\nMini RAG over", len(NOTICES), "notices, vocabulary", len(VOCAB))
print(build_prompt("When is the library open on Sunday?", vector_search("When is the library open on Sunday?", 2)))


# %% 4. Experiments
def experiment_batching() -> None:
    """Simulate 200 requests with random lengths on 8 slots: static vs continuous batching."""
    print("\n[Experiment 1] Static vs continuous batching (8 slots, 200 requests)")
    lens = rng.integers(10, 300, size=200)
    static_steps = sum(lens[i:i + 8].max() for i in range(0, 200, 8))
    slots, queue, steps = [0] * 8, list(lens), 0
    while queue or any(slots):
        for j in range(8):
            if slots[j] == 0 and queue:
                slots[j] = queue.pop(0)
        slots = [max(0, s - 1) for s in slots]
        steps += 1
    total = int(lens.sum())
    print(f"  static    : {static_steps:5d} steps, utilisation {total / (8 * static_steps):.0%}")
    print(f"  continuous: {steps:5d} steps, utilisation {total / (8 * steps):.0%}"
          f"  ({static_steps / steps:.2f}x throughput)")


def experiment_hybrid() -> None:
    """Keyword vs vector vs hybrid on queries with rare codes or reworded phrases."""
    print("\n[Experiment 2] Vector, BM25 and hybrid (RRF) retrieval")
    for qtext in ["form 27B-X", "late payment penalty for exam fee", "bus to the station"]:
        v, kw = vector_search(qtext), bm25_search(qtext)
        fused = [int(d) for d, _ in rrf([[str(i) for i in kw], [str(i) for i in v]])][:3]
        print(f"  {qtext!r:38}: vector {v}  bm25 {kw}  hybrid {fused}")
    print("  Note: our TF-IDF 'embedding' is itself word-based, so it also finds the rare code 27B-X.")
    print("  A learned neural embedding often blurs such codes; that is where BM25 + RRF earns its keep.")


def experiment_metrics() -> None:
    """recall@k and MRR on a labelled test set for this mini index."""
    print("\n[Experiment 3] Retrieval metrics on 6 labelled questions")
    tests = [("library timing on weekdays", 0), ("sunday library hours", 1), ("dinner time in the mess", 2),
             ("exam fee deadline", 3), ("gym opening time", 7), ("hostel wifi password", 8)]
    rk = []
    for qtext, gold in tests:
        found = vector_search(qtext, k=5)
        rk.append(found.index(gold) + 1 if gold in found else None)
    print(f"  ranks {rk}: recall@1 {sum(r == 1 for r in rk) / len(rk):.2f}, "
          f"recall@3 {sum(r is not None and r <= 3 for r in rk) / len(rk):.2f}, "
          f"MRR {np.mean([1 / r if r else 0 for r in rk]):.2f}")


experiment_batching()
experiment_hybrid()
experiment_metrics()


# %% 5. Student TODOs
def cost_per_million(dollars_per_hour: float, tokens_per_second: float) -> float:
    """Serving cost in dollars per million output tokens."""
    # TODO(student): tokens per hour = tokens_per_second * 3600.
    raise NotImplementedError


def max_chunks(context: int, system: int, question: int, answer: int, chunk_tokens: int) -> int:
    """How many retrieved chunks fit in the context window."""
    # TODO(student): integer division of what is left.
    raise NotImplementedError


def mrr(ranks: list) -> float:
    """Mean reciprocal rank (None = not retrieved, counts as 0)."""
    # TODO(student): average of 1 / rank.
    raise NotImplementedError


checks = [
    ("cost_per_million", lambda: np.isclose(cost_per_million(2, 2000), 0.2778, atol=1e-3)),
    ("max_chunks", lambda: max_chunks(4096, 200, 50, 500, 500) == 6 and max_chunks(8192, 300, 100, 1000, 800) == 8),
    ("mrr", lambda: np.isclose(mrr([2, 1, None, 4]), 0.4375)),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
