# Section 11 — Solutions: Encoder vs Decoder vs Both

---

## A. Warm-up

**A1.** (1) The **attention mask**: bidirectional, causal, or both together with cross-attention. (2) The **training objective**: masked LM, causal LM or span corruption. *(1 mark each)*

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **True.** The prefix is attended bidirectionally, so every prompt token sees the whole prompt.
2. **False.** Causal LM supervises **every** position. The 15% figure belongs to MLM.
3. **True.** For example, the target `<X> study late <Y> before <Z>` lists each missing span after its sentinel.

**A3.** *(½ mark each)*
- Encoder-only: BERT, ModernBERT
- Decoder-only: LLaMA, GPT-2
- Encoder–decoder: T5, Whisper

---

## B. Multiple choice

**B1. (c).** This is not a reason: encoders are usually better or cheaper for embeddings, and decoders won for *generation* and generality.
- (a), (b) and (d) are all real reasons from the notes.

**B2. (b).** A small, fast encoder is ideal for high-volume labelling.
- (a) Wrong: a 70B model is far too expensive at 50 million tickets a day.
- (c) Wrong: it works, but generating text is slower than a classification head, with no benefit here.
- (d) Wrong: training a new model from scratch is unnecessary.

**B3. (b).** $10 \cdot 11/2 = 55$.
- (a) 45 leaves out the diagonal.
- (c) Wrong: 90 is $n(n-1)$, the number of ordered pairs of *different* tokens.
- (d) 100 is the full $n^2$ of an encoder.

**B4. (c).** Causal LM supervises every token.
- (a) and (b) supervise about 15% of the tokens each.
- (d) Next sentence prediction gives one label per sentence pair.

**B5. (b).**
- (a) Wrong: no weights change in in-context learning.
- (c) Wrong: a longer context window is a different idea.
- (d) Wrong: that describes RAG (Section 18).

---

## C. Calculation

**C1.**
1. *(2 marks)*
   - Encoder: $6^2 = 36$
   - Decoder: $6 \cdot 7/2 = 21$
   - Prefix-LM: $4^2 + 5 + 6 = 27$
2. *(2 marks)*

```
        k1 k2 k3 k4 k5 k6
   q1   ✓  ✓  ✓  ✓  ·  ·
   q2   ✓  ✓  ✓  ✓  ·  ·
   q3   ✓  ✓  ✓  ✓  ·  ·
   q4   ✓  ✓  ✓  ✓  ·  ·
   q5   ✓  ✓  ✓  ✓  ✓  ·
   q6   ✓  ✓  ✓  ✓  ✓  ✓
```

3. Encoder $36/36 = 100\%$; decoder $21/36 = 58\%$; prefix-LM $27/36 = 75\%$. *(1 mark)*

**C2.** *(3 marks)*
- Encoder self-attention: $4^2 = 16$
- Decoder self-attention: $3 \cdot 4/2 = 6$
- Cross-attention: $3 \times 4 = 12$
- Total: $34$

**C3.** *(2 marks per part)*
1. One example: input `the train [MASK] the station early this morning`, target `left` at position 3. **1** supervised position. (Any single hidden token is correct.)
2. Input: `<bos> the train left the station early this morning`. Targets: `the train left the station early this morning <eos>`. **9** supervised positions.
3. Input: `the train <X> station early <Y>`. Target: `<X> left the <Y> this morning <Z>`.

**C4.** MLM: $0.15 \times 50 = 7.5$. Span corruption: about 7–8 tokens. Causal LM: 50 (51 with `<eos>`). *(2 marks)*

---

## D. Conceptual

**D1.** *(1 mark each)*
- **Encoder:** input `[CLS] the team won the final [SEP]`. The output is a label from a classifier on the `[CLS]` vector. It needs a classification head, fine-tuned on labelled examples.
- **Encoder–decoder:** input `topic: the team won the final`, output the text `sport`. No new parts are needed; the label is generated as text (after fine-tuning on the task, or if the model was instruction-tuned).
- **Decoder-only:** input `Text: the team won the final. Topic (sport or politics):`, output the next token ` sport`. No new parts and no fine-tuning are needed for a large enough model; the prompt defines the task.

**D2.** The input (thousands of audio frames) is long and must be understood as a whole, so it suits a **bidirectional** encoder that reads it once. *(1 mark)* The output (the transcript) is much shorter, so a small causal decoder can generate it, **cross-attending** to the encoded audio at every step. *(1 mark)* The expensive part, encoding, happens only once rather than at every generated token, and the decoder can stay small, which suits limited hardware. *(1 mark)*

**D3.** In a bidirectional encoder, every token's vector depends on **all** the other tokens, including later ones. *(1 mark)* Appending a new token changes the keys and values of every earlier position, so nothing can be reused, and each step must recompute the whole sequence. *(1 mark)* In a causal decoder, earlier tokens never see later ones, so their keys and values are final once computed and can be cached. *(1 mark)*

---

## E. Applied / coding

**E1.** *(4 marks: 1 per pattern, 1 for checking the formula)*

```python
import numpy as np

def mask(kind: str, n: int, p: int | None = None) -> np.ndarray:
    """Boolean n x n attention pattern (True = allowed)."""
    if kind == "encoder":
        return np.ones((n, n), dtype=bool)
    causal = np.tril(np.ones((n, n), dtype=bool))
    if kind == "decoder":
        return causal
    if kind == "prefix":
        m = causal.copy()
        m[:p, :p] = True                     # the prefix sees itself bidirectionally
        return m
    raise ValueError(kind)

print([int(mask(k, 6, 4).sum()) for k in ("encoder", "decoder", "prefix")])   # [36, 21, 27]
for n in range(1, 30):
    for p in range(0, n + 1):
        assert mask("prefix", n, p).sum() == p * p + sum(range(p + 1, n + 1))
print("prefix formula verified for n < 30")
```

**E2.** *(4 marks: 2 input, 2 target)*

```python
def span_corrupt(tokens: list[str], spans: list[tuple[int, int]]) -> tuple[list[str], list[str]]:
    """T5-style span corruption with sentinels <X>, <Y>, <Z>, ..."""
    sentinels = ["<X>", "<Y>", "<Z>", "<W>", "<V>"]
    inp, tgt, pos = [], [], 0
    for s, (start, end) in zip(sentinels, spans):
        inp += tokens[pos:start] + [s]
        tgt += [s] + tokens[start:end]
        pos = end
    inp += tokens[pos:]
    tgt += [sentinels[len(spans)]]           # final sentinel marks the end
    return inp, tgt

sent = "the train left the station early this morning".split()
inp, tgt = span_corrupt(sent, [(2, 4), (6, 8)])
print(" ".join(inp))   # the train <X> station early <Y>
print(" ".join(tgt))   # <X> left the <Y> this morning <Z>
```

---

## F. ★ Challenge

1. $\frac{n(n+1)/2}{n^2} = \frac{n + 1}{2n} = \frac12 + \frac{1}{2n} \to \frac12$. *(1 mark)*
2. $p^2 + \sum_{i=p+1}^{n} i = p^2 + \frac{n(n+1) - p(p+1)}{2}$. With $p = n/2$: *(2 marks)*
$$\frac{n^2}{4} + \frac{n^2 + n - \frac{n^2}{4} - \frac{n}{2}}{2} = \frac{n^2}{4} + \frac{3n^2}{8} + \frac{n}{4} = \frac{5n^2}{8} + \frac{n}{4},$$
so the fraction is $\frac58 + \frac{1}{4n} \to \frac58$.
3. *(1 mark for one of each)*
   - **Advantage:** each prompt token sees the *whole* prompt, so the prompt can be understood better, as in an encoder.
   - **Disadvantage:** if the user adds a new turn and it becomes part of the prefix, every earlier prefix token should now attend to it. Their keys and values change, so the KV cache can't simply be extended. With a plain causal decoder, old cache entries stay valid forever.
