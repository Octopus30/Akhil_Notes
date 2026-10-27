# Section 9 — Solutions: The Encoder Stack

---

## A. Warm-up

**A1.** *(1 mark each)*
- **Input:** token embeddings plus positions, $X^{(0)}$, with shape $n \times d_{model}$.
- **Processing:** $N$ identical blocks of bidirectional self-attention and an FFN, each mapping $n \times d_{model}$ to $n \times d_{model}$.
- **Output:** the contextual vectors $H$ (after the final norm), $n \times d_{model}$, one per input token.

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **False.** All tokens see all other tokens at the same time, in one attention step. There is no second pass.
2. **False.** The loss is computed only at the picked positions $\mathcal{M}$, about 15% of them.
3. **True.** The padding mask stops attention to padding; every real token may attend to every other real token.

**A3.** *(1 mark each)*
1. $128^2 = 16{,}384$
2. 80, 10, 10
3. `[CLS]`

---

## B. Multiple choice

**B1. (b).**
- (a) Wrong: encoders can have any head on top.
- (c) Wrong: the training objective doesn't depend on the position scheme.
- (d) Wrong: it is precisely because the task becomes trivial that the loss would go to near zero without the model learning anything.

**B2. (b).**
- (a) Wrong: there is no saving in compute.
- (c) Wrong: any token can be replaced by `[MASK]`.
- (d) Wrong: the rule is the same throughout training.

**B3. (b).**
- (a) Wrong: the vocabulary size isn't the reason.
- (c) Wrong: ELECTRA still uses a tokenizer.
- (d) Wrong: it is a standard Transformer encoder.

**B4. (b).** Documents are embedded once, offline, and comparing a query with all of them is cheap (and can be done approximately at large scale, Section 18).
- (a) A cross-encoder would need 10 million forward passes per query. It is used only to re-rank the top results.
- (c) and (d) don't produce a ranking.

**B5. (c).**
- (a) The embedding layer isn't contextual at all.
- (b) The lower layers capture surface features.
- (d) The pooler is a single projection on `[CLS]`.

---

## C. Calculation

**C1.**
1. light's scaled scores are $[0.2, 2, 1.5]/\sqrt2 = [0.14, 1.41, 1.06]$. *(2 marks)* $e^{0.14} = 1.15$, $e^{1.41} = 4.11$, $e^{1.06} = 2.89$, with sum $8.15$, so the weights are $[0.14, 0.50, 0.35]$.
2. The attention output is $0.14[0.1, 0.1] + 0.50[1, 1] + 0.35[1.5, 0] = [1.05, 0.52]$. Adding the residual gives $X^{(1)}_{\text{light}} = [1, 1] + [1.05, 0.52] = [2.05, 1.52]$. *(2 marks)*
3. The lamp sentence is the bag sentence with the two axes swapped, so light gives $[1.52, 2.05]$. *(1 mark)*

**C2.**
1. *(2 marks)*
   - Layer 1: $\cos = \frac{2(2.05)(1.52)}{2.05^2 + 1.52^2} = \frac{6.23}{6.51} = 0.96$
   - Layer 2: $\frac{2(4.30)(2.53)}{4.30^2 + 2.53^2} = \frac{21.76}{24.89} = 0.87$
   - Layer 3: $\frac{2(8.94)(4.41)}{8.94^2 + 4.41^2} = \frac{78.85}{99.37} = 0.79$
2. The similarity drops: $1.00 \to 0.96 \to 0.87 \to 0.79$. The same word becomes more and more context-specific, "light as not heavy" versus "light as brightness". These are contextual vectors. *(1 mark)*
3. Each layer **adds** an attention output to the residual stream and nothing rescales it. Real encoders apply LayerNorm, which keeps the scale steady while keeping the directions. *(1 mark)*

**C3.**
1. $0.15 \times 40 = 6$ picked. Of those, on average $4.8$ become `[MASK]`, $0.6$ become random and $0.6$ stay unchanged. *(1 mark)*
2. $-\log p = [0.693, 1.386, 0.223, 0.916, 0.105, 0.357]$. The sum is $3.681$, so $\mathcal{L} = 3.681/6 = 0.613$. *(2 marks)*
3. Only the 6 picked positions contribute; the other 34 add nothing. *(1 mark)*

**C4.**
1. $(30{,}522 + 512 + 2)\times1024 + 2\times1024 = 31{,}782{,}912$ *(1 mark)*
2. $4d^2 + 4d + 2d\,d_{ff} + d_{ff} + d + 4d = 4{,}194{,}304 + 4{,}096 + 8{,}388{,}608 + 4{,}096 + 1{,}024 + 4{,}096 = 12{,}596{,}224$ *(1 mark)*
3. $31{,}782{,}912 + 24 \times 12{,}596{,}224 + (1024^2 + 1024) = 31{,}782{,}912 + 302{,}309{,}376 + 1{,}049{,}600 = 335{,}141{,}888 \approx$ **335M** *(1 mark)*

---

## D. Conceptual

**D1.** *(3 marks: 1 for a sensible task, 2 for the reasoning)* An example is spam or sentiment classification, or named-entity tagging, at high volume. A fine-tuned BERT-sized encoder (about 110M parameters) is tens of times smaller and faster than a 7B LLM, so it is cheap to run on millions of messages. Being bidirectional, every token's vector uses context from **both sides**, which is ideal for understanding a fixed input. It also gives one vector per token for tagging, without generating any text.

**D2.**
- A **bi-encoder** embeds the query and each document **separately** and compares them with cosine similarity. Document vectors can be computed in advance, so searching millions is fast, but the query and document never "see" each other. *(1 mark)*
- A **cross-encoder** feeds the query and document **together** into one encoder and outputs a relevance score. Attention between the two gives much better accuracy, but it needs one forward pass per pair. *(1 mark)*
- Systems use a bi-encoder to fetch the top ~100 candidates, then a cross-encoder to re-rank them. This gives speed and accuracy together. *(1 mark)*

**D3.** Mean pooling uses information from **every** token. It is often better when the model wasn't trained to put a sentence summary into `[CLS]`, for example when no NSP or classification objective was used, and especially for embedding models trained with contrastive learning on mean-pooled vectors. *(2 marks)*

---

## E. Applied / coding

**E1.** *(5 marks: 1 picking 15%, 2 the 80/10/10 split, 1 statistics check, 1 explanation)*

```python
import numpy as np

def mlm_corrupt(ids: np.ndarray, vocab_size: int, mask_id: int, rng,
                p: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Return (corrupted ids, boolean array of picked positions) using the 80/10/10 rule."""
    picked = rng.random(ids.shape) < p
    roll = rng.random(ids.shape)
    out = ids.copy()
    to_mask = picked & (roll < 0.8)
    to_random = picked & (roll >= 0.8) & (roll < 0.9)
    out[to_mask] = mask_id
    out[to_random] = rng.integers(0, vocab_size, size=to_random.sum())
    return out, picked                                   # the remaining 10% stay unchanged

rng = np.random.default_rng(0)
vocab_size, mask_id = 30_000, 103
ids = rng.integers(1_000, vocab_size, size=10_000)
out, picked = mlm_corrupt(ids, vocab_size, mask_id, rng)

print(f"picked    {picked.mean():.3f}")
print(f"[MASK]    {(out[picked] == mask_id).mean():.3f}")
print(f"changed   {((out[picked] != mask_id) & (out[picked] != ids[picked])).mean():.3f}")
print(f"unchanged {(out[picked] == ids[picked]).mean():.3f}")
```

The output is close to 0.15, 0.80, 0.10 and 0.10. A random replacement has a $1/\text{vocabulary size}$ chance of drawing the original token. That only nudges the "unchanged" share up by a negligible amount, and the model still has to predict the original token, so nothing breaks.

**E2.** *(2 marks)*

```python
def mlm_loss(probs_of_original: np.ndarray, picked: np.ndarray) -> float:
    """Average -log p over the picked positions only."""
    return float(-np.log(probs_of_original[picked]).mean())

probs = np.full(40, 0.01)                   # unpicked positions: any value, they are ignored
picked40 = np.zeros(40, dtype=bool)
picked40[[2, 7, 11, 19, 25, 33]] = True
probs[picked40] = [0.5, 0.25, 0.8, 0.4, 0.9, 0.7]
print(round(mlm_loss(probs, picked40), 3))  # 0.613
```

---

## F. ★ Challenge

1. MLM gives about $0.15n$ targets; next-token prediction gives about $n$. For $n = 512$ that is about **77 versus 512**, so roughly 6.7× more signal per pass for next-token prediction. *(1 mark)*
2. The generator is a real (small) masked LM, so its replacements are **plausible** words that fit the context, such as "cooked" → "ate", not random junk. Spotting them requires understanding the whole sentence: grammar, meaning and facts. Because the label applies at **every** token, the discriminator learns from all positions. *(2 marks)*
3. The NSP negatives came from a different document, so the task could mostly be solved by spotting a **topic change**, which is too easy to teach much. Training on longer, contiguous text with plain MLM gives the model more useful context, and MLM already teaches the cross-sentence understanding NSP was meant to teach. *(1 mark)*
