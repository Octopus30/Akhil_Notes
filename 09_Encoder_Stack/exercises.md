# Section 9 — Exercises: The Encoder Stack

**Total: 45 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 55 minutes**

Simplified block used in the notes: $X^{(\ell)} = X^{(\ell-1)} + \text{softmax}\!\left(X^{(\ell-1)}X^{(\ell-1)\top}/\sqrt{2}\right)X^{(\ell-1)}$ (no LayerNorm, no FFN). Round to 2 decimal places unless told otherwise.

---

## A. Warm-up (recall)

**A1.** Describe the input, the processing and the output of an encoder, with shapes. *(3 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. "Bidirectional" means the encoder reads the sentence once left-to-right and once right-to-left.
2. In masked language modelling, the loss is computed at every position of the sentence.
3. An encoder's self-attention uses a padding mask but no causal mask.

**A3.** Fill in. *(3 marks)*

1. For $n = 128$ real tokens, one encoder head computes `__` allowed attention connections.
2. In BERT's recipe, of the picked tokens, `__`% become `[MASK]`, `__`% become a random token and `__`% stay unchanged.
3. For sentence classification, BERT uses the output vector of the `__` token.

---

## B. Multiple choice (1 mark each)

**B1.** Why can't a bidirectional encoder be trained with plain next-token prediction?
- (a) Encoders have no output head.
- (b) Every position can already see the next token, so it would just copy it.
- (c) Next-token prediction needs RoPE.
- (d) The loss would be too small.

**B2.** Why are 10% of the picked tokens left unchanged in MLM?
- (a) To save compute.
- (b) So the model must build a good representation for every token, not only for `[MASK]` positions, since real text has no `[MASK]`.
- (c) Because the tokenizer can't mask some words.
- (d) To make the task harder for the first epoch.

**B3.** ELECTRA's replaced-token-detection objective is more sample-efficient than MLM mainly because:
- (a) It uses a larger vocabulary.
- (b) It gets a training signal from every token, not just the 15% that were picked.
- (c) It doesn't need a tokenizer.
- (d) It removes attention.

**B4.** For semantic search over 10 million documents, which setup do you use first?
- (a) A cross-encoder over every (query, document) pair.
- (b) A bi-encoder: embed the documents once, then compare each query embedding by cosine similarity.
- (c) A decoder generating every document.
- (d) An MLM head on the query.

**B5.** In which layers do encoders tend to capture word sense and coreference?
- (a) The embedding layer.
- (b) The lower layers.
- (c) The upper layers.
- (d) Only the pooler.

---

## C. Calculation

Use the toy embeddings a = [0.1, 0.1], light = [1, 1], bag = [1.5, 0], lamp = [0, 1.5].

**C1. One encoder layer.** For *"a light bag"*: *(5 marks)*

1. Compute the layer-1 attention weights of "light" (with scaling by $\sqrt2$).
2. Compute light's layer-1 vector $X^{(1)}_{\text{light}}$ (remember the residual).
3. By symmetry, write light's layer-1 vector in *"a light lamp"*, without recomputing.

**C2. Contextual divergence.** A computer continues to layer 3 and gets light's vectors in the two sentences: *(4 marks)*

| Layer | "a light bag" | "a light lamp" |
|---|---|---|
| 0 | [1, 1] | [1, 1] |
| 1 | your C1 answer | your C1 answer |
| 2 | [4.30, 2.53] | [2.53, 4.30] |
| 3 | [8.94, 4.41] | [4.41, 8.94] |

1. Compute the cosine similarity between the two "light" vectors at layers 1, 2 and 3.
2. What trend do you see, and what does it mean?
3. Why do the numbers keep growing, and what in a real encoder stops that?

**C3. MLM loss.** A 40-token sentence goes through the BERT recipe. *(4 marks)*

1. How many tokens are picked on average, and how many of those become `[MASK]`, random or unchanged?
2. Suppose 6 are picked, and the model gives the original tokens probabilities 0.5, 0.25, 0.8, 0.4, 0.9 and 0.7. Compute the MLM loss.
3. How many of the 40 positions contribute to the loss?

**C4. BERT-large parameters.** Use the notes' recipe with $d = 1024$, 24 layers, $d_{ff} = 4096$, vocabulary 30,522, 512 positions, 2 segments. *(3 marks)*

1. Compute the embedding parameters (including the embedding LayerNorm).
2. Compute the parameters of one layer.
3. Compute the total, including the pooler.

---

## D. Conceptual

**D1.** Give one task where an encoder is a better choice than a decoder-only LLM, and explain why, mentioning speed or size and bidirectionality. *(3 marks)*

**D2.** Compare a bi-encoder and a cross-encoder for ranking documents: how each works, and why search systems often use both. *(3 marks)*

**D3.** Mean pooling versus the `[CLS]` vector: when might mean pooling give a better sentence embedding? *(2 marks)*

---

## E. Applied / coding (NumPy)

**E1.** Implement *(5 marks)*

```python
def mlm_corrupt(ids: np.ndarray, vocab_size: int, mask_id: int, rng,
                p: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Return (corrupted ids, boolean array of picked positions) using the 80/10/10 rule."""
```

Run it on 10,000 random token IDs, and check that about 15% are picked, and that of those about 80% became `mask_id`, about 10% changed to another token, and about 10% are unchanged. (A random replacement can occasionally equal the original token; explain why that is fine.)

**E2.** Implement `mlm_loss(probs_of_original, picked)` that averages $-\log p$ over the picked positions only. Check it against C3. *(2 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

1. In one forward pass over $n$ tokens, how many prediction targets does MLM give (expected value), and how many does next-token prediction give? For $n = 512$?
2. ELECTRA trains a small generator and a discriminator. Explain why the discriminator's task ("was this token replaced?") is still hard enough to learn good representations, even though most tokens are original.
3. Suggest why RoBERTa could drop next-sentence prediction without hurting downstream accuracy.
