# Section 11 — Exercises: Encoder vs Decoder vs Both

**Total: 46 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 50 minutes**

Rows are queries and columns are keys. "Allowed pairs" counts the entries of the attention pattern that are not masked.

---

## A. Warm-up (recall)

**A1.** All Transformers are built from the same block. Name the **two** things that make a model an encoder, a decoder or an encoder–decoder. *(2 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. In a prefix-LM, a prompt token can attend to later prompt tokens.
2. Causal language modelling supervises only about 15% of the positions.
3. In T5's span corruption, the decoder's target contains the missing spans, each introduced by its sentinel token.

**A3.** Match each model to its architecture: BERT, LLaMA, T5, Whisper, GPT-2, ModernBERT. *(3 marks)*

---

## B. Multiple choice (1 mark each)

**B1.** Which is **not** a reason decoder-only models became the default for LLMs?
- (a) Every token of any raw text is a training signal.
- (b) Any task can be posed as "continue this text".
- (c) They give better fixed-size sentence embeddings than encoders, at the same size.
- (d) Causal attention allows a KV cache, which makes generation efficient.

**B2.** A company must tag 50 million support tickets per day with one of 12 categories, cheaply. The best first choice is:
- (a) A 70B decoder-only LLM with a long prompt.
- (b) A fine-tuned encoder with a classification head.
- (c) An encoder–decoder that generates the category name.
- (d) A prefix-LM trained from scratch.

**B3.** For $n = 10$ tokens, a causal decoder has how many allowed pairs?
- (a) 45
- (b) 55
- (c) 90
- (d) 100

**B4.** Which objective yields the most supervised predictions per sentence?
- (a) Masked LM
- (b) Span corruption
- (c) Causal LM
- (d) Next sentence prediction

**B5.** In-context learning means:
- (a) Fine-tuning the model on the user's data.
- (b) The model performs a new task from instructions or examples given in the prompt, with no weight updates.
- (c) Training with a longer context window.
- (d) Retrieving documents from a database.

---

## C. Calculation

**C1. Attention patterns.** A sequence has $n = 6$ tokens: a 4-token prompt and a 2-token answer. *(5 marks)*

1. Count the allowed pairs for an encoder, a decoder and a prefix-LM with $p = 4$.
2. Draw the $6 \times 6$ prefix-LM pattern with ✓ and ·.
3. What fraction of the full $n^2$ does each pattern allow?

**C2. Encoder–decoder.** A source has 4 tokens and the target has 3. Count the encoder self-attention pairs, the decoder self-attention pairs, the cross-attention pairs, and the total. *(3 marks)*

**C3. Three objectives.** Sentence (8 tokens): **the train left the station early this morning**. *(6 marks)*

1. Give an MLM input and target that hides 1 token. How many positions are supervised?
2. Give the causal-LM input and targets. How many positions are supervised (count `<eos>`)?
3. Corrupt the spans "left the" and "this morning" T5-style. Write the input and the target.

**C4. Signals per sentence.** For a 50-token sentence, give the expected number of supervised predictions under MLM (15%), span corruption (15%) and causal LM. *(2 marks)*

---

## D. Conceptual

**D1.** Frame the task "decide whether *'the team won the final'* is about sport or politics" for an encoder, an encoder–decoder and a decoder-only model. Give the input, the output, and any extra parts needed for each. *(3 marks)*

**D2.** Why does a long-input, short-output task such as speech-to-text suit an encoder–decoder? *(3 marks)*

**D3.** Why can't a bidirectional encoder use a KV cache to generate text efficiently, while a causal decoder can? *(3 marks)*

---

## E. Applied / coding (NumPy)

**E1.** Write `mask(kind, n, p=None)` that returns an $n \times n$ boolean pattern for `"encoder"`, `"decoder"` and `"prefix"` (with prefix length `p`). Check your counts from C1, and check the formula $p^2 + \sum_{i=p+1}^{n} i$ for several $n$ and $p$. *(4 marks)*

**E2.** Write `span_corrupt(tokens, spans)`, where `spans` is a list of `(start, end)` index pairs (end exclusive, non-overlapping, sorted). It returns the T5-style `(input, target)` using the sentinels `<X>`, `<Y>`, `<Z>`, …, with a final sentinel at the end of the target. Check it against C3. *(4 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

1. Show that the fraction of allowed pairs in a causal decoder, $\frac{n(n+1)/2}{n^2}$, tends to $\frac12$ as $n$ grows.
2. For a prefix-LM, write the allowed pairs in closed form, and show that with $p = n/2$ the fraction tends to $\frac58$.
3. A prefix-LM sees more context per prompt token than a decoder. Give one advantage and one disadvantage of that for a chat model that keeps a KV cache across turns.
