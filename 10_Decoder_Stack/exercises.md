# Section 10 — Exercises: The Decoder Stack

**Total: 45 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 55 minutes**

Use natural logarithms. Round to 3 decimal places for losses and to 2 elsewhere.

---

## A. Warm-up (recall)

**A1.** Write the sub-layers of (a) a decoder-only block and (b) a decoder block in an encoder–decoder model, in order. *(2 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. Cross-attention uses a causal mask over the source sentence.
2. Teacher forcing feeds the model its own previous predictions during training.
3. Generating 200 new tokens with a decoder needs at least 200 sequential forward passes (without tricks such as speculative decoding).

**A3.** Fill in the shapes. A decoder writes $n_{tgt} = 7$ tokens while cross-attending to $n_{src} = 12$ encoder outputs, with $d_k = 64$ and vocabulary size 32,000. *(3 marks)*

1. Cross-attention $Q$: `__ × __`, $K$: `__ × __`
2. Cross-attention score matrix: `__ × __`
3. Output-head logits for all positions: `__ × __`

---

## B. Multiple choice (1 mark each)

**B1.** For the training sentence `<bos> I like tea <eos>`, the decoder's **input** under teacher forcing is:
- (a) `I like tea <eos>`
- (b) `<bos> I like tea`
- (c) `<bos> I like tea <eos>`
- (d) `tea like I <bos>`

**B2.** In cross-attention, the keys and values come from:
- (a) The decoder's own previous tokens.
- (b) The encoder's output $H$.
- (c) The output head.
- (d) Random noise.

**B3.** How many training signals (predicted tokens) does one teacher-forced pass over a target of $T$ tokens give?
- (a) 1
- (b) About $0.15T$
- (c) $T$
- (d) $T^2$

**B4.** What is **exposure bias**?
- (a) The model sees too many examples of common words.
- (b) Training always uses correct prefixes, but generation uses the model's own outputs, so errors can push it into unfamiliar states.
- (c) The output head is biased toward `<eos>`.
- (d) Attention weights are biased toward the first token.

**B5.** Compared with an encoder layer of the same width, a decoder layer in an encoder–decoder model has:
- (a) The same number of parameters.
- (b) About a third more parameters, because of cross-attention.
- (c) Twice as many parameters.
- (d) Fewer parameters, because of masking.

---

## C. Calculation

**C1. Teacher forcing.** Training sentence: `<bos> rain falls softly <eos>`. *(5 marks)*

1. Write the input sequence, the target sequence, and a table of (position, context, target).
2. The model gives the correct targets probabilities 0.3, 0.5, 0.8 and 0.6. Compute the loss.
3. Compute the perplexity $e^{\mathcal{L}}$, and explain in one sentence what the number means.

**C2. Generation.** Use this toy next-token table (the probabilities depend only on the last token): *(4 marks)*

| After… | next-token probabilities |
|---|---|
| `<bos>` | rain 0.6, sun 0.4 |
| rain | falls 0.5, stops 0.3, `<eos>` 0.2 |
| sun | shines 0.9, `<eos>` 0.1 |
| falls | softly 0.4, `<eos>` 0.6 |
| stops | `<eos>` 1.0 |
| softly | `<eos>` 1.0 |
| shines | `<eos>` 1.0 |

1. Run greedy generation from `<bos>`. List each pass and the token it chose.
2. Compute the probability of the greedy sentence.
3. Find a complete sentence with a **higher** probability than the greedy one. What does this tell you about greedy decoding?

**C3. Cross-attention.** An encoder has read "light bag" and produced $H = [\text{light}: [1, 1];\ \text{bag}: [1.5, 0]]$. Use identity projections and $d_k = 2$. *(4 marks)*

1. The decoder's query is $q = [3, 0]$. Compute the weights and the output.
2. The decoder's query is $q = [0, 2]$. Compute the weights and the output.
3. Which source word does each query align with?

**C4. Parameters.** Ignore biases and norms. *(3 marks)*

1. A decoder-only model has $d = 1024$, 24 blocks and $d_{ff} = 4d$. How many parameters are in its blocks?
2. An encoder–decoder model has $d = 768$, 12 encoder and 12 decoder layers, $d_{ff} = 4d$, and a shared vocabulary of 32,128 tokens. How many parameters does it have in total?

---

## D. Conceptual

**D1.** Explain why training a decoder is parallel but generation is sequential. Include the role of the causal mask. *(3 marks)*

**D2.** A decoder-only chatbot has no encoder. How does it "understand" a long user prompt before answering? *(3 marks)*

**D3.** Name one practical way that exposure bias is reduced or made less harmful. *(2 marks)*

---

## E. Applied / coding (NumPy)

**E1.** Write `teacher_forcing_pairs(tokens)` that takes a list such as `["<bos>", "rain", "falls", "softly", "<eos>"]` and returns `(inputs, targets)`. Then write `sequence_loss(probs)`, the mean of $-\log p$. Check both against C1. *(3 marks)*

**E2.** Write `greedy_generate(table, start="<bos>", max_len=10)` for a dictionary table like the one in C2, stopping at `<eos>`. Also write `sentence_prob(table, tokens)`. Then search all complete sentences (a depth-first search is fine) and print the most probable one. *(5 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

A decoder-only model generates $T$ tokens after a prompt of $p$ tokens, **without** a KV cache. Each pass reprocesses the whole sequence so far.

1. How many token positions are processed in total over the $T$ passes? Give a formula.
2. Evaluate it for $p = 100$ and $T = 400$, and compare with the $p + T$ positions a KV cache would process (one per new token, after a prefill of $p$).
3. Teacher forcing on the same 500-token text processes how many positions? Explain why training doesn't have this problem.
