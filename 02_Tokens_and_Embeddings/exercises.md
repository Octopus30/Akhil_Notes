# Section 2 — Exercises: Tokens & Embeddings

**Total: 49 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 55 minutes**

Use only ideas from Sections 1–2. Write the vocabulary size as $|\mathcal{V}|$ and shapes as `rows × columns`.

---

## A. Warm-up (recall)

**A1.** Define, in one sentence each: **tokenizer**, **token ID**, **embedding matrix**. *(3 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. The tokenizer is trained together with the model, so its merges keep changing during pre-training.
2. Multiplying a one-hot vector $o_i$ by $E$ gives the same result as reading row $i$ of $E$.
3. Two embeddings with cosine similarity 1 must be exactly equal vectors.

**A3.** Fill in the shapes. A model has $|\mathcal{V}| = 32{,}000$ and $d_{model} = 512$. It reads $n = 20$ tokens. *(3 marks)*

1. $E$ has shape `__ × __`.
2. The one-hot vector of one token has length `__`.
3. The embedded sentence $X$ has shape `__ × __`.

---

## B. Multiple choice (1 mark each)

**B1.** Why do modern models use sub-word tokens rather than whole words?
- (a) Sub-words make every sentence shorter than whole words would.
- (b) Any new word can still be built from known pieces, without a huge vocabulary.
- (c) Sub-word tokens don't need embeddings.
- (d) Whole-word tokenizers can't handle punctuation.

**B2.** Why not simply use single characters as tokens?
- (a) There are too many distinct characters.
- (b) Characters can't be embedded.
- (c) Sequences become several times longer, and attention cost grows with $n^2$.
- (d) Characters can't represent numbers.

**B3.** In BPE, which pair is merged at each step?
- (a) The pair that appears in the most distinct words.
- (b) The pair with the highest total count, weighting each word by its frequency.
- (c) The pair of the two most frequent single characters.
- (d) A random pair, chosen to keep the vocabulary balanced.

**B4.** What does **weight tying** do?
- (a) It forces every embedding to have length 1.
- (b) It reuses $E^{\top}$ as the output head, saving $|\mathcal{V}| \cdot d_{model}$ parameters.
- (c) It shares one embedding between tokens that have the same spelling.
- (d) It freezes $E$ during training.

**B5.** Vector $a = [1, 2]$ and $b = [10, 20]$. Which statement is true?
- (a) Their cosine similarity is 0.1, because $b$ is ten times longer.
- (b) Their cosine similarity is 1, because they point in the same direction.
- (c) Their cosine similarity is 50, the dot product.
- (d) Cosine similarity can't be computed for vectors of different lengths.

---

## C. Calculation

**C1. BPE by hand.** Train BPE on this corpus (word × count): `sun` × 5, `sung` × 3, `snug` × 2, `gun` × 1. *(6 marks)*

1. Write the starting vocabulary.
2. Perform **three** merges. For each round, list all the pair counts and say which pair is merged.
3. Write the final vocabulary and the merge rules in order.
4. Tokenize the new words `unsung` and `suns` using your merges.

**C2. Embedding lookup.** A toy vocabulary has $d_{model} = 3$: *(4 marks)*

| ID | Token | Row of $E$ |
|---|---|---|
| 0 | `<pad>` | [0.0, 0.0, 0.0] |
| 1 | rain | [0.5, −0.1, 0.7] |
| 2 | falls | [0.2, 0.6, −0.3] |
| 3 | softly | [−0.4, 0.3, 0.2] |

1. Write the one-hot vector for "falls" and compute $o^{\top} E$ in full, showing that it equals row 2.
2. Write $X$ for the sentence "rain falls softly" and give its shape.
3. Compute $u^{\top} E$ for $u = [0, 0.5, 0.5, 0]$. What does this vector represent?

**C3. Cosine similarity.** Toy embeddings: mango = [3, 4, 0], apple = [4, 3, 0], car = [0, 3, 4]. *(4 marks)*

1. Compute $\cos(\text{mango}, \text{apple})$ and $\cos(\text{mango}, \text{car})$.
2. Let big_mango = [6, 8, 0]. Compute $\cos(\text{mango}, \text{big\_mango})$, and the Euclidean distances $\lVert\text{mango} - \text{big\_mango}\rVert$ and $\lVert\text{mango} - \text{apple}\rVert$.
3. Which measure says mango is closer to big_mango, and which says it is closer to apple? Which matches "same meaning" better here?

**C4. Table sizes.** *(4 marks)*

1. BERT-base has $|\mathcal{V}| = 30{,}522$ and $d_{model} = 768$. How many parameters are in its embedding table? Roughly what share is that of its ≈110 M total?
2. LLaMA-3 8B has $|\mathcal{V}| = 128{,}256$ and $d_{model} = 4{,}096$, and does **not** tie its weights. How many parameters do the embedding table and the output head hold together? How much memory is that at 2 bytes per number?
3. Using "1,000 tokens ≈ 750 words", roughly how many tokens is a 3,000-word English essay?

---

## D. Conceptual

**D1.** A tokenizer was trained only on English web text. Explain why a Telugu or Hindi sentence of the same meaning will usually become many more tokens, and give two practical consequences for the user. *(3 marks)*

**D2.** During training, a token appears in a batch. Which rows of $E$ receive a gradient from that batch? What does this tell you about the embeddings of very rare tokens? *(3 marks)*

**D3.** What would happen if we fed raw token IDs (e.g. 1, 2, 3) straight into the model as numbers, instead of looking up embeddings? Give two reasons this is a bad idea. *(3 marks)*

---

## E. Applied / coding (NumPy / Python)

**E1. BPE from scratch.** Write two functions *(6 marks)*

```python
def train_bpe(corpus: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    """Return the merge rules, in the order they were learned."""

def tokenize(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Split word into characters and apply the merges in order."""
```

Check that training on the notes' corpus (`tea` × 4, `ten` × 3, `tent` × 2, `neat` × 1) with 4 merges gives `[('t','e'), ('te','n'), ('te','a'), ('ten','t')]`, and that `tokenize("neaten", merges)` returns `['n', 'e', 'a', 'ten']`.

**E2. Lookup two ways.** Given `E` of shape `|V| × d_model` and `ids` of shape `(n,)`, compute $X$ (1) by building a one-hot matrix of shape `n × |V|` and multiplying by `E`, and (2) by indexing `E[ids]`. Assert that both give the same $X$ of shape `n × d_model`. *(2 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

A team must choose between two tokenizers for a model with $d_{model} = 4{,}096$:

| Tokenizer | $|\mathcal{V}|$ | Average characters per token (assume) |
|---|---|---|
| A | 32,000 | 4.0 |
| B | 128,000 | 4.8 |

1. How many tokens does a 24,000-character document become with each?
2. Attention work grows with $n^2$. By what factor is the attention work for that document smaller with B?
3. How many more embedding parameters does B need (untied, embedding + output head)?
4. Give one reason a larger vocabulary helps and one reason it hurts, beyond these numbers.
