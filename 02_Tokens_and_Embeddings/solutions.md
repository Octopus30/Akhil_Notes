# Section 2 — Solutions: Tokens & Embeddings

---

## A. Warm-up

**A1.** *(1 mark each)*
- **Tokenizer:** the fixed procedure, built before training, that splits text into tokens and maps each token to an ID.
- **Token ID:** a token's position in the vocabulary, an integer from $0$ to $|\mathcal{V}| - 1$.
- **Embedding matrix $E$:** a learned table of shape $|\mathcal{V}| \times d_{model}$ with one row (one embedding) per token.

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **False.** The tokenizer is built before training and never changes afterwards; only the model's weights, including $E$, are trained.
2. **True.** Every zero in $o_i$ wipes out a row of $E$, and the single 1 keeps row $i$, so $o_i^{\top}E = E[i,:]$.
3. **False.** Cosine similarity ignores length. $[1, 2]$ and $[2, 4]$ have cosine 1 but are different vectors.

**A3.** *(1 mark each)*
1. `32000 × 512`
2. `32000`: one entry per vocabulary token.
3. `20 × 512`: one row per token ($n \times d_{model}$).

---

## B. Multiple choice

**B1. (b).** With sub-words, an unseen word such as "neaten" is still built from known pieces, and the vocabulary stays manageable (around 30k–130k tokens).
- (a) Wrong: sub-word sequences are usually *longer* than word sequences, because some words split into several tokens.
- (c) Wrong: every token, sub-word or not, has an embedding.
- (d) Wrong: punctuation can simply be treated as separate word tokens. The real problem with whole words is the unbounded vocabulary and unknown words.

**B2. (c).** A character-level sequence is about 4× longer than a sub-word one for English, and attention cost grows with $n^2$, so roughly 16× more attention work.
- (a) Wrong: there are few distinct characters. That is the one thing character tokens do well.
- (b) Wrong: characters can be embedded like any other token.
- (d) Wrong: digits are characters too.

**B3. (b).** BPE counts every adjacent pair, weighting each word by how often it occurs, and merges the pair with the highest total.
- (a) Wrong: the number of distinct words that contain a pair isn't what is counted. A pair inside one very common word can win.
- (c) Wrong: BPE counts adjacent pairs, not single characters.
- (d) Wrong: BPE is fully deterministic.

**B4. (b).** The output head needs a $d_{model} \times |\mathcal{V}|$ matrix, and $E^{\top}$ has exactly that shape.
- (a) Wrong: that would be normalisation, which is a different idea.
- (c) Wrong: each token already has exactly one embedding, so there is nothing to share.
- (d) Wrong: tied weights are still trained, and receive gradients from both uses.

**B5. (b).** $b = 10a$, so $\cos = \frac{50}{\sqrt5 \cdot \sqrt{500}} = \frac{50}{50} = 1$.
- (a) Wrong: cosine ignores length completely.
- (c) Wrong: 50 is the dot product before dividing by the lengths.
- (d) Wrong: cosine works for vectors with the same number of entries. Their lengths (norms) can differ.

---

## C. Calculation

**C1.**
1. Starting vocabulary: `{s, u, n, g}`. *(1 mark)*

| Word | Tokens | Count |
|---|---|---|
| sun | s · u · n | 5 |
| sung | s · u · n · g | 3 |
| snug | s · n · u · g | 2 |
| gun | g · u · n | 1 |

2. *(3 marks, one per round)*

**Round 1.**
- `u n`: 5 (sun) + 3 (sung) + 1 (gun) = **9**
- `s u`: 5 + 3 = 8
- `n g`: 3
- `s n`, `n u`, `u g`: 2 each (snug)
- `g u`: 1

**Merge 1: u + n → un.** Words are now `s·un`, `s·un·g`, `s·n·u·g`, `g·un`. (In snug the `u` and `n` are in the wrong order, so they don't merge.)

**Round 2.**
- `s un`: 5 + 3 = **8**
- `un g`: 3
- `s n`, `n u`, `u g`: 2
- `g un`: 1

**Merge 2: s + un → sun.** Words: `sun`, `sun·g`, `s·n·u·g`, `g·un`.

**Round 3.**
- `sun g`: **3**
- `s n`, `n u`, `u g`: 2
- `g un`: 1

**Merge 3: sun + g → sung.**

3. The final vocabulary is `{s, u, n, g, un, sun, sung}`, which is 7 tokens. The merge rules, in order, are (1) u + n, (2) s + un, (3) sun + g. *(1 mark)*
4. *(1 mark)*
   - `unsung`: u·n·s·u·n·g → (merge 1) un·s·un·g → (merge 2) un·sun·g → (merge 3) un·sung, giving `["un", "sung"]`.
   - `suns`: s·u·n·s → un gives s·un·s → sun gives sun·s, so `["sun", "s"]`.

**Marking:** 1 for the starting vocabulary; 1 per round, each with correct counts and the right merge; 1 for the final vocabulary and rules; 1 for both tokenizations. Count a slip in the counts that doesn't change the winning pair as an error worth ½ mark.

**C2.**
1. "falls" is ID 2, so $o = [0, 0, 1, 0]$. *(2 marks)*
   - Column 1: $0(0.0) + 0(0.5) + 1(0.2) + 0(-0.4) = 0.2$
   - Column 2: $0 + 0 + 1(0.6) + 0 = 0.6$
   - Column 3: $0 + 0 + 1(-0.3) + 0 = -0.3$
   - The result $[0.2, 0.6, -0.3]$ equals row 2.
2. The IDs are $[1, 2, 3]$, so *(1 mark)*

$$X = \begin{bmatrix} 0.5 & -0.1 & 0.7 \\ 0.2 & 0.6 & -0.3 \\ -0.4 & 0.3 & 0.2 \end{bmatrix} \qquad 3 \times 3$$

3. $u^{\top}E = 0.5 \cdot [0.5, -0.1, 0.7] + 0.5 \cdot [0.2, 0.6, -0.3] = [0.35, 0.25, 0.20]$. *(1 mark)* This is the **average** of the embeddings for "rain" and "falls". A non-one-hot vector mixes rows. (This weighted mixing of vectors comes back as attention in Section 3.)

**C3.**
1. *(2 marks)*
   - $\lVert\text{mango}\rVert = \lVert\text{apple}\rVert = \lVert\text{car}\rVert = 5$.
   - mango · apple $= 12 + 12 + 0 = 24$, so $\cos = 24/25 = \mathbf{0.96}$.
   - mango · car $= 0 + 12 + 0 = 12$, so $\cos = 12/25 = \mathbf{0.48}$.
2. *(1 mark)*
   - big_mango $= 2 \times$ mango, so $\cos = 1$.
   - $\lVert\text{mango} - \text{big\_mango}\rVert = \lVert[-3,-4,0]\rVert = 5$.
   - $\lVert\text{mango} - \text{apple}\rVert = \lVert[-1, 1, 0]\rVert = \sqrt2 \approx 1.41$.
3. **Cosine** says mango is closest to big_mango (1 > 0.96). **Euclidean distance** says it is closer to apple (1.41 < 5). If we treat the direction as the meaning and the length as something like strength or frequency, then cosine matches "same meaning" better, which is why it is the standard similarity measure for embeddings. *(1 mark)*

**C4.**
1. $30{,}522 \times 768 = 23{,}440{,}896 \approx 23.4$ M, about **21%** of 110 M. *(1 mark)*
2. One table: $128{,}256 \times 4{,}096 = 525{,}336{,}576$. Untied, the embedding plus the head is $2 \times 525{,}336{,}576 = 1{,}050{,}673{,}152 \approx 1.05$ B parameters. *(1 mark)* At 2 bytes each that is $\approx 2.1$ GB. *(1 mark)*
3. $3{,}000 / 0.75 = $ **≈ 4,000 tokens**. *(1 mark)*

---

## D. Conceptual

**D1.** BPE merges whatever is frequent in its training corpus. Telugu or Hindi script was rare there, so few merges were learned for it, and words split into many short pieces, often single characters or even bytes. *(1 mark)* Consequences *(1 mark each, any two)*:
- The same request costs more, because APIs charge per token.
- It uses up the context window faster, so less text fits.
- It is slower to process, since attention grows with $n^2$ and generation emits more tokens.
- Quality can be worse, because the model sees less meaningful units.

**D2.** Only the rows of tokens that appear in the batch receive a gradient. An embedding is a lookup, so other rows aren't used in the forward pass and get zero gradient. *(2 marks)* A very rare token is updated only on the few steps where it appears, so its embedding stays close to its random starting point and is poorly learned. *(1 mark)*

**D3.** *(1.5 marks each, any two)*
- **False order and distance.** IDs are arbitrary labels: ID 3 is not "three times" ID 1, and IDs 41 and 42 needn't be related. The model would read meaning into the size of the number.
- **One number is too little.** A single scalar can't express the many-sided meaning of a token. A $d_{model}$-long vector can encode many features at once.
- **Nothing to learn.** With embeddings the model learns where to place each token; a fixed ID offers no learned representation.

---

## E. Applied / coding

**E1.** *(6 marks: 2 pair counting with frequencies, 2 applying a merge, 1 tokenize in order, 1 checks pass)*

```python
from collections import Counter

def merge_pair(tokens: list[str], pair: tuple[str, str]) -> list[str]:
    """Replace every adjacent occurrence of pair with the merged token."""
    out, i = [], 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            out.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out

def train_bpe(corpus: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    """Return the merge rules, in the order they were learned."""
    words = {word: list(word) for word in corpus}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for word, tokens in words.items():
            for a, b in zip(tokens, tokens[1:]):
                pairs[(a, b)] += corpus[word]
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        words = {w: merge_pair(t, best) for w, t in words.items()}
    return merges

def tokenize(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Split word into characters and apply the merges in order."""
    tokens = list(word)
    for pair in merges:
        tokens = merge_pair(tokens, pair)
    return tokens

merges = train_bpe({"tea": 4, "ten": 3, "tent": 2, "neat": 1}, 4)
print(merges)
assert merges == [("t", "e"), ("te", "n"), ("te", "a"), ("ten", "t")]
assert tokenize("neaten", merges) == ["n", "e", "a", "ten"]
print(tokenize("neaten", merges))

merges_c1 = train_bpe({"sun": 5, "sung": 3, "snug": 2, "gun": 1}, 3)
print(merges_c1, tokenize("unsung", merges_c1), tokenize("suns", merges_c1))
```

Output:

```
[('t', 'e'), ('te', 'n'), ('te', 'a'), ('ten', 't')]
['n', 'e', 'a', 'ten']
[('u', 'n'), ('s', 'un'), ('sun', 'g')] ['un', 'sung'] ['sun', 's']
```

**E2.** *(2 marks)*

```python
import numpy as np

rng = np.random.default_rng(0)
vocab_size, d_model = 5, 3
E = rng.normal(size=(vocab_size, d_model))
ids = np.array([1, 2, 3])

one_hot = np.eye(vocab_size)[ids]        # n x |V|
X_mult = one_hot @ E                      # (n x |V|) @ (|V| x d_model)
X_index = E[ids]

assert one_hot.shape == (3, vocab_size)
assert X_mult.shape == X_index.shape == (3, d_model)
assert np.allclose(X_mult, X_index)
print("both give the same X, shape", X_index.shape)
```

---

## F. ★ Challenge

1. A: $24{,}000 / 4.0 = 6{,}000$ tokens. B: $24{,}000 / 4.8 = 5{,}000$ tokens. *(1 mark)*
2. $(6000/5000)^2 = 1.2^2 = 1.44$, so B needs about **1.44× less** attention work, a 31% reduction. *(1 mark)*
3. $2 \times (128{,}000 - 32{,}000) \times 4{,}096 = 786{,}432{,}000 \approx$ **786 M** more parameters. *(1 mark)*
4. *(1 mark for one of each)*
   - **Helps:** shorter sequences mean cheaper attention and more text in the same context. Other languages and code also get better coverage.
   - **Hurts:** more parameters and memory, and a slower softmax over $|\mathcal{V}|$ scores at every output step. Each rare token is also seen less often in training, so its embedding is learned worse (see D2).
