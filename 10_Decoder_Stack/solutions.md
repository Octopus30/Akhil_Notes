# Section 10 — Solutions: The Decoder Stack

---

## A. Warm-up

**A1.** *(1 mark each)*
- (a) Masked (causal) self-attention, then the FFN, each with a residual connection and a norm.
- (b) Masked self-attention, then cross-attention to the encoder output $H$, then the FFN, each with a residual connection and a norm.

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **False.** The whole source is already known, so cross-attention has no causal mask. Only the decoder's *self*-attention is causal.
2. **False.** Teacher forcing feeds the **true** previous tokens. Feeding the model its own outputs happens at generation time.
3. **True.** Each new token needs the previous one first, so the passes must run one after another.

**A3.** *(1 mark each)*
1. $Q$: `7 × 64`, $K$: `12 × 64`
2. `7 × 12`
3. `7 × 32000`

---

## B. Multiple choice

**B1. (b).** The input is shifted right, so it starts with `<bos>` and drops the last token.
- (a) is the target sequence.
- (c) is the full sequence; position $t$ would then have to predict a token it is fed.
- (d) Wrong: order is never reversed.

**B2. (b).**
- (a) Keys and values from the decoder's own tokens would be self-attention.
- (c) Wrong: the output head comes after the blocks and plays no part in attention.
- (d) Wrong: the keys and values are computed from the encoder's output, not from noise.

**B3. (c).** Every position predicts its next token.
- (a) Wrong: one signal per pass is generation, not training.
- (b) $0.15T$ is MLM.
- (d) Wrong: $T^2$ counts pairs of tokens, not targets.

**B4. (b).**
- (a) Wrong: that is about frequency, not about prefixes.
- (c) Wrong: the output head isn't biased toward `<eos>`.
- (d) Wrong: exposure bias isn't about where attention goes.

**B5. (b).** A decoder layer has $16d^2$ against an encoder layer's $12d^2$.
- (a) Wrong: cross-attention adds a full $4d^2$.
- (c) Wrong: $16d^2$ is a third more than $12d^2$, not twice.
- (d) Wrong: masking adds no parameters and removes none.

---

## C. Calculation

**C1.**
1. Input: `<bos> rain falls softly`. Target: `rain falls softly <eos>`. *(2 marks)*

| Position | Context | Target |
|---|---|---|
| 1 | `<bos>` | rain |
| 2 | `<bos> rain` | falls |
| 3 | `<bos> rain falls` | softly |
| 4 | `<bos> rain falls softly` | `<eos>` |

2. $-\log p = [1.204, 0.693, 0.223, 0.511]$. The sum is $2.631$, so $\mathcal{L} = 2.631/4 = 0.658$. *(2 marks)*
3. $e^{0.658} = 1.93$. On average the model is about as uncertain as if it were choosing uniformly between roughly 1.9 tokens at each step. *(1 mark)*

**C2.**
1. *(2 marks)*

| Pass | Input | Chosen |
|---|---|---|
| 1 | `<bos>` | rain (0.6) |
| 2 | `<bos> rain` | falls (0.5) |
| 3 | `<bos> rain falls` | `<eos>` (0.6) |

2. $P(\text{rain falls}) = 0.6 \times 0.5 \times 0.6 = 0.18$ *(1 mark)*
3. "sun shines" has probability $0.4 \times 0.9 \times 1.0 = 0.36$, twice the greedy sentence. Greedy decoding picks the best token at each step, and that need not give the most probable **sequence**: here it was led away by rain (0.6) at step 1. Beam search tackles this (Section 13). *(1 mark)*

**C3.**
1. With $q = [3, 0]$, the scores are $[3, 4.5]/\sqrt2 = [2.12, 3.18]$. *(1.5 marks)* $e^{2.12} = 8.34$ and $e^{3.18} = 24.1$, so the weights are $[0.26, 0.74]$. The output is $0.26[1, 1] + 0.74[1.5, 0] = [1.37, 0.26]$.
2. With $q = [0, 2]$, the scores are $[2, 0]/\sqrt2 = [1.41, 0]$, so the weights are $[0.80, 0.20]$. The output is $0.80[1, 1] + 0.20[1.5, 0] = [1.10, 0.80]$. *(1.5 marks)*
3. Query 1 aligns mostly with **bag** (0.74); query 2 aligns with **light** (0.80). *(1 mark)*

**C4.**
1. $12 \times 1024^2 \times 24 = 301{,}989{,}888 \approx 302$ M *(1 mark)*
2. *(2 marks)*
   - Each encoder layer has $12d^2 = 7{,}077{,}888$, so 12 layers have $84{,}934{,}656$.
   - Each decoder layer has $16d^2 = 9{,}437{,}184$, so 12 layers have $113{,}246{,}208$.
   - The embeddings are $32{,}128 \times 768 = 24{,}674{,}304$.
   - The total is $222{,}855{,}168 \approx 223$ M.

---

## D. Conceptual

**D1.** In training, the whole true sentence is known in advance. *(1 mark)* The causal mask lets every position see only its own prefix, so all $T$ predictions are honest, and they can be computed in one batched pass. *(1 mark)* In generation, the token at step $t + 1$ doesn't exist until step $t$ has chosen it, so each pass waits for the previous one. *(1 mark)*

**D2.** The prompt is simply the start of the sequence. *(1 mark)* In the prefill pass, causal self-attention lets each prompt token attend to all earlier ones, building contextual vectors layer by layer, so the last positions summarise the whole prompt. *(1 mark)* Each generated token then attends back to every prompt token in every layer, through the KV cache. This plays the role an encoder plus cross-attention would play. *(1 mark)*

**D3.** *(2 marks for one, well explained)*
- Large-scale training on diverse text, so that "unusual" states are rarely truly unseen.
- Decoding strategies such as sampling with top-p or temperature (Section 13), which avoid low-probability derailments.
- Fine-tuning on the model's own outputs with feedback (RLHF/DPO, Section 12), so the model learns from the states it actually reaches.
- Scheduled sampling: occasionally feeding the model its own predictions during training.

---

## E. Applied / coding

**E1.** *(3 marks)*

```python
import numpy as np

def teacher_forcing_pairs(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Input = shifted right (drop last); target = shifted left (drop first)."""
    return tokens[:-1], tokens[1:]

def sequence_loss(probs) -> float:
    """Mean negative log-probability of the target tokens."""
    return float(-np.log(np.asarray(probs)).mean())

inp, tgt = teacher_forcing_pairs(["<bos>", "rain", "falls", "softly", "<eos>"])
print(inp, tgt)
print(round(sequence_loss([0.3, 0.5, 0.8, 0.6]), 3))   # 0.658
```

**E2.** *(5 marks: 2 greedy loop, 1 sentence probability, 2 exhaustive search)*

```python
TABLE = {
    "<bos>": {"rain": 0.6, "sun": 0.4},
    "rain": {"falls": 0.5, "stops": 0.3, "<eos>": 0.2},
    "sun": {"shines": 0.9, "<eos>": 0.1},
    "falls": {"softly": 0.4, "<eos>": 0.6},
    "stops": {"<eos>": 1.0},
    "softly": {"<eos>": 1.0},
    "shines": {"<eos>": 1.0},
}

def greedy_generate(table, start="<bos>", max_len=10) -> list[str]:
    """Append the most likely next token until <eos>."""
    seq = [start]
    while seq[-1] != "<eos>" and len(seq) < max_len:
        nxt = table[seq[-1]]
        seq.append(max(nxt, key=nxt.get))
    return seq

def sentence_prob(table, tokens) -> float:
    """Product of next-token probabilities along the sequence."""
    return float(np.prod([table[a][b] for a, b in zip(tokens, tokens[1:])]))

def all_sentences(table, seq=("<bos>",)):
    """Yield every complete sequence (depth-first)."""
    if seq[-1] == "<eos>":
        yield list(seq)
        return
    for tok in table[seq[-1]]:
        yield from all_sentences(table, seq + (tok,))

g = greedy_generate(TABLE)
print("greedy:", g, round(sentence_prob(TABLE, g), 2))
best = max(all_sentences(TABLE), key=lambda s: sentence_prob(TABLE, s))
print("best:  ", best, round(sentence_prob(TABLE, best), 2))
```

Output:

```
greedy: ['<bos>', 'rain', 'falls', '<eos>'] 0.18
best:   ['<bos>', 'sun', 'shines', '<eos>'] 0.36
```

---

## F. ★ Challenge

1. Pass $k$ (for $k = 1, \ldots, T$) processes the prompt plus the $k - 1$ tokens generated so far: $p + k - 1$ positions. The total is $\sum_{k=1}^{T}(p + k - 1) = Tp + \frac{T(T-1)}{2}$. *(1 mark)*
2. $400 \times 100 + \frac{400 \times 399}{2} = 40{,}000 + 79{,}800 = 119{,}800$ positions. With a KV cache: a prefill of 100, then one new position per later pass, $100 + 399 = 499$. That is about **240× less** work on the per-token parts (projections and FFN). *(2 marks)*
3. Teacher forcing processes the 500 positions **once**, in one pass, because all the targets are known in advance and the causal mask keeps each prediction honest. Training never has to wait for the model's own outputs. *(1 mark)*
