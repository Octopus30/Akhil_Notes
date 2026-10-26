# Course style guide (shared by all course skills)

## Audience
Second- or third-year bachelor's students in CS or a related field. Assume they know:
- Python, and basic NumPy
- Linear algebra basics (vectors, matrices, dot product, matrix multiply, transpose)
- Calculus basics (derivative, chain rule) and probability basics (distribution, expectation)
- A rough idea of what a neural network and gradient descent are

Don't assume: prior NLP, attention, PyTorch internals, information theory, or research-paper fluency.

## Tone
- Plain, friendly, direct. Write as a good lecturer speaks: short sentences, active voice, "we" for working through things together.
- Define a term the first time it appears, in **bold**. Afterwards use exactly the same term.
- Explain *why* before *how*. Every formula is followed by one sentence saying what it means in words.
- No hype ("revolutionary", "game-changing") and no filler ("it's worth noting").

## Notation (keep it identical across all sections)
| Symbol | Meaning |
|---|---|
| $n$ | sequence length (number of tokens) |
| $d_{model}$ | model / embedding width |
| $d_k, d_v$ | key/query width, value width per head |
| $h$ | number of attention heads |
| $N$ | number of stacked layers |
| $V$ | vocabulary size (write "vocabulary size", not just $V$, wherever it could be confused with the value matrix) |
| $X$ | input matrix, shape $n \times d_{model}$, one token per **row** |
| $Q, K, V$ | query, key, value matrices |
| $W_Q, W_K, W_V, W_O$ | learned projection matrices |

Vectors are rows. Always state shapes as `rows × columns`.

## Originality
- Write every explanation, example, analogy and exercise in your own words.
- Don't copy or closely paraphrase text, figures or worked examples from blog posts, cheat sheets, LinkedIn posts, books or course pages. Learn from them, then write fresh material.
- Standard formulas and well-known facts (model sizes, paper titles) are fine. Cite the original paper for them.

## Accuracy
- Recompute every number. Never state a model size, benchmark or date you are unsure of; leave it out or mark it "approximately" with a source.
- Prefer the original paper over secondary sources.

## Files per section folder
| File | Made by skill |
|---|---|
| `notes.md` | bachelor-notes |
| `exercises.md` and `solutions.md` | practice-problems |
| `lab.py` (or `lab.ipynb` if asked) | code-lab |
