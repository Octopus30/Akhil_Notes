# Section 15 — Exercises: Vision Transformers (ViT)

**Total: 48 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 55 minutes**

$N = (H/P)(W/P)$ patches, each with $P^2C$ numbers; $z_0 = [x_{\text{cls}}; x_1E; \ldots; x_NE] + E_{\text{pos}}$. Flatten patches row by row.

---

## A. Warm-up (recall)

**A1.** List the steps that turn an image into the input sequence $z_0$ of a ViT. *(3 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. A ViT's blocks use a causal mask, so earlier patches can't see later ones.
2. The patch embedding can be written as a convolution with kernel size $P$ and stride $P$.
3. ViTs generally beat CNNs when trained from scratch on small datasets.

**A3.** Fill in. *(3 marks)*

1. A $224 \times 224 \times 3$ image with $P = 16$ gives `__` patches, each with `__` numbers.
2. With a `[CLS]` token, the sequence length is `__`.
3. MAE masks `__`% of the patches during pre-training.

---

## B. Multiple choice (1 mark each)

**B1.** Why does a ViT need position embeddings?
- (a) To tell the model the image's size.
- (b) Attention is order-blind, so without them the model can't tell where each patch came from.
- (c) To reduce the number of patches.
- (d) They replace the `[CLS]` token.

**B2.** Halving the patch size (e.g. 16 → 8) at fixed resolution multiplies the attention cost by about:
- (a) 2
- (b) 4
- (c) 16
- (d) 64

**B3.** What makes CLIP capable of zero-shot classification?
- (a) It is trained on ImageNet labels.
- (b) Images and captions are embedded in a shared space, so an image can be compared with any text prompt.
- (c) It generates captions with beam search.
- (d) It uses a larger patch size.

**B4.** How does Swin Transformer keep its cost linear in image size?
- (a) It uses no attention at all.
- (b) It attends within local windows that shift between layers, with a hierarchy of patch merging.
- (c) It drops 75% of the patches.
- (d) It uses a causal mask.

**B5.** In a LLaVA-style vision–language model, what does the projector do?
- (a) It converts text into pixels.
- (b) It maps ViT patch vectors into the LLM's embedding space, so they can be fed in like tokens.
- (c) It compresses the LLM.
- (d) It replaces the tokenizer.

---

## C. Calculation

**C1. Patch arithmetic.** Fill in the table. Give the attention cost relative to a 196-patch ViT, using (number of patches)². *(5 marks)*

| Image | $P$ | Patches $N$ | Numbers per patch | Tokens with `[CLS]` | Attention cost vs 196 |
|---|---|---|---|---|---|
| 256 × 256 × 3 | 16 | | | | |
| 512 × 512 × 3 | 16 | | | | |
| 224 × 224 × 3 | 14 | | | | |
| 336 × 336 × 3 | 14 | | | | |

**C2. Patch embeddings by hand.** A $4 \times 4$ grey-scale image, $P = 2$: *(6 marks)*

$$\text{image} = \begin{bmatrix} 8 & 8 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 8 & 8 & 0 \\ 0 & 8 & 8 & 0 \end{bmatrix}, \qquad
E = \begin{bmatrix} 0.25 & 0.5 \\ 0.25 & 0.5 \\ 0.25 & -0.5 \\ 0.25 & -0.5 \end{bmatrix}$$

1. Flatten the four patches (top-left, top-right, bottom-left, bottom-right).
2. Column 1 of $E$ measures average brightness. What does column 2 measure?
3. Compute all four patch embeddings.
4. Two patches get the same embedding even though they look different. Which ones, and why? What would $E$ need in order to tell them apart?

**C3. ViT-L/16 parameters.** $d = 1024$, 24 layers, $d_{ff} = 4096$, $P = 16$, a 224 × 224 RGB input, 1,000 classes. Use the same recipe as the notes (patch embedding with bias, `[CLS]`, positions, BERT-style blocks with biases and LayerNorms, final LN, head). *(4 marks)*

**C4. CLIP toy.** Image embedding $[0.28, 0.96]$. Prompts: "a sunset" $[0.6, 0.8]$, "a beach" $[0.96, 0.28]$, "a mountain" $[0, 1]$. *(3 marks)*

1. Compute the three cosine similarities.
2. Compute the softmax with logit scale 10, and with scale 100.
3. Which class is chosen? Why does the scale matter?

---

## D. Conceptual

**D1.** Explain "inductive bias" by comparing a CNN with a ViT, and say why this makes ViTs data-hungry. *(3 marks)*

**D2.** A ViT pre-trained at $224 \times 224$ is fine-tuned at $448 \times 448$ with the same patch size. What happens to the number of tokens, and to the position embeddings? *(3 marks)*

**D3.** Why is MAE pre-training much cheaper than running the encoder on every patch? *(2 marks)*

---

## E. Applied / coding (NumPy)

**E1.** Implement *(5 marks)*

```python
def patchify(img: np.ndarray, P: int) -> np.ndarray:
    """img: H x W x C  ->  N x (P*P*C), patches in row-major order, each flattened row by row."""
```

using only `reshape` and `transpose` (no Python loops). Check it against C2 (with $C = 1$), and check that a $224 \times 224 \times 3$ image gives shape `196 × 768`.

**E2.** Build $z_0$ for a random $32 \times 32 \times 3$ image with $P = 8$ and $d_{model} = 16$: patchify, project with a random $E$, prepend a `[CLS]` vector, and add random position embeddings. Assert the shape. *(3 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

1. Show that applying $E$ to each flattened patch is the same as a convolution with $d_{model}$ filters of size $P \times P \times C$ and stride $P$. How many parameters does each form have?
2. A ViT trained at $224^2$ with $P = 16$ has a $14 \times 14$ grid of position embeddings. Describe how to get position embeddings for a $24 \times 24$ grid. Why interpolate in 2-D rather than treating the positions as a 1-D list?
3. For the LLaVA-style setup, a $336 \times 336$ image with $P = 14$ becomes how many visual tokens in the LLM's context? Compare that with a 300-word question.
