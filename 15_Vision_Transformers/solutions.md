# Section 15 — Solutions: Vision Transformers (ViT)

---

## A. Warm-up

**A1.** *(3 marks: 1 for patchify and flatten, 1 for the projection, 1 for `[CLS]` and positions)*
1. Split the image into $N$ non-overlapping $P \times P$ patches.
2. Flatten each patch into $P^2C$ numbers.
3. Project each one with $E$ (plus a bias) to $d_{model}$.
4. Prepend the learned `[CLS]` vector.
5. Add the learned position embeddings, giving $z_0$ of shape $(N + 1) \times d_{model}$.

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **False.** A ViT is an **encoder**: its attention is bidirectional, with no mask.
2. **True.** Each filter covers one patch, and the stride moves exactly one patch, so it computes $x_iE$ for every patch.
3. **False.** With little data, a CNN's built-in assumptions of locality and translation equivariance help it. ViTs need large-scale pre-training or tricks such as DeiT's recipe.

**A3.** *(1 mark each)*
1. 196 patches, $16 \cdot 16 \cdot 3 = 768$ numbers each
2. 197
3. 75

---

## B. Multiple choice

**B1. (b).**
- (a) Wrong: positions say where each patch came from, not how big the image is.
- (c) Wrong: they don't change the number of patches.
- (d) Wrong: the `[CLS]` token is still used.

**B2. (c).** There are 4× more patches, and attention cost grows with $N^2$, so $4^2 = 16$.
- (a) Wrong: 2× doesn't follow from any count here.
- (b) 4× is the growth in the number of tokens, not in attention cost.
- (d) Wrong: 64× would be $N^3$ growth.

**B3. (b).**
- (a) Wrong: CLIP is trained on image–caption pairs, not ImageNet labels.
- (c) Wrong: CLIP doesn't generate captions.
- (d) Wrong: patch size has nothing to do with zero-shot ability.

**B4. (b).**
- (a) Wrong: Swin does use attention.
- (c) Wrong: masking 75% of patches is MAE.
- (d) Wrong: there is no causal mask.

**B5. (b).**
- (a) Wrong: nothing is converted into pixels.
- (c) Wrong: the LLM isn't compressed.
- (d) Wrong: text still goes through the tokenizer.

---

## C. Calculation

**C1.** *(5 marks, about 1¼ per row)*

| Image | $P$ | $N$ | Numbers per patch | Tokens | Cost vs 196 |
|---|---|---|---|---|---|
| 256 × 256 × 3 | 16 | $16^2 = 256$ | 768 | 257 | $(256/196)^2 = 1.7$× |
| 512 × 512 × 3 | 16 | $32^2 = 1{,}024$ | 768 | 1,025 | $27.3$× |
| 224 × 224 × 3 | 14 | $16^2 = 256$ | 588 | 257 | $1.7$× |
| 336 × 336 × 3 | 14 | $24^2 = 576$ | 588 | 577 | $8.6$× |

**C2.**
1. Flattened row by row, the patches are: top-left $[8, 8, 0, 0]$, top-right $[0, 0, 0, 0]$, bottom-left $[0, 8, 0, 8]$, bottom-right $[8, 0, 8, 0]$. *(1 mark)*
2. Column 2 is $+0.5$ on the top row of a patch and $-0.5$ on the bottom row, so it measures **top minus bottom**: a horizontal-edge detector. *(1 mark)*
3. *(2 marks)*

| Patch | Embedding [brightness, top − bottom] |
|---|---|
| top-left | $[0.25 \times 16,\; 0.5(16) - 0.5(0)] = [4, 8]$ |
| top-right | $[0, 0]$ |
| bottom-left | $[0.25 \times 16,\; 0.5(8) - 0.5(8)] = [4, 0]$ |
| bottom-right | $[4, 0]$ |

4. **Bottom-left** and **bottom-right** both get $[4, 0]$. They have the same brightness and no horizontal edge, but their *vertical* edges point opposite ways: dark→bright versus bright→dark. *(1 mark)* $E$ only "sees" what its columns measure. Telling them apart needs a column such as $[-0.5, 0.5, -0.5, 0.5]$ (right minus left), or a wider $d_{model}$ with more learned detectors. *(1 mark)*

**C3.** *(4 marks: 1 for the embedding parts, 2 for the blocks, 1 for the total)*

| Part | Parameters |
|---|---|
| Patch embedding $768 \times 1024 + 1024$ | 787,456 |
| `[CLS]` | 1,024 |
| Positions $197 \times 1024$ | 201,728 |
| 24 blocks × 12,596,224 (BERT-large block, Section 9) | 302,309,376 |
| Final LN | 2,048 |
| Head $1024 \times 1000 + 1000$ | 1,025,000 |
| **Total** | **304,326,632 ≈ 304M** |

(The ViT paper's model table lists ViT-Large as 307M. Our recipe counts a 1,000-class head and no extra parts; small differences in what is counted explain the gap.)

**C4.**
1. *(1 mark)*
   - sunset: $0.28(0.6) + 0.96(0.8) = 0.936$
   - beach: $0.28(0.96) + 0.96(0.28) = 0.538$
   - mountain: $0.960$
2. *(1 mark)*
   - Scale 10: the softmax of $[9.36, 5.38, 9.60]$ is $[0.44, 0.01, 0.56]$.
   - Scale 100: the softmax of $[93.6, 53.8, 96.0]$ is $[0.08, 0.00, 0.92]$.
3. **Mountain** is chosen either way. *(1 mark)* The scale controls how confident the softmax is: sunset and mountain have similar cosines, so at scale 10 the model looks unsure (0.56), while CLIP's learned scale of about 100 makes a clear decision (0.92).

---

## D. Conceptual

**D1.** **Inductive bias** means the assumptions built into an architecture. A CNN assumes **locality** (nearby pixels matter together) and **translation equivariance** (the same filter is used everywhere), so it starts out knowing a lot about images. *(1.5 marks)* A ViT treats patches as a set, with only learned positions, and has to learn spatial structure from data. With little data it lacks those useful assumptions; with huge data, its flexibility lets it keep improving where CNNs level off. *(1.5 marks)*

**D2.** The grid goes from $14 \times 14 = 196$ to $28 \times 28 = 784$ patches, so there are 4× more tokens and about 16× the attention cost. *(1.5 marks)* Only 196 (+1) position embeddings were learned. They are reshaped into a $14 \times 14$ grid, **interpolated** (for example bilinearly) to $28 \times 28$, and then fine-tuned. *(1.5 marks)*

**D3.** The encoder processes only the visible 25% of patches, e.g. 49 of 196. Attention cost is proportional to $N^2$, so it falls by $(196/49)^2 = 16$×, and the FFN cost falls by 4×. *(1 mark)* A small, light decoder handles the masked positions, so most of the compute goes to the encoder on a quarter of the tokens. *(1 mark)*

---

## E. Applied / coding

**E1.** *(5 marks: 3 correct reshape and transpose, 2 checks)*

```python
import numpy as np

def patchify(img: np.ndarray, P: int) -> np.ndarray:
    """img: H x W x C  ->  N x (P*P*C), patches in row-major order, each flattened row by row."""
    H, W, C = img.shape
    x = img.reshape(H // P, P, W // P, P, C)        # (grid rows, P, grid cols, P, C)
    x = x.transpose(0, 2, 1, 3, 4)                  # (grid rows, grid cols, P, P, C)
    return x.reshape(-1, P * P * C)

img = np.array([[8, 8, 0, 0], [0, 0, 0, 0], [0, 8, 8, 0], [0, 8, 8, 0]], float)[:, :, None]
E = np.array([[0.25, 0.5], [0.25, 0.5], [0.25, -0.5], [0.25, -0.5]])
patches = patchify(img, 2)
print(patches)
print(patches @ E)                                  # [[4,8],[0,0],[4,0],[4,0]]
assert patchify(np.zeros((224, 224, 3)), 16).shape == (196, 768)
```

**E2.** *(3 marks)*

```python
rng = np.random.default_rng(0)
P, d = 8, 16
image = rng.random((32, 32, 3))
x = patchify(image, P)                              # 16 x 192
E_proj = rng.normal(0, 0.02, (P * P * 3, d))
cls = rng.normal(0, 0.02, (1, d))
E_pos = rng.normal(0, 0.02, (x.shape[0] + 1, d))
z0 = np.vstack([cls, x @ E_proj]) + E_pos
assert z0.shape == (17, d)
print("z0 shape:", z0.shape)
```

---

## F. ★ Challenge

1. A convolution filter $f$ of size $P \times P \times C$ at a position computes $\sum f \odot \text{patch} = \text{flatten(patch)} \cdot \text{flatten}(f)$. With $d_{model}$ filters, the outputs at one position are $\text{flatten(patch)} \cdot [\text{flatten}(f_1), \ldots]$, which is $x_iE$ with the column $j$ of $E$ equal to $\text{flatten}(f_j)$. Stride $P$ visits each patch exactly once. Both forms have $P^2C \cdot d_{model}$ weights plus $d_{model}$ biases. *(2 marks)*
2. Drop the `[CLS]` position, reshape the other 196 vectors into a $14 \times 14 \times d$ grid, resize to $24 \times 24 \times d$ with bilinear (or bicubic) interpolation, flatten again, and put the `[CLS]` position back. *(1 mark)* 2-D interpolation keeps **neighbours** in both directions close: the patch below another stays "below" it. Interpolating a 1-D list would mix up the ends of one row with the start of the next.
3. $(336/14)^2 = 24^2 = 576$ visual tokens. A 300-word question is about $300/0.75 = 400$ text tokens, so the image costs more context than the question itself. *(1 mark)*
