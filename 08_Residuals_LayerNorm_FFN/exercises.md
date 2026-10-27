# Section 8 — Exercises: Residuals, LayerNorm & the FFN

**Total: 47 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 55 minutes**

Pre-norm block: $A = X + \text{MHA}(\text{LN}_1 X)$, $X' = A + \text{FFN}(\text{LN}_2 A)$. Take $\epsilon \approx 0$ in hand calculations. Round to 2 decimal places unless told otherwise.

---

## A. Warm-up (recall)

**A1.** Write the derivative of $y = x + F(x)$ and explain in one sentence why it helps deep networks. *(2 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. LayerNorm normalizes each feature across all the examples in a batch.
2. RMSNorm subtracts the mean, but has no $\beta$.
3. In a typical block with $d_{ff} = 4d_{model}$, the FFN has twice as many parameters as the attention.

**A3.** Fill in the shapes. $d_{model} = 512$, $d_{ff} = 2048$, $n = 30$. *(4 marks)*

1. $W_1$: `__ × __`
2. $W_2$: `__ × __`
3. The hidden activations for all tokens: `__ × __`
4. LayerNorm's $\gamma$: `__`

---

## B. Multiple choice (1 mark each)

**B1.** Why does LayerNorm suit text better than BatchNorm?
- (a) It is the only norm with learned parameters.
- (b) It normalizes within one token, so it doesn't depend on batch size or on other, variable-length sequences.
- (c) It removes the need for residuals.
- (d) It makes all vectors the same.

**B2.** In pre-norm, where is LayerNorm applied?
- (a) After adding the residual: $\text{LN}(x + F(x))$.
- (b) On the input to each sub-layer: $x + F(\text{LN}(x))$, leaving the residual path clean.
- (c) Only once, at the very start.
- (d) Only on the output head.

**B3.** The FFN is applied:
- (a) Once to the whole sequence, mixing tokens.
- (b) To each token separately, with the same weights at every position.
- (c) To each token with position-specific weights.
- (d) Only to the first token.

**B4.** Why does SwiGLU use $d_{ff} \approx \frac{8}{3}d_{model}$ instead of $4d_{model}$?
- (a) SiLU needs a smaller hidden layer.
- (b) It has three matrices instead of two, so a smaller $d_{ff}$ keeps the parameter count near $8d_{model}^2$.
- (c) It makes $d_{ff}$ a power of 2.
- (d) Larger $d_{ff}$ doesn't work with gating.

**B5.** Ignoring embeddings, biases and norms, how many parameters does one standard block have?
- (a) $4d^2$
- (b) $8d^2$
- (c) $12d^2$
- (d) $16d^2$

---

## C. Calculation

**C1. LayerNorm and RMSNorm.** One token has $x = [1, 3, 5, 11]$. *(5 marks)*

1. Compute $\mu$, $\sigma^2$ and $\text{LN}(x)$ with $\gamma = 1$, $\beta = 0$.
2. Apply $\gamma = [2, 1, 1, 0.5]$ and $\beta = [0, 0, 1, 0]$.
3. Compute $\text{RMSNorm}(x)$ with $\gamma = 1$.

**C2. FFN by hand.** $d_{model} = 2$, $d_{ff} = 3$, no biases: *(5 marks)*

$$W_1 = \begin{bmatrix} 1 & 0 & -1 \\ 0.5 & 1 & 1 \end{bmatrix}, \qquad W_2 = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix}, \qquad x = [2, -1]$$

1. Compute $h = xW_1$.
2. Apply ReLU. Which hidden units fire?
3. Compute $\text{FFN}(x)$ and the residual output $x + \text{FFN}(x)$.
4. GELU gives $\phi(h) \approx [1.40, -0.16, -0.004]$. Compute the residual output with GELU.

**C3. Gradients through depth.** Every sub-layer has $F'(x) = 0.2$. Compute the gradient factor through 12 layers without residuals, and with residuals. *(2 marks)*

**C4. Counting parameters.** *(4 marks)*

1. A model has $d_{model} = 2048$ and $N = 24$ standard blocks ($d_{ff} = 4d$). How many parameters are in the blocks (no biases or norms)?
2. For $d_{model} = 4096$, compare a standard FFN ($d_{ff} = 4d$) with a SwiGLU FFN with $d_{ff} = 11{,}008$.
3. What is $\frac{8}{3} \times 4096$, and why do you think LLaMA-2 uses 11,008 instead?

---

## D. Conceptual

**D1.** Compare post-norm and pre-norm: write both formulas, and explain why pre-norm trains more stably for very deep models. *(3 marks)*

**D2.** "LayerNorm destroys information, because every token ends up with mean 0 and variance 1." Is this right? Explain, mentioning $\gamma$ and $\beta$ and what is kept. *(3 marks)*

**D3.** Attention moves information between tokens. What is the job of the FFN, and what is the "detector → write" picture of it? *(3 marks)*

---

## E. Applied / coding (NumPy)

**E1.** Implement `layer_norm(X, gamma, beta, eps=1e-5)`, `rms_norm(X, gamma, eps=1e-5)` and `ffn(X, W1, b1, W2, b2, act)` for a whole matrix $X$ of shape $n \times d_{model}$, working row by row. Check them against C1 and C2. *(5 marks)*

**E2.** Build a pre-norm block `block(X, params)` from your functions and the multi-head attention of Section 6 (you may use a single head). Assert that the output shape equals the input shape. Then stack 6 blocks with random weights and print the norm of each row after every block. *(3 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

1. Write $x_N = x_0 + \sum_{l=1}^{N} F_l(x_{l-1})$ for a stack of residual layers. Use it to explain the name "residual stream".
2. Using the chain rule on $x_N$, show that $\frac{\partial x_N}{\partial x_0}$ contains the identity term $I$ plus other terms. What does this mean for the gradient reaching layer 1?
3. If each $F_l$ adds a vector of roughly the same size, and these are independent, how does $\lVert x_N\rVert$ grow with $N$? Why does that make a final LayerNorm before the output head useful?
