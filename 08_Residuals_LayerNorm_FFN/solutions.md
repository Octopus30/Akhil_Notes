# Section 8 — Solutions: Residuals, LayerNorm & the FFN

---

## A. Warm-up

**A1.** $\frac{\partial y}{\partial x} = 1 + F'(x)$. *(1 mark)* The "1" means the gradient never falls below about 1 per layer, even when $F'$ is tiny, so it doesn't vanish through deep stacks. *(1 mark)*

**A2.** *(1 mark each, and the justification is needed for the mark)*
1. **False.** That is BatchNorm. LayerNorm normalizes across the features of a **single token**.
2. **False.** RMSNorm skips the mean subtraction as well as $\beta$; it only divides by the root-mean-square.
3. **True.** The FFN has $2 \times d \times 4d = 8d^2$ parameters, against $4d^2$ for attention.

**A3.** *(1 mark each)* `512 × 2048`, `2048 × 512`, `30 × 2048`, `512`.

---

## B. Multiple choice

**B1. (b).**
- (a) Wrong: BatchNorm has learned $\gamma$ and $\beta$ too.
- (c) Wrong: blocks still need residuals.
- (d) Wrong: the pattern within each vector is kept.

**B2. (b).**
- (a) Wrong: $\text{LN}(x + F(x))$ is post-norm.
- (c) Wrong: a norm is applied before *every* sub-layer.
- (d) Wrong: pre-norm models do add one final norm, but that isn't the defining placement.

**B3. (b).**
- (a) Wrong: mixing tokens is attention's job.
- (c) Wrong: the weights are shared by every position.
- (d) Wrong: every token goes through the FFN.

**B4. (b).** The parameters are $3 d\, d_{ff}$, and $3 d \cdot \frac{8}{3}d = 8d^2$.
- (a) Wrong: the activation function doesn't set the width.
- (c) Wrong: $\frac{8}{3}d$ usually isn't a power of 2.
- (d) Wrong: a larger $d_{ff}$ would work; it would just add parameters.

**B5. (c).** Attention $4d^2$ plus FFN $8d^2$.
- (a) $4d^2$ is attention only.
- (b) $8d^2$ is the FFN only.
- (d) $16d^2$ counts one matrix too many: $16d^2 = 4d^2 + 12d^2$.

---

## C. Calculation

**C1.**
1. $\mu = (1 + 3 + 5 + 11)/4 = 5$. *(3 marks)*
   - $\sigma^2 = (16 + 4 + 0 + 36)/4 = 14$, so $\sigma = 3.742$.
   - $\text{LN}(x) = [-4, -2, 0, 6]/3.742 = [-1.07, -0.53, 0.00, 1.60]$.
   - Check: the mean is 0 and the variance is 1.
2. $[2(-1.07),\; -0.53,\; 0 + 1,\; 0.5(1.60)] = [-2.14, -0.53, 1.00, 0.80]$. *(1 mark)*
3. $\text{rms} = \sqrt{(1 + 9 + 25 + 121)/4} = \sqrt{39} = 6.245$, so $\text{RMSNorm}(x) = [0.16, 0.48, 0.80, 1.76]$. *(1 mark)* All the entries stay positive, because nothing is centred.

**C2.**
1. $h = [2(1) + (-1)(0.5),\; 2(0) + (-1)(1),\; 2(-1) + (-1)(1)] = [1.5, -1, -3]$. *(1 mark)*
2. ReLU gives $[1.5, 0, 0]$. Only unit 1 fires. *(1 mark)*
3. $\text{FFN}(x) = 1.5 \cdot [1, 0] = [1.5, 0]$, so $x + \text{FFN}(x) = [3.5, -1]$. *(2 marks)*
4. $\phi(h)W_2 = 1.40[1, 0] - 0.16[0, 2] - 0.004[1, 1] = [1.40, -0.32]$, so the residual output is $[3.40, -1.32]$. *(1 mark)* The silent units leak a small negative amount, which changes dimension 2.

**Marking:** 1 for $h$; 1 for ReLU; 1 for the FFN output; 1 for the residual; 1 for the GELU version.

**C3.** Without residuals: $0.2^{12} \approx 4.1 \times 10^{-9}$, effectively zero. *(1 mark)* With residuals: $1.2^{12} \approx 8.9$. *(1 mark)*

**C4.**
1. $12 \times 2048^2 \times 24 = 1{,}207{,}959{,}552 \approx 1.21$ B. *(1 mark)*
2. Standard: $8 \times 4096^2 = 134{,}217{,}728$. SwiGLU: $3 \times 4096 \times 11{,}008 = 135{,}266{,}304$. They are almost the same, with SwiGLU about 0.8% larger. *(2 marks)*
3. $\frac{8}{3} \times 4096 = 10{,}922.7$. LLaMA-2 rounds **up** to 11,008, a multiple of 256 ($11{,}008 = 43 \times 256$), because GPU matrix multiplications run fastest on sizes that are multiples of large powers of 2. *(1 mark)*

---

## D. Conceptual

**D1.** Post-norm: $\text{LN}(x + F(x))$. Pre-norm: $x + F(\text{LN}(x))$. *(1 mark)* In post-norm, the residual path itself passes through a LayerNorm in every layer, so the "highway" is rescaled $N$ times and the gradients to early layers can become badly scaled. That is why post-norm needs careful warm-up. *(1 mark)* In pre-norm, $x$ is added through untouched, giving a clean identity path from the output back to the input, so even very deep models train stably. *(1 mark)*

**D2.** It is mostly wrong. LayerNorm removes only **two numbers** per token, its mean and its overall scale. The **pattern**, meaning the relative sizes and signs across features, is kept. *(1.5 marks)* The learned $\gamma$ and $\beta$ let the model rescale and shift each feature, even undoing the normalization where that helps. In pre-norm, the un-normalized $x$ also stays on the residual path, so nothing is lost there at all. *(1.5 marks)*

**D3.** The FFN processes **each token on its own**: it transforms what the token has just gathered through attention. It adds non-linear computation and stores much of the model's knowledge. *(1 mark)* In the detector picture, each hidden unit's column of $W_1$ is a pattern detector. If the token matches it, the unit fires, and the corresponding row of $W_2$ is written (added) into the token's vector. *(2 marks)*

---

## E. Applied / coding

**E1.** *(5 marks: 2 LayerNorm, 1 RMSNorm, 2 FFN)*

```python
import numpy as np
from math import erf, sqrt

def layer_norm(X, gamma, beta, eps=1e-5):
    """Normalize each row (token) to mean 0, variance 1, then scale and shift."""
    mu = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    return gamma * (X - mu) / np.sqrt(var + eps) + beta

def rms_norm(X, gamma, eps=1e-5):
    """Divide each row by its root-mean-square, then scale."""
    return gamma * X / np.sqrt((X ** 2).mean(axis=-1, keepdims=True) + eps)

relu = lambda z: np.maximum(z, 0)
gelu = np.vectorize(lambda z: 0.5 * z * (1 + erf(z / sqrt(2))))

def ffn(X, W1, b1, W2, b2, act=relu):
    """Expand, activate, project back; applied to every row independently."""
    return act(X @ W1 + b1) @ W2 + b2

x = np.array([[1.0, 3.0, 5.0, 11.0]])
print(np.round(layer_norm(x, 1, 0, eps=0), 2))
print(np.round(layer_norm(x, np.array([2, 1, 1, 0.5]), np.array([0, 0, 1, 0]), eps=0), 2))
print(np.round(rms_norm(x, 1, eps=0), 2))

W1 = np.array([[1.0, 0.0, -1.0], [0.5, 1.0, 1.0]])
W2 = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
xt = np.array([[2.0, -1.0]])
print(xt + ffn(xt, W1, 0, W2, 0, relu), np.round(xt + ffn(xt, W1, 0, W2, 0, gelu), 2))
```

Output:

```
[[-1.07 -0.53  0.    1.6 ]]
[[-2.14 -0.53  1.    0.8 ]]
[[0.16 0.48 0.8  1.76]]
[[ 3.5 -1. ]] [[ 3.4  -1.32]]
```

**E2.** *(3 marks: 1 pre-norm wiring, 1 shape assert, 1 stacking loop)*

```python
def attention(X, W_Q, W_K, W_V, W_O):
    """Single-head scaled dot-product self-attention with an output projection."""
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    S = Q @ K.T / np.sqrt(Q.shape[1])
    A = np.exp(S - S.max(axis=1, keepdims=True))
    A /= A.sum(axis=1, keepdims=True)
    return A @ V @ W_O

def block(X, p):
    """Pre-norm block: A = X + Attn(LN1 X); X' = A + FFN(LN2 A)."""
    d = X.shape[1]
    A = X + attention(layer_norm(X, 1, 0), *p["attn"])
    out = A + ffn(layer_norm(A, 1, 0), *p["ffn"])
    assert out.shape == X.shape
    return out

rng = np.random.default_rng(0)
n, d = 5, 8
X = rng.normal(size=(n, d))
for layer in range(1, 7):
    p = {"attn": [rng.normal(0, d**-0.5, (d, d)) for _ in range(4)],
         "ffn": [rng.normal(0, d**-0.5, (d, 4 * d)), np.zeros(4 * d),
                 rng.normal(0, (4 * d)**-0.5, (4 * d, d)), np.zeros(d)]}
    X = block(X, p)
    print(f"after block {layer}: row norms {np.round(np.linalg.norm(X, axis=1), 2)}")
```

The row norms tend to grow with depth (from about 2–4 after block 1 to 6–8 after block 6), because each block *adds* to the residual stream. See F3.

---

## F. ★ Challenge

1. Unroll $x_l = x_{l-1} + F_l(x_{l-1})$ to get $x_N = x_0 + F_1(x_0) + F_2(x_1) + \cdots + F_N(x_{N-1})$. The token's vector is a running sum: each layer reads the current sum and **adds** its contribution to it. It flows like a stream that every layer reads from and writes into. *(1 mark)*
2. $\frac{\partial x_N}{\partial x_0} = \prod_l \left(I + \frac{\partial F_l}{\partial x_{l-1}}\right) = I + \sum_l \frac{\partial F_l}{\partial x_{l-1}} + (\text{higher-order products})$. *(1 mark)* The $I$ term means the gradient of the loss reaches layer 1 directly, unchanged, whatever the other layers do. So early layers always get a usable signal. *(1 mark)*
3. A sum of $N$ independent vectors of similar size $s$ has a norm of about $s\sqrt{N}$, growing with depth (or faster, if they are correlated). The output head would then see vectors whose scale depends on depth. A final LayerNorm resets the scale, so the logits stay in a sensible range. *(1 mark)*
