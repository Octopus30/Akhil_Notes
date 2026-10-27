# Section 16 — Exercises: Fine-Tuning & LoRA

**Total: 47 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 55 minutes**

LoRA uses the paper's column-vector convention: $h = W_0x + \frac{\alpha}{r}BAx$, with $B$ of shape $d \times r$ (initialised to 0) and $A$ of shape $r \times k$ (random). Trainable parameters per matrix: $r(d + k)$. Full fine-tuning ≈ 16 bytes per parameter.

---

## A. Warm-up (recall)

**A1.** For a team that wants a model to answer from a large, changing set of company documents, compare three options: prompting, RAG and LoRA fine-tuning. Say which you would try first and why. *(3 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. At initialisation, a LoRA model gives exactly the same outputs as the base model.
2. A merged LoRA model is slower at inference than the base model.
3. QLoRA stores the trainable LoRA matrices in 4-bit.

**A3.** Fill in. *(3 marks)*

1. $W_0$ is $d \times k$, so $B$ is `__ × __` and $A$ is `__ × __`.
2. With $\alpha = 32$ and $r = 8$, the scale factor $\alpha/r$ is `__`.
3. A typical LoRA learning rate is around `__`, higher than full fine-tuning's ≈ $10^{-5}$.

---

## B. Multiple choice (1 mark each)

**B1.** Why is $B$ initialised to zero rather than $A$?
- (a) It doesn't matter which one is zero.
- (b) So that $BA = 0$ at the start (the model is unchanged), while gradients can still flow, because $A$ is random.
- (c) To reduce memory.
- (d) Because $B$ is larger than $A$.

**B2.** Why does a low-rank update usually suffice for adapting to a task?
- (a) Pre-trained weights are already low-rank.
- (b) The *change* needed for a task tends to lie in a few important directions.
- (c) Rank doesn't affect what can be learned.
- (d) GPUs can only multiply low-rank matrices.

**B3.** Which method cannot be merged into the base weights, and so adds a little inference latency?
- (a) LoRA
- (b) Adapters (bottleneck layers inserted inside each block)
- (c) Full fine-tuning
- (d) QLoRA after merging

**B4.** After narrow fine-tuning, the model gets worse at general questions. This is called:
- (a) Exposure bias
- (b) Catastrophic forgetting
- (c) Softmax saturation
- (d) Quantization error

**B5.** A common silent failure when fine-tuning a chat model is:
- (a) Using too few GPUs.
- (b) Formatting the training data with a different chat template than the model expects.
- (c) Using AdamW.
- (d) Training for one epoch.

---

## C. Calculation

**C1. Parameter counts.** *(5 marks)*

1. For one $5120 \times 5120$ matrix, how many trainable LoRA parameters are there with $r = 8$ and with $r = 32$? What share of the matrix is the $r = 8$ adapter?
2. LoRA with $r = 8$ is applied to $W_Q$ and $W_V$ in all 40 layers of a model with $d = 5120$. How many trainable parameters is that? How big is the adapter file in 16-bit?
3. For a non-square FFN matrix of shape $4096 \times 11{,}008$ with $r = 16$, give the LoRA count and its share of the matrix.

**C2. LoRA forward pass.** $W_0 = \text{diag}(1, 2, 1)$, $r = 1$, $\alpha/r = 2$, $A = [1, 1, 0]$, $B = [1, 2, 0]^{\top}$, $x = [2, -1, 1]^{\top}$. *(5 marks)*

1. Compute $W_0x$, $Ax$, $\frac{\alpha}{r}B(Ax)$ and $h$.
2. Compute the merged $W = W_0 + \frac{\alpha}{r}BA$, and check that $Wx = h$.
3. What is $h$ at initialisation?

**C3. Memory for a 13B model.** Use the LoRA setup of C1.2 (6.55M trainable parameters). *(3 marks)* Estimate the memory before activations for: full fine-tuning; LoRA with a 16-bit frozen base; QLoRA with a 4-bit base.

**C4. How much rank?** The ideal update for a $8 \times 8$ matrix is a sum of four orthogonal rank-1 pieces with strengths 6, 3, 2 and 1. The relative error of the best rank-$r$ approximation is $\sqrt{\sum_{i > r} s_i^2 / \sum_i s_i^2}$. Fill in the error for $r = 0, 1, 2, 3, 4$, and the parameters $r(8 + 8)$. *(3 marks)*

**C5. Other methods.** *(2 marks)*

1. Adapters in a 24-layer model with $d = 1024$, bottleneck $m = 32$ and two adapters per layer (down, up, both biases): how many parameters?
2. Prompt tuning with 50 virtual tokens on $d = 5120$: how many parameters?

---

## D. Conceptual

**D1.** One company needs 20 differently fine-tuned variants of a 7B model, one per customer. Compare full fine-tuning with LoRA for storage and serving. *(3 marks)*

**D2.** Why does LoRA use a higher learning rate than full fine-tuning? *(2 marks)*

**D3.** Name two ways to limit catastrophic forgetting during fine-tuning. *(2 marks)*

---

## E. Applied / coding (NumPy)

**E1.** Implement a `LoRALinear` class with a frozen `W0` (`d × k`), trainable `A` (random, `r × k`) and `B` (zeros, `d × r`), a `forward(x)` method for a batch $X$ of shape `n × k` (rows are inputs, so compute $XW_0^{\top} + \frac{\alpha}{r} XA^{\top}B^{\top}$), and a `merge()` method that returns $W$. Check C2, check that the outputs at initialisation equal the base layer's, and check that merged and unmerged outputs agree. *(5 marks)*

**E2.** Train only $A$ and $B$ by gradient descent so that `LoRALinear` with $W_0 = 0$ fits a target $W^*$ that is exactly rank 2 ($6 \times 6$), using $r = 2$. Print the error before and after. Then try $r = 1$ and explain the result. *(3 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

1. (Reminder: the **singular value decomposition** writes any matrix as $\Delta W = U\Sigma V^{\top} = \sum_i \sigma_i u_i v_i^{\top}$, a sum of rank-1 pieces along orthonormal directions $u_i$, $v_i$ with strengths $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$; `np.linalg.svd` computes it.) Using the SVD, explain why the best rank-$r$ approximation keeps the $r$ largest singular values, and show where the error formula in C4 comes from (use the Frobenius norm).
2. Show that if **both** $A$ and $B$ started at 0, gradient descent could never move them. (Hint: compute $\partial h/\partial A$ and $\partial h/\partial B$.)
3. Why scale by $\alpha/r$ rather than using $BA$ directly? Think about what happens to a good learning rate when you change $r$.
