# Section 12 — Exercises: Training an LLM

**Total: 49 marks** (+ 4 bonus marks for the ★ challenge) · **Estimated time: 60 minutes**

Use natural logarithms. $C \approx 6ND$; training memory ≈ 16 bytes per parameter; Chinchilla ≈ 20 tokens per parameter.

---

## A. Warm-up (recall)

**A1.** Name the three stages of making an assistant LLM, and say what data each stage uses. *(3 marks)*

**A2.** True or false? Justify each answer in one sentence. *(3 marks)*

1. In SFT, the loss is computed on both the prompt tokens and the response tokens.
2. Perplexity 1 would mean the model is certain of every next token and always right.
3. DPO needs a separately trained reward model.

**A3.** Fill in. *(3 marks)*

1. For a model guessing uniformly over 32,000 tokens, the loss is `__` and the perplexity is `__`.
2. Of the 16 bytes per parameter, Adam's $m$ and $v$ take `__` bytes together.
3. In $C \approx 6ND$, roughly `__N` FLOPs per token are spent on the forward pass and `__N` on the backward pass.

---

## B. Multiple choice (1 mark each)

**B1.** Why is the learning rate warmed up at the start of training?
- (a) To save memory.
- (b) Early on, the weights are random and Adam's statistics are unreliable, so large steps are unstable.
- (c) Because the loss is too small at first.
- (d) Warm-up is only needed for SFT.

**B2.** What does Adam's bias correction fix?
- (a) $m$ and $v$ start at 0, so in early steps they underestimate the true averages.
- (b) Gradients that are too large.
- (c) Weight decay pulling the weights to 0.
- (d) The learning rate becoming negative.

**B3.** LLaMA-3 8B was trained on about 15T tokens, far beyond Chinchilla's 20 tokens per parameter. The main reason:
- (a) Chinchilla's result was wrong.
- (b) A smaller model trained longer reaches a given quality and is much cheaper to run for many users.
- (c) More tokens reduce memory use.
- (d) It was required by the tokenizer.

**B4.** In RLHF, why is there a KL penalty towards the SFT model?
- (a) To speed up training.
- (b) To stop the policy drifting into strange text that games the reward model.
- (c) To reduce the vocabulary.
- (d) It replaces the reward model.

**B5.** A loss spike suddenly appears during pre-training. Which response is standard practice?
- (a) Increase the learning rate.
- (b) Restart from a checkpoint before the spike, possibly skipping the bad batches, with gradient clipping on.
- (c) Remove the LayerNorms.
- (d) Switch to post-norm.

---

## C. Calculation

**C1. Loss and perplexity.** A model gives the correct next tokens probabilities 0.6, 0.3, 0.9, 0.05 and 0.5. *(4 marks)*

1. Compute the loss and the perplexity.
2. Which single position contributes the most loss? What share of the total loss is it?

**C2. AdamW by hand.** $\theta = 2.0$, $g = -0.2$, $\eta = 0.01$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\lambda = 0.1$, $\epsilon \approx 0$. *(5 marks)*

1. Do the first step ($t = 1$): compute $m_1, v_1, \hat m_1, \hat v_1$ and $\theta_1$.
2. The gradient at step 2 is again $-0.2$. Compute $\hat m_2$, $\hat v_2$ and $\theta_2$.
3. Why is the Adam part of the step the same size in both steps?

**C3. Learning-rate schedule.** Peak $6 \times 10^{-4}$, warm-up 1,000 steps, 50,000 steps in total, cosine decay to $6 \times 10^{-5}$. Find the learning rate at steps 500, 1,000, 25,500, 37,750 and 50,000. *(4 marks)*

**C4. Budget.** A 13B-parameter model is trained on 1T tokens. *(4 marks)*

1. Estimate the training FLOPs.
2. GPUs that each sustain $4 \times 10^{14}$ FLOP/s are available. How many GPU-hours are needed? How many days on 1,024 GPUs?
3. How many tokens would Chinchilla call compute-optimal for this model?
4. How much memory do the weights, gradients and optimizer states need? At least how many 80 GB GPUs is that, before activations?

**C5. DPO.** $\beta = 0.2$. The winner's log-probability has risen by 1.5 relative to the reference, and the loser's has fallen by 0.5. Compute the DPO loss, and compare it with the starting value. *(2 marks)*

---

## D. Conceptual

**D1.** Explain why, in SFT, the prompt tokens are kept in the input but masked out of the loss. What would go wrong otherwise? *(3 marks)*

**D2.** Explain why Adam lets one learning rate work for layers whose gradients differ in size by a factor of 1,000. *(2 marks)*

**D3.** Compare RLHF and DPO: what each needs, and one practical advantage of DPO. *(3 marks)*

---

## E. Applied / coding (NumPy)

**E1.** Implement `adamw_step(theta, g, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0)`, returning `(theta, m, v)`. Check it against C2 and the notes' example ($\theta = 1.0 \to 0.899$). Then use it to minimise $f(\theta) = (\theta - 3)^2$ from $\theta = 0$ with $\eta = 0.1$ for 200 steps, and print $\theta$. *(5 marks)*

**E2.** Implement `lr_schedule(step, peak, warmup, total, min_lr)` and check it against C3. *(3 marks)*

---

## F. ★ Challenge (bonus, 4 marks)

1. Using $C = 6ND$ and $D = 20N$ (the Chinchilla rule), write $N$ and $D$ as functions of $C$.
2. For a budget of $C = 10^{24}$ FLOPs, find the compute-optimal $N$ and $D$.
3. A team instead trains a model with $N/4$ parameters on the same budget. How many tokens does it see, and how many tokens per parameter is that? Give one reason a team might choose this.
