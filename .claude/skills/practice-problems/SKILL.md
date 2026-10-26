---
name: practice-problems
description: Create graded exercises with full worked solutions for a section folder, at bachelor's (undergraduate) level, like a tutorial sheet or exam prep. Use when asked for exercises, practice questions, a problem set, quiz, or exam questions for a section. Produces <section>/exercises.md and <section>/solutions.md.
---

# Practice problems for a section

Follow `../bachelor-notes/style-guide.md` for audience, notation and originality. Read the section's `notes.md` first. Exercises must only use ideas taught in that section or in earlier sections.

## exercises.md
Group the questions in this order, with the marks in brackets after each question:

1. **Warm-up (recall)**, 3 questions: definitions, true/false *with justification*, fill-in-the-shape ("Q has shape __").
2. **Calculation**, 3–4 questions: by-hand computations with small matrices (2×2 to 3×4). Use numbers that differ from the worked example in `notes.md`.
3. **Conceptual**, 2–3 questions: "why" and "what would happen if" questions, e.g. "What happens to attention weights if we drop the √d_k scaling and d_k = 512?"
4. **Applied / coding**, 1–2 questions: implement or modify something in NumPy, with a precise spec of the inputs, outputs and shapes.
5. **Challenge**, 1 question (optional, marked ★): a derivation or a design trade-off that stretches the strongest students.

Also include:
- Total marks, and an estimated time at the top (aim for 45–60 minutes).
- One multiple-choice block of 5 questions, each with 4 plausible options. The distractors should come from real misconceptions, not obvious nonsense.

## solutions.md
- Give a full worked solution for every question, showing every step, not just the answer.
- Add a marking scheme for calculation and conceptual questions, e.g. "1 mark: correct QKᵀ; 1 mark: scaling; 1 mark: softmax rows sum to 1".
- For multiple-choice questions, explain why each wrong option is wrong.
- Give coding solutions as complete, runnable snippets.

## Verify
Recompute every numeric answer with `uv run python -c "..."` (NumPy is available once `uv add numpy` has been run; add it if missing). Run every coding solution. Fix any mismatch before finishing.
