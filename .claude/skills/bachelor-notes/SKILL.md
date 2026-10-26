---
name: bachelor-notes
description: Write lecture notes for one section folder (e.g. 05_Scaled_Dot_Product_Attention) pitched at bachelor's (undergraduate) students. Use when asked to prepare, write, or redo the notes for a section, or to start the course notes. Produces <section>/notes.md.
---

# Bachelor-level section notes

Write `notes.md` inside the section folder the user names (or each folder in order, if asked to do all). Follow `style-guide.md` in this folder for level, tone, notation and originality rules. Read it before writing.

## Before writing
1. Read `../../../COURSE.md` if it exists, for course-wide decisions (notation, running example, what earlier sections covered).
2. Read the `notes.md` of the previous one or two sections so terms and notation stay consistent and nothing is re-taught from scratch.
3. Identify the 3–5 ideas a student must leave with. Everything else supports these.

## Structure of notes.md
Use these headings, in this order:

1. **Title and one-line summary**
2. **Learning objectives**: 3–5 bullets, each starting with a measurable verb (explain, compute, derive, compare, implement).
3. **Prerequisites**: what to know first, linking to earlier section folders by name. Add a 3–5 line refresher for any maths needed (matrix multiply, softmax, log, gradients) instead of assuming it.
4. **Motivation**: the problem this idea solves, starting from something the student already knows.
5. **Intuition first**: an everyday analogy or picture *before* any formula. Say where the analogy breaks down.
6. **The formal idea**: definitions, then formulas in LaTeX (`$...$`, `$$...$$`). Name every symbol and give its shape or units the first time it appears.
7. **Worked example**: small numbers (2–4 tokens, dimensions 2–4) a student can check by hand. Show every arithmetic step and round to 2 decimals.
8. **Diagram**: a Mermaid diagram or ASCII sketch of the data flow, with tensor shapes on the arrows.
9. **In practice**: where this shows up in real models (name models and real sizes), plus common hyperparameter values.
10. **Common mistakes and misconceptions**: 3–5 bullets in the form "Students often think X; actually Y because Z".
11. **Summary**: 5–7 bullet recap.
12. **Check your understanding**: 3 quick self-test questions, with answers in a collapsed `<details>` block.
13. **Further reading**: 2–4 reliable sources (the original paper, a textbook chapter, official documentation) with full titles. Cite only sources you are confident exist. Don't invent URLs.

## Length
About 1,500–2,500 words. Split anything longer into subsections, don't cut the worked example.

## After writing
- Re-check every number in the worked example by recomputing it (use `uv run python -c` for anything non-trivial).
- Add or update the section's line in `COURSE.md` (create the file if needed): section name, the key notation introduced, the running example used.
- Suggest running the `practice-problems` and `code-lab` skills for the same section.
