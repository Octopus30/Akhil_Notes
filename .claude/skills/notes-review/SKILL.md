---
name: notes-review
description: Review a section's course material (notes.md, exercises.md, solutions.md, lab.py) for correctness, bachelor's-level clarity, consistency with other sections, and originality, then fix what it finds. Use when asked to review, check, proofread, or quality-check notes or a section, or after a section has been written.
---

# Review a section

Follow `../bachelor-notes/style-guide.md`. Review the section folder the user names, or every folder that has a `notes.md` if asked to review all.

## Checklist
**Correctness**
- Recompute every number in the notes, exercises and solutions (`uv run python -c ...`).
- Check that formulas, shapes and dimensions agree with each other and with the notation table.
- Check that the claims about real models (sizes, dates, papers) are accurate. Remove anything you can't stand behind.
- Run `lab.py` and confirm it completes and matches the notes.

**Level and clarity**
- Every new term is defined at first use, and no concept is used before it's taught (check against earlier sections and `COURSE.md`).
- An intuition comes before every formula, and every formula is explained in words.
- No paragraph runs longer than about 6 lines. Jargon a second-year student wouldn't know gets explained or cut.

**Consistency**
- Same symbols, terms and running example as the other sections.
- `exercises.md` only tests what the notes teach, and `solutions.md` covers every question.

**Originality**
- No text, examples or figures copied or lightly reworded from outside sources. Rewrite anything that reads like it was.

## Output
Fix the problems directly in the files. Then report a short list to the user: what you changed, and anything you couldn't resolve that needs their decision.
