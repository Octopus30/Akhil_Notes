---
name: code-lab
description: Build a hands-on, runnable Python lab for a section folder that lets bachelor's students implement and experiment with the section's idea (NumPy first, PyTorch only when needed). Use when asked for a lab, code example, notebook, implementation, or demo for a section. Produces <section>/lab.py (or lab.ipynb if asked).
---

# Code lab for a section

Follow `../bachelor-notes/style-guide.md`. Read the section's `notes.md` so the code uses the same notation and the same example numbers.

## Principles
- **From scratch first.** Implement the idea in NumPy so every step is visible. Then, if a library version exists, show the one-line library equivalent (e.g. `torch.nn.functional.scaled_dot_product_attention`) and check that both give the same answer.
- **Readable over clever.** Use short functions with type hints and one docstring each. Use variable names that match the notes (`Q`, `K`, `V`, `d_k`).
- **Shapes everywhere.** Add an `assert x.shape == (...)` after each important step, and print shapes in the walkthrough.
- **Small and deterministic.** Use tiny inputs and `np.random.default_rng(0)`. The lab should run in seconds on a laptop CPU.

## Layout of lab.py
Split the file into numbered parts with `# %%` cell markers, so it also runs as cells in VS Code:
1. Setup: imports and seed.
2. Reproduce the worked example from `notes.md`, and check the result matches the numbers printed in the notes.
3. A general implementation as a function.
4. Experiments: 2–3 "try this" experiments that print or plot something revealing (e.g. how softmax sharpens as d_k grows, what masking does). Put each behind a clearly labelled function.
5. Student TODOs: 2–3 small extensions marked `# TODO(student):`, each with a hint and a check the student can run.

## Dependencies
Add packages with `uv add` from the project root (numpy first, matplotlib for plots, torch only if the lab needs it). Never pip-install into the global Python.

## Verify
Run `uv run python <section>/lab.py` and confirm it completes without errors, and that the values it prints match `notes.md`. Make sure the student TODOs are left unimplemented but don't crash, e.g. guard each one with a `NotImplementedError` check that the script catches and reports.
