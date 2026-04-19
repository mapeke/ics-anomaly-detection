# Thesis draft

Working markdown draft, ported to LaTeX for final submission.

## Layout

- `outline.md` — abstract, chapter map, figure/table list, writing order, risks.
- `chapters/01_introduction.md` — motivation, three contributions, scope.
- `chapters/02_background.md` — datasets, detectors, metrics, transfer, attribution (stub with cite list).
- `chapters/03_method.md` — implementation walk-through; mirrors `src/` layout.
- `chapters/04_results_detection.md` — Phase-1/2 detection grid.
- `chapters/05_results_transfer.md` — headline chapter; Phase-3 findings.
- `chapters/06_results_attribution.md` — Phase-4 attribution findings.
- `chapters/07_discussion.md` — objections and practical implications.
- `chapters/08_conclusion.md` — recap + future work.

## Writing conventions

- Every number in these files is cross-referenced to a cell in a notebook or a row in a parquet file. If a number needs updating (new seeds, new model), update the source first, re-execute the notebook, then fix the chapter.
- Figure placeholders (`Insert Figure F1 ...`) refer to the table in `outline.md`.
- Prefer concrete numbers over hedging. "Dense AE p@5 on P3 is 0.34 vs. random baseline 0.09" is better than "Dense AE shows strong attribution on P3."

## Next actions

Per outline.md writing order:
1. Flesh out Chapter 3 bullets into full prose (method is the most deterministic).
2. Chapter 4 — Phase 1/2 results — should read like a report, not a lab notebook.
3. Chapter 5 — biggest chapter, most reviewer scrutiny; three sub-findings each get ~2 pages.
4. Chapter 6 — attribution is short but load-bearing; don't skimp on the null finding discussion.
5. Chapter 2 — easy once the results are written.
6. Chapter 1 + 7 + abstract last.
