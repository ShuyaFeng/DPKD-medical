# Archived paper-draft files

These four files were superseded on 2026-06-09 by a single
consolidated `paper_draft/section_3.tex`.

| File | Why archived |
|------|--------------|
| `section_3_body.tex` | Was the main §3 body. Now merged into `section_3.tex`. Predated PATE empirical results and α-loss section. |
| `section_3_appendix.tex` | Was the §3 appendix. Now merged into `section_3.tex` after `\appendix`. |
| `section_3_privacy.tex` | Thin wrapper that `\input`'d both halves above. No longer needed with a single file. |
| `main.tex` | Standalone preview document that wrapped `section_3_privacy.tex`. Obsolete; compile via `PaperForReview.tex` instead. |

Kept here for reference; do not edit. If you need the old per-file
split for any reason, restore from this directory and update
`PaperForReview.tex` `\input` paths accordingly.
