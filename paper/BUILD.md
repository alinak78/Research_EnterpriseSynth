# Build the anonymous paper

Use XeLaTeX and BibTeX through `latexmk` with Times New Roman available:

```
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The original conference `svproc.cls`, support styles, and `splncs03.bst` are included unmodified. Do not substitute a two-column IEEE class or compress the official margins. Font files are not bundled. Figures are vector PDFs. The main text has embedded table bodies matching the generated table fragments.

From the full repository root, `python scripts/finalize_pdf.py` clears metadata and copies the result into `output/pdf`. The source archive can be compiled without the experiment data. Human authors must review the science and the truthful GenAI declaration before submission. Source compilation alone does not certify scientific validity or conference policy compliance.
