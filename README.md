# EnterpriseSynth: IntelliSys 2027 constraint-repair audit

**New measured study:** *Validity Is Not Utility: A Paired Audit of Constraint Repair and Class Coverage in Synthetic Tabular Data.* Prepared for the Artificial Intelligence / Trustworthy & Responsible AI topic at IntelliSys 2027.

The paper asks whether making synthetic rows valid preserves downstream minority-class utility and whether feature-only rejection distorts class coverage. This repository has independent Git history and a newly implemented experiment. It is not a mirror of the upstream project and makes no claim of a new generator or privacy guarantee.

## Paper and evidence

- [Anonymous manuscript PDF](output/pdf/IntelliSys2027_EnterpriseSynth_Draft.pdf)
- [Springer LaTeX source](paper/main.tex)
- [Protocol recorded before the run](paper/STUDY_PROTOCOL.md)
- [Raw measurements: 210 predictive evaluations](results/metrics.csv)
- [Paired effects and conditional stability intervals](results/paired_effects.csv)
- [195 actual generated releases](data/generated/)
- [Submission checks and remaining human review](paper/SUBMISSION_CHECKLIST.md)
- [Source provenance and differences from the earlier project](SOURCE_PROVENANCE.md)

Three datasets, five stratified folds, three generators, four postprocessing methods, plus bootstrap and real-training references. All three processing methods remove the selected violations. Adult conditional projection reduces positive-class F1 in all five folds (mean 66.8% to 65.4%). Rejection changes class coverage; quotas preserve counts but do not reliably restore utility. These are limited empirical findings, not a neural-generator benchmark, regulatory certification, or privacy evaluation.

## Reproduction

Python 3.11+ and a fresh virtual environment are recommended. Install `requirements.txt`, then:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/validate_evidence.py
PYTHONPATH=src MPLCONFIGDIR=/tmp/enterprise-mpl python scripts/report_results.py
```

The checked-in run is protected from accidental overwrite. To execute all 210 fits again, copy this repository into a **separate experiment directory**, retain `data/raw`, `src`, `scripts`, and the protocol, and use empty `results` and `data/generated` directories. Then run:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_study.py
```

The runner uses no API keys, paid services, LLM inference, or external private data. All candidate generators fit training rows only. The preprocessing is fitted on real training data and reused; this is a curator-side audit, not a release-only recipient scenario. Floating-point environments can change borderline model predictions. Fit timings are descriptive and machine-specific.

Build the paper with a TeX installation containing XeLaTeX, `latexmk`, `newtxmath`, and Times New Roman:

```sh
latexmk -cd -xelatex -interaction=nonstopmode -halt-on-error -outdir=build paper/main.tex
python scripts/finalize_pdf.py
```

`paper/utility_table.tex` and `audit_table.tex` are generated evidence tables; their rows are embedded in `main.tex` for reliable TeX alignment. The validator checks that the manuscript rows still match the generated tables. Fonts are system dependencies, not bundled licensed font files.

## Responsible use and status

This is an **AI-assisted research draft requiring human scientific review**, not an automatic submission. The manuscript truthfully discloses design, coding, analysis, and writing assistance. IntelliSys restricts the scientific role of GenAI and requires human development and responsibility; independent author review and policy assessment are outstanding. The PDF omits author identities and project links. Public repository URLs and history identify its owner: do not submit this repository URL as an anonymous-review artifact without considering conference rules.

The user reported the earlier EnterpriseSynth paper as “not submitted or rejected”; no concurrent-review claim is made. No original results, estimated utility curves, regulatory epsilon tiers, or source manuscript prose are used. Attribution and dataset licenses are recorded separately. Do not infer legal compliance or differential privacy from the absence of exact training-row duplicates.
