# Prospective local protocol: constraint repair and synthetic-data utility

Recorded 2026-11-01 before the first downstream evaluation. This is a local frozen protocol, not an external preregistration. The runner fingerprints the protocol, source, data, splits, and environment before running models. No previous EnterpriseSynth results will populate new tables.

## Research questions

1. Does eliminating feature-constraint violations improve held-out real-data minority-class F1, or can validity and utility move differently?
2. Does feature-only rejection change the target-class mix, and does class-quota rejection preserve prevalence without guaranteeing utility?
3. How do repair methods compare in marginal fidelity, exact training-row replication, and candidate examinations against an empirical-resampling reference?

## Data and splits

Use UCI Adult `adult.data` (first 6,000 complete rows, no outcome-dependent sampling), Bank Marketing `bank.csv` (4,521 rows), and Wine Quality red (1,599 rows). Report omissions and raw-data checks. Targets: income >50K; subscription yes; wine quality >=7. The target is excluded from all validity rules and projection edits.

Use five shuffled, stratified outer folds with random state 2027: 80% training, 20% test, disjoint test folds. No hyperparameter or threshold selection on the test data. Candidate pool size is 10 times the training row count. Each fold fits generators anew, with a deterministic seed based on dataset and fold. One pool per generator/fold is shared by all four processing arms.

## Validity rules

Adult: education category must agree with its canonical education-num code. Bank: pdays=-1 iff previous=0, with previous nonnegative; these are metadata-derived record-consistency checks, not universal business rules. Wine: total sulfur dioxide >= free sulfur dioxide; other nonnegative quantities remain supported by empirical marginals. Audit real-row violation rates rather than silently removing inconvenient real records.

Projection changes only education-num (Adult), previous (Bank: set to zero if pdays=-1, otherwise at least one), or total sulfur dioxide (Wine: raise to at least free). It never changes target labels. Rules are limited, dataset-specific checks, not proof of complete realism.

## Generators and postprocessing

- Independent empirical marginals: diagnostic loss-of-dependence baseline, not a competitive state-of-the-art claim.
- Joint Gaussian-rank copula: encode nominal categories by a fixed alphabetical training-category ordering; transform training ranks with normal quantiles; estimate correlation and apply 5% identity shrinkage; sample and invert empirical marginals. This is a transparent custom baseline, not SDV or CTGAN.
- Label-conditional copula: fit a separate feature copula per target class and draw class counts from training prevalence. A stronger supervised dependency baseline; no test labels enter fitting.
- Empirical row bootstrap: retain whole training rows with replacement, same release size. Explicit record-copying reference, not a privacy-preserving synthesizer.

Four postprocessing arms: first n pool rows unchanged; deterministic projection of exactly those rows; first n valid pool rows (ordinary rejection); first n valid rows subject to training-prevalence class quotas. No backfill if the pool is exhausted. Record incomplete releases and candidate examinations. Pool generation cost is shared; prefix examinations are not represented as separately executed generator calls.

## Predictive and data-quality evaluation

Fit a fixed class-balanced logistic regression (C=1, max_iter=1000), with numeric standardization and one-hot categorical encoding fit on the training partition and reused across arms. Use threshold .5; report positive-class F1, recall, balanced accuracy, average precision, accuracy. Also report feature-rule violation rate, target prevalence, absolute prevalence drift, marginal Wasserstein/IQR or categorical TV distance, and exact full-row training match rate. Exact copying is descriptive, not a membership attack or DP guarantee.

Keep all 180 generator/postprocessing/fold releases, 15 bootstrap releases, and 15 real-data predictive references. Export actual synthetic CSVs, predictions, raw metrics, split identities and source hashes. No simulated utility, fitted epsilon curves, or legal-compliance tiers are permitted.

## Statistical interpretation

Report paired differences within generator/dataset/fold. Five folds are the repeated units within a dataset; overlap among their training sets limits independence. Report per-dataset fold ranges and paired effects; bootstrap paired folds with 2,000 draws only as conditional stability intervals, not population confidence guarantees. Three datasets are insufficient for broad domain or regulatory conclusions. Do not assert novelty of projection, rejection, copulas, or class quotas.

## Scope

No new DP mechanism, formal privacy guarantee, legal-compliance mapping, neural synthesizer superiority, or LLM experiment is claimed. The source project motivated a narrower, new measured study. The author reports the original is not currently submitted/under review (not submitted or rejected). Preserve anonymous attribution to closely related prior work and maintain accurate GenAI disclosure and pending human-author review.
