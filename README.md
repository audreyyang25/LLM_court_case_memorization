# Do language models remember court opinions?

No. They remember the statutes quoted inside them.

This repository reproduces every number and figure in `paper/main.tex`: a
memorization audit of seven closed-weight models on 106 US federal court
opinions, with a measured floor, two positive controls, a scrambled negative
control, a pre/post-training-cutoff arm, a teacher-forced replication on open
weights, and hand adjudication of the 399 highest-probability spans.

| corpus | role | mean contiguous tokens | share above 10 |
|---|---|---|---|
| US Constitution | positive control | **23.4** | 92% |
| The Great Gatsby | positive control | **13.0** | 61% |
| Gatsby, scrambled | negative control | **1.1** | 0% |
| **Court opinions** | **target** | **2.1** | **2.1%** |

Measured floor: 1.0 contiguous tokens. The harness finds memorization where it
exists and does not find it in court opinions, and the distribution — not the
mean — is what carries the argument: real memorization is bimodal, and the
court-opinion distribution has no upper mode at all.

## Run it

```bash
pip install -r requirements.txt
python -m src.run_all
```

About ten seconds. No API key, no network, no GPU — the expensive stages are
committed in `data/` (765 KB). Writes `results/summary.json` and
`results/figures/*.png`.

```bash
python -m src.run_all --no-figures     # numbers only, no matplotlib
```

Bootstraps are seeded, so repeated runs agree exactly.

## Self-contained

This repository stands alone. `src/`, `data/`, `paper/` and `results/` have no
dependency on any other project: `python -m src.run_all` regenerates every number
and figure from the committed artifacts with numpy and matplotlib, and nothing
else — no API key, no GPU, no rented box.

The single exception is `regenerate/export.py`, which rebuilds `data/` from the
upstream probing pipeline and takes its location as `--src`. You never need it to
reproduce the paper — only to re-run the probe on new models or new documents.

`results/` is committed on purpose, so a reader can check the paper's numbers
without running anything at all.

## Checking it

```bash
python regenerate/verify.py
```

Re-scores all 15,190 raw model outputs with an independent implementation of the
metric and compares row by row against the committed scores, confirms no prompt
leaks the continuation it asks for, and tests the attempt predicate against the
prediction strings.

## Layout

```
data/       committed artifacts + MANIFEST.md (schemas, provenance, corpora)
data/raw/   the audit layer: prompts, gold continuations, verbatim model outputs
src/
  load.py     artifacts -> rows, with the attempt predicate defined in one place
  stats.py    the summaries behind every claim
  figures.py  the five figures
  run_all.py  entry point
paper/      main.tex, refs.bib
results/    generated; committed so the paper's numbers are checkable without running anything
```

## What the controls are for

A null result on obscure regulatory text looks exactly like a broken harness.
Every control below exists to tell those apart.

**The floor is measured, not assumed.** Each gold answer is scored against other
cases' gold answers — same register, same boilerplate, no shared provenance.
That floor sits at 1.0 contiguous tokens, so a raw score of 2.1 is not "2.1 above
zero" and reporting it as such would overstate recall by a factor of two.

**Two positive controls.** The Constitution is heavily duplicated public-domain
legal text; Gatsby is a novel the extraction literature reports as substantially
memorized. Finding 23.4 and 13.0 tokens on them is what licenses reading 2.1 as a
real null rather than as instrument failure.

**A negative control that runs the model.** Scrambled Gatsby has the same
vocabulary, length and register with word order destroyed. It scores 1.06 —
the floor exactly. This bounds something the gold-versus-gold floor structurally
cannot: whether the metric rewards fluent generic prose rather than sequence.

**A pre/post-cutoff arm.** A model cannot have memorized an opinion filed after
its training cutoff, so post-cutoff recall is what it produces with zero
knowledge of the case. Six of seven models' differences are indistinguishable
from zero and the seventh's lower bound is +0.02 tokens, at interval widths near
one token against a positive-control effect of roughly twenty. That is a precise
null, not an underpowered one.

**A white-box replication.** The sampled probe is a proxy for a metric that needs
logits. Running the real teacher-forced metric on Llama 3.1 base and instruct at
8B and 70B reproduces the black-box ordering, which rules out the possibility
that sampling was missing memorization the logprobs would reveal.

**A checkable chain.** `regenerate/verify.py` re-scores all 15,190 raw model
outputs with an independent implementation of the metric and compares against the
committed scores, checks that no prompt leaks the continuation it asks for, and
tests the attempt predicate against the prediction strings. It caught two real
errors during development; see `regenerate/README.md`.

**Adjudication of the tail.** Means hide tails, so the 399 highest-probability
spans were classified one at a time. Seven survive as memorization candidates
(1.8%), and all seven are themselves quoted external text — passages of an agency
memorandum, statutory enumerations — not a court's own reasoning. Half the spans
in the tail come from opinions filed *after* the training cutoff, which cannot
have been memorized at all.

## What this does not claim

Black-box results are lower bounds. A model that has memorized text but declines
to reproduce it registers here as unmemorized, which is why the disposition flags
are reported per model and why the white-box replication exists. The claim is
about these opinions, these models, and this period — not about legal text in
general.
