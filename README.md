# Do language models remember court opinions?

No. What they do reproduce is mostly statute text quoted inside the opinions.

This repository has the data, analysis code and figures for a memorization audit
of seven closed-weight models on 106 US federal court opinions. The study includes
a measured chance floor, two positive controls, a scrambled negative control, a
pre/post-training-cutoff comparison, a teacher-forced replication on open-weight
models, and hand labels for the 399 highest-probability spans. The paper itself
is kept outside this repository; all of its numbers and figures are regenerated
here with one command.

| corpus | role | mean contiguous tokens | share above 10 |
|---|---|---|---|
| US Constitution | positive control | **23.4** | 92% |
| The Great Gatsby | positive control | **13.0** | 61% |
| Gatsby, scrambled | negative control | **1.1** | 0% |
| **Court opinions** | **target** | **2.1** | **2.1%** |

Measured floor: 1.0 contiguous tokens. The probe picks up memorization on the
controls but not on court opinions. The distribution is also informative:
memorized text tends to be bimodal, and the court-opinion scores have no upper mode.

## Running

```bash
pip install -r requirements.txt
python -m src.run_all
```

Takes about ten seconds and needs no API key, network, or GPU, since the
expensive steps are already saved in `data/`. It writes `results/summary.json`
and the five figures in `results/figures/`.

```bash
python -m src.run_all --no-figures     # numbers only, no matplotlib
```

Bootstraps are seeded, so reruns give identical output.

## Verification

```bash
python regenerate/verify.py
```

Re-scores all 15,190 raw model outputs with a separate implementation of the
metric and compares each row to the committed scores. It also checks that no
prompt contains the continuation it asks for, and checks the attempt flag against
the prediction strings. This caught two errors during development; see
`regenerate/README.md`.

## Dependencies

`src/`, `data/` and `results/` don't depend on any other project. `python -m
src.run_all` needs only numpy and matplotlib.

Rebuilding the data files (re-probing models or generating questions for new
documents) needs the upstream harness. A copy is included in `vendor/`: the probe
harness, question extractors, scorer, the 106 opinion texts and the case
manifest. Scripts use that copy by default, so `--upstream` is optional; pass a
`Memorization/` checkout to use upstream instead. `vendor/VENDORED.md` lists
where the files came from, their hashes, and the one-line path change made to
four of them.

`results/` is committed so the numbers can be checked without running anything.

### Levels of reproduction

| | command | needs |
|---|---|---|
| **reproduce the reported numbers** | `python -m src.run_all` | numpy + matplotlib |
| **check the scoring** | `python regenerate/verify.py` | nothing extra |
| **rebuild from scratch** | `regenerate/`, `vendor/` | an OpenRouter key; code is in `vendor/` |

## Layout

```
data/       committed data files + MANIFEST.md (schemas, provenance, corpora)
data/raw/   prompts, gold continuations, model outputs
vendor/     copy of the upstream harness and its data (see vendor/VENDORED.md)
src/
  load.py     reads data files; defines the attempt filter
  stats.py    summary statistics
  figures.py  the five figures
  run_all.py  entry point
regenerate/ scripts to rebuild data/ from an upstream run, plus the verifier
results/    generated output, committed
```

## Method

**1. Corpus.** 106 US federal opinions from CourtListener. Controls are Project
Gutenberg texts. Documents are normalized to rejoin hard-wrapped lines before
scoring. Otherwise a stray newline every ~11 tokens (about 9% of positions) lowers
the teacher-forced probability enough to hide memorization.

**2. Question generation.** A standard-library-only script. Prompts and gold
answers are extracted from the source with regexes, so no model is involved in
building the test set (which could introduce the same contamination being
measured). Generative AI was not used in this step. Twenty questions per opinion:
4 metadata and 16 verbatim continuation.

**3. Black-box probe.** Seven closed-weight models via OpenRouter at temperature
0. Retrieval is disabled: the adapter raises an error on any online-enabled model
string and doesn't send tool or plugin parameters.

**4. Scoring and floor.** Each gold answer is also scored against other cases'
gold answers to estimate the chance level.

**5. Teacher-forced scoring.** Llama 3.1 8B and 70B, base and instruct, bf16, no
quantization, on rented A100s. `regenerate/UPSTREAM_METHODS.md` describes four
alignment issues the code handles: the off-by-one between logits and positions,
BPE merges across the prefix/suffix boundary, logits at padded positions, and bf16
softmax precision over a 128k vocabulary.

Steps 1–3 and 5 run from `vendor/`. `regenerate/export.py` and
`regenerate/export_raw.py` rebuild `data/` and `data/raw/` from a finished run;
see `regenerate/README.md` for commands.

## Controls

A null result on obscure legal text could also mean the probe doesn't work. The
controls are there to separate the two.

**Floor.** Scoring gold answers against other cases' gold answers (same register
and boilerplate, unrelated content) gives 1.0 contiguous tokens. So 2.1 on court
opinions is about 1 token above chance.

**Positive controls.** The Constitution is widely duplicated; Gatsby is a novel
that prior extraction work reports as memorized. They score 23.4 and 13.0 tokens,
so the probe can detect memorization.

**Negative control.** Scrambled Gatsby keeps the vocabulary and length but
shuffles word order. It scores 1.06, at the floor. This checks that the metric
isn't just rewarding fluent generic prose.

**Pre/post cutoff.** Opinions filed after a model's training cutoff can't have
been memorized. For all seven models the pre-minus-post difference is
indistinguishable from zero, with interval widths around one token, compared with
a positive-control effect of about twenty.

**White-box replication.** The sampled probe approximates a metric that needs
logits. Running the teacher-forced metric on Llama 3.1 (8B and 70B, base and
instruct) gives the same ordering as the black-box probe.

**Verification.** `regenerate/verify.py` (see above).

**Top spans.** The 399 highest-probability spans were labeled by hand. Seven
(1.8%) remain as memorization candidates, and all seven are quoted external text
(an agency memorandum, statutory lists) rather than the court's own writing. Half
of the spans in this set come from opinions filed after the training cutoff.

## Uncertainty

Intervals are seeded 2,000-draw percentile bootstraps that resample documents
instead of items, since the questions drawn from one document share its text. On
court opinions the intraclass correlation by document is 0.057 with mean cluster
size 128, so the design effect is 8.2 and the effective sample size is about 1,600
out of 13,533. Item-level intervals are about 2.7x too narrow. Both are computed
in `src/stats.py` and saved in `results/summary.json` as `ci` and `ci_item_level`.

## Limitations

Black-box results are lower bounds. A model that has memorized text but won't
reproduce it counts as unmemorized here. Refusal rates are reported per model for
this reason, and the white-box replication partly addresses it.

This measures verbatim reproduction, not knowledge of the case. A model may know
the outcome of a case from summaries or commentary without being able to
reproduce the opinion.

Scope: 106 US federal opinions from 2014–2026 and seven closed-weight models at
one point in time (providers can change what a model identifier points to). State
courts, administrative decisions and non-US courts weren't tested. The sample
leans toward securities and ERISA cases because of how it was drawn.
