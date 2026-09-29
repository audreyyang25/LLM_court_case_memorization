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
committed in `data/`. Writes `results/summary.json` and `results/figures/*.png`.

```bash
python -m src.run_all --no-figures     # numbers only, no matplotlib
```

Bootstraps are seeded, so repeated runs agree exactly.

## Checking it

```bash
python regenerate/verify.py
```

Re-scores all 15,190 raw model outputs with an independent implementation of the
metric and compares row by row against the committed scores, confirms no prompt
leaks the continuation it asks for, and tests the attempt predicate against the
prediction strings. It caught two real errors during development; see
`regenerate/README.md`.

## Self-contained

`src/`, `data/`, `paper/` and `results/` have no dependency on any other
project: `python -m src.run_all` regenerates every number and figure from the
committed artifacts with numpy and matplotlib, and nothing else.

Re-deriving those artifacts — re-probing new models, or building questions for
new documents — needs the upstream harness, which is vendored in `vendor/`
rather than referenced: the probe harness, the question extractors, the scorer,
together with the 106 opinion texts and the case manifest they read. Every entry
point defaults to that copy, so `--upstream` is optional; pass a real
`Memorization/` checkout to run against upstream instead. `vendor/VENDORED.md`
records provenance, per-file hashes, and the single one-line path change applied
to four of them.

`results/` is committed on purpose, so a reader can check the paper's numbers
without running anything at all.

### Three levels of reproduction

| | command | needs |
|---|---|---|
| **reproduce the paper** | `python -m src.run_all` | nothing but numpy + matplotlib |
| **audit the measurement** | `python regenerate/verify.py` | nothing |
| **rederive from scratch** | `regenerate/`, `vendor/` | an OpenRouter key; code is vendored |

## Layout

```
data/       committed artifacts + MANIFEST.md (schemas, provenance, corpora)
data/raw/   the audit layer: prompts, gold continuations, verbatim model outputs
vendor/     the upstream harness, vendored + its data (see vendor/VENDORED.md)
src/
  load.py     artifacts -> rows, with the attempt predicate defined in one place
  stats.py    the summaries behind every claim
  figures.py  the five figures
  run_all.py  entry point
regenerate/ rebuilding the artifacts from an upstream run, plus the verifier
paper/      main.tex, appendix.tex, refs.bib
results/    generated; committed so the paper's numbers are checkable without running anything
```

## Building the paper

`paper/` needs two files from the ICLR author kit that are not redistributable
and are therefore gitignored: `iclr2026_conference.sty` and
`iclr2026_conference.bst`. **Both are required.** `\bibliographystyle{iclr2026_conference}`
fails without the `.bst`, which produces an empty bibliography and `[?]` for
every citation.

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The four passes are necessary: BibTeX runs against the `.aux` file the first
pass writes, and the last two resolve citation and cross-reference numbers.

## The procedure

**1. Corpus.** 106 US federal opinions crawled from CourtListener, which serves
`plain_text` / `html` / `html_with_citations` as distinct fields. Controls are
Project Gutenberg texts. Documents are normalized to reflow paragraphs before
anything is scored: no mainstream PDF or HTML extractor rejoins hard-wrapped
lines, and an artifact newline every ~11 tokens is a high-surprisal token at ~9%
of positions, which depresses a teacher-forced product enough to make a
perfectly memorized document look unmemorized.

**2. Question generation.** A script importing only the Python standard library.
Every prompt and gold answer is regex-extracted from the source, so ground truth
is mechanically derived rather than model-produced — in a study of whether models
memorized documents, using a model to build the test would introduce the
contamination under study. This is the one place in the project where generative
AI was categorically excluded. Twenty questions per opinion, in two tiers:
metadata (4) and verbatim continuation (16).

**3. Black-box probe.** Seven closed-weight models through OpenRouter,
temperature 0, resumable, with retrieval disabled in the adapter: it raises on
any online-enabled model string and never transmits tool or plugin parameters.

**4. Scoring and the measured floor.** Each gold answer is also scored against
*other* cases' gold answers, which share register and boilerplate but no
provenance, to measure chance rather than assume it is zero.

**5. Teacher-forced scoring.** Llama 3.1 at 8B and 70B, base and instruct, bf16,
no quantization, on rented A100s. `regenerate/UPSTREAM_METHODS.md` documents the
four alignment traps this code handles: the off-by-one between logits and
positions, BPE merges across the prefix/suffix seam, logits at padded positions,
and bf16 softmax precision over a 128k vocabulary.

Stages 1–3 and 5 are driven from `vendor/`; `regenerate/export.py` and
`regenerate/export_raw.py` rebuild `data/` and `data/raw/` from a completed run.
See `regenerate/README.md` for the commands.

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
knowledge of the case. All seven models' differences are indistinguishable from
zero at interval widths near one token, against a positive-control effect of
roughly twenty. That is a precise null, not an underpowered one.

**A white-box replication.** The sampled probe is a proxy for a metric that needs
logits. Running the real teacher-forced metric on Llama 3.1 base and instruct at
8B and 70B reproduces the black-box ordering, which rules out the possibility
that sampling was missing memorization the logprobs would reveal.

**A checkable chain.** `regenerate/verify.py` re-scores all 15,190 raw model
outputs with an independent implementation of the metric and compares against the
committed scores, checks that no prompt leaks the continuation it asks for, and
tests the attempt predicate against the prediction strings.

**Adjudication of the tail.** Means hide tails, so the 399 highest-probability
spans were classified one at a time. Seven survive as memorization candidates
(1.8%), and all seven are themselves quoted external text — passages of an agency
memorandum, statutory enumerations — not a court's own reasoning. Half the spans
in the tail come from opinions filed *after* the training cutoff, which cannot
have been memorized at all.

## What this does not claim

Black-box results are lower bounds. A model that has memorized text but declines
to reproduce it registers here as unmemorized, which is why the disposition flags
are reported per model and why the white-box replication exists.

The claim is about verbatim reproduction, not about case knowledge. A model that
cannot reproduce a line of an opinion may still know how the case came out,
having learned it from summaries and commentary rather than from the opinion
itself. Nothing here measures that.

The claim is also about these opinions, these models, and this period — 106
opinions, US federal courts, 2014–2026, seven closed-weight models at one point
in time behind identifiers that providers re-point. State courts, administrative
adjudications, and non-US jurisdictions are untested, and the corpus is biased
toward securities and ERISA matters by the sampling frame it came from.
