# Regenerating the artifacts from scratch

`../data/` holds the output of the expensive stages: API probing of seven closed-weight
models, and teacher-forced scoring on rented GPUs. Everything in `../src/` runs offline
from those files, so the paper reproduces with no key and no hardware. This directory
documents how they were produced.

## Files here

| file | |
|---|---|
| `UPSTREAM_METHODS.md` | the full methods document from the upstream project — corpus construction, question generation, iteration history, scoring rules |
| `export.py` | rebuilds the scored artifacts in `../data/` from a completed upstream run |
| `export_raw.py` | rebuilds the audit layer in `../data/raw/` — prompts, gold continuations, verbatim model outputs |
| `verify.py` | walks the chain from `../data/raw/` to `../data/` and asserts they agree |

## What the verifier caught

`verify.py` is not decoration. Writing it surfaced two real errors in this
repository's own analysis code, both of the kind that produce plausible numbers
rather than crashes:

1. **The attempt predicate treated `nonanswer` as a boolean.** It is a taxonomy
   (`attempt` / `empty` / `refusal_policy` / `unknown_explained`), so the test
   never fired and 537 blank responses were averaged in as zero-token attempts.
   Fixing it moved the novel positive control from 10.7 to 13.0 contiguous tokens
   and the scrambled negative control from 0.62 to 1.06 — the latter mattering
   most, because a negative control sitting *below* the measured floor should
   have been implausible on its face. This is the same bug `UPSTREAM_METHODS.md`
   §6.1 reports having fixed once already upstream; it was reintroduced here
   independently.

2. **Policy refusal was undercounted forty-fold on court opinions.** Providers
   deliver a blocked response either as an explicit refusal or as an empty body
   with `finish=content_filter`, and the taxonomy files the second under `empty`
   alongside budget truncation. Counting only explicit refusals gives 3 of 13,776
   court items; counting content-filtered empties with them gives 118. The
   scored artifact now carries `finish` and `provider_refused` so the
   distinction is available without the raw layer.

Neither error was visible from the outputs alone. Both were found by rebuilding
the derived numbers from the raw text and asking why the two disagreed.

## The four stages

**1. Corpus.** 106 US federal opinions crawled from CourtListener, which serves
`plain_text` / `html` / `html_with_citations` as distinct fields. Controls are Project
Gutenberg texts. Documents are normalized to reflow paragraphs before anything is scored:
no mainstream PDF or HTML extractor rejoins hard-wrapped lines, and an artifact newline
every ~11 tokens is a high-surprisal token at ~9% of positions, which depresses a
teacher-forced product enough to make a perfectly memorized document look unmemorized.

**2. Question generation.** A script importing only the Python standard library. Every
prompt and gold answer is regex-extracted from the source, so ground truth is mechanically
derived rather than model-produced — in a study of whether models memorized documents,
using a model to build the test would introduce the contamination under study. This is the
one place in the project where generative AI was categorically excluded.

**3. Black-box probe.** Seven closed-weight models through OpenRouter, temperature 0,
resumable, with retrieval disabled in the adapter: it raises on any online-enabled model
string and never transmits tool or plugin parameters.

**4. Teacher-forced scoring.** Llama 3.1 at 8B and 70B, base and instruct, bf16, no
quantization, on rented A100s. `UPSTREAM_METHODS.md` and the upstream `score_logprobs/`
directory document the four alignment traps this code handles: the off-by-one between
logits and positions, BPE merges across the prefix/suffix seam, logits at padded
positions, and bf16 softmax precision over a 128k vocabulary.

## Running it

The pipeline lives in the separate upstream project (`Memorization/` in the original
monorepo). Note that the upstream data directory is
`Data Collection and Training Material Generation/datasets/`, **not**
`Memorization/datasets/` — the paths printed in `UPSTREAM_METHODS.md` §9 are relative to
the former and do not resolve from the latter.

```bash
export CRAWL_CONTACT="you@example.com"      # sec.gov's automated-access policy
export COURTLISTENER_TOKEN="..."

python Memorization/crawl/fetch_courtlistener.py
python Memorization/qa/build_questions.py --v2
python Memorization/control/build_controls.py --text ...
python Memorization/probe/run_probe.py --model <m> --cutoff <date> --qa <qa.csv>
python Memorization/qa/score_answers.py --null
python Memorization/qa/score_answers.py --predictions <f> --out-dir <d>
```

Then rebuild the compact artifacts, pointing `--src` at the upstream datasets directory:

```bash
python regenerate/export.py --src "/path/to/Data Collection and Training Material Generation/datasets"
```

`export.py` is the only file in this repository that reaches outside it.

## Discrepancies with UPSTREAM_METHODS.md

`UPSTREAM_METHODS.md` is preserved as written and is **not** fully consistent with the
committed data. Read it for the design reasoning, not as a description of these files.

- **Score directories.** It refers to `scores_v2/`, `scores_forced/` and `scores_merged/`.
  The committed run has a single flat `scores/` directory.
- **The forced-attempt arm.** It describes re-asking every declined item with the escape
  hatch removed, producing voluntary/forced/merged populations with a `prompt_mode`
  column. The shipped probe replaced that design with a single compulsory prompt plus a
  separate `RECALL: yes|no` self-report, and no committed score file carries a mode
  column. The paper in `../paper/` describes only what the committed data supports.
- **Five control corpora.** Its results table lists Pride and Prejudice and Moby-Dick
  alongside Gatsby. The committed per-item scores cover Constitution, Gatsby and scrambled
  Gatsby only, so the paper reports three.
- **Model roster.** Its tables predate the re-run that produced these scores, which cover
  seven models including newer versions.
