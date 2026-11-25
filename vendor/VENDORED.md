# Vendored code

The analysis in `src/` doesn't need any of this; it runs from the committed
artifacts in `data/`. This directory is here so the *upstream* half of the
pipeline can be re-run from a clone: re-probing new models, building questions
for new documents, re-scoring, re-judging. Without it, `--upstream` would point
at a path only the original author has.

## Provenance

Taken from the `Memorization/` directory of the author's `FinAITrainingData`
project at commit `201a786`, on 2026-09-15.

| file | lines | sha256 (first 12) | path-patched |
|---|---|---|---|
| `__init__.py` | 0 | `e3b0c44298fc` | no |
| `crawl/__init__.py` | 0 | `e3b0c44298fc` | no |
| `extract/normalize.py` | 79 | `bc8aa0283bf2` | no |
| `outcome/judge_outcome.py` | 602 | `d14033beb021` | yes |
| `outcome/models.py` | 108 | `c25340702b5c` | no |
| `outcome/summarize.py` | 142 | `e397c2f713c2` | yes |
| `probe/providers.py` | 146 | `fcdf92693056` | no |
| `probe/run_probe.py` | 380 | `6489c233a382` | yes |
| `qa/build_questions.py` | 507 | `1155880a42f4` | yes |
| `qa/classify_nonanswers.py` | 170 | `87c980945a07` | yes |
| `qa/score_answers.py` | 433 | `d66d1486d806` | yes |

## The only modification

Four of these files find their data root by walking three directories up from
their own location to the monorepo root:

```python
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "Data Collection and Training Material Generation")
```

Vendored here, that walk ends up outside the repository. Each is replaced with:

```python
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("MEMAUDIT_DATA") or os.path.join(REPO, "data")
```

The replacement is marked `VENDORED PATH PATCH` in every file it touches.
Nothing else is changed (no logic, prompts, thresholds, or scoring). Diff any
file against the upstream original to confirm.

Set `MEMAUDIT_DATA` to override the root without editing anything.

## The data under `vendor/data/`

The upstream tools read documents and a manifest as well as code, so those are
included with the same internal layout:

| path | contents |
|---|---|
| `data/datasets/court_opinions_courtlistener/` | 106 opinion texts, one per case in the audit |
| `data/datasets/court_opinions_qa_v2.csv` | the 2,120 court-opinion probe questions |
| `data/datasets/controls_qa.csv` | the Constitution / Gatsby / scrambled-Gatsby questions |
| `data/crawl_manifest/case_ids.json` | captions, dockets, dates and arm labels |

Court opinions are US government edicts and uncopyrightable, as is the
Constitution. *The Great Gatsby* entered the US public domain in 2021. Nothing
here is restricted.

Roughly 11 MB. The full upstream project is much larger and contains material
unrelated to this paper: a separate reasoning-fingerprint study, a judge panel,
and 6,064 source PDFs including law-firm memoranda and academic articles that we
can't redistribute. Vendoring the seven files this paper uses is smaller than
pointing a reader at all of that.

## Running against a real upstream checkout instead

Every entry point takes `--upstream` (or `$1` for the shell scripts) and defaults
to this directory. Pass a path to an actual `Memorization/` checkout to use that
instead:

```bash
python outcome/run_arms.py --condition closed --all --upstream /path/to/Memorization
bash regenerate/run_landmark_probe.sh /path/to/Memorization
```
