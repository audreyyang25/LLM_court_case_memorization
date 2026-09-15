# Data manifest

Everything in `../src/` runs from the top four files alone — no API key, no
network, no GPU, no rented box.

| file | contents | rows |
|---|---|---|
| `blackbox_scores.csv.gz` | per-item sampled-probe scores | 15,190 |
| `logprobs.csv.gz` | teacher-forced logprobs, 4 truncation lengths | 27,194 |
| `hotspots_classified.csv.gz` | the 399 highest-probability spans, adjudicated | 399 |
| `null_floor.json` | the measured chance level | — |
| `raw/questions.csv.gz` | every prompt and gold continuation | 2,322 |
| `raw/predictions.csv.gz` | every model output, verbatim | 15,190 |

`raw/` is the audit layer. Nothing in `../src/` reads it; it exists so the
measurement can be checked rather than only re-run, and
`../regenerate/verify.py` walks the chain from it to the scored artifacts.
Delete it and the paper still reproduces.

## Provenance

Extracted from the parent project's memorization pipeline. Black-box probing ran
2025-08-13 against seven closed-weight models through OpenRouter at temperature 0
with retrieval disabled in the adapter (the model adapter raises on any
online-enabled model string and never transmits tool or plugin parameters — a
model able to fetch the opinion would make the measurement vacuous). Teacher-forced
scoring ran 2025-08-17/18 on rented A100s against Llama 3.1 weights at bf16, no
quantization.

## Corpora

| corpus | role | why |
|---|---|---|
| US Constitution | positive control | heavily duplicated public-domain legal text; if the harness cannot find this, it cannot find anything |
| The Great Gatsby | positive control | reported in the extraction literature as substantially memorized by open-weight models |
| Gatsby, scrambled | negative control | same vocabulary, same length, same register, word order destroyed within 40-word windows — no reproducible sequence |
| Court opinions | target | 106 US federal opinions, 2014–2026, from CourtListener |

The two controls bound different things and are often conflated. The
gold-versus-gold floor in `null_floor.json` runs no model and bounds *corpus
formulaicity* — how much unrelated legal text overlaps by chance. The scrambled
control runs a model and bounds *model and metric jointly* — whether fluent
generic prose scores above the floor, i.e. whether the metric rewards style
rather than sequence. The second catches a failure the first structurally cannot.

## Schemas

### `blackbox_scores.csv.gz`

One row per (case, question, model).

| column | |
|---|---|
| `corpus`, `work` | `controls`/`court_opinions`; `constitution`/`gatsby`/`gatsby_shuf`/`court_opinion` |
| `case_id`, `qid` | document and question identifiers |
| `tier` | question difficulty class (`A` short factual, `D` passage continuation) |
| `arm` | `pre_cutoff` / `post_cutoff_control` / `positive` / `negative` |
| `model`, `model_cutoff` | the probed model and its stated training cutoff |
| `longest_run` | **the headline metric** — longest contiguous run of tokens reproduced from the true continuation |
| `prefix_tokens` | leading tokens matched before the first divergence |
| `rouge_l`, `token_f1`, `exact_match` | secondary overlap metrics |
| `gold_tokens`, `pred_tokens` | lengths |
| `nonanswer` | **the authoritative disposition taxonomy**: `attempt` / `empty` / `refusal_policy` / `unknown_explained`. Not a boolean — reading it as one silently readmits the 549 empty responses as zero-token attempts |
| `refusal`, `unknown`, `errored` | booleans derived from `nonanswer`; kept for convenience, but `nonanswer` is the column to test |
| `finish`, `provider_refused` | the provider's finish reason and refusal flag, joined from `raw/predictions.csv.gz`. Needed because a blocked response arrives as an empty body with `finish=content_filter`, which the taxonomy files under `empty`; 240 of the 549 empties are declinations rather than scoring failures |

### `logprobs.csv.gz`

One row per (case, question, model, truncation).

| column | |
|---|---|
| `variant` | `base` or `instruct` — base is the headline condition |
| `condition` | `raw` (plain text, matching the original metric) rather than chat-templated |
| `trunc_n` | 10 / 25 / 50 / `full`; `logp_sum` is **not** comparable across different `n`, and these suffixes run 19–33 words, so a fixed `n` is what makes the numbers mean anything |
| `logp_sum` | $\sum \log P$ — the quantity thresholded for $(n,p)$-discoverable extraction |
| `logp_per_token`, `p_geomean` | comparable across rows of different length; the figures use `p_geomean` |
| `min_token_logprob`, `first_break_index` | memorized passages usually fail at one surprising token rather than degrading evenly; this says where |

### `hotspots_classified.csv.gz`

The 399 spans with the highest teacher-forced probability under Llama-3.1-70B
base, each adjudicated.

| column | |
|---|---|
| `rank`, `mean_logp`, `n_tokens` | position in the tail and span size |
| `text` | the span itself |
| `category`, `category_near` | content type: statutory citation, quoted precedent, SCOTUS boilerplate, court furniture, pleading template |
| `self_repetition`, `recur_in_post`, `recur_in_other_pre` | whether the span recurs within the document, in post-cutoff opinions, or across other pre-cutoff opinions |
| `verdict` | the adjudication; `MEMORISATION_CANDIDATE` is what survives after every other explanation is ruled out |

### `raw/questions.csv.gz`

One row per (document, question): the prompt, the gold continuation, and the
metadata identifying the arm. This is what lets a reader check that no prompt
leaks the continuation it asks for, and that the nested-prefix family holds the
target byte-identical while growing the prefix backward.

### `raw/predictions.csv.gz`

One row per (document, question, model): the model's output verbatim, plus
`finish`, `refused`, `served_by` and the self-reported `recall` line. Shipping
the strings is what makes the attempt predicate falsifiable rather than something
a reader has to trust.

### `null_floor.json`

Each gold answer scored against *other cases'* gold answers: same register, same
boilerplate, no shared provenance. Chance is measured rather than assumed to be
zero, because legal prose is formulaic enough that unrelated documents overlap by
construction.

## Known gaps

The upstream project also built Pride and Prejudice and Moby-Dick control sets
(`controls_qa_all5.csv` in the parent repo). The committed scores cover
Constitution, Gatsby and scrambled Gatsby only, so this repository reports those
three. Any table elsewhere listing five control corpora comes from a run whose
per-item scores were not retained.
