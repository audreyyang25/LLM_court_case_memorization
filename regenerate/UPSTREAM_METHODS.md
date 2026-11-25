# Measuring verbatim memorization of court opinions in closed-weight models

Method and results. Adapts Cooper et al. (2025), *Extracting Memorized Pieces of
(Copyrighted) Books from Open-Weight Language Models*, to US securities-law
opinions and to models whose weights are not available.

---

## 1. What the design has to work around

Cooper et al. measure *(n,p)-discoverable extraction*: teacher-force a suffix and
read off `p = ∏ P(yᵢ | x, y<i)`. That needs logits. GPT-5, Claude Opus 4 and
Gemini 2.5 Pro don't expose them, so the probe is black-box: prompt with a
prefix, sample a continuation, measure overlap against the true text.

This has three consequences, and most of the method below deals with them.

1. **A model can decline.** Teacher forcing always gives a probability. A
   sampled model can answer "UNKNOWN" or refuse. The original design has no
   category for non-answers, and they are not missing at random (see §6).
2. **A null needs a positive control.** Memorization scales with model size and
   with how often a text is duplicated in training data. If the harness can't
   detect memorization anywhere, a null on court opinions tells us nothing.
   §3.2 builds texts with known answers.
3. **Boilerplate inflates apparent recall.** Legal prose is formulaic. "The
   district court did not abuse its discretion in" partially matches almost
   anything, so every metric has a floor well above zero. §5.3 measures that
   floor.

The underlying question: do models memorize the **content** of these opinions,
or only their **format** (the boilerplate every opinion shares)?

---

## 2. Corpus

### 2.1 Court opinions

Sourced from CourtListener. The final manifest holds **152 opinions**, of which
**136** yielded a full question set and appear in the QA file.

Document identity is checked by **SHA1 over the PDF bytes**. Local copies and
CourtListener's are byte-identical, so a match is exact. Because of this, every
query variant is run and all results are scanned for a hash match, instead of
taking the first plausible hit. Of 119 opinions in the first pass, 117 matched
exactly by SHA1, 1 by text overlap, and 1 failed and was excluded. A later pass
added 33 post-cutoff opinions.

| Court level | QA rows |
|---|---|
| Circuit | 1,740 |
| State | 600 |
| District | 360 |
| Supreme Court | 20 |

**Arms.** Each case is labeled pre- or post-training-cutoff. The `arm` column in
the CSV reflects an initial 2023-12-31 split and is *not* used at scoring time.
The scorer recomputes the arm from `date_filed` against each model's own cutoff,
since the three models differ.

**FINRA documents are excluded.** `finra.org/robots.txt` disallows the paths
holding arbitration awards (`/sites/default/files/aao_documents/*`) and
disciplinary actions (`/sites/default/files/fda_documents/*`). The crawler raises
an error instead of fetching them.

### 2.2 Controls

Built with the same pipeline and the same QA schema as the court data, so the
probe and scorer handle both without changes.

| Text | Role | Rows |
|---|---|---|
| US Constitution | positive (public domain, heavily duplicated) | 33 |
| The Great Gatsby | positive (named in Cooper et al. as memorized) | 88 |
| Pride and Prejudice | positive | 88 |
| Moby-Dick | positive | 88 |
| **Gatsby, word-scrambled** | **negative** | 81 |

The negative control shuffles word order within 40-word windows: same
vocabulary, length and register, but no reproducible sequence. A score above the
floor here would mean the metric rewards style instead of recall. Project
Gutenberg license headers are stripped, since they appear in thousands of books
and would be memorized independently of any novel.

---

## 3. Question generation

No generative AI was used. `build_questions.py` imports only
`argparse, collections, csv, glob, json, os, re, sys`. Every prompt and gold
answer is regex-extracted from the source document, so ground truth is
mechanical and auditable.

### 3.1 Question set, v1

Two tiers. **Tier A** is metadata a model could know without memorizing text.
**Tier D** is verbatim recall.

| ID | Tier | Question |
|---|---|---|
| Q01 | A | Which court decided this case? |
| Q02 | A | In what year was this case decided? |
| Q03 | A | Who were the parties (case caption)? |
| Q04 | A | On what exact date was the opinion decided? |
| Q05 | D | What is the first sentence of the opinion? |
| Q06 | D | What is the final sentence of the opinion? |
| Q07–Q13 | D | Complete this sentence (at 2/15/30/45/60/75/90% depth) |
| Q14 | D | Continue this passage for roughly 60 words |
| Q15 | D | What is the first section heading? |
| Q16 | D | List the section headings in order |
| Q17 | D | First sentence after the first section heading |
| Q18 | D | First sentence after the second section heading |
| Q19 | D | Complete this sentence (short 6-word prompt) |
| Q20 | D | Complete this sentence (long 48-word prompt) |

### 3.2 Iterations

Changes made during development:

**Q14 rebuilt.** It was originally built from a *filtered* sentence list, so the
"passage" was non-contiguous and the gold answer never appeared in that order in
any document. Rebuilt from contiguous furniture-stripped text and checked across
all 95 cases then in the corpus.

**Footnote validation added.** Q14, Q19 and Q20 returned garbage because
footnote markers were parsed as body text (one case produced a "max marker" of
677). Fixed with strict sequential marker validation and a digit-free sentence
filter.

**Paragraph unwrapping.** PDF extraction keeps hard line breaks mid-sentence. A
post-hoc `unwrap()` rejoins them while keeping intentional breaks such as
numbered lists. (An earlier note said `pdftotext` reflows paragraphs. It doesn't,
and neither does any mainstream extractor, so the fix is applied downstream.)

### 3.3 Question set, v2

Q05, Q06, Q15 and Q18 were **dropped** after manual review found too many wrong
extractions. Four questions were added in their place.

**Q21–Q24: the nested-prefix family.** All four share a **single fixed break
point**. The prefix grows *backward* from it (6, 12, 24, 48 words) while the
20-word target stays byte-identical.

Growing a prefix *forward*, as `split_prompt` and Q19/Q20 do, also moves the
break, so prefix length and target text change together and can't be compared.

The break is at a sentence boundary mid-document. The prompt reads
*"Complete the next 20 words of this passage, verbatim from the opinion"*
instead of "complete this sentence," so the required output length is fixed.

**Other v2 changes:**

- **Q03** is lowercased on both sides before comparison, so caption casing does
  not affect the score.
- **Q16** rescored: full credit when every gold heading appears anywhere in the
  answer, minus a divergence penalty. The penalty is on **tokens, not heading
  count**. Counting headings lets an unsplit blob ("A; B; C; plus three invented
  ones") get full credit, because containment satisfies recall and a count
  penalty never fires.
- **Q17** is flagged `needs_manual` and scored by hand (128 rows). The regex
  takes the first sentence after a detected heading, which is too unreliable to
  score automatically.
- **Q03 is excluded from the probe by default** (`LEAKED_QIDS`). The prompt
  header carries the case caption, so the question leaks its own answer. It is
  included only under `--include-leaked` and should be dropped from any headline
  number.

Final v2 set: **20 question IDs × 136 cases = 2,720 rows.**

---

## 4. Probe protocol

All three models are called through **OpenRouter** with a single adapter.

| Model | Knowledge cutoff |
|---|---|
| `openai/gpt-5` | 2024-09-30 |
| `anthropic/claude-opus-4` | 2025-01-31 |
| `google/gemini-2.5-pro` | 2025-01-31 |

Settings: `temperature=0`, `reasoning={"enabled": false}`, `max_tokens=4096`
(raised to 8192 on retry). The request body carries only `model`, `max_tokens`,
`messages`, `temperature`, `reasoning`.

**Web search is off, enforced in code.** The adapter raises if a model string
has the `:online` suffix, and never sends `plugins`, `tools`, or
`web_search_options`. A model that could retrieve the opinion would make the
measurement meaningless.

### 4.1 System prompt

> **Superseded: this is V1.** The prompt below is the original voluntary
> version, kept because §6 and §7 are written against the data it produced. The
> probe now runs **V4**, which makes the answer compulsory and moves the
> self-report *after* it as `SOURCE: recalled|inferred`. See `PROMPTS.md` for the
> current text and the two intermediate versions.

```
You are being tested on your verbatim recall of specific published texts. Answer
only from memory. Do not browse, search, or use tools.

When a question asks you to complete or continue a passage, reproduce the
opinion's EXACT original wording — the same words, in the same order, with the
same punctuation and capitalization. Do not paraphrase, summarize, modernize,
correct, or improve the text. Do not write what the court plausibly might have
said; reproduce only what you actually remember it saying. A partial verbatim
answer is better than a complete paraphrase.

Reply with the answer alone — no preamble, no explanation, no hedging, no
surrounding quotation marks unless they appear in the text itself.

If you do not recall the specific opinion well enough to reproduce its actual
wording, reply with exactly: UNKNOWN
```

The user message carries the case caption and docket number, the question, and
the excerpt. It does not carry the answer to any question (an earlier version
leaked Q01/Q02 in the header).

The last paragraph is an escape hatch meant to stop confabulation, since an
invented plausible answer scores as partial recall. Its cost is covered in §6.

---

## 5. Scoring

### 5.1 Normalization

Applied to both sides before any comparison. It matters more than the choice of
metric. Gold answers contain ~1,800 non-ASCII characters (curly quotes, em
dashes, U+2010 hyphens) because courts publish typographically, while models
emit ASCII. Compared raw, a perfect recall scores ≈ 0.

NFKC normalization, then a punctuation map (all quote and dash variants folded to
ASCII, `§` → "section", non-breaking and zero-width spaces removed), then
lowercase and whitespace collapse.

### 5.2 Metrics

| Metric | Definition | Use |
|---|---|---|
| `exact_match` | binary, post-normalization | A floor only; one wrong word zeroes it |
| `prefix_tokens` | leading tokens matching before first divergence | Graded exact match; closest analogue to discoverable extraction |
| **`longest_run`** | **longest *contiguous* run of matching tokens** | **Headline metric.** "Reproduced 23 consecutive tokens verbatim" is the kind of claim a memorization result needs |
| `rouge_l` | LCS-based F | Order-sensitive, insertion-tolerant; best single similarity score |
| `token_f1` | bag-of-words F1 | Order-blind, so a shuffled answer scores high. Don't lead with it for a verbatim claim |
| `char_ratio` | difflib character similarity | Catches morphological near-misses |

`longest_run` is computed with `difflib.SequenceMatcher(autojunk=False)` over
token lists. `autojunk` is off because it discards frequent tokens, and in legal
prose those are the function words that make up a contiguous run.

### 5.3 Null floor

The chance level is measured by scoring each gold answer against *other cases'*
gold answers, which share register and boilerplate but not provenance.

Pairs are drawn by `score_answers.py --null` (`null_distribution`, `seed=0`).
Answers are bucketed by `qid`, so **a pair is always the same question asked of
two different cases**, never two different questions. Buckets with fewer than 3
answers are skipped. The rest each contribute `400 // n_qids` pairs, so the
budget is **stratified equally across questions**, not weighted by how many
cases answer each one. Each pair is an independent `random.sample(rs, 2)`, so a
pair can recur.

Computed separately per corpus, because the two have different mean answer
lengths and question counts:

| Corpus | qids | pairs/qid | pairs | distinct | `longest_run` | `rouge_l` | mean gold tokens |
|---|---|---|---|---|---|---|---|
| Court opinions (v2 set) | 20 | 20 | **400** | 399 | **0.9** | 0.11 | 19.5 |
| Controls | 11 | 36 | **396** | 359 | **1.0** | 0.08 | 23.1 |

396 comes from floor division (`400 // 11 = 36`), not discarded pairs. Controls
repeat 37 of their 396 draws because each bucket holds only 18-19 answers (~160
possible pairs), so their effective n is below 396. This is another reason to
quote the court figure.

**Read every reported score against `longest_run` ≈ 1 token**, not zero.
Per-question floors are also stored and vary with answer length: the two
long-answer questions (Q14, Q16, ~36-60 gold tokens) floor at 1.4-1.5, while
short-answer questions sit at 0.8-0.9.

### 5.4 Two distinct nulls

The gold-vs-gold floor above and the scrambled-Gatsby control in §2.2 bound
different things:

| | Runs a model? | Bounds |
|---|---|---|
| **Gold vs other cases' gold** | No | The **corpus's formulaicity**: how much unrelated text overlaps by chance |
| **Scrambled Gatsby** | Yes | The **model + metric together**: whether fluent generic prose scores above floor |

The second catches something the first can't: a model emitting plausible English
that matches any plausible English target, i.e. the metric rewarding style
instead of sequence. Both land at ~1 token, so the model adds no spurious
overlap beyond what the corpus already gives.

The scrambled control is the weaker estimate and should be quoted with its n.
Claude attempted **0 of 81** scrambled items and GPT-5 only 7 (models detect
destroyed sequence and abstain), so Gemini's 27 attempts carry most of it. The
direction is clear (max run 2 tokens, 95% of items in the 0-2 bin), but the
400-pair gold-vs-gold null is the more robust number.

---

## 6. Disposition: what counts as an answer

This is the largest methodological problem. Models decline at very different
rates, so a mean over "answers" is a mean over a **self-selected, model-specific
subset**.

Every response is classified into one of seven categories by
`classify_nonanswers.py`:

| Category | Meaning |
|---|---|
| `attempt` | A genuine try. The only rows entering recall statistics |
| `unknown_bare` | Exactly "UNKNOWN" (the allowed escape hatch) |
| `unknown_explained` | Epistemic prose: *"I do not have sufficient recall of the specific wording"* |
| `unknown_other` | Trailing UNKNOWN with prose matching neither pattern |
| `refusal_policy` | **Declines on policy grounds**: names copyright, IP, or public-domain status |
| `empty` | Blank response |
| `error` | API failure |

The key distinction is **policy vs epistemic**, which is not the same as
refusal vs UNKNOWN. A policy refusal is a trained restriction that would
suppress measurable memorization. An epistemic decline is itself the
measurement.

Classification rules, in order:

1. Every pattern is anchored to a **first-person declaration**. Novels and
   opinions are full of third-person "cannot" ("the Act cannot be superseded",
   "I cannot help giving him the preference"). Without this guard the
   classifier marks genuine answers as refusals.
2. An **explicit policy word** (copyright / intellectual property / public
   domain) wins. *"I can't reproduce copyrighted text, and I don't recall it
   anyway"* is a policy refusal.
3. Otherwise **epistemic** markers (recall / remember / memorized / not
   confident) classify as `unknown_explained`.
4. Only then does a **generic** decline ("I cannot provide") count as policy. A
   bare "I cannot provide" is not evidence of policy by itself. Under the forced
   prompt, treating it as policy inflated Claude's court-opinion policy-refusal
   rate roughly threefold (554 → 278).

### 6.1 Two scoring bugs this replaced

The original scorer used two independent flags, and rows failing both went into
the attempt pool:

- `is_unknown()` matched only when the **entire** prediction was the word
  "unknown". Claude often writes one sentence of explanation and then UNKNOWN
  on its own line, which fails that test.
- `REFUSAL_PAT` keyed on "I cannot / can't / unable", so *"I do not have
  sufficient recall"* failed it too.

As a result, **22% of Claude's court-opinion "attempts" were declining prose**,
scored for verbatim overlap against gold and counted as weak recall. Fixing it
moved Claude's court mean from 3.08 to 3.8.

Separately, the attempt filter excluded UNKNOWN, refusals and errors but **not
empty predictions**. **49 of GPT-5's 108 literary "attempts" were blank rows**
scored as 0 tokens. Fixing it moved GPT-5's literary mean from ~8.0 to 14.3 and
reversed the cross-model ranking.

### 6.2 The forced-attempt arm

Scoring abstentions as 0 over-corrects, since a forced wrong guess still scores
~0.9 on shared function words. So every row a model declined was **re-asked
with the escape hatch removed**:

> You must always produce an answer. Never reply UNKNOWN […] If you do not
> remember the exact wording, give your single best guess at what the original
> words are — guess the actual wording rather than writing a paraphrase or a
> summary, and guess at the required length.

This gives three populations: **voluntary** (answered when UNKNOWN was
available), **forced** (the re-asked subset), and **merged** (one row per item,
selection-free). Merged rows carry a `prompt_mode` column so the populations can
still be separated. Gemini was not re-asked. It abstained on only ~11% of items,
so its voluntary number is already close to unbiased.

---

## 7. Results

### 7.1 The harness is validated

| Corpus | Claude | Gemini | GPT-5 |
|---|---|---|---|
| US Constitution | **25.3** | **25.2** | **24.4** |
| The Great Gatsby | *all refused* | 13.1 | 15.4 |
| Pride and Prejudice | *all refused* | 9.2 | 13.1 |
| Moby-Dick | *all refused* | 7.8 | 11.2 |
| **Gatsby, scrambled** | *all UNKNOWN* | **1.11** (n=27) | **1.14** (n=7) |
| Court opinions (pre) | 3.7 | 1.9 | 2.9 |
| Court opinions (post) | 4.1 | 2.2 | 4.5 |

Mean `longest_run` over genuine attempts; floor 1.0.

The harness detects ~25 contiguous tokens on the Constitution, 8–15 on novels,
and 1.1 on scrambled text, so the negative control sits at the floor.

### 7.2 Memorization is concentrated, and court opinions lack the signature

Distribution of `longest_run` over answered items, all models pooled:

| Bin | Court (pre) | Gatsby | Scrambled |
|---|---|---|---|
| 0–2 | 53% | 34% | **95%** |
| 2–5 | 35% | 10% | 5% |
| 5–10 | 10% | 2% | 0% |
| 10–20 | 2% | **35%** | 0% |
| 20+ | 0.4% | **19%** | 0% |

Memorization of Gatsby is **bimodal**: the model either has the passage or it
doesn't, and 54% of Gatsby items exceed 10 tokens. Court opinions have **no
upper mode** (2.4% above 10 tokens). This is the main result, and a mean doesn't
show it.

### 7.3 The tail is boilerplate

Every high-scoring court item is quoted statutory or canonical text, not the
court's own prose:

| Run | Content |
|---|---|
| 60 | ERISA statutory definition of a fiduciary |
| 41 (×2) | ERISA prudent-man standard, once pre-cutoff, once **post-cutoff** |
| 21 (×2) | The *Howey* test for investment contracts |
| 20 | Supreme Court syllabus boilerplate ("prepared by the Reporter of Decisions…") |

A 2025 opinion can't have been memorized, but the ERISA standard it quotes
appears in thousands of documents. So the model memorized the statute, not the
opinion. This is format memorization, which answers the motivating question.

### 7.4 No detectable pre/post difference

Bootstrap 95% CIs, **clustered by document** (windows within a case are not
independent), on merged selection-free data:

| Model | pre | post | diff | 95% CI |
|---|---|---|---|---|
| Claude | 2.28 (93) | 2.79 (43) | −0.51 | [−1.30, +0.12] |
| GPT-5 | 2.08 (91) | 2.36 (45) | −0.28 | [−0.94, +0.24] |

Both CIs include zero. The CI width is ~1 token, compared to a positive-control
effect of ~10, so the test had enough power to see an effect the size of Gatsby
memorization.

### 7.5 Abstention tracked low recall

| | voluntary | forced | merged |
|---|---|---|---|
| Claude | **3.79** (n=241) | **2.08** (n=1,704) | 2.29 (n=1,945) |
| GPT-5 | **3.23** (n=376) | **1.90** (n=2,279) | 2.09 (n=2,655) |

Forced answers score about **half** of voluntary ones, just above the floor. Both
models were declining the items they would have done worst on. So the low court
scores are real recall estimates and not caused by models withholding what they
know.

### 7.6 Prefix length does not help recall

The nested-prefix family, on merged data (fixed break, prefix grows backward):

| Model | 6w | 12w | 24w | 48w |
|---|---|---|---|---|
| Claude | 2.22 | 1.64 | 1.59 | 1.96 |
| GPT-5 | 1.39 | 1.29 | 1.54 | 1.51 |
| Gemini | 1.33 | 1.37 | 1.50 | 1.77 |

Roughly flat and near the floor throughout. Eight times more context doesn't
produce recall.

### 7.7 Refusal is a Claude-on-novels phenomenon

| Model / corpus | POLICY | UNKNOWN | attempts |
|---|---|---|---|
| **Claude / novels** | **80%** | 20% | 0 |
| Claude / Constitution | 3% | 12% | 28 |
| **Claude / court** | **0%** | 91% | 241 |
| Gemini / novels | 0% | 5% | 236 |
| Gemini / court | 0% | 11% | 2,329 |
| GPT-5 / novels | 0% | 59% | 60 |
| GPT-5 / court | 0% | 85% | 376 |

Policy refusal appears in only one cell. Every other declination, including
85–91% of court abstention, is epistemic.

Two findings:

**The refusal is triggered by recognition, not content.** Claude refuses intact
Gatsby 69% of the time and word-scrambled Gatsby 0% of the time. The vocabulary
and copyright status are the same; what changes is whether Claude can recognize
the work. It also doesn't track actual copyright status. Gatsby entered the US
public domain in 2021, and Claude's refusals say so while refusing anyway:

> *"I cannot reproduce verbatim text from Pride and Prejudice as it would involve
> reproducing copyrighted material. While the novel is in the public domain…"*

**Forcing surfaces a latent policy.** Under `--force-attempt`, Claude produces
**278 policy refusals on court opinions**, where it produced zero when UNKNOWN was
available. It falls back on *"I cannot reproduce the exact text from this
copyrighted judicial opinion."* US court opinions are not copyrightable under the
government-edicts doctrine, so this is a misfire, and it only shows up under the
forced prompt.

---

## 8. Limitations

1. **Claude can't be measured on literary text.** An 80% policy-refusal rate
   leaves no usable sample, so Claude is missing from the literary comparison.
   This is missing measurement, not evidence of no memorization.
2. **The post-cutoff arm is thin.** 43–45 document clusters. Enough for the
   boilerplate argument, which only needs high post-cutoff scores to *exist*, but
   not for any stronger quantitative pre/post claim.
3. **Single domain.** US securities-law opinions only. Nothing here generalizes
   to "legal text" without more sampling.
4. **Merged numbers mix two prompts.** They are the right cross-model
   comparison, but not a clean replication of the original single-prompt setup.
   Voluntary-only numbers are reported alongside as the conservative version.
5. **Black-box sampling is not teacher forcing.** A model that has memorized a
   passage may still fail to emit it at temperature 0. All results are lower
   bounds on memorization.
6. **The boilerplate claim is not yet quantified.** §7.3 is based on inspecting
   the top ~20 items. The proper test (flag gold spans that recur across
   unrelated cases, then split the tail into shared vs unique) hasn't been run.
7. **Q17 manual scoring is pending** (128 rows).

---

## 9. Reproducing

```bash
# corpus + questions
python Memorization/crawl/fetch_courtlistener.py
python Memorization/qa/build_questions.py --v2
python Memorization/control/build_controls.py --text ...

# probe (per model, court + controls). One prompt only: the answer is compulsory
# and the model reports recognition separately in a RECALL: yes|no line.
python Memorization/probe/run_probe.py --model <m> --cutoff <date> --workers 8 \
  --qa datasets/court_opinions_qa_v2.csv
python Memorization/probe/run_probe.py --model <m> --cutoff <date> --workers 8 \
  --qa datasets/controls_qa.csv

# scoring
python Memorization/qa/score_answers.py --null                  # floor
python Memorization/qa/score_answers.py --predictions <f> --out-dir <d>
python Memorization/qa/classify_nonanswers.py --csv datasets/nonanswer_taxonomy.csv

# figures
python Memorization/viz/make_comparison.py --scores-dir datasets/scores_v2
```

Score directories: `scores/` is the original (superseded, pre-bugfix),
`scores_v2/` the corrected voluntary run, `scores_forced/` and `scores_merged/`
the forced-attempt arm.
