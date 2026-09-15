"""load.py -- read the committed artifacts.

Four files, all small, all committed, so every number and figure reproduces
offline with no API key, no GPU and no rented box:

  data/blackbox_scores.csv.gz      per-item sampled-probe scores, 7 models
  data/logprobs.csv.gz             teacher-forced logprobs, 4 Llama variants
  data/hotspots_classified.csv.gz  the 399 highest-probability spans, adjudicated
  data/null_floor.json             the measured chance level

No pandas: these are small enough for the standard library, and one fewer
dependency is one fewer version to pin.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# The corpora, in the order they appear in every table and figure: a heavily
# duplicated public-domain legal text, a novel the literature reports as
# memorized, that same novel with word order destroyed, and the target.
CORPUS_ORDER = ["constitution", "gatsby", "gatsby_shuf", "court_opinion"]
CORPUS_LABEL = {
    "constitution": "US Constitution",
    "gatsby": "The Great Gatsby",
    "gatsby_shuf": "Gatsby, scrambled",
    "court_opinion": "Court opinions",
}
CORPUS_ROLE = {
    "constitution": "positive control",
    "gatsby": "positive control",
    "gatsby_shuf": "negative control",
    "court_opinion": "target",
}


def _read_gz_csv(path: Path):
    with gzip.open(path, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_blackbox():
    """Per-item scores from the sampled probe. One row per (case, question, model).

    `is_attempt` marks rows that actually produced a scoreable answer. Getting
    this predicate right is load-bearing rather than cosmetic: a mean taken over
    "answers" is a mean over a model-specific, self-selected subset, and two
    earlier versions of it produced wrong numbers silently. An unknown-detector
    that fired only when the entire response was the word "unknown" missed a
    model that writes one explanatory sentence first, scoring 22% of its
    declines as weak recall; and an attempt filter that excluded refusals but not
    EMPTY responses scored blank rows as zero tokens, which moved one model's
    literary mean from ~8 to ~14 and reversed the cross-model ranking.
    """
    rows = _read_gz_csv(DATA / "blackbox_scores.csv.gz")
    for r in rows:
        r["longest_run"] = _num(r["longest_run"], 0.0)
        r["rouge_l"] = _num(r["rouge_l"], 0.0)
        r["token_f1"] = _num(r["token_f1"], 0.0)
        r["gold_tokens"] = _num(r["gold_tokens"], 0.0)
        r["exact_match"] = _num(r["exact_match"], 0.0)
        # `nonanswer` is the authoritative disposition taxonomy, not a boolean:
        # attempt / empty / refusal_policy / unknown_explained. The `refusal`,
        # `unknown` and `errored` columns are strictly derived from it, so
        # testing this one column is both necessary and sufficient. Reading it
        # as a 0/1 flag silently readmits the 549 EMPTY responses as attempts
        # scoring zero tokens, which is the exact bug described above -- caught
        # here by regenerate/verify.py after being reintroduced once.
        r["is_attempt"] = r["nonanswer"] == "attempt"
    return rows


def load_logprobs(trunc_n="25"):
    """Teacher-forced scores: p = prod_i P(y_i | x, y_<i).

    Filtered to one truncation length by default. `logp_sum` is not comparable
    across different suffix lengths, and these suffixes run 19-33 words, so a
    fixed n is what makes the numbers mean anything. `p_geomean` is the
    per-token geometric mean, which IS comparable across lengths and is what the
    figures use.
    """
    rows = _read_gz_csv(DATA / "logprobs.csv.gz")
    out = []
    for r in rows:
        if trunc_n is not None and r["trunc_n"] != trunc_n:
            continue
        r["p_geomean"] = _num(r["p_geomean"])
        r["logp_per_token"] = _num(r["logp_per_token"])
        r["logp_sum"] = _num(r["logp_sum"])
        if r["p_geomean"] is None:
            continue
        out.append(r)
    return out


def load_hotspots():
    """The 399 highest-probability spans, each adjudicated by hand.

    `verdict` is the adjudication; MEMORISATION_CANDIDATE is what survives after
    in-document repetition, post-cutoff-predictable text, quoted material and
    recurring furniture are ruled out.
    """
    return _read_gz_csv(DATA / "hotspots_classified.csv.gz")


def load_floor():
    """The measured chance level, in contiguous tokens.

    Each gold answer scored against OTHER cases' gold answers: same register,
    same boilerplate, no shared provenance. Chance is measured rather than
    assumed to be zero, because legal prose is formulaic enough that unrelated
    documents overlap by construction.
    """
    return json.loads((DATA / "null_floor.json").read_text())
