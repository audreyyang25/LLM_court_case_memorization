"""load.py: read the committed data files.

  data/blackbox_scores.csv.gz      per-item scores from the sampled probe, 7 models
  data/logprobs.csv.gz             teacher-forced logprobs, 4 Llama variants
  data/hotspots_classified.csv.gz  the 399 highest-probability spans, hand-labeled
  data/null_floor.json             the measured chance level

All small enough for the standard library, so no pandas.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# Corpus order used in all tables and figures: the Constitution, a novel known
# to be memorized, the same novel with words shuffled, and the court opinions.
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
    """Per-item scores from the sampled probe, one row per (case, question, model).

    `is_attempt` marks rows with a scoreable answer. Means are only taken over
    attempts. Two earlier versions of this filter were wrong: one missed
    "unknown" answers that had an explanation sentence before them, and one
    counted empty responses as 0-token attempts (this moved one model's
    literary mean from ~8 to ~14).
    """
    rows = _read_gz_csv(DATA / "blackbox_scores.csv.gz")
    for r in rows:
        r["longest_run"] = _num(r["longest_run"], 0.0)
        r["rouge_l"] = _num(r["rouge_l"], 0.0)
        r["token_f1"] = _num(r["token_f1"], 0.0)
        r["gold_tokens"] = _num(r["gold_tokens"], 0.0)
        r["exact_match"] = _num(r["exact_match"], 0.0)
        # `nonanswer` is a category (attempt / empty / refusal_policy /
        # unknown_explained), not a 0/1 flag. The refusal/unknown/errored
        # columns are derived from it. Treating it as a flag lets the 549 empty
        # responses back in as attempts; verify.py checks for this.
        r["is_attempt"] = r["nonanswer"] == "attempt"
    return rows


def load_logprobs(trunc_n="25"):
    """Teacher-forced scores: p = prod_i P(y_i | x, y_<i).

    Filtered to one truncation length by default, since `logp_sum` depends on
    suffix length (suffixes are 19-33 words). `p_geomean` is the per-token
    geometric mean, which is comparable across lengths; the figures use it.
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
    """The 399 highest-probability spans, each labeled by hand.

    `verdict` is the label. MEMORISATION_CANDIDATE is what is left after
    excluding in-document repetition, text predictable after the cutoff,
    quoted material and recurring boilerplate.
    """
    return _read_gz_csv(DATA / "hotspots_classified.csv.gz")


def load_floor():
    """Chance level in contiguous tokens.

    Each gold answer is scored against other cases' gold answers. Legal text
    is formulaic, so unrelated documents overlap somewhat and chance is not zero.
    """
    return json.loads((DATA / "null_floor.json").read_text())
