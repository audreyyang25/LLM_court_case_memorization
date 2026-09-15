"""stats.py -- the summaries behind every claim in the paper.

The central quantity is `longest_run`: the longest contiguous run of tokens the
model reproduced from the true continuation. It is read against a MEASURED floor
(see load.load_floor), never against zero.

Two things this module is careful about, both of which have silently produced
wrong numbers in earlier versions of this pipeline:

  * Means are taken over ATTEMPTS only, and the attempt predicate is explicit.
    Models decline at wildly different rates, so a mean over "answers" is a mean
    over a self-selected, model-specific subset.

  * The distribution matters more than the mean. Real memorization is bimodal --
    the model either has the passage or it does not -- so a corpus that is truly
    memorized shows an upper mode, and a mean alone cannot distinguish "half the
    items at 15 tokens" from "every item at 7.5".
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict

from src.load import CORPUS_ORDER


def _mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else float("nan")


def bootstrap_ci(xs, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for a mean. Seeded, so it reproduces exactly."""
    import random
    xs = list(xs)
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    means = []
    for _ in range(n_boot):
        means.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return (lo, hi)


def by_corpus_model(rows, metric="longest_run"):
    """(corpus, model) -> {n_rows, n_attempts, attempt_rate, mean, ci, frac_over_10}."""
    g = defaultdict(list)
    tot = Counter()
    for r in rows:
        tot[(r["work"], r["model"])] += 1
        if r["is_attempt"]:
            g[(r["work"], r["model"])].append(r[metric])

    out = {}
    for key, n_all in tot.items():
        vals = g.get(key, [])
        out[key] = {
            "n_rows": n_all,
            "n_attempts": len(vals),
            "attempt_rate": len(vals) / n_all if n_all else float("nan"),
            "mean": _mean(vals),
            "ci": bootstrap_ci(vals),
            "frac_over_10": (sum(1 for v in vals if v > 10) / len(vals)) if vals else float("nan"),
        }
    return out


def corpus_summary(rows, metric="longest_run"):
    """Pooled over models, per corpus -- the headline table."""
    g = defaultdict(list)
    tot = Counter()
    for r in rows:
        tot[r["work"]] += 1
        if r["is_attempt"]:
            g[r["work"]].append(r[metric])

    out = {}
    for w in CORPUS_ORDER:
        vals = g.get(w, [])
        if not vals:
            continue
        out[w] = {
            "n_rows": tot[w],
            "n_attempts": len(vals),
            "attempt_rate": len(vals) / tot[w],
            "mean": _mean(vals),
            "ci": bootstrap_ci(vals),
            "frac_over_10": sum(1 for v in vals if v > 10) / len(vals),
            "frac_over_20": sum(1 for v in vals if v > 20) / len(vals),
            "median": statistics.median(vals),
        }
    return out


def histogram(rows, work, bins=(0, 2, 4, 6, 8, 10, 15, 20, 30, 1e9), metric="longest_run"):
    """Distribution of contiguous-token runs for one corpus, over attempts."""
    vals = [r[metric] for r in rows if r["work"] == work and r["is_attempt"]]
    counts = [0] * (len(bins) - 1)
    for v in vals:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                break
    return counts, len(vals)


def pre_post(rows, metric="longest_run"):
    """Pre- vs post-cutoff court opinions, per model.

    A model cannot have memorized an opinion filed after its training cutoff, so
    the post-cutoff arm is what verbatim text it produces with zero knowledge of
    the case. If pre-cutoff recall does not exceed it, nothing was memorized that
    was not already reachable from the statutes and boilerplate quoted inside.
    """
    g = defaultdict(list)
    for r in rows:
        if r["work"] != "court_opinion" or not r["is_attempt"]:
            continue
        arm = "pre" if r["arm"] == "pre_cutoff" else "post"
        g[(r["model"], arm)].append(r[metric])

    models = sorted({m for m, _ in g})
    out = {}
    for m in models:
        pre, post = g.get((m, "pre"), []), g.get((m, "post"), [])
        if not pre or not post:
            continue
        diff = _mean(pre) - _mean(post)
        # CI on the difference, resampling each arm independently.
        import random
        rng = random.Random(1)
        draws = []
        for _ in range(2000):
            a = sum(pre[rng.randrange(len(pre))] for _ in range(len(pre))) / len(pre)
            b = sum(post[rng.randrange(len(post))] for _ in range(len(post))) / len(post)
            draws.append(a - b)
        draws.sort()
        out[m] = {
            "pre_mean": _mean(pre), "post_mean": _mean(post),
            "n_pre": len(pre), "n_post": len(post),
            "diff": diff, "diff_ci": (draws[50], draws[-51]),
        }
    return out


def disposition(rows):
    """Per-model response disposition on each corpus.

    The distinction that matters is POLICY versus EPISTEMIC declination, and it
    is not the same as refusal versus unknown. A policy declination is a trained
    restriction that would suppress measurable memorization; an epistemic
    declination IS the measurement. Conflating them makes a well-behaved model
    look unmemorized.
    """
    # Read the taxonomy column directly. The `refusal` / `unknown` / `errored`
    # columns are derived from it, and a cascade over those booleans silently
    # files EMPTY responses under "attempt" -- see load.load_blackbox.
    #
    # "empty" is then split by finish reason, because it is not one thing. A
    # blocked response and a response truncated by the token budget both arrive
    # as an empty string, and only the first is a declination. Counting them
    # together understates policy refusal roughly threefold and does so
    # unevenly across corpora, which would misstate the one claim in the paper
    # that rests on refusal rates.
    label = {
        "attempt": "attempt",
        "refusal_policy": "policy_refusal",
        "unknown_explained": "epistemic_unknown",
    }
    out = defaultdict(Counter)
    for r in rows:
        key = (r["work"], r["model"])
        if r["errored"] == "1":
            k = "error"
        elif r["nonanswer"] == "empty":
            fin = r.get("finish", "")
            if fin == "content_filter" or r.get("provider_refused") == "1":
                k = "policy_refusal"        # blocked, delivered as an empty body
            elif fin == "length":
                k = "truncated"             # budget exhausted before the answer
            elif fin == "error":
                k = "error"
            else:
                k = "empty_other"
        else:
            k = label.get(r["nonanswer"], f"other:{r['nonanswer']}")
        out[key][k] += 1
    return {k: dict(v) for k, v in out.items()}


def logprob_summary(lp_rows):
    """(corpus, model, variant) -> mean per-token geometric-mean probability.

    This is the teacher-forced metric the black-box probe cannot access. Base
    models are the headline condition: instruction tuning suppresses verbatim
    reproduction, and the question is whether the model CARRIES the text.
    """
    g = defaultdict(list)
    for r in lp_rows:
        g[(r["work"], r["model"], r["variant"])].append(r["p_geomean"])
    return {k: {"n": len(v), "mean_p_geomean": _mean(v),
                "ci": bootstrap_ci(v), "median": statistics.median(v)}
            for k, v in g.items()}


def hotspot_summary(hs_rows):
    """Adjudication of the 399 highest-probability spans."""
    verdicts = Counter(r["verdict"] for r in hs_rows)
    cats = Counter(r["category_near"] for r in hs_rows)
    cands = [r for r in hs_rows if r["verdict"] == "MEMORISATION_CANDIDATE"]
    return {
        "n_spans": len(hs_rows),
        "verdicts": dict(verdicts),
        "n_candidates": len(cands),
        "candidate_rate": len(cands) / len(hs_rows) if hs_rows else float("nan"),
        "categories_near": dict(cats),
        "candidates": [{"case_id": r["case_id"], "arm": r["arm"],
                        "category": r["category_near"],
                        "n_tokens": r["n_tokens"],
                        "text": r["text"][:220]} for r in cands],
        # Post-cutoff opinions cannot have been memorized, so a high-probability
        # span from one is proof that the metric's tail is reachable without
        # memorization at all. The arm label is "post_cutoff_control".
        "post_cutoff_spans": sum(1 for r in hs_rows
                                 if r["arm"].startswith("post_cutoff")),
        "candidates_post_cutoff": sum(1 for r in hs_rows
                                      if r["verdict"] == "MEMORISATION_CANDIDATE"
                                      and r["arm"].startswith("post_cutoff")),
    }
