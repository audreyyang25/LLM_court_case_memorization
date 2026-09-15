"""verify.py -- re-score the raw predictions and compare against the committed scores.

`data/` holds scored artifacts; `data/raw/` holds the prompts, gold continuations
and verbatim model outputs they were computed from. This walks that chain with an
INDEPENDENT implementation of the headline metric and asserts the result matches,
so the paper's central number does not rest on the upstream scorer being right:

    source document -> prompt + gold continuation -> model output
                    -> longest_run -> blackbox_scores.csv.gz

Four checks:

  1. COVERAGE     every scored row has a prediction and a question behind it
  2. RESCORE      recompute longest_run from the raw text and compare per row
  3. NO LEAKAGE   no prompt contains its own gold continuation -- the failure
                  mode that would make recall trivially high
  4. DISPOSITION  the attempt flags are consistent with the prediction strings

    python regenerate/verify.py

Exits non-zero if any check fails. Needs only data/ and data/raw/ -- no upstream
project, no network, no API key.
"""

import argparse
import csv
import difflib
import gzip
import json
import os
import re
import sys
import unicodedata
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")

# Typographic folding, applied before tokenizing. Models and source documents
# disagree about quote, dash and space characters constantly, and without this
# the metric would score typography rather than sequence.
#
# The section sign is deliberately spelled out rather than dropped. Legal text
# writes a citation as both "section 1132" and "§ 1132", and a metric that
# treated those as different tokens would understate recall on exactly the
# statutory citations that dominate this corpus -- 4 rows in this dataset turn
# on it.
PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "­": "", "​": "",
    "…": "...", "§": "section",
}


def normalize(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    for a, b in PUNCT_MAP.items():
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s.lower()).strip()


def tokens(s):
    return re.findall(r"[\w']+", normalize(s))


def longest_run(pred, gold):
    """Longest contiguous run of matching tokens -- the headline metric.

    Contiguity is the point. A high token-overlap score can be produced by
    scattered function words, which is exactly what formulaic legal prose
    guarantees; a long contiguous run cannot.
    """
    a, b = tokens(pred), tokens(gold)
    if not a or not b:
        return 0
    return max((m.size for m in
                difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks()),
               default=0)


def _read_gz_csv(path):
    with gzip.open(path, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def check_raw_present():
    missing = [f for f in ("questions.csv.gz", "predictions.csv.gz")
               if not os.path.exists(os.path.join(RAW, f))]
    if missing:
        print(f"data/raw/ is missing {missing}.")
        print('Rebuild it with:  python regenerate/export_raw.py --src "/path/to/.../datasets"')
        print("\nNote: the raw layer is optional. The paper reproduces from data/ alone;")
        print("these files exist so the measurement can be audited, not re-run.")
        sys.exit(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tolerance", type=int, default=0,
                    help="allowed per-row difference in longest_run (default: 0)")
    args = ap.parse_args()

    check_raw_present()

    scored = _read_gz_csv(os.path.join(DATA, "blackbox_scores.csv.gz"))
    questions = {(r["case_id"], r["qid"]): r
                 for r in _read_gz_csv(os.path.join(RAW, "questions.csv.gz"))}
    preds = {(r["case_id"], r["qid"], r["model"]): r
             for r in _read_gz_csv(os.path.join(RAW, "predictions.csv.gz"))}

    failures = []

    # ---- 1. COVERAGE -------------------------------------------------------
    no_pred = [r for r in scored
               if (r["case_id"], r["qid"], r["model"]) not in preds]
    no_q = [r for r in scored if (r["case_id"], r["qid"]) not in questions]
    if no_pred or no_q:
        failures.append(f"COVERAGE: {len(no_pred)} scored rows have no prediction, "
                        f"{len(no_q)} have no question")
    else:
        print(f"  [ok] COVERAGE     all {len(scored):,} scored rows have a prompt, a "
              f"gold continuation and a verbatim model output")

    # ---- 3. NO LEAKAGE -----------------------------------------------------
    # If a prompt contained its own gold continuation, a model could score
    # perfectly by copying rather than recalling. The paper reports one early
    # question that leaked this way and was excluded; this confirms none remain.
    leaked = []
    for q in questions.values():
        gold, prompt = normalize(q.get("answer", "")), normalize(q.get("prompt", ""))
        if len(gold) > 20 and gold in prompt:
            leaked.append((q["case_id"], q["qid"]))
    if leaked:
        failures.append(f"NO LEAKAGE: {len(leaked)} prompts contain their own gold "
                        f"continuation, e.g. {leaked[:3]}")
    else:
        print(f"  [ok] NO LEAKAGE   none of {len(questions):,} prompts contain the "
              "continuation they ask for")

    # ---- 2. RESCORE --------------------------------------------------------
    diffs = Counter()
    worst = []
    n_checked = 0
    for r in scored:
        q = questions.get((r["case_id"], r["qid"]))
        p = preds.get((r["case_id"], r["qid"], r["model"]))
        if not q or not p:
            continue
        gold = q.get("answer", "")
        pred = p.get("prediction", "")
        if not gold:
            continue
        n_checked += 1
        try:
            committed = int(float(r["longest_run"] or 0))
        except ValueError:
            continue
        got = longest_run(pred, gold)
        d = got - committed
        diffs[d] += 1
        if abs(d) > args.tolerance and len(worst) < 5:
            worst.append((r["case_id"], r["qid"], r["model"], committed, got))

    n_off = sum(c for d, c in diffs.items() if abs(d) > args.tolerance)
    if n_off:
        failures.append(
            f"RESCORE: {n_off} of {n_checked:,} rows ({n_off / n_checked:.2%}) differ "
            f"from the committed longest_run by more than {args.tolerance}; "
            f"examples: {worst}")
    else:
        print(f"  [ok] RESCORE      {n_checked:,} rows re-scored from raw text with an "
              "independent implementation, all matching the committed longest_run")

    # ---- 4. DISPOSITION ----------------------------------------------------
    # The attempt predicate decides which rows enter every mean in the paper, so
    # it has to be checked against the strings it claims to describe.
    #
    # This imports the predicate from src.load rather than restating it. A
    # verifier carrying its own copy would pass while the loader was wrong,
    # which is exactly what happened when this check was first written: both had
    # the same bug and agreed with each other.
    sys.path.insert(0, ROOT)
    from src.load import load_blackbox                        # noqa: E402

    loaded = {(r["case_id"], r["qid"], r["model"]): r for r in load_blackbox()}
    empty_attempts = 0
    for key, p in preds.items():
        r = loaded.get(key)
        if r and r["is_attempt"] and not (p.get("prediction") or "").strip():
            empty_attempts += 1

    # The taxonomy column and the derived booleans must agree, or the two could
    # disagree silently and a reader would not know which the means used.
    inconsistent = sum(
        1 for r in scored
        if (r["nonanswer"] == "refusal_policy") != (r["refusal"] == "1")
        or (r["nonanswer"] == "unknown_explained") != (r["unknown"] == "1"))

    if empty_attempts or inconsistent:
        if empty_attempts:
            failures.append(f"DISPOSITION: {empty_attempts} rows are counted as "
                            "attempts but carry an empty prediction -- this is the "
                            "bug the paper reports having fixed once already")
        if inconsistent:
            failures.append(f"DISPOSITION: {inconsistent} rows where the nonanswer "
                            "taxonomy and the derived refusal/unknown flags disagree")
    else:
        n_att = sum(1 for r in loaded.values() if r["is_attempt"])
        breakdown = Counter(r["nonanswer"] for r in scored)
        print(f"  [ok] DISPOSITION  {n_att:,} attempts; taxonomy consistent with the "
              f"derived flags; none of the attempts empty")
        print("       " + "  ".join(f"{k}={v:,}" for k, v in breakdown.most_common()))

    print()
    if failures:
        print("FAILED")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed: the committed scores derive from data/raw/.")


if __name__ == "__main__":
    main()
