"""export_raw.py: export the raw prompts and outputs from a completed upstream run.

`export.py` writes the scored files the analysis uses. This writes the raw data
behind them so the scoring can be checked:

    data/raw/questions.csv.gz     every prompt and gold continuation, for court
                                  opinions and controls alike
    data/raw/predictions.csv.gz   what each model actually produced, verbatim

With `verify.py` this covers:

    source document -> prompt + gold continuation -> model output
                    -> longest_run -> blackbox_scores.csv.gz

`verify.py` re-scores the raw predictions with a separate implementation of
the metric and compares against the committed scores.

    python regenerate/export_raw.py --src "/path/to/.../datasets"

About 2 MB gzipped. `src/run_all.py` doesn't need it; it's for checking the scores.

Redistribution: court opinions and the Constitution are US government works and
not copyrighted; The Great Gatsby entered the US public domain in 2021.
"""

import argparse
import csv
import glob
import gzip
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

Q_KEEP = ["corpus", "work", "case_id", "qid", "tier", "arm", "date_filed",
          "caption", "question", "prompt", "prompt_words", "answer"]

P_KEEP = ["corpus", "work", "case_id", "qid", "arm", "tier", "model",
          "model_cutoff", "served_by", "recall", "prediction", "refused",
          "finish", "error", "out_tokens"]


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="upstream datasets directory (holds predictions/, *_qa*.csv)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "data", "raw"),
                    help="where to write (default: ../data/raw)")
    args = ap.parse_args()
    for probe in ("predictions", "court_opinions_qa_v2.csv", "controls_qa.csv"):
        if not os.path.exists(os.path.join(args.src, probe)):
            sys.exit(f"error: {args.src} has no {probe!r} -- "
                     "is --src the upstream datasets directory?")
    os.makedirs(args.out, exist_ok=True)
    return args


def _work_of(corpus, case_id):
    """Corpus label. Court rows collapse to one label; controls split by text."""
    return "court_opinion" if corpus == "court_opinions" else case_id.split("_ch")[0]


def export_questions(args):
    """Prompts and gold continuations.

    Used to check that prompts don't contain their answers, and that the
    nested-prefix questions keep the same target while the prefix grows
    backward from a fixed break point.
    """
    path = os.path.join(args.out, "questions.csv.gz")
    n = 0
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=Q_KEEP)
        w.writeheader()
        for name, corpus in (("court_opinions_qa_v2.csv", "court_opinions"),
                             ("controls_qa.csv", "controls")):
            with open(os.path.join(args.src, name)) as fin:
                for x in csv.DictReader(fin):
                    r = {k: x.get(k, "") for k in Q_KEEP}
                    r["corpus"] = corpus
                    r["work"] = _work_of(corpus, x["case_id"])
                    w.writerow(r)
                    n += 1
    print(f"questions     {n:6d} rows  {os.path.getsize(path):,} bytes")


def export_predictions(args):
    """Raw model outputs, as returned.

    The attempt/refusal flags in the scored file are derived from these, so
    including them lets the flags be checked.
    """
    path = os.path.join(args.out, "predictions.csv.gz")
    n = 0
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=P_KEEP)
        w.writeheader()
        for p in sorted(glob.glob(os.path.join(args.src, "predictions", "*.csv"))):
            b = os.path.basename(p)
            corpus = "controls" if b.startswith("CONTROL__") else "court_opinions"
            with open(p) as fin:
                for x in csv.DictReader(fin):
                    r = {k: x.get(k, "") for k in P_KEEP}
                    r["corpus"] = corpus
                    r["work"] = _work_of(corpus, x["case_id"])
                    w.writerow(r)
                    n += 1
    print(f"predictions   {n:6d} rows  {os.path.getsize(path):,} bytes")


def main():
    args = parse_args()
    export_questions(args)
    export_predictions(args)
    print(f"\nwrote to {args.out}")
    print("run `python regenerate/verify.py` to re-score these independently "
          "and compare against the committed scores")


if __name__ == "__main__":
    main()
