"""export.py: rebuild ../data/ from a completed upstream run.

Only this file needs the upstream project. The analysis in ../src/ runs from
the committed files; this is for rebuilding them or pointing at a new run.

    python regenerate/export.py --src /path/to/datasets

The upstream project can be downloaded here: https://github.com/audreyyang25/FinAITrainingData.git.
--src must be the upstream `datasets` directory (the one holding scores/,
logprobs/ and controls_qa.csv). In the original project that is
"Data Collection and Training Material Generation/datasets".
"""
import argparse, csv, gzip, glob, os, json, collections
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", required=True,
                help="upstream datasets directory (holds scores/, logprobs/, controls_qa.csv)")
ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "data"),
                help="where to write the artifacts (default: ../data)")
args = ap.parse_args()

SRC, OUT = args.src, args.out
for probe in ("scores", "logprobs", "controls_qa.csv"):
    if not os.path.exists(os.path.join(SRC, probe)):
        sys.exit(f"error: {SRC} has no {probe!r} -- is --src the upstream datasets directory?")
os.makedirs(OUT, exist_ok=True)

# ---------- 1. black-box per-item scores ----------
KEEP = ["corpus", "work", "case_id", "qid", "tier", "arm", "model", "model_cutoff",
        "longest_run", "prefix_tokens", "rouge_l", "token_f1", "exact_match",
        "gold_tokens", "pred_tokens", "refusal", "unknown", "nonanswer", "errored",
        "finish", "provider_refused"]

ctl_work = {}
for x in csv.DictReader(open(f"{SRC}/controls_qa.csv")):
    ctl_work[x["case_id"]] = x["case_id"].split("_ch")[0]

# finish reason / provider refusal flag, keyed by (case, question, model)
finish = {}
for p in sorted(glob.glob(f"{SRC}/predictions/*.csv")):
    for x in csv.DictReader(open(p)):
        finish[(x["case_id"], x["qid"], x["model"])] = (
            x.get("finish", ""), x.get("refused", ""))

rows = []
for p in sorted(glob.glob(f"{SRC}/scores/scores_*.csv")):
    b = os.path.basename(p)
    corpus = "controls" if b.startswith("scores_controls__") else "court_opinions"
    for x in csv.DictReader(open(p)):
        r = {k: x.get(k, "") for k in KEEP if k in x}
        fin, refused = finish.get((x["case_id"], x["qid"], x["model"]), ("", ""))
        r["finish"], r["provider_refused"] = fin, refused
        r["corpus"] = corpus
        r["work"] = ctl_work.get(x["case_id"], "court_opinion") if corpus == "controls" else "court_opinion"
        rows.append({k: r.get(k, "") for k in KEEP})

with gzip.open(f"{OUT}/blackbox_scores.csv.gz", "wt", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=KEEP); w.writeheader(); w.writerows(rows)
print("blackbox rows", len(rows), os.path.getsize(f"{OUT}/blackbox_scores.csv.gz"))

# ---------- 2. teacher-forced logprobs ----------
LKEEP = ["corpus", "work", "case_id", "qid", "arm", "model", "variant", "condition",
         "trunc_n", "n_suffix_tokens", "logp_sum", "logp_per_token", "p_geomean",
         "min_token_logprob", "first_break_index"]
lrows = []
for p in sorted(glob.glob(f"{SRC}/logprobs/logprobs__*.csv")):
    b = os.path.basename(p)
    parts = b[:-4].split("__")           # logprobs, corpus, org, model, condition
    corpus = "controls" if parts[1] == "controls" else "court_opinions"
    model = parts[3]
    variant = "instruct" if model.endswith("-Instruct") else "base"
    for x in csv.DictReader(open(p)):
        # `work` should be the corpus. Court rows have their case_id in the
        # source's work column, which would give 106 separate "works".
        if corpus == "court_opinions":
            base = "court_opinion"
        else:
            base = (x.get("work", "") or x.get("case_id", "")).split("_ch")[0]
        r = {k: x.get(k, "") for k in LKEEP}
        r.update(corpus=corpus, work=base, model=model,
                 variant=variant, condition=parts[4])
        lrows.append({k: r.get(k, "") for k in LKEEP})

with gzip.open(f"{OUT}/logprobs.csv.gz", "wt", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=LKEEP); w.writeheader(); w.writerows(lrows)
print("logprob rows", len(lrows), os.path.getsize(f"{OUT}/logprobs.csv.gz"))

# ---------- 3. hotspot classification ----------
src = f"{SRC}/logprobs/classified__meta-llama__Llama-3.1-70B__raw.csv"
hs = list(csv.DictReader(open(src)))
with gzip.open(f"{OUT}/hotspots_classified.csv.gz", "wt", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(hs[0])); w.writeheader(); w.writerows(hs)
print("hotspots", len(hs), os.path.getsize(f"{OUT}/hotspots_classified.csv.gz"))

# ---------- 4. null floor ----------
floor = {}
for name, f in (("court_opinions", "null_floor__court_opinions_qa_v2.json"),
                ("controls", "null_floor__controls.json")):
    d = json.load(open(f"{SRC}/scores/{f}"))
    floor[name] = {"n_pairs": d["pooled"]["n"],
                   "longest_run": d["pooled"]["longest_run"],
                   "rouge_l": d["pooled"]["rouge_l"],
                   "token_f1": d["pooled"]["token_f1"],
                   "exact": d["pooled"]["exact"],
                   "gold_tokens": d["pooled"]["gold_tok"]}
floor["description"] = ("Measured chance level: each gold answer scored against OTHER "
                        "cases' gold answers, which share register and boilerplate but no "
                        "provenance. Every reported score is read against this, not zero.")
json.dump(floor, open(f"{OUT}/null_floor.json", "w"), indent=1)
print("floor", floor["court_opinions"], floor["controls"])
