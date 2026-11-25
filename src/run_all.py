"""run_all.py: regenerate all numbers and figures in the paper.

    python -m src.run_all                # everything
    python -m src.run_all --no-figures   # numbers only (no matplotlib needed)

Writes results/summary.json and results/figures/*.png. The console output is a
short summary; the full numbers are in the JSON. Bootstraps are seeded, so
reruns give identical output. Takes about ten seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import load, stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)

    bb = load.load_blackbox()
    lp = load.load_logprobs()
    hs = load.load_hotspots()
    floor = load.load_floor()

    court_floor = floor["court_opinions"]["longest_run"]
    ctl_floor = floor["controls"]["longest_run"]

    print("=" * 78)
    print("  1. THE MEASURED FLOOR")
    print("=" * 78)
    print(f"  court opinions  {court_floor:.1f} contiguous tokens "
          f"({floor['court_opinions']['n_pairs']} gold-vs-other-gold pairs)")
    print(f"  controls        {ctl_floor:.1f} contiguous tokens "
          f"({floor['controls']['n_pairs']} pairs)")
    print("  Scores below should be compared to this floor.")

    print("\n" + "=" * 78)
    print("  2. BLACK-BOX PROBE, POOLED OVER 7 MODELS")
    print("=" * 78)
    cs = stats.corpus_summary(bb)
    print(f"  {'corpus':22s} {'role':17s} {'docs':>5s} {'attempts':>9s} {'rate':>6s} "
          f"{'mean':>6s} {'95% CI':>14s} {'>10 tok':>8s} {'deff':>5s}")
    for w in load.CORPUS_ORDER:
        if w not in cs:
            continue
        d = cs[w]
        print(f"  {load.CORPUS_LABEL[w]:22s} {load.CORPUS_ROLE[w]:17s} "
              f"{d['n_documents']:5d} {d['n_attempts']:9d} {d['attempt_rate']:6.0%} "
              f"{d['mean']:6.2f} [{d['ci'][0]:5.2f},{d['ci'][1]:5.2f}] "
              f"{d['frac_over_10']:8.1%} {d['design_effect']:5.1f}")
    print("\n  CIs resample documents, since questions from the same text are correlated.")
    print(f"  On court opinions this widens the interval {(cs['court_opinion']['ci'][1]-cs['court_opinion']['ci'][0]) / (cs['court_opinion']['ci_item_level'][1]-cs['court_opinion']['ci_item_level'][0]):.1f}x"
          f" (ICC {cs['court_opinion']['icc']:.3f}, effective n"
          f" {cs['court_opinion']['n_effective']:.0f} of {cs['court_opinion']['n_attempts']:,}).")

    print(f"\n  Controls: {cs['constitution']['mean']:.1f} tokens on the Constitution,"
          f" {cs['gatsby']['mean']:.1f} on Gatsby,\n"
          f"  {cs['gatsby_shuf']['mean']:.2f} on scrambled Gatsby (floor {ctl_floor:.1f}). "
          f"Court opinions: {cs['court_opinion']['mean']:.2f},\n"
          f"  with {cs['court_opinion']['frac_over_10']:.1%} of items above 10 tokens "
          f"(Gatsby: {cs['gatsby']['frac_over_10']:.0%}).")

    print("\n" + "=" * 78)
    print("  3. PER MODEL")
    print("=" * 78)
    bcm = stats.by_corpus_model(bb)
    models = sorted({m for _, m in bcm})
    print(f"  {'model':34s} " + "".join(f"{load.CORPUS_LABEL[w][:13]:>15s}"
                                        for w in load.CORPUS_ORDER))
    for m in models:
        cells = []
        for w in load.CORPUS_ORDER:
            d = bcm.get((w, m))
            cells.append(f"{d['mean']:7.2f} ({d['attempt_rate']:3.0%})" if d and d["n_attempts"]
                         else f"{'no sample':>15s}")
        print(f"  {m:34s} " + "".join(f"{c:>15s}" for c in cells))
    print("  cell = mean contiguous tokens (share of items attempted)")

    print("\n" + "=" * 78)
    print("  4. PRE- VS POST-CUTOFF COURT OPINIONS")
    print("=" * 78)
    pp = stats.pre_post(bb)
    print(f"  {'model':34s} {'pre':>7s} {'post':>7s} {'diff':>7s} {'95% CI on diff':>18s}")
    for m, d in sorted(pp.items()):
        excl = "" if d["diff_ci"][0] <= 0 <= d["diff_ci"][1] else "  <- excludes 0"
        print(f"  {m:34s} {d['pre_mean']:7.2f} {d['post_mean']:7.2f} "
              f"{d['diff']:+7.2f}   [{d['diff_ci'][0]:+.2f}, {d['diff_ci'][1]:+.2f}]{excl}")
    if pp:
        # Arms depend on each model's cutoff, so document counts vary by model.
        pre = [d["n_pre_docs"] for d in pp.values()]
        post = [d["n_post_docs"] for d in pp.values()]
        print(f"  (documents per model: {min(pre)}-{max(pre)} pre-cutoff, "
              f"{min(post)}-{max(post)} post-cutoff, by each model's cutoff;\n"
              "   intervals resample documents within each arm)")
    n_zero = sum(1 for d in pp.values() if d["diff_ci"][0] <= 0 <= d["diff_ci"][1])
    print(f"  {n_zero} of {len(pp)} intervals include zero (widths ~1 token;")
    print("  positive-control effect is ~20 tokens).")

    print("\n" + "=" * 78)
    print("  5. TEACHER-FORCED LOGPROBS (white-box replication, Llama 3.1)")
    print("=" * 78)
    ls = stats.logprob_summary(lp)
    variants = sorted({(m, v) for _, m, v in ls})
    print(f"  {'model / variant':34s} " + "".join(f"{load.CORPUS_LABEL[w][:13]:>15s}"
                                                  for w in load.CORPUS_ORDER))
    for m, v in variants:
        cells = []
        for w in load.CORPUS_ORDER:
            d = ls.get((w, m, v))
            cells.append(f"{d['mean_p_geomean']:.3f}" if d else "-")
        print(f"  {m + ' (' + v + ')':34s} " + "".join(f"{c:>15s}" for c in cells))
    print("  cell = mean per-token geometric-mean probability of the TRUE continuation")
    print("  Same ordering as the black-box probe.")

    print("\n" + "=" * 78)
    print("  6. HIGHEST-PROBABILITY SPANS")
    print("=" * 78)
    hsum = stats.hotspot_summary(hs)
    print(f"  {hsum['n_spans']} highest-probability spans, hand-labeled:")
    for k, v in sorted(hsum["verdicts"].items(), key=lambda kv: -kv[1]):
        print(f"    {v:4d}  {k}")
    print(f"\n  memorization candidates: {hsum['n_candidates']} "
          f"({hsum['candidate_rate']:.1%})")
    print(f"  spans from POST-cutoff opinions: {hsum['post_cutoff_spans']} of "
          f"{hsum['n_spans']} ({hsum['post_cutoff_spans'] / hsum['n_spans']:.0%})"
          f", which can't have been memorized")
    print("\n  what the spans are, by content:")
    for k, v in sorted(hsum["categories_near"].items(), key=lambda kv: -kv[1]):
        print(f"    {v:4d}  {k}")
    print("\n  the surviving candidates:")
    for c in hsum["candidates"]:
        print(f"    [{c['arm']:>14s}] {c['category']:20s} {c['n_tokens']:>3s} tok  "
              f"{c['text'][:80]}...")

    out = {
        "floor": floor,
        "corpus_summary": {w: {k: v for k, v in d.items()} for w, d in cs.items()},
        "by_corpus_model": {f"{w}|{m}": d for (w, m), d in bcm.items()},
        "pre_post": pp,
        "logprobs": {f"{w}|{m}|{v}": d for (w, m, v), d in ls.items()},
        "hotspots": hsum,
        "disposition": {f"{w}|{m}": d for (w, m), d in stats.disposition(bb).items()},
    }
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {RESULTS}/summary.json")

    if not args.no_figures:
        from src import figures
        figures.render(bb, lp, hs, floor, RESULTS / "figures")


if __name__ == "__main__":
    main()
