"""figures.py -- the five figures in the paper.

  fig1_corpus_ladder.png  mean contiguous tokens by corpus, against the measured floor
  fig2_distributions.png  the bimodality test -- what a mean cannot show
  fig3_prepost.png        pre- vs post-cutoff difference per model, with CIs
  fig4_whitebox.png       teacher-forced replication on open weights
  fig5_hotspots.png       adjudication of the 399 highest-probability spans

Palette is Okabe-Ito (colour-blind safe). Positive controls are blue, the
negative control grey, the target orange -- held constant across every figure so
the reader learns the coding once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import load, stats

# Two shades of blue for the two positive controls -- same family, still
# distinguishable when they sit side by side in the grouped bars of fig4.
C_POS = "#0072B2"      # positive control: heavily duplicated public text
C_POS2 = "#56B4E9"     # positive control: memorized novel
C_NEG = "#999999"      # negative control: same novel, order destroyed
C_TGT = "#D55E00"      # court opinions, the target
C_FLOOR = "#C1272D"

CORPUS_COLOR = {
    "constitution": C_POS,
    "gatsby": C_POS2,
    "gatsby_shuf": C_NEG,
    "court_opinion": C_TGT,
}


def _short_model(m):
    return m.split("/")[-1]


def fig_corpus_ladder(bb, floor, out):
    """The instrument-validation figure: does the harness detect anything?

    A null on obscure legal text looks exactly like a broken harness, so the
    positive controls are load-bearing rather than decorative.
    """
    cs = stats.corpus_summary(bb)
    works = [w for w in load.CORPUS_ORDER if w in cs]
    means = [cs[w]["mean"] for w in works]
    los = [cs[w]["mean"] - cs[w]["ci"][0] for w in works]
    his = [cs[w]["ci"][1] - cs[w]["mean"] for w in works]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    y = np.arange(len(works))
    ax.barh(y, means, xerr=[los, his], color=[CORPUS_COLOR[w] for w in works],
            error_kw=dict(ecolor="#333", lw=1.2, capsize=4), height=0.62)

    fl = floor["controls"]["longest_run"]
    ax.axvline(fl, color=C_FLOOR, ls="--", lw=1.4, zorder=0)
    # y is inverted below, so -0.62 puts this above the top bar rather than
    # under the x-axis label.
    ax.text(fl + 0.5, -0.62, f"measured floor = {fl:.1f} tokens",
            color=C_FLOOR, fontsize=8.5, va="center")

    for i, w in enumerate(works):
        d = cs[w]
        ax.text(d["mean"] + max(his) + 0.9, i,
                f"{d['mean']:.1f}   n={d['n_attempts']:,}   {d['frac_over_10']:.0%} over 10 tok",
                va="center", fontsize=8.2)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{load.CORPUS_LABEL[w]}\n({load.CORPUS_ROLE[w]})" for w in works],
                       fontsize=9)
    ax.invert_yaxis()
    ax.set_ylim(len(works) - 0.4, -0.95)
    ax.set_xlim(0, max(means) * 1.55)
    ax.set_xlabel("mean longest contiguous run reproduced (tokens), over attempts", fontsize=9)
    ax.set_title("The harness detects memorization where it exists, and not in court opinions\n"
                 "7 closed-weight models, temperature 0, retrieval disabled in code",
                 fontsize=10.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_distributions(bb, out):
    """Memorization is bimodal; the court-opinion distribution has no upper mode.

    This is the figure the paper leans on hardest, because a mean cannot make
    the distinction. "Half the items at 15 tokens" and "every item at 7.5" give
    the same average and mean opposite things about whether a passage is carried
    in the weights.
    """
    bins = (0, 2, 4, 6, 8, 10, 15, 20, 30, 1e9)
    labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-15", "15-20", "20-30", "30+"]
    show = ["gatsby", "court_opinion", "gatsby_shuf"]
    cs = stats.corpus_summary(bb)

    fig, axes = plt.subplots(1, len(show), figsize=(13.0, 4.2), sharey=True)
    for ax, w in zip(axes, show):
        counts, n = stats.histogram(bb, w, bins=bins)
        frac = np.array(counts) / n if n else np.zeros(len(counts))
        ax.bar(np.arange(len(labels)), frac, color=CORPUS_COLOR[w])
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=7.6, rotation=45, ha="right")
        # Computed from the same strict ">10" predicate the tables use, not by
        # summing the bins -- the 10-15 bin is half-open and includes exactly 10,
        # which would disagree with the reported figure by a couple of points.
        over = cs[w]["frac_over_10"]
        ax.set_title(f"{load.CORPUS_LABEL[w]}  ({load.CORPUS_ROLE[w]})\n"
                     f"n={n:,}   {over:.0%} above 10 tokens", fontsize=9.5)
        ax.axvline(4.5, color="#666", ls=":", lw=1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("share of attempts", fontsize=9)
    axes[len(show) // 2].set_xlabel("longest contiguous run reproduced (tokens)", fontsize=9)
    fig.suptitle("Real memorization is bimodal. Court opinions have no upper mode.",
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_prepost(bb, out):
    """The precise null: pre-cutoff minus post-cutoff recall, per model.

    Width is what makes this a null rather than an absence of evidence. An
    effect the size of the novel controls could not have been missed at these
    interval widths.
    """
    pp = stats.pre_post(bb)
    models = sorted(pp, key=lambda m: pp[m]["diff"])
    diffs = [pp[m]["diff"] for m in models]
    lo = [pp[m]["diff"] - pp[m]["diff_ci"][0] for m in models]
    hi = [pp[m]["diff_ci"][1] - pp[m]["diff"] for m in models]

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    y = np.arange(len(models))
    ax.errorbar(diffs, y, xerr=[lo, hi], fmt="o", color=C_TGT,
                ecolor="#555", lw=0, elinewidth=1.3, capsize=4, markersize=7)
    ax.axvline(0, color="#333", lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([_short_model(m) for m in models], fontsize=8.5)
    ax.set_xlabel("pre-cutoff minus post-cutoff mean contiguous tokens", fontsize=9)
    ax.set_title("No model recalls opinions it could have seen better than ones it could not\n"
                 "every interval includes zero, at a width of about one token",
                 fontsize=10.5)

    span = max(max(diffs) + max(hi), abs(min(diffs) - max(lo))) * 1.6
    ax.set_xlim(-span, span)
    ax.invert_yaxis()
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_whitebox(lp, out):
    """Teacher-forced replication: the metric the black-box probe cannot reach."""
    ls = stats.logprob_summary(lp)
    order = [("Llama-3.1-8B", "base"), ("Llama-3.1-8B-Instruct", "instruct"),
             ("Llama-3.1-70B", "base"), ("Llama-3.1-70B-Instruct", "instruct")]
    order = [o for o in order if any(k[1] == o[0] for k in ls)]
    works = [w for w in load.CORPUS_ORDER if any(k[0] == w for k in ls)]

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    x = np.arange(len(order))
    width = 0.8 / len(works)
    for i, w in enumerate(works):
        vals = [ls.get((w, m, v), {}).get("mean_p_geomean", np.nan) for m, v in order]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width,
               color=CORPUS_COLOR[w], label=load.CORPUS_LABEL[w])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m.replace('Llama-3.1-', '').replace('-Instruct', '')}\n{v}"
                        for m, v in order], fontsize=8.5)
    ax.set_ylabel("mean per-token probability of the true continuation", fontsize=9)
    ax.set_title("Teacher-forced replication on open weights reproduces the black-box ordering\n"
                 r"$p = \prod_i P(y_i \mid x, y_{<i})$, suffix truncated to 25 tokens, "
                 "no quantization", fontsize=10.5)
    ax.legend(fontsize=8.2, frameon=False, ncol=4)
    ax.set_ylim(0, 1.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_hotspots(hs, out):
    """What the top of the distribution actually contains."""
    h = stats.hotspot_summary(hs)
    pretty = {
        "in_document_repetition": "repeated elsewhere in the same document",
        "post_cutoff_predictable": "from an opinion filed after the cutoff",
        "quoted_in_context": "quoted material, flagged as such in context",
        "recurring_furniture": "recurring structural furniture",
        "furniture_by_pattern": "furniture, matched by pattern",
        "MEMORISATION_CANDIDATE": "memorization candidate",
    }
    items = sorted(h["verdicts"].items(), key=lambda kv: kv[1])
    labels = [pretty.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    colors = [C_FLOOR if k == "MEMORISATION_CANDIDATE" else C_NEG for k, _ in items]

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + 3, i, f"{v}  ({v / h['n_spans']:.1%})", va="center", fontsize=8.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.set_xlim(0, max(vals) * 1.30)
    ax.set_xlabel("number of spans", fontsize=9)
    ax.set_title(f"The {h['n_spans']} highest-probability spans, adjudicated one by one\n"
                 f"{h['n_candidates']} survive as memorization candidates "
                 f"({h['candidate_rate']:.1%}); all are themselves quoted external text",
                 fontsize=10.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def render(bb, lp, hs, floor, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig_corpus_ladder(bb, floor, outdir / "fig1_corpus_ladder.png")
    fig_distributions(bb, outdir / "fig2_distributions.png")
    fig_prepost(bb, outdir / "fig3_prepost.png")
    fig_whitebox(lp, outdir / "fig4_whitebox.png")
    fig_hotspots(hs, outdir / "fig5_hotspots.png")
    print(f"\nwrote 5 figures to {outdir}")
