"""Observation histories and threshold effects at their final journal size.

All statistical marks use the supplied summaries or already reported tables.
The named examples are read from the frozen-artifact extraction in domain_cases.json.
"""

from __future__ import annotations

import json

from matplotlib.patches import Patch, Rectangle

from dense_style import (
    HERE, BLUE, PURPLE, PINK, GRAY, INK, axes, figure, label, strip,
    csv_rows, effect, interval, manuscript_table, marker_handle, number,
    numeric, save, stored_effect,
)

COHORTS = ("adjacent", "graph_first_observed", "raw_first_observed")
COHORT_LABELS = ("Adjacent", "Graph-first", "Raw-first")
THRESHOLDS = (10, 20, 30)
COL = (170 - 4 - 6) / 3
X = (2, 2 + COL + 3, 2 + 2 * (COL + 3))


def select(rows, **criteria):
    """Require one stored comparison for each requested figure entry."""
    selected = [r for r in rows if all(r.get(k) == v for k, v in criteria.items())]
    if len(selected) != 1:
        raise ValueError(f"Missing/duplicated observation comparison: {criteria}")
    return selected[0]


def history_cases(fig):
    """Draw processed annual counts, keeping raw first/last records separate."""
    source = json.loads((HERE / "domain_cases.json").read_text())
    cases = source["fig3"]["history_cases"]
    years = list(range(2010, 2017))
    if len(cases) != 3:
        raise ValueError("Fig3a requires the three prespecified history cases")
    label(fig, "(a)", "Named pairs and annual processed observations", X[0], 194)
    ax = axes(fig, 60, 168.5, 50, 21.5)
    ax.set(xlim=(-.5, 6.5), ylim=(-.5, 2.5), xticks=range(7),
           xticklabels=[str(y)[2:] for y in years], yticks=[])
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, pad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    receipt = []
    for i, case in enumerate(cases):
        y = 2 - i
        names = case["pair_names"]
        counts = [int(case["processed_counts"][str(year)]) for year in years]
        if len(names) != 2 or any(c < 0 for c in counts):
            raise ValueError(f"Invalid frozen history case: {case}")
        for j, count in enumerate(counts):
            ax.add_patch(Rectangle((j - .45, y - .37), .90, .74,
                                   fc=BLUE if count else "white", ec=GRAY, lw=.45,
                                   alpha=.18 if count else 1))
            ax.text(j, y, str(count) if count else "–", ha="center", va="center", fontsize=8)
        text_y = 168.5 + (y + .5) / 3 * 21.5
        strip(fig, f"{names[0]}\n× {names[1]}", 3, text_y, va="center", linespacing=1.0)
        first, last = case["raw_first_year"], case["raw_last_year"]
        if first not in years or (last is not None and last not in years):
            raise ValueError("Raw endpoint lies outside the explicitly displayed history")
        ax.plot(years.index(first), y + .32, marker="^", color=INK, ms=3.3, ls="none", clip_on=False)
        if last is not None:
            ax.plot(years.index(last), y - .32, marker="v", color=INK, mfc="white", ms=3.3,
                    mew=.65, ls="none", clip_on=False)
        receipt.append({**case, "processed_counts_2010_2016": counts})
    fig.legend(handles=[marker_handle(INK, "^", "Raw first"), marker_handle(INK, "v", "Last prior raw", False)],
               loc="center", bbox_to_anchor=(54 / 170, 163 / 200), ncol=2, frameon=False,
               fontsize=8, handlelength=.6, handletextpad=.3, columnspacing=1.2)
    return receipt


def membership_flow(fig, row, x, letter):
    """Show shared membership plus both directions of non-nested turnover."""
    df = row[0]
    candidate, shared, entered, exited = [int(number(row[k])) for k in (2, 3, 4, 5)]
    if candidate != shared + entered or 51589 != shared + exited:
        raise ValueError(f"Inconsistent membership partition: {row}")
    if entered != int(number(row[6])):
        raise ValueError("New DF candidates must retain the missing-label disclosure")
    label(fig, letter, f"DF20 ↔ {df} membership", x, 113)
    ax = axes(fig, x + 1, 85.5, COL - 2, 24)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    scale = .70 / max(candidate, 51589)
    base = .10
    left, right, bar_w = .13, .77, .12
    shared_h = shared * scale
    ax.add_patch(Rectangle((left + bar_w, base), right - left - bar_w, shared_h,
                           fc=BLUE, alpha=.15, ec="none"))
    for bar_x in (left, right):
        ax.add_patch(Rectangle((bar_x, base), bar_w, shared_h, fc=BLUE, ec=INK, lw=.4))
    ax.add_patch(Rectangle((left, base + shared_h), bar_w, exited * scale,
                           fc=GRAY, hatch="xx", ec=INK, lw=.4))
    ax.add_patch(Rectangle((right, base + shared_h), bar_w, entered * scale,
                           fc=PINK, hatch="///", ec=INK, lw=.4))
    ax.text(.50, base + shared_h / 2, f"Shared\n{shared:,}", ha="center", va="center", fontsize=8)
    for bar_x, title, total in ((left, "DF20", 51589), (right, df, candidate)):
        ax.text(bar_x + bar_w / 2, base + total * scale + .04,
                f"{title}\n{total:,}", ha="center", va="bottom", fontsize=8, linespacing=1.0)
    strip(fig, f"Exit {exited:,}", x + 2, 84.5, va="top")
    strip(fig, f"Enter {entered:,}*", x + COL - 1, 84.5, ha="right", va="top")
    strip(fig, "* entries unlabeled", x + COL / 2, 78.8, ha="center")
    return {"threshold": df, "DF20": 51589, "candidate": candidate, "shared": shared,
            "entered": entered, "exited": exited, "entered_label_missing": entered}


def main():
    rows = csv_rows("frozen_comparisons.csv")
    history = sorted(json.loads((HERE / "observation_history.json").read_text())["rows"],
                     key=lambda r: int(r["condition"][2:]))
    if [r["condition"] for r in history] != ["df10", "df20", "df30"]:
        raise ValueError("The complete DF10/20/30 observation history is required")
    counts = manuscript_table("tab:frozen-history-counts")
    pair_rows = [r for r in counts if len(r) == 8 and r[1] == "Pair"]
    term_rows = [r for r in counts if len(r) == 8 and r[1] == "Term"]
    if len(pair_rows) != 3 or len(term_rows) != 3:
        raise ValueError("Missing pair or term history table rows")
    fig = figure(200)
    cases = history_cases(fig)

    # (b) Three pair-only entry denominators for the same DF20 year.
    label(fig, "(b)", "Entry cohorts · 2016", X[2], 194)
    ax = axes(fig, 132, 169.5, 32, 20)
    pair20 = next(r for r in pair_rows if r[0] == "DF20")
    cohort_n = [int(number(pair20[2])), int(number(pair20[7].split("/")[0])), int(number(pair20[3]))]
    for y, n, color, hatch in zip((2, 1, 0), cohort_n, (BLUE, PURPLE, PINK), ("", "///", "..."), strict=True):
        ax.barh(y, n / 1000, height=.55, color=color, hatch=hatch, ec="white", lw=.5)
        ax.text(n / 1000 + 1, y, f"{n:,}", va="center", fontsize=8)
    ax.set(yticks=[2, 1, 0], yticklabels=COHORT_LABELS, xlim=(0, 68), ylim=(-.5, 2.5),
           xticks=[0, 30, 60], xlabel="Pairs (k)")
    numeric(ax)

    # (c) Complete raw-history partitions, normalized only for graphical composition.
    label(fig, "(c)", "Histories of entry pairs", X[0], 152)
    ax = axes(fig, 12, 133.5, 41, 13.5)
    fields = ("first_observed", "present_previous", "reentry_gap_1", "reentry_gap_ge2")
    names = ("Raw-first", "Previous year", "Reentry 1 year", "Reentry ≥2 years")
    colors, hatches = (BLUE, GRAY, PURPLE, PINK), ("", "xx", "///", "...")
    history_receipt = []
    for y, row in zip((2, 1, 0), history, strict=True):
        parts = [int(row[f"n_raw_{field}"]) for field in fields]
        if sum(parts) != row["n_candidates"]:
            raise ValueError(f"Non-partitioning raw history: {row}")
        left = 0.0
        for n, color, hatch in zip(parts, colors, hatches, strict=True):
            pct = 100 * n / row["n_candidates"]
            ax.barh(y, pct, left=left, height=.65, color=color, hatch=hatch, ec="white", lw=.35)
            left += pct
        history_receipt.append({"condition": row["condition"], "n": row["n_candidates"],
                                "partition": dict(zip(fields, parts, strict=True))})
    ax.set(yticks=[2, 1, 0], yticklabels=["DF10", "DF20", "DF30"], xlim=(0, 100),
           ylim=(-.5, 2.5), xticks=[0, 50, 100], xticklabels=["0", "50", "100%"])
    ax.legend(handles=[Patch(fc=c, hatch=h, ec="white", label=n)
                       for c, h, n in zip(colors, hatches, names, strict=True)],
              loc="upper left", bbox_to_anchor=(-.28, -.40), ncol=2, frameon=False,
              handlelength=.8, handletextpad=.3, columnspacing=.7, labelspacing=.15,
              fontsize=8, borderaxespad=0)

    # (d) Prior raw observations among graph-first pairs.
    label(fig, "(d)", "Prior raw in graph-first", X[1], 152)
    ax = axes(fig, 72.5, 132.5, 30.5, 14.5)
    graph_counts = [[int(number(v)) for v in r[7].split("/")] for r in pair_rows]
    for y, (total, prior) in zip((2, 1, 0), graph_counts, strict=True):
        pct = 100 * prior / total
        ax.plot(pct, y, marker="^", ms=4.2, color=PURPLE, ls="none")
        ax.text(pct + 1.2, y, f"{pct:.1f}%", va="center", fontsize=8)
    ax.set(yticks=[2, 1, 0], yticklabels=["DF10", "DF20", "DF30"], xlim=(0, 42),
           ylim=(-.5, 2.5), xticks=[0, 20, 40], xlabel="Graph-first pairs (%)")
    numeric(ax)

    # (e) Graph-first term entries are nevertheless all previously raw-observed.
    label(fig, "(e)", "Term entries", X[2], 152)
    ax = axes(fig, 128, 133, 31, 14)
    term_receipt = []
    for y, row in zip((2, 1, 0), term_rows, strict=True):
        total = int(number(row[2]))
        graph, prior = [int(number(v)) for v in row[7].split("/")]
        if number(row[3]) != 0 or graph != prior:
            raise ValueError("Entering-term raw-first zero must remain visible")
        ax.barh(y, total, height=.56, color=GRAY, ec="white", lw=.4)
        ax.barh(y, graph, height=.56, color=PURPLE, hatch="///", ec="white", lw=.4)
        ax.text(total + 7, y, str(total), va="center", fontsize=8)
        term_receipt.append({"DF": row[0], "total": total, "graph_first": graph, "prior_raw": prior, "raw_first": 0})
    ax.set(yticks=[2, 1, 0], yticklabels=["DF10", "DF20", "DF30"], xlim=(0, 320),
           ylim=(-.5, 2.5), xticks=[0, 150, 300])
    numeric(ax)
    fig.legend(handles=[Patch(fc=PURPLE, hatch="///", ec="white", label="Graph-first"),
                        Patch(fc=GRAY, ec="white", label="Other")],
               loc="center", bbox_to_anchor=(142 / 170, 125 / 200), ncol=2, frameon=False,
               fontsize=8, handlelength=.8, handletextpad=.3, columnspacing=.8)
    strip(fig, "Raw-first = 0 at every DF", X[2] + COL / 2, 120.5, ha="center")

    # (f,g) Independent bidirectional partitions relative to DF20.
    membership = [r for r in manuscript_table("tab:df-membership") if len(r) == 8 and r[1] == "Pair"]
    if len(membership) != 3:
        raise ValueError("Missing pair membership rows")
    flow_receipts = [membership_flow(fig, next(r for r in membership if r[0] == df), x, letter)
                     for df, x, letter in (("DF10", X[0], "(f)"), ("DF30", X[1], "(g)"))]

    # (h) Same IDs, two different histories, three actual coverage levels.
    label(fig, "(h)", "Matched prior coverage", X[2], 113)
    ax = axes(fig, 129, 92.5, 35, 15)
    matched = {}
    for i, cohort in enumerate(("adjacent", "raw_first_observed")):
        cell = [select(rows, condition=f"matched_df{df}_{cohort}") for df in THRESHOLDS]
        if len({r["n_fronts"] for r in cell}) != 1:
            raise ValueError("Matched-ID effects have different denominators")
        ax.plot(THRESHOLDS, [100 * r["n_prox_defined"] / r["n_fronts"] for r in cell],
                color=(BLUE, PURPLE)[i], marker=("o", "^")[i], lw=.8, ms=4)
        matched[cohort] = cell
    ax.set(xlim=(8, 32), xticks=THRESHOLDS, xticklabels=["10", "20", "30"],
           ylim=(60, 100), yticks=[60, 80, 100], yticklabels=["60%", "80%", "100%"])
    numeric(ax, "y")
    ax.legend(handles=[marker_handle(BLUE, "o", "Adjacent"), marker_handle(PURPLE, "^", "Raw-first")],
              loc="upper center", bbox_to_anchor=(.42, -.25), ncol=2, frameon=False,
              handlelength=.6, handletextpad=.2, columnspacing=.6, fontsize=8, borderaxespad=0)
    strip(fig, "DF threshold", 147, 81.0, ha="center")

    # (i–k) Nine effects in their original populations, with a common x axis.
    limits = (-.014, .043)
    effects_receipt = []
    for index, (letter, title, cohort) in enumerate((("(i)", "Original DF20 · ΔAP", None),
                                                   ("(j)", "Matched adjacent · ΔAP", "adjacent"),
                                                   ("(k)", "Matched raw-first · ΔAP", "raw_first_observed"))):
        x = X[index]
        label(fig, letter, title, x, 74)
        ax = axes(fig, x + (18 if index == 0 else 13), 48.0, 33 if index == 0 else 39, 20.5)
        if cohort is None:
            cell = [select([r for r in rows if not r["condition"].startswith("matched_")],
                           analysis="legacy_mixed", cohort=c) for c in COHORTS]
            labels = COHORT_LABELS
        else:
            cell = matched[cohort]
            labels = [f"DF{df}" for df in THRESHOLDS]
        for y, row in zip((2, 1, 0), cell, strict=True):
            effect(ax, row, y, PURPLE if cohort == "raw_first_observed" else BLUE,
                   "^" if cohort == "raw_first_observed" else "o")
            effects_receipt.append(row)
        ax.set(yticks=[2, 1, 0], yticklabels=labels, xlim=limits, ylim=(-.55, 2.55),
               xticks=[0, .02, .04], xlabel="ΔAP")
        numeric(ax, signed=True)
        if cohort:
            strip(fig, f"Matched N = {cell[0]['n_fronts']:,}", x + COL / 2, 36.5, ha="center")
        else:
            strip(fig, "Original mixed fronts", x + COL / 2, 36.5, ha="center")
    if len(effects_receipt) != 9:
        raise ValueError("The figure must retain all nine observation effects")
    positive = matched["raw_first_observed"][-1]
    if positive["conditional_dependence_p2_5"] <= 0:
        raise ValueError("Positive matched raw-first DF30 interval was lost")

    # (l) All direct paired contrasts; no inference from interval overlap.
    label(fig, "(l)", "Direct paired threshold contrasts", 2, 28)
    contrasts = [r for r in manuscript_table("tab:threshold-contrasts")
                 if len(r) == 7 and r[0] in {"Adjacent", "Raw-first"}]
    if len(contrasts) != 4:
        raise ValueError("The complete set of four threshold contrasts is required")
    ax = axes(fig, 34, 11.0, 130, 13.0)
    for y, row in zip((3, 2, 1, 0), contrasts, strict=True):
        color, marker = (BLUE, "o") if row[0] == "Adjacent" else (PURPLE, "^")
        effect(ax, stored_effect(number(row[4]), interval(row[6])), y, color, marker)
    ax.set(yticks=[3, 2, 1, 0], yticklabels=["Adjacent 10−20", "Adjacent 30−20", "Raw-first 10−20", "Raw-first 30−20"],
           xlim=(-.012, .030), ylim=(-.5, 3.5), xticks=[-.01, 0, .01, .02, .03], xlabel="ΔΔAP")
    numeric(ax, signed=True)
    save(fig, "Fig3", "Observation histories and non-nested threshold comparisons", {
        "panels": 12, "panel_labels": [f"({c})" for c in "abcdefghijkl"],
        "layout_rows_mm": [39, 36, 36, 43, 30], "named_history_cases": cases,
        "annual_processed_case_cells": 21, "raw_first_last_not_continuous_history": True,
        "DF20_pair_entry_denominators": dict(zip(COHORTS, cohort_n, strict=True)),
        "raw_history_partition": history_receipt, "history_conditions": 3, "raw_history_partition_cells": 12,
        "graph_first_pair_counts_prior_raw": graph_counts, "entering_terms": term_receipt,
        "non_nested_membership_flows": flow_receipts, "matched_prior_coverage": matched,
        "matched_P_coverage_conditions": 6,
        "stored_observation_effects": effects_receipt, "legacy_DF20_effects": 3,
        "matched_threshold_effects": 6, "paired_threshold_contrasts": len(contrasts),
        "paired_threshold_contrast_rows": contrasts,
        "positive_raw_first_DF30_preserved": True,
        "interval_definition": "stored conditional dependence percentile endpoints; not model-seed SD",
    })


if __name__ == "__main__":
    main()
