"""Same-snapshot reference views, all effects, and selection counterevidence."""

from __future__ import annotations

import json

from matplotlib.patches import Patch, Rectangle
import numpy as np

from dense_style import (
    HERE, BLUE, PURPLE, PINK, GRAY, INK, GRID, axes, figure, label, strip,
    csv_rows, effect, interval, manuscript_table, marker_handle, number,
    numeric, save, stored_effect,
)

COHORTS = ("adjacent", "raw_first_observed")
VIEWS = ("full", "api_order", *(f"random_s{k}" for k in range(10)))
VIEW_LABELS = ("Full", "API", *(f"R{k}" for k in range(10)))
COL = (170 - 4 - 6) / 3
X = (2, 2 + COL + 3, 2 + 2 * (COL + 3))
SETTINGS = (("composition", "adjacent"), ("composition", "raw_first_observed"),
            ("common", "adjacent"), ("common", "raw_first_observed"))
SETTING_LABELS = ("View adj.", "View raw", "ID adj.", "ID raw")


def require_view_coverage(cell):
    """Reject missing or duplicated source views before plotting an effect cell."""
    actual = [r["view"] for r in cell]
    if len(actual) != 12 or set(actual) != set(VIEWS):
        raise ValueError(f"Missing or duplicate reference views: {actual}")


def view_style(view):
    """Share colour, shape, and fill across all source-view panels."""
    if view == "full":
        return BLUE, "o", True
    if view == "api_order":
        return PURPLE, "^", True
    return PINK, "s", False


def fingerprints(fig):
    """Keep a named pair fixed while showing its actual observed view status."""
    cases = json.loads((HERE / "domain_cases.json").read_text())["fig5"]["pair_view_cases"]
    if len(cases) != 3:
        raise ValueError("Fig5a needs the three prespecified pair cases")
    label(fig, "(a)", "Same MeSH pairs across observation views", 2, 204)
    ax = axes(fig, 60, 179.0, 50, 20.5)
    ax.set(xlim=(-.5, 11.5), ylim=(-.5, 2.5), xticks=range(12),
           xticklabels=["F", "A", *map(str, range(10))], yticks=[])
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, pad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    state_style = {"raw_first": (BLUE, "", "R"), "hidden_history": (PURPLE, "///", "H"),
                   "reentry": (PINK, "...", "E"), "not_entering": ("white", "", "—")}
    receipt = []
    for i, case in enumerate(cases):
        if set(case["views"]) != set(VIEWS) or len(case["pair_names"]) != 2:
            raise ValueError(f"Incomplete actual pair/view case: {case}")
        y = 2 - i
        text_y = 179 + (y + .5) / 3 * 20.5
        strip(fig, f"{case['pair_names'][0]}\n× {case['pair_names'][1]}", 3, text_y,
              va="center", linespacing=1.0)
        for j, view in enumerate(VIEWS):
            state = case["views"][view]["state"]
            if state not in state_style:
                raise ValueError(f"Unrecognized observed view state: {state}")
            color, hatch, symbol = state_style[state]
            ax.add_patch(Rectangle((j - .46, y - .35), .92, .70, fc=color,
                                   ec=GRAY, hatch=hatch, lw=.45, alpha=.22 if color != "white" else 1))
            ax.text(j, y, symbol, ha="center", va="center", fontsize=8, color=INK)
        receipt.append(case)
    strip(fig, "R raw-first   H hidden history   E reentry   — not entering", 3, 171.0)
    return receipt


def selection_flow(fig, cells):
    """Show exact Full-to-common N and positive counts for both cohorts."""
    label(fig, "(d)", "Common-ID selection", X[2], 161)
    strip(fig, "Full", 132, 155, ha="center")
    strip(fig, "Common", 157, 155, ha="center")
    ax = axes(fig, X[2], 134, COL, 20)
    ax.set(xlim=(0, COL), ylim=(134, 154))
    ax.axis("off")
    receipt = []
    for cohort, short, y in (("adjacent", "Adj", 151), ("raw_first_observed", "Raw", 141)):
        full = next(r for r in cells[("composition", cohort)] if r["view"] == "full")
        common = cells[("common", cohort)][0]
        strip(fig, short, X[2] + 1, y - 1, va="center")
        for x, row in ((132, full), (157, common)):
            strip(fig, f"N {row['n_fronts']:,}", x, y, ha="center")
            strip(fig, f"Y {row['n_positive']:,}", x, y - 3, ha="center", color=BLUE)
            width = 18 * row["n_fronts"] / full["n_fronts"]
            p_width = 18 * row["n_positive"] / full["n_fronts"]
            ax.add_patch(Rectangle((x - X[2] - 9, y - 6.2), width, 1.2,
                                   fc="white", ec=GRAY, lw=.35))
            ax.add_patch(Rectangle((x - X[2] - 9, y - 6.2), p_width, 1.2,
                                   fc=BLUE, ec=BLUE, hatch="///", lw=.35))
        strip(fig, "→", 145, y - .5, ha="center")
        receipt.append({"cohort": cohort, "Full": {"N": full["n_fronts"], "Y": full["n_positive"]},
                        "common": {"N": common["n_fronts"], "Y": common["n_positive"]}})
    return receipt


def main():
    rows = csv_rows("reference_comparisons.csv")
    if len(rows) != 48:
        raise ValueError("Fig5 must retain all 48 stored reference effects")
    cells = {key: [r for r in rows if (r["population"], r["cohort"]) == key] for key in SETTINGS}
    for cell in cells.values():
        require_view_coverage(cell)
    fig = figure(210)
    cases = fingerprints(fig)

    # (b) Identical limited budgets; full reference retained for outcome collection.
    label(fig, "(b)", "Annual observed works", X[2], 204)
    year_counts = [r for r in manuscript_table("tab:reference-year-counts")
                   if len(r) == 4 and r[0].isdigit() and 2010 <= int(r[0]) <= 2016]
    if len(year_counts) != 7:
        raise ValueError("All seven feature/history years are required")
    ax = axes(fig, 128.5, 181, 35.5, 18)
    years = [int(r[0]) for r in year_counts]
    full_n = [number(r[1]) / 1000 for r in year_counts]
    limited_n = [number(r[2]) / 1000 for r in year_counts]
    ax.plot(years, full_n, color=BLUE, marker="o", lw=.9, ms=3.5, label="Full")
    ax.plot(years, limited_n, color=PURPLE, marker="^", ls="--", lw=.9, ms=3.5, label="API / random")
    ax.set(xlim=(2009.7, 2016.3), ylim=(10.5, 18.5), xticks=[2010, 2012, 2014, 2016],
           xticklabels=["10", "12", "14", "16"], yticks=[12, 15, 18], ylabel="Works (k)",
           xlabel="Year (20xx)")
    numeric(ax, "y")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.61), frameon=False, ncol=2,
              fontsize=8, handlelength=1.0, handletextpad=.25, columnspacing=.7,
              labelspacing=.15, borderaxespad=0)

    # One source-view key also covers the four complete effect matrices below.
    fig.legend(handles=[marker_handle(BLUE, "o", "Full"), marker_handle(PURPLE, "^", "API"),
                        marker_handle(PINK, "s", "Random 0–9", False), marker_handle(INK, "D", "Common IDs")],
               loc="center", bbox_to_anchor=(56 / 170, 167 / 210), ncol=4, frameon=False,
               fontsize=8, handlelength=.65, handletextpad=.3, columnspacing=.85)

    # (c) Every view-specific composition, and the fixed common-ID point.
    label(fig, "(c)", "Candidate size and prevalence", X[0], 161)
    composition_receipt = []
    for j, cohort in enumerate(COHORTS):
        ax = axes(fig, 17 + 53 * j, 142.5, 37, 13.0)
        cell = cells[("composition", cohort)]
        common = cells[("common", cohort)]
        if len({(r["n_fronts"], r["n_positive"]) for r in common}) != 1:
            raise ValueError("Common population is not fixed across views")
        fixed = common[0]
        for row in cell:
            color, marker, filled = view_style(row["view"])
            ax.plot(row["n_fronts"] / 1000, 100 * row["n_positive"] / row["n_fronts"],
                    marker=marker, color=color, mfc=color if filled else "white", ms=3.7, mew=.7, ls="none")
            composition_receipt.append({k: row[k] for k in ("view", "cohort", "n_fronts", "n_positive")})
        ax.plot(fixed["n_fronts"] / 1000, 100 * fixed["n_positive"] / fixed["n_fronts"],
                "D", color=INK, ms=3.7)
        ax.set(ylim=(4, 25), yticks=[5, 15, 25], xlabel="Test fronts (k)")
        ax.set(xlim=(0, 57), xticks=[0, 25, 50]) if j == 0 else ax.set(xlim=(0, 20), xticks=[0, 10, 20])
        if j == 0:
            ax.set_ylabel("Positive (%)")
        ax.set_title("Adjacent" if j == 0 else "Raw-first", fontsize=8, pad=2)
        numeric(ax, "y")
    selection = selection_flow(fig, cells)

    # (e,f,h,i) Complete 12-view effects with identical view order and limits.
    limits = (-.015, .055)
    if min(r["conditional_dependence_p2_5"] for r in rows) < limits[0] or max(r["conditional_dependence_p97_5"] for r in rows) > limits[1]:
        raise ValueError("Reference effect interval outside the shared limits")
    effect_receipts = []
    matrix = (("composition", "adjacent", "(e)", "View-specific · adjacent", X[0], 92, 127),
              ("composition", "raw_first_observed", "(f)", "View-specific · raw-first", X[1], 92, 127),
              ("common", "adjacent", "(h)", "Common IDs · adjacent", X[0], 43, 78),
              ("common", "raw_first_observed", "(i)", "Common IDs · raw-first", X[1], 43, 78))
    for population, cohort, letter, title, x, y, title_y in matrix:
        label(fig, letter, title, x, title_y)
        ax = axes(fig, x + 10, y, 41, 33.6)
        for k, view in enumerate(VIEWS):
            row = next(r for r in cells[(population, cohort)] if r["view"] == view)
            color, marker, filled = view_style(view)
            effect(ax, row, 11 - k, color, marker, filled, size=3.1)
            effect_receipts.append(row)
        ax.axhline(9.5, color=GRID, lw=.6)
        ax.set(yticks=range(11, -1, -1), yticklabels=VIEW_LABELS, ylim=(-.55, 11.55),
               xlim=limits, xticks=[0, .02, .04], xlabel="ΔAP")
        numeric(ax, signed=True)
    if len(effect_receipts) != 48:
        raise ValueError("Reference coverage must equal 48 effects")
    full_common_raw = next(r for r in cells[("common", "raw_first_observed")] if r["view"] == "full")
    if full_common_raw["conditional_dependence_p2_5"] <= 0 or abs(full_common_raw["mean_delta_ap"] - .011577) >= 5e-7:
        raise ValueError("The positive Full common-ID raw-first result must be retained")

    # (g) Three direct paired contrasts per population-by-cohort setting.
    label(fig, "(g)", "View contrasts · ΔΔAP", X[2], 127)
    contrasts = [r for r in manuscript_table("tab:view-contrasts")
                 if len(r) == 6 and r[0] in {"View-specific", "Common-ID"}]
    if len(contrasts) != 12:
        raise ValueError("All twelve direct paired view contrasts are required")
    ax = axes(fig, 130.5, 96.5, 34, 26.5)
    styles = ((PURPLE, "^", "API−F"), (PINK, "s", "μR−F"), (GRAY, "o", "API−μR"))
    for index, row in enumerate(contrasts):
        group, comparison = divmod(index, 3)
        color, marker, _ = styles[comparison]
        lo, hi = interval(row[5])
        if not lo < 0 < hi:
            raise ValueError("Changed source interval in paired view contrasts")
        effect(ax, stored_effect(number(row[3]), (lo, hi)), 3 - group + (.22, 0, -.22)[comparison],
               color, marker, comparison != 1, size=3.1)
    ax.set(yticks=[3, 2, 1, 0], yticklabels=SETTING_LABELS, ylim=(-.5, 3.5),
           xlim=(-.016, .027), xticks=[-.01, 0, .02], xlabel="ΔΔAP")
    numeric(ax, signed=True)
    fig.legend(handles=[marker_handle(c, m, n, i != 1) for i, (c, m, n) in enumerate(styles)],
               loc="center", bbox_to_anchor=(141.5 / 170, 86.5 / 210), ncol=3, frameon=False,
               fontsize=8, handlelength=.6, handletextpad=.2, columnspacing=.5)

    # (j) Fitted-seed SD versus SD across ten views of the same snapshot.
    label(fig, "(j)", "Model and view variation", X[2], 78)
    ax = axes(fig, 130.5, 47, 34, 26.5)
    dispersion = []
    reported_dispersion = [r for r in manuscript_table("tab:random-dispersion") if len(r) == 6 and r[2] == "AP increment"]
    if len(reported_dispersion) != 4:
        raise ValueError("Missing reported across-view increment dispersion")
    for group, key in enumerate(SETTINGS):
        random = [next(r for r in cells[key] if r["view"] == f"random_s{k}") for k in range(10)]
        view_sd = float(np.std([r["mean_delta_ap"] for r in random], ddof=1))
        if abs(view_sd - number(reported_dispersion[group][4])) > 5e-7:
            raise ValueError("Across-view SD disagrees with its reported source")
        for k, row in enumerate(random):
            ax.plot(row["seed_delta_sd"], 3 - group + .17 + .055 * (k % 3 - 1),
                    marker="o", color=GRAY, ms=2.2, ls="none")
        ax.plot(view_sd, 3 - group - .17, marker="s", color=PINK, ms=4, ls="none")
        dispersion.append({"population": key[0], "cohort": key[1], "view_sample_sd_divisor9": view_sd,
                           "within_view_seed_sds": [r["seed_delta_sd"] for r in random]})
    ax.set(yticks=[3, 2, 1, 0], yticklabels=SETTING_LABELS, xlim=(0, .018),
           ylim=(-.5, 3.5), xticks=[0, .01], xlabel="SD of ΔAP")
    numeric(ax)
    fig.legend(handles=[marker_handle(GRAY, "o", "Model seed"), marker_handle(PINK, "s", "Random view")],
               loc="center", bbox_to_anchor=(141.5 / 170, 35.3 / 210), ncol=2, frameon=False,
               fontsize=8, handlelength=.6, handletextpad=.3, columnspacing=.7)

    # (k) Pair-only denominators and the API's smaller history omission.
    label(fig, "(k)", "Omitted full-reference history", X[0], 29)
    history = [r for r in manuscript_table("tab:full-history")
               if len(r) == 7 and (r[0] in {"Full", "API order"} or r[0].startswith("Random "))]
    if len(history) != 12:
        raise ValueError("All twelve pair-only full-history summaries are required")
    ax = axes(fig, 13.5, 10.4, 96.5, 12.6)
    fractions, history_receipt = [], []
    for x, (view, row) in enumerate(zip(VIEWS, history, strict=True)):
        n, omitted, last2015, last2014, earlier = [int(number(row[k])) for k in (1, 2, 4, 5, 6)]
        if omitted != last2015 + last2014 + earlier or earlier != 0:
            raise ValueError("Unexpected full-history partition")
        fraction = omitted / n
        fractions.append(fraction)
        ax.bar(x, 100 * last2015 / n, color=GRAY, hatch="///", width=.70, ec=INK, lw=.3)
        ax.bar(x, 100 * last2014 / n, bottom=100 * last2015 / n,
               color="white", hatch="...", width=.70, ec=INK, lw=.3)
        prediction_n = next(r for r in cells[("composition", "raw_first_observed")] if r["view"] == view)["n_fronts"]
        excluded_terms = int(view in {"random_s2", "random_s5"})
        if prediction_n != n + excluded_terms:
            raise ValueError("Pair-only history denominator mixed with entering terms")
        history_receipt.append({"view": view, "pair_denominator": n, "omitted": omitted,
                                "last_full_2015": last2015, "last_full_2014": last2014,
                                "earlier": earlier, "excluded_terms": excluded_terms})
    if fractions[1] != 143 / 14317 or history_receipt[4]["pair_denominator"] != 12650 or history_receipt[7]["pair_denominator"] != 12680:
        raise ValueError("The API or Random 2/5 pair-only history boundary changed")
    ax.set(xticks=range(12), xticklabels=["F", "A", *map(str, range(10))],
           xlim=(-.6, 11.6), ylim=(0, 4.5), yticks=[0, 2, 4], ylabel="Pairs (%)")
    numeric(ax, "y")
    strip(fig, "API 143 / 14,317 = 0.999%", 29, 24.6)
    fig.legend(handles=[Patch(fc=GRAY, ec=INK, hatch="///", label="2015"),
                        Patch(fc="white", ec=INK, hatch="...", label="2014")],
               loc="center", bbox_to_anchor=(91 / 170, 30 / 210), ncol=2, frameon=False,
               fontsize=8, handlelength=.8, handletextpad=.3, columnspacing=.8)
    strip(fig, "Pair-only N · R2 12,650 · R5 12,680", 57, 2.8, ha="center")

    # (l) Selection in later training cohorts as well as the 2016 test cohort.
    label(fig, "(l)", "Annual common retention", X[2], 29)
    membership = [r for r in manuscript_table("tab:common-membership")
                  if len(r) == 7 and r[0].isdigit()]
    if [int(r[0]) for r in membership] != list(range(2011, 2017)):
        raise ValueError("All 2011–2016 common-membership years are required")
    ax = axes(fig, 128.5, 10.5, 35.5, 11.8)
    retention = []
    for cohort, full_col, common_col, color, marker in (("adjacent", 1, 2, BLUE, "o"),
                                                       ("raw_first_observed", 4, 5, PURPLE, "^")):
        full = [int(number(r[full_col].split("/")[0])) for r in membership]
        common = [int(number(r[common_col].split("/")[0])) for r in membership]
        pct = [100 * c / f for c, f in zip(common, full, strict=True)]
        ax.plot(range(2011, 2017), pct, color=color, marker=marker, mfc="white", ms=3.5, lw=.8)
        ax.plot(2016, pct[-1], marker=marker, color=color, ms=4.5, ls="none")
        retention.append({"cohort": cohort, "years": list(range(2011, 2017)), "full": full,
                          "common": common, "retained_percent": pct})
    ax.set(xlim=(2010.7, 2016.3), ylim=(0, 108), xticks=[2011, 2013, 2016],
           xticklabels=["11", "13", "16"], yticks=[0, 50, 100], yticklabels=["0%", "50%", "100%"])
    numeric(ax, "y")
    fig.legend(handles=[marker_handle(BLUE, "o", "Adjacent", False), marker_handle(PURPLE, "^", "Raw-first", False)],
               loc="center", bbox_to_anchor=(142 / 170, 25 / 210), ncol=2, frameon=False,
               fontsize=8, handlelength=.6, handletextpad=.25, columnspacing=.7)
    strip(fig, "2011–15 train · 2016 test", 142, 2.8, ha="center")
    save(fig, "Fig5", "Same-snapshot views, composition selection, and omitted history", {
        "panels": 12, "panel_labels": [f"({c})" for c in "abcdefghijkl"],
        "layout_rows_mm": [40, 31, 46, 46, 31], "named_pair_view_cases": cases,
        "pair_view_state_cells": 36, "not_entering_is_not_raw_absence": True,
        "annual_observation_budgets": year_counts, "composition_views": len(composition_receipt),
        "composition_rows": composition_receipt,
        "Full_to_common_selection": selection, "reference_effects": len(effect_receipts),
        "reference_effect_rows": effect_receipts,
        "effect_population_cohort_cells": 4, "views_per_effect_cell": list(VIEWS),
        "paired_view_contrasts": len(contrasts), "paired_view_contrast_rows": contrasts,
        "all_view_contrast_intervals_span_zero": True,
        "zero_spanning_intervals_do_not_establish_equivalence": True,
        "variation_summary": dispersion, "views_are_same_snapshot_not_independent_corpora": True,
        "history_audit": history_receipt, "history_omission_fractions": fractions,
        "annual_common_retention": retention, "full_common_raw_positive_preserved": True,
        "interval_definition": "stored conditional dependence percentile endpoints; separate from seed SD",
    })


if __name__ == "__main__":
    main()

