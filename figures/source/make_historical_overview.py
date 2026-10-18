"""Reconstruct Fig. 2 at the approved 170 × 198 mm publication size.

All quantitative marks come from the supplied historical controls, reported
manuscript tables, or frozen domain cases. The network positions are layout
coordinates and do not represent an embedding or a scientific distance.
"""

from __future__ import annotations

import json
from textwrap import fill

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import numpy as np

from dense_style import (
    HERE, BLUE, PURPLE, PINK, GRAY, CORAL, INK, GRID, axes, figure, label,
    manuscript_table, number, numeric, save,
)


HEIGHT = 198
MARGIN = 2
GUTTER = 3
COLUMN = (170 - 2 * MARGIN - 2 * GUTTER) / 3
ROW_HEIGHTS = (52, 45, 45, 43)
FAMILIES = {"control": GRAY, "structural": PURPLE, "semantic": PINK, "tsnd": BLUE}
SHAPES = {"control": "x", "structural": "^", "semantic": "s", "tsnd": "o"}
CMAP = LinearSegmentedColormap.from_list("pass_fraction", ["#FFFFFF", BLUE])


def panel(column, row, span=1):
    """Return the approved panel rectangle in physical millimetres."""
    top = HEIGHT - MARGIN - sum(ROW_HEIGHTS[:row]) - row * GUTTER
    return (MARGIN + column * (COLUMN + GUTTER), top - ROW_HEIGHTS[row],
            span * COLUMN + (span - 1) * GUTTER, ROW_HEIGHTS[row])


PANELS = {
    "a": panel(0, 0, 2), "b": panel(2, 0),
    "c": panel(0, 1), "d": panel(1, 1), "e": panel(2, 1),
    "f": panel(0, 2), "g": panel(1, 2), "h": panel(2, 2),
    "i": panel(0, 3), "j": panel(1, 3), "k": panel(2, 3),
}


def heading(fig, letter, title, subtitle=None):
    """Keep one restrained title hierarchy throughout the approved grid."""
    x, y, _, height = PANELS[letter]
    label(fig, letter, title, x, y + height - 4.3)
    if subtitle:
        physical_text(fig, x, y + height - 8.5, subtitle)


def physical_text(fig, x, y, text, **kwargs):
    """Position ordinary eight-point text without changing canvas geometry."""
    defaults = {"fontsize": 8, "color": INK, "va": "bottom"}
    defaults.update(kwargs)
    return fig.text(x / 170, y / HEIGHT, text, **defaults)


def _matrix(ax, values, *, percent=False):
    """Show every input cell with a printed value as a second encoding."""
    rows, columns = values.shape
    ax.pcolormesh(np.arange(columns + 1) - .5, np.arange(rows + 1) - .5,
                  values, vmin=0, vmax=1, cmap=CMAP,
                  edgecolors="white", linewidth=.65, rasterized=False)
    for i in range(rows):
        for j in range(columns):
            text = str(round(values[i, j] * 100)) if percent else f"{values[i, j]:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if values[i, j] > .85 else INK,
                    fontweight="bold" if not percent and j == columns - 1 else "normal")
    ax.set(xlim=(-.5, columns - .5), ylim=(rows - .5, -.5))
    ax.tick_params(which="both", length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def reported_baselines(data):
    """Read all nine reported AP/SD/ECE rows and verify supplied AP values."""
    reported = [r for r in manuscript_table("tab:generic-baselines")
                if len(r) == 4 and (r[0].startswith("B") or r[0].startswith("TSND"))]
    assert len(reported) == 9
    lookup = {r[0].split()[0]: {"ap": number(r[1]), "sd": number(r[2]), "ece": number(r[3])}
              for r in reported}
    assert set(lookup) == {f"B{i}" for i in range(1, 9)} | {"TSND"}
    for row in data["individual"]:
        assert abs(row["mean_ap"] - lookup[row["label"].split()[0]]["ap"]) < 6e-7
    return lookup


def draw_historical_panels(fig, data):
    """Render c–j without changing stored outcomes or SD definitions."""
    assert set(data) == {"surrogate", "native_response", "individual", "increments", "robustness"}
    assert (len(data["individual"]), len(data["increments"]), len(data["robustness"])) == (9, 7, 7)

    # c: percentages preserve the complete 8 × 6 matrix at eight points.
    heading(fig, "c", "Document probes", "Document surrogate · pass (%)")
    x, y, width, _ = PANELS["c"]
    values = np.asarray(data["surrogate"]["pass_fractions"], dtype=float)
    assert values.shape == (8, 6) and np.isfinite(values).all()
    assert ((0 <= values) & (values <= 1)).all()
    assert np.allclose(values * 100, np.rint(values * 100))
    ax = axes(fig, x + 13, y + 6.5, width - 13, 27.0)
    _matrix(ax, values, percent=True)
    ax.set_xticks(range(6), ["A1", "A2", "A3", "A7", "A8", "A9"])
    ax.set_yticks(range(8), ["RND", "SN", "Yin", "FTL", "SN ≤t", "Yin ≤t", "FTL ≤t", "TSND"])

    # d: failed relabel response remains below the original 0.8 criterion.
    heading(fig, "d", "Native response", "Criterion 0.8 · model-seed SD")
    x, y, width, _ = PANELS["d"]
    native = data["native_response"]
    assert native["means"][0] < .8
    ax = axes(fig, x + 14, y + 10, width - 14, 23)
    for index, (mean, sd) in enumerate(zip(native["means"], native["seed_sd"], strict=True)):
        failed = mean < .8
        ax.errorbar(mean, 2 - index, xerr=sd, color=CORAL if failed else BLUE,
                    marker="x" if failed else "o", ms=4, mew=.9,
                    elinewidth=.8, capsize=2, ls="none")
        ax.text(mean, 2 - index + .23, f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axvline(.8, color=INK, ls=(0, (3, 2)), lw=.7)
    ax.set(xlim=(0, 1.14), ylim=(-.5, 2.8), xticks=[0, .8, 1],
           yticks=[2, 1, 0], yticklabels=native["labels"])
    numeric(ax)
    physical_text(fig, x + width / 2, y + .3, f"Primary mean {native['primary_mean']:.3f}", ha="center")

    # e: stricter attribution differs from primary response.
    heading(fig, "e", "Attribution rules", "Same native perturbations")
    x, y, width, _ = PANELS["e"]
    component = [r for r in manuscript_table("tab:component-response") if r[0] in {"Strict", "Dominance"}]
    assert [r[0] for r in component] == ["Strict", "Dominance"]
    attribution = np.array([[number(cell) for cell in r[1:4]] +
                           [native["strict_mean"] if r[0] == "Strict" else native["dominance_mean"]]
                           for r in component])
    assert np.isclose(attribution[0, 3], .4888888888888889)
    ax = axes(fig, x + 14, y + 16, width - 14, 14)
    _matrix(ax, attribution)
    ax.set_xticks(range(4), ["Relab.", "Pair", "Drift", "Mean"])
    ax.set_yticks(range(2), ["Strict", "Domin."])
    ax.axvline(2.5, color="white", lw=1.6)
    physical_text(fig, x + width / 2, y + 2.7, "Coupled components", ha="center")

    # f/g: all controls; family also uses a non-colour marker shape.
    lookup = reported_baselines(data)
    rows = sorted(data["individual"], key=lambda r: r["mean_ap"])
    heading(fig, "f", "Standalone ranking", "Generic · 2021 · model-seed SD")
    x, y, width, _ = PANELS["f"]
    ax = axes(fig, x + 17, y + 7.5, width - 17, 27.3)
    names = {"B1": "B1 ctrl", "B2": "B2 degree", "B3": "B3 TF-IDF", "B4": "B4 atyp.",
             "B5": "B5 SP2", "B6": "B6 static", "B7": "B7 I4C", "B8": "B8 struct.", "TSND": "TSND"}
    for row_index, row in enumerate(rows):
        code, family = row["label"].split()[0], row["family"]
        point = lookup[code]
        ax.errorbar(point["ap"], row_index, xerr=point["sd"], marker=SHAPES[family],
                    color=FAMILIES[family], mfc=FAMILIES[family], ms=3.6,
                    elinewidth=.7, capsize=1.5, mew=.8, ls="none")
        ax.text(point["ap"] + .018, row_index, f"{point['ap']:.3f}", va="center", fontsize=8)
    ax.set(yticks=range(9), yticklabels=[names[r["label"].split()[0]] for r in rows],
           xlim=(0, .68), ylim=(-.55, 8.55), xticks=[0, .2, .4, .6], xlabel="AP")
    ax.tick_params(axis="y", length=0, pad=2)
    numeric(ax)

    heading(fig, "g", "Ranking / calibration", "Generic · 2021 · means")
    x, y, width, _ = PANELS["g"]
    ax = axes(fig, x + 9, y + 7.5, width - 10, 27.3)
    text_positions = {
        "B1": (.025, .33), "B2": (.021, .48), "B3": (.016, .145),
        "B4": (.0008, .22), "B5": (.018, .25), "B6": (.016, .035),
        "B7": (.001, .57), "B8": (.038, .59), "TSND": (.0008, .32),
    }
    for row in rows:
        code, family = row["label"].split()[0], row["family"]
        point = lookup[code]
        ax.plot(point["ece"], point["ap"], marker=SHAPES[family], color=FAMILIES[family], ms=3.7, ls="none")
        ax.annotate(code, (point["ece"], point["ap"]), xytext=text_positions[code],
                    textcoords="data", ha="left", va="center", fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": GRAY, "lw": .45,
                                "shrinkA": 1.5, "shrinkB": 2})
    ax.set(xlim=(0, .05), ylim=(0, .65), xticks=[0, .02, .04],
           yticks=[0, .2, .4, .6], xlabel="ECE", ylabel="AP")
    numeric(ax)

    # h: preserve adverse oncology result and both seed-SD conventions.
    heading(fig, "h", "Structural increments", "SD divisors: 9 (●10), 3 (□3)")
    x, y, width, _ = PANELS["h"]
    ax = axes(fig, x + 17, y + 7.5, width - 17, 27.3)
    positions = [7.2, 6.2, 5.2, 4.2, 2.2, 1.2, .2]
    for position, row in zip(positions, data["increments"], strict=True):
        assert row["n_seeds"] in {3, 10}
        ten = row["n_seeds"] == 10
        assert row["sd_divisor"] == (9 if ten else 3)
        ax.errorbar(row["increment"], position, xerr=row["seed_sd"],
                    color=CORAL if row["increment"] < 0 else BLUE,
                    marker="o" if ten else "s", mfc=BLUE if ten else "white", mew=.85,
                    ms=3.6, elinewidth=.8, capsize=1.7, ls="none")
    ax.text(-.005, 8.2, "Over B2+B7", fontsize=8, va="center")
    ax.text(-.005, 3.25, "Over B2+B7+B8", fontsize=8, va="center")
    ax.axhline(3.7, color=GRID, lw=.55)
    ax.set(yticks=positions,
           yticklabels=["Onco GF", "Onco adj.", "Generic", "Cardiol.", "Generic", "Oncology", "Cardiol."],
           ylim=(-.6, 8.9), xlim=(-.006, .046), xticks=[0, .02, .04], xlabel="ΔAP")
    ax.tick_params(axis="y", length=0, pad=2)
    numeric(ax, signed=True)

    # i/j: historical rolling years and a joint block, with original seed SD.
    rolling = [r for r in data["robustness"] if r["group"] == "Test year"]
    assert [r["label"] for r in rolling] == ["2013", "2014", "2015", "2016"]
    heading(fig, "i", "Rolling test years", "Oncology · 3-seed SD (divisor 3)")
    x, y, width, _ = PANELS["i"]
    ax = axes(fig, x + 13, y + 7.5, width - 14, 24.5)
    ax.errorbar(range(4), [r["increment"] for r in rolling],
                yerr=[r["seed_sd"] for r in rolling], color=BLUE, marker="s",
                mfc="white", ms=3.8, lw=.8, capsize=2)
    ax.set(xlim=(-.35, 3.35), ylim=(0, .03), xticks=range(4),
           xticklabels=[r["label"] for r in rolling], yticks=[0, .01, .02, .03],
           ylabel="ΔAP", xlabel="Test year")
    numeric(ax, "y")

    ablations = [r for r in data["robustness"] if r["group"] == "Feature block"]
    assert [r["label"] for r in ablations] == ["Full", "Without LE", "Without RC+P"]
    assert all(r["n_seeds"] == 3 and r["sd_divisor"] == 3 for r in rolling + ablations)
    heading(fig, "j", "Joint-block ablation", "ΔAP over B2+B7")
    x, y, width, _ = PANELS["j"]
    ax = axes(fig, x + 17, y + 14, width - 18, 18)
    for row_index, row in enumerate(reversed(ablations)):
        ax.errorbar(row["increment"], row_index, xerr=row["seed_sd"], color=BLUE,
                    marker="s", mfc="white", ms=3.8, elinewidth=.8, capsize=1.7, ls="none")
    ax.set(yticks=range(3), yticklabels=["No RC+P", "No LE", "Full"], ylim=(-.6, 2.6),
           xlim=(-.004, .028), xticks=[0, .01, .02], xlabel="ΔAP")
    ax.tick_params(axis="y", length=0, pad=2)
    numeric(ax, signed=True)
    physical_text(fig, x + width / 2, y + 1.8, "3-seed SD; divisor 3", ha="center")

    return {"surrogate_cells": int(values.size), "native_primary_components": 3,
            "native_failed_relabel_mean": native["means"][0], "native_criterion": .8,
            "strict_dominance_cells_including_means": int(attribution.size),
            "strict_mean": native["strict_mean"], "dominance_mean": native["dominance_mean"],
            "individual_feature_sets": len(rows), "standalone_AP_ECE_points": len(rows),
            "cross_domain_increments": len(data["increments"]), "rolling_years": len(rolling),
            "ablation_rows": len(ablations), "historical_json_coverage": "all",
            "negative_cross_domain_increments": sum(r["increment"] < 0 for r in data["increments"]),
            "native_not_identification": True, "SD_protocols": {"10_seed": 9, "3_seed": 3},
            "surrogate_display_scale": "integer percent; exact source hundredths",
            "surrogate_key": {"SN": "SemNovel", "FTL": "FastTextLOF", "≤t": "time-restricted"}}


def draw_domain_panels(fig, domain):
    """Render the verified frozen cases, including all three fixed rank examples."""
    if domain is None:
        raise ValueError("Fig2 a/b/k require verified domain_cases.json; no placeholder evidence is exported")
    data = domain["fig2"]
    network = data["network"]
    nodes, edges = network["nodes"], network["edges"]
    names = [node["name"] for node in nodes]
    assert len(names) == len(set(names)) == 12
    assert len(network["focus_pairs"]) == 3
    assert {case["component"] for case in data["component_cases"]} == {"LE", "NS", "RC"}
    assert len(data["rank_cases"]["cases"]) == 3
    focus = {key: chr(65 + index) for index, key in enumerate(network["focus_pairs"])}
    assert set(focus) == {case["key"] for case in data["rank_cases"]["cases"]}

    # a: all local graph edges are drawn. Coordinates only separate the labels.
    heading(fig, "a", "Local MeSH co-occurrence", "2015 → 2016 · oncology · DF20 · seed 0")
    x, y, width, _ = PANELS["a"]
    ax = axes(fig, x, y + 5.2, width, 35)
    left = [node for node in nodes if node["anchor"]]
    right = [node for node in nodes if not node["anchor"]]
    assert len(left) == len(right) == 6
    positions = {}
    for side, collection in ((.36, left), (.65, right)):
        for node, vertical in zip(collection, np.linspace(.95, .05, 6), strict=True):
            positions[node["name"]] = (side, vertical)
    styles = {"retained": {"color": GRAY, "ls": "-", "alpha": .32},
              "entering": {"color": BLUE, "ls": (0, (3, 2)), "alpha": .75},
              "prior_only": {"color": GRAY, "ls": (0, (1, 2)), "alpha": .8}}
    edge_counts = {state: 0 for state in styles}
    edge_keys, rendered_focus = set(), set()
    for edge in sorted(edges, key=lambda e: e["state"] != "retained"):
        source, target, state = edge["source"], edge["target"], edge["state"]
        assert source in positions and target in positions and source != target
        assert state == ("retained" if edge["count_2015"] > 0 and edge["count_2016"] > 0
                         else "entering" if edge["count_2016"] > 0 else "prior_only")
        edge_counts[state] += 1
        start, end = positions[source], positions[target]
        radius = -.35 if start[0] == end[0] and start[0] < .5 else .25 if start[0] == end[0] else 0
        key = "|".join(sorted((source, target)))
        assert key not in edge_keys
        edge_keys.add(key)
        is_focus = key in focus
        if is_focus:
            rendered_focus.add(key)
        style = styles[state]
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-", mutation_scale=1,
                                     connectionstyle=f"arc3,rad={radius}", shrinkA=2.3, shrinkB=2.3,
                                     lw=1.25 if is_focus else .55,
                                     color=INK if is_focus else style["color"],
                                     linestyle=style["ls"], alpha=1 if is_focus else style["alpha"],
                                     zorder=2 if is_focus else 1))
    assert rendered_focus == set(focus)
    for node in nodes:
        horizontal, vertical = positions[node["name"]]
        entering = node["entering_term"]
        ax.plot(horizontal, vertical, marker="s" if entering else "o", ls="none",
                ms=4.1, mew=.8, mfc=BLUE if entering else "white", mec=INK, zorder=4)
        ax.text(horizontal - .035 if horizontal < .5 else horizontal + .035, vertical,
                fill(node["name"], 22), ha="right" if horizontal < .5 else "left",
                va="center", fontsize=8, linespacing=1.05, zorder=5)
    # The letters bind the same real pairs to the ranking panel without long repeats.
    for key, location in zip(network["focus_pairs"], [( .402, .77), (.439, .67), (.463, .40)], strict=True):
        ax.text(*location, focus[key], ha="center", va="center", fontsize=8, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": .3}, zorder=6)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    for left_mm, state, text in [(0, "retained", "retained"), (29, "entering", "entering"),
                                 (58, "prior_only", "2015 only")]:
        ax_legend = axes(fig, x + left_mm, y + .7, 5.4, 2.5)
        ax_legend.plot([0, 1], [.5, .5], color=INK, ls=styles[state]["ls"], lw=.8)
        ax_legend.axis("off")
        physical_text(fig, x + left_mm + 6.5, y + .5, text)
    physical_text(fig, x + 87, y + .5, "□ new term")

    # b: these values apply to different objects and have no shared radar axis.
    heading(fig, "b", "Typed component readings", "2016 · DF20 · seed 0")
    x, y, width, height = PANELS["b"]
    component_values = {}
    for index, case in enumerate(data["component_cases"]):
        top = y + height - 14 - index * 13.2
        component_values[case["component"]] = case["value"]
        physical_text(fig, x, top, f"{case['component']} · {case['kind']}")
        physical_text(fig, x + width, top, f"{case['value']:.4f}", ha="right", fontweight="bold")
        physical_text(fig, x, top - 4.0, fill(case["key"].replace("|", " × "), 32), va="bottom", linespacing=1.05)
        metadata = f"DF {case['current_df']}" if "current_df" in case else (
            f"P {case['proximity']:.3f} · Z {case['degree_product_log']:.3f}")
        physical_text(fig, x, top - 7.6, metadata)

    # k: a common absolute-rank scale, with worse and modest changes retained.
    rank_data = data["rank_cases"]
    heading(fig, "k", "Selected pair ranks", f"○ S → ■ S+TSND · N = {rank_data['n_fronts']:,}")
    x, y, width, _ = PANELS["k"]
    ax = axes(fig, x + 13, y + 7.5, width - 13, 25)
    history_names = {"raw_first_observed": "RF", "graph_first_with_raw_history": "GF",
                     "graph_reentry_gap_ge2": "RE"}
    rank_changes = []
    for index, case in enumerate(rank_data["cases"]):
        vertical = 2 - index
        before, after = case["rank_s"], case["rank_tsnd"]
        assert 1 <= before <= rank_data["n_fronts"] and 1 <= after <= rank_data["n_fronts"]
        color = CORAL if after > before else BLUE
        ax.plot([before, after], [vertical + .04, vertical - .04], color=color, lw=.85)
        ax.plot(before, vertical + .04, marker="o", ms=3.8, mew=.85, mfc="white", mec=INK, ls="none")
        ax.plot(after, vertical - .04, marker="s", ms=3.8, mew=.85, mfc=color, mec=color, ls="none")
        ax.text(30000, vertical + .26, f"{before:,} → {after:,}", ha="center", va="bottom", fontsize=8)
        rank_changes.append({"case": focus[case["key"]], "history_class": case["history_class"],
                             "rank_s": before, "rank_tsnd": after, "label": case["label"]})
    ax.set(xlim=(1, 60000), ylim=(-.55, 2.85), xticks=[1, 25000, 50000], xticklabels=["1", "25k", "50k"],
           yticks=[2, 1, 0], yticklabels=[f"{focus[case['key']]} · {history_names[case['history_class']]}"
                                        for case in rank_data["cases"]], xlabel="Rank (lower is earlier)")
    ax.tick_params(axis="y", length=0, pad=2)
    numeric(ax)
    return {"domain_case_status": "all_frozen_cases_rendered", "network_terms": len(nodes),
            "network_edges": len(edges), "network_edge_states": edge_counts,
            "network_focus_pairs": len(focus), "network_layout": "fixed label layout; no distance meaning",
            "typed_component_cases": len(component_values), "component_case_values": component_values,
            "rank_cases": rank_changes, "rank_denominator": rank_data["n_fronts"],
            "rank_tie_policy": rank_data["tie_policy"], "rank_outcomes_used_for_selection": False,
            "fixed_rank_case_worsens": any(c["rank_tsnd"] > c["rank_s"] for c in rank_changes),
            "all_rank_case_labels": [c["label"] for c in rank_changes]}


def build_figure(data, domain=None, *, include_domain=True):
    """Build the complete figure, or c–j for a private preliminary layout check."""
    fig = figure(HEIGHT)
    coverage = draw_historical_panels(fig, data)
    coverage.update({"panels": 11 if include_domain else 8, "planned_panels": 11,
                     "panel_letters": list(PANELS) if include_domain else list("cdefghij"),
                     "domain_case_status": "required" if include_domain else "not_rendered_in_private_draft"})
    if include_domain:
        coverage.update(draw_domain_panels(fig, domain))
    return fig, coverage


def main():
    data = json.loads((HERE / "historical_controls.json").read_text())
    path = HERE / "domain_cases.json"
    domain = json.loads(path.read_text()) if path.is_file() else None
    fig, coverage = build_figure(data, domain)
    save(fig, "Fig2", "Historical measurement and ranking controls", coverage)


if __name__ == "__main__":
    main()
