"""Shared physical-size drawing and source checks for the journal composites.

Only the supplied frozen scientific CSV/JSON tables are read.
No experiment, resampling, or remote data access is performed by these scripts.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent
PAPER = OUTPUT.parent
PREVIEW = Path(os.environ["TSND_FIGURE_PREVIEW_DIR"]) if os.environ.get("TSND_FIGURE_PREVIEW_DIR") else None
WIDTH = 170.0
BLUE, PURPLE, PINK, GRAY, CORAL = "#4C72B0", "#8172B3", "#DA8BC3", "#767676", "#C44E52"
INK, GRID = "#202020", "#DFDFDF"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.0,
    "axes.labelsize": 8.0, "axes.titlesize": 8.5,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "hatch.linewidth": .45,
})


def figure(height):
    """Create an untrimmed canvas at its final 170 mm insertion width."""
    return plt.figure(figsize=(WIDTH / 25.4, height / 25.4))


def axes(fig, x, y, width, height):
    """Add an axes using millimetres from the bottom-left of the canvas."""
    h = fig.get_figheight() * 25.4
    return fig.add_axes([x / WIDTH, y / h, width / WIDTH, height / h])


def label(fig, letter, title, x, y):
    """Place panel letter and title without consuming the data rectangle."""
    h = fig.get_figheight() * 25.4
    tag = letter if letter.startswith("(") else f"({letter})"
    fig.text(x / WIDTH, y / h, tag, fontsize=10, fontweight="bold", va="bottom")
    fig.text((x + 7.0) / WIDTH, (y + .1) / h, title, fontsize=8.5, va="bottom")


def strip(fig, text, x, y, **kwargs):
    """Draw a short section/encoding label in physical coordinates."""
    h = fig.get_figheight() * 25.4
    fig.text(x / WIDTH, y / h, text, fontsize=8, **kwargs)


def numeric(ax, which="x", signed=False):
    """Use a light common numerical grid and honest zero reference."""
    ax.grid(axis=which, color=GRID, linewidth=.45, zorder=0)
    ax.set_axisbelow(True)
    if signed:
        ax.axvline(0, color=INK, ls=(0, (3, 2)), lw=.65, zorder=1.6)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if abs(x) < 1e-10 else f"{x:+.2f}"))


def csv_rows(name):
    """Read complete supplied tables with numerical columns explicitly parsed."""
    with (HERE / name).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if key.startswith("n_") and value:
                row[key] = int(value)
            elif key in {"mean_delta_ap", "baseline_ap_mean", "augmented_ap_mean", "s_ap_mean", "tsnd_ap_mean", "seed_delta_sd", "conditional_dependence_p2_5", "conditional_dependence_p97_5"} and value:
                row[key] = float(value)
                if not np.isfinite(row[key]):
                    raise ValueError(f"Nonfinite {key} in {name}")
    return rows


def manuscript_table(label_name):
    """Read complete reported table values independently of printed-table selection."""
    data = json.loads((HERE / "reported_tables.json").read_text())
    if label_name not in data or not data[label_name]["rows"]:
        raise ValueError(f"Missing reported table values: {label_name}")
    return data[label_name]["rows"]


def number(cell):
    """Parse one scalar cell, rejecting ambiguous or empty contents."""
    values = re.findall(r"[-+]?\d+(?:\.\d+)?", cell.replace(",", ""))
    if len(values) != 1:
        raise ValueError(f"Expected one number in {cell!r}, got {values}")
    return float(values[0])


def interval(cell):
    """Parse the two already-reported percentile endpoints, without fitting."""
    values = re.findall(r"[-+]?\d+(?:\.\d+)?", cell.replace(",", ""))
    if len(values) != 2:
        raise ValueError(f"Expected two endpoints in {cell!r}, got {values}")
    return tuple(map(float, values))


def effect(ax, row, y, color=BLUE, marker="o", filled=True, size=3.8):
    """Draw stored exact percentile endpoints and a ten-seed mean increment."""
    lo, hi = row["conditional_dependence_p2_5"], row["conditional_dependence_p97_5"]
    point = row["mean_delta_ap"]
    if not lo <= point <= hi:
        raise ValueError("Mean outside supplied percentile endpoints")
    ax.plot([lo, hi], [y, y], color=color, lw=1, solid_capstyle="butt", zorder=2)
    ax.plot([lo, lo], [y - .065, y + .065], color=color, lw=.6)
    ax.plot([hi, hi], [y - .065, y + .065], color=color, lw=.6)
    ax.plot(point, y, marker=marker, color=color, mfc=color if filled else "white",
            ms=size, mew=.75, ls="none", zorder=3)


def stored_effect(delta, bounds):
    """Adapt a reported table contrast to the shared plotting helper."""
    return {"mean_delta_ap": delta, "conditional_dependence_p2_5": bounds[0],
            "conditional_dependence_p97_5": bounds[1]}


def marker_handle(color, marker, text, filled=True):
    return Line2D([], [], color=color, marker=marker, mfc=color if filled else "white",
                  ls="none", ms=4, mew=.75, label=text)


def save(fig, name, title, coverage):
    """Verify text/canvas bounds and preserve an actual final-size preview."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside, small = [], []
    visible = []
    for item in fig.findobj(match=matplotlib.text.Text):
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        if box.width == 0 or box.height == 0:
            continue
        visible.append(item.get_text())
        if box.x0 < -.8 or box.y0 < -.8 or box.x1 > fig.bbox.width + .8 or box.y1 > fig.bbox.height + .8:
            outside.append(item.get_text())
        if item.get_fontsize() < 8:
            small.append(item.get_text())
    if outside or small:
        raise ValueError(f"{name}: out-of-canvas={outside}; below-8pt={small}")
    fig.savefig(OUTPUT / f"{name}.pdf", metadata={"Title": title, "Author": ""})
    record = {"figure": name, "width_mm": WIDTH, "height_mm": round(fig.get_figheight() * 25.4, 3),
              "minimum_nominal_font_pt": 8, "canvas_text_bounds": "pass",
              "coverage": coverage, "visible_text": visible}
    if PREVIEW is not None:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW / f"{name}.png", dpi=200)
        (PREVIEW / f"{name}-coverage.json").write_text(json.dumps(record, indent=2) + "\n")
    plt.close(fig)
    print(f"Wrote {name}.pdf ({WIDTH:g} × {record['height_mm']:g} mm)")
