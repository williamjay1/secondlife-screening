"""Publication-grade figures for the second-life screening manuscript.

Every figure is written as a vector PDF (typographically scalable, equivalent to
900 dpi and above when rasterised) plus a 1200 dpi PNG preview.  Each figure is
passed through the layout audit in :mod:`figstyle`, which measures real rendered
text boxes and drawn line segments, so text-text overlap, text-line crossing and
out of canvas text are reported instead of assumed absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle as fs
from figstyle import (
    CLASS_COLORS,
    CLASS_LABELS,
    DOUBLE_COL,
    PALETTE,
    panel_label,
    save_figure,
    short_stage_labels,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

TARGET_LABELS = {
    "any_second_life": "Source-defined second-life eligibility",
    "survives_100_at_70pct": "100-cycle retention at 70 percent",
}

WANG_LABELS = {
    "capacity_only": "Capacity only",
    "explicit_snapshot": "Five explicit diagnostics",
    "implicit_single_cycle": "108 implicit features",
    "explicit_plus_implicit": "Explicit and implicit",
}


def num(series: pd.Series) -> pd.Series:
    """Numeric conversion for source files that use a comma decimal separator.

    The released Salinas semicolon delimited files store decimals with a comma,
    so a direct numeric cast would silently yield all NaN and produce an empty
    panel. Values are normalised before the cast and the conversion is verified
    by the caller through :func:`require_finite`.
    """
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def require_finite(frame: pd.DataFrame, columns: list[str], context: str, min_fraction: float = 0.90) -> None:
    """Stop the build when a plot input is effectively empty.

    A figure built from an all NaN block renders as bare axes, which passes a
    text layout audit while being useless. This guard fails loudly instead.
    """
    block = frame[columns].apply(num)
    finite = int(np.isfinite(block.to_numpy(dtype=float)).sum())
    total = block.shape[0] * block.shape[1]
    fraction = finite / total if total else 0.0
    if fraction < min_fraction:
        raise SystemExit(
            f"{context}: only {fraction:.1%} of {total} values are numeric "
            f"({finite} finite). Check the decimal separator or the column mapping."
        )
    print(f"    data guard {context}: {fraction:.1%} of {total} values numeric")


def load_metrics() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "salinas_grouped_metrics.csv")


def load_frontier() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "salinas_resource_frontier_any_second_life.csv")


def load_trajectories() -> tuple[pd.DataFrame, list[str]]:
    perf = pd.read_csv(DATA / "salinas_performance.csv", sep=";", dtype=str)
    frame = pd.read_csv(DATA / "salinas_2021_decision_frame.csv", dtype=str)
    cols = [str(i) for i in range(1, 101)]
    require_finite(perf, cols, "salinas_performance capacity cycles")

    perf[cols] = perf[cols].apply(num)
    key = frame[["Code", "N Zell", "y_int", "cap_cycle_100", "n_cycles_observed"]].copy()
    merged = key.merge(perf[["Code", "N Zell"] + cols], on=["Code", "N Zell"], how="inner", validate="one_to_one")
    merged["cap_cycle_100"] = num(merged["cap_cycle_100"])
    merged["n_cycles_observed"] = num(merged["n_cycles_observed"])
    merged["y_int"] = num(merged["y_int"]).astype(int)
    if len(merged) != len(frame):
        raise SystemExit(f"trajectory merge lost rows: {len(merged)} of {len(frame)}")
    for cls in (0, 1, 2):
        subset = merged.loc[merged.y_int == cls, cols]
        if subset.empty or not np.isfinite(subset.to_numpy(dtype=float)).any():
            raise SystemExit(f"class {cls} has no finite trajectory values after the merge")
    return merged, cols


# ---------------------------------------------------------------------------
# Figure 1. Screening information timeline
# ---------------------------------------------------------------------------
def figure_timeline() -> None:
    fig = plt.figure(figsize=(DOUBLE_COL, 88 * fs.MM))
    grid = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 1.25], hspace=0.72, left=0.055, right=0.985, top=0.905, bottom=0.045
    )
    ax = fig.add_subplot(grid[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Screening information timeline for retired lithium-ion cells", pad=9)

    stages = [
        ("First-life\noperation", "BMS logs,\ncycle count", PALETTE["history"]),
        ("Recovery and\ninspection", "Cell mass,\nopen-circuit voltage", PALETTE["physical"]),
        ("Early electrical\ndiagnostics", "Transient and\nshort-circuit response", PALETTE["early"]),
        ("Full electrical\ncharacterization", "Incremental capacity,\ndischarge energy", PALETTE["full"]),
        ("Second-life\ncycling", "Retention at\ncycle 100", PALETTE["muted"]),
        ("Delayed storage\nmeasurement", "Float voltage after\nthree months", PALETTE["delayed"]),
    ]
    n = len(stages)
    width, gap, left0 = 1.30, 0.335, 0.10
    top_box, box_h = 0.74, 0.26
    for i, (name, detail, colour) in enumerate(stages):
        x0 = left0 + i * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x0, top_box),
                width,
                box_h,
                boxstyle="round,pad=0.0,rounding_size=0.045",
                linewidth=0.7,
                edgecolor=colour,
                facecolor=colour,
                alpha=0.15,
                zorder=2,
            )
        )
        ax.text(
            x0 + width / 2,
            top_box + box_h / 2,
            name,
            ha="center",
            va="center",
            fontsize=6.4,
            fontweight="bold",
            color=PALETTE["ink"],
            linespacing=1.3,
            zorder=3,
        )
        ax.text(
            x0 + width / 2,
            0.555,
            detail,
            ha="center",
            va="top",
            fontsize=5.9,
            color=PALETTE["muted"],
            linespacing=1.35,
        )
        if i < n - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x0 + width + 0.03, top_box + box_h / 2),
                    (x0 + width + gap - 0.03, top_box + box_h / 2),
                    arrowstyle="-|>",
                    mutation_scale=5.5,
                    linewidth=0.7,
                    color=PALETTE["muted"],
                    shrinkA=0,
                    shrinkB=0,
                )
            )

    band_y, band_h = 0.235, 0.075
    bands = [
        (0, 2, "Available before any paid electrical test", PALETTE["history"]),
        (2, 4, "Paid electrical testing with recorded duration", PALETTE["early"]),
        (4, 6, "Observed only after the screening decision", PALETTE["muted"]),
    ]
    for i0, i1, text, colour in bands:
        xa = left0 + i0 * (width + gap)
        xb = left0 + (i1 - 1) * (width + gap) + width
        ax.add_patch(
            Rectangle((xa, band_y), xb - xa, band_h, facecolor=colour, edgecolor="none", alpha=0.15, zorder=2)
        )
        ax.text(
            (xa + xb) / 2,
            band_y + band_h / 2,
            text,
            ha="center",
            va="center",
            fontsize=5.9,
            color=PALETTE["ink"],
            zorder=3,
        )

    dec_y = 0.165
    ax.add_patch(
        FancyArrowPatch(
            (left0, dec_y),
            (left0 + 2 * (width + gap) - gap + 0.02, dec_y),
            arrowstyle="<|-|>",
            mutation_scale=5.5,
            linewidth=0.7,
            color=PALETTE["history"],
            shrinkA=0,
            shrinkB=0,
        )
    )
    ax.text(
        left0 + (2 * (width + gap) - gap) / 2,
        0.095,
        "Screening decision window",
        ha="center",
        va="center",
        fontsize=6.1,
        fontweight="bold",
        color=PALETTE["history"],
    )

    # ---------------- panel b: block by property matrix -------------------
    bx = fig.add_subplot(grid[1])
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 1)
    bx.axis("off")
    bx.set_title("Information blocks, availability and role in the evaluation", pad=7)

    blocks = [
        ("Battery history and passport-compatible fields", "Pre-existing", "No", "Deployable screening input"),
        ("Physical observations after recovery", "Pre-existing", "No", "Deployable screening input"),
        ("Early electrical diagnostics", "Pre-test", "Yes", "Deployable screening input"),
        ("Full electrical characterization", "Post-completion", "Yes", "Discrimination benchmark"),
        ("Delayed storage measurement", "Delayed", "Yes", "Chronology sensitivity only"),
        ("Pack yield summary field", "Post-test", "No", "Excluded as outcome-derived"),
    ]
    x_block, x_time, x_paid, x_role = 0.012, 0.452, 0.567, 0.652
    header_y = 0.905
    row_h = 0.132
    bx.text(x_block, header_y, "Information block", ha="left", va="center", fontsize=6.3, fontweight="bold")
    bx.text(x_time, header_y, "Timing", ha="center", va="center", fontsize=6.3, fontweight="bold")
    bx.text(x_paid, header_y, "Paid test", ha="center", va="center", fontsize=6.3, fontweight="bold")
    bx.text(x_role, header_y, "Role in the evaluation", ha="left", va="center", fontsize=6.3, fontweight="bold")
    bx.plot([0.006, 0.994], [header_y - 0.072, header_y - 0.072], color=PALETTE["ink"], linewidth=0.6)

    role_colour = {
        "Deployable screening input": PALETTE["history"],
        "Discrimination benchmark": PALETTE["full"],
        "Chronology sensitivity only": PALETTE["muted"],
        "Excluded as outcome-derived": "#B03A2E",
    }
    for i, (name, timing, paid, role) in enumerate(blocks):
        y = header_y - 0.072 - (i + 0.60) * row_h
        if i % 2 == 0:
            bx.add_patch(
                Rectangle((0.004, y - 0.061), 0.992, 0.122, facecolor="#F4F4F4", edgecolor="none", zorder=0)
            )
        bx.text(x_block, y, name, ha="left", va="center", fontsize=6.0, color=PALETTE["ink"], zorder=2)
        bx.text(x_time, y, timing, ha="center", va="center", fontsize=6.0, color=PALETTE["muted"], zorder=2)
        bx.text(x_paid, y, paid, ha="center", va="center", fontsize=6.0, color=PALETTE["muted"], zorder=2)
        bx.add_patch(
            Rectangle((x_role - 0.009, y - 0.048), 0.006, 0.096, facecolor=role_colour[role], edgecolor="none", zorder=2)
        )
        bx.text(x_role + 0.010, y, role, ha="left", va="center", fontsize=6.0, color=PALETTE["ink"], zorder=2)

    empty = fs.require_panel_content(fig, {"1a timeline": ax, "1b block matrix": bx}, min_ink=0.02)
    if empty:
        raise SystemExit(f"Figure 1 has an empty panel: {empty}")
    report = save_figure(fig, "fig1_screening_information_timeline", OUT, name="Figure 1")
    print(report.render())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2. Delayed performance trajectories
# ---------------------------------------------------------------------------
def figure_trajectories() -> None:
    merged, cols = load_trajectories()
    x = np.arange(1, 101)
    fig = plt.figure(figsize=(DOUBLE_COL, 74 * fs.MM))
    grid = fig.add_gridspec(
        1, 2, width_ratios=[1.60, 1.0], wspace=0.40, left=0.078, right=0.988, top=0.88, bottom=0.315
    )
    ax = fig.add_subplot(grid[0])
    counts = {}
    for cls in [0, 1, 2]:
        arr = merged.loc[merged.y_int == cls, cols].to_numpy(float)
        counts[cls] = arr.shape[0]
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(arr, axis=0)
            sd = np.nanstd(arr, axis=0, ddof=1)
        ax.fill_between(x, mean - sd, mean + sd, color=CLASS_COLORS[cls], alpha=0.16, linewidth=0)
        ax.plot(x, mean, color=CLASS_COLORS[cls], linewidth=1.2, label=f"{CLASS_LABELS[cls]} (n = {counts[cls]})")
    ax.plot([1, 100], [0.70, 0.70], color=PALETTE["ink"], linewidth=0.7, linestyle=(0, (4, 2.5)))
    ax.text(98.5, 0.725, "70 percent retention", ha="right", va="bottom", fontsize=5.9, color=PALETTE["ink"])
    ax.set_xlabel("Second-life cycle")
    ax.set_ylabel("Relative capacity")
    ax.set_xlim(1, 100)
    ax.set_ylim(0.10, 1.12)
    ax.set_yticks(np.arange(0.2, 1.01, 0.2))
    ax.set_title("Delayed capacity trajectories by source-defined class")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.30),
        ncol=3,
        handlelength=1.4,
        columnspacing=1.5,
        borderaxespad=0,
    )
    panel_label(ax, "a", dx=-0.115, dy=1.025)

    bx = fig.add_subplot(grid[1])
    positions = [0, 1, 2]
    rng = np.random.default_rng(20260910)
    for cls, pos in zip([0, 1, 2], positions):
        vals = merged.loc[merged.y_int == cls, "cap_cycle_100"].astype(float).dropna().to_numpy()
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.13, 0.13, size=vals.size)
        bx.scatter(
            np.full(vals.size, pos) + jitter, vals, s=6, color=CLASS_COLORS[cls], alpha=0.55, linewidths=0, zorder=3
        )
        bx.plot([pos - 0.28, pos + 0.28], [float(np.mean(vals))] * 2, color=PALETTE["ink"], linewidth=1.1, zorder=4)
    bx.plot([-0.5, 2.5], [0.70, 0.70], color=PALETTE["ink"], linewidth=0.7, linestyle=(0, (4, 2.5)))
    complete = int((merged.n_cycles_observed == 100).sum())
    bx.set_xticks(positions, [f"{CLASS_LABELS[c]}\n(n = {counts[c]})" for c in [0, 1, 2]])
    bx.set_ylabel("Relative capacity at cycle 100")
    bx.set_ylim(0.10, 1.10)
    bx.set_yticks(np.arange(0.2, 1.01, 0.2))
    bx.set_xlim(-0.58, 2.58)
    bx.set_title("Capacity at cycle 100")
    bx.yaxis.set_label_coords(-0.30, 0.5)
    panel_label(bx, "b", dx=-0.30, dy=1.025)

    empty = fs.require_panel_content(fig, {"2a trajectories": ax, "2b cycle 100 capacity": bx})
    if empty:
        raise SystemExit(f"Figure 2 has an empty panel: {empty}")
    report = save_figure(fig, "fig2_delayed_capacity_trajectories", OUT, name="Figure 2")
    print(report.render())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3. Screening performance progression
# ---------------------------------------------------------------------------
def figure_progression(metrics: pd.DataFrame) -> None:
    """Screening performance as information accumulates, with the decision errors."""
    ba = metrics[metrics.metric == "balanced_accuracy"].set_index(["target", "stage"])
    stages = [
        "passport",
        "passport_plus_physical",
        "early_diagnostics",
        "full_diagnostics",
        "full_plus_delayed_x29",
    ]
    labels = short_stage_labels()
    fig = plt.figure(figsize=(DOUBLE_COL, 106 * fs.MM))
    grid = fig.add_gridspec(
        2, 1, height_ratios=[1.28, 1.0], hspace=0.58, left=0.175, right=0.985, top=0.90, bottom=0.125
    )

    # ---------------- panel a: grouped horizontal bars ---------------------
    ax = fig.add_subplot(grid[0])
    series = [
        ("any_second_life", PALETTE["history"], 0.195),
        ("survives_100_at_70pct", PALETTE["early"], -0.195),
    ]
    bar_h = 0.30
    for target, colour, dy in series:
        for i, stage in enumerate(stages):
            row = ba.loc[(target, stage)]
            y = (len(stages) - 1 - i) + dy
            full_estimate = stage in {"full_diagnostics", "full_plus_delayed_x29"}
            ax.barh(
                y,
                row.estimate,
                height=bar_h,
                color=colour,
                alpha=0.95 if full_estimate else 0.72,
                linewidth=0,
                zorder=3,
            )
            ax.plot(
                [row.ci_low, row.ci_high],
                [y, y],
                color=PALETTE["ink"],
                linewidth=0.75,
                zorder=4,
            )
            for bound in (row.ci_low, row.ci_high):
                ax.plot([bound, bound], [y - 0.115, y + 0.115], color=PALETTE["ink"], linewidth=0.75, zorder=4)
            ax.text(
                1.014,
                y,
                f"{row.estimate:.3f}",
                ha="left",
                va="center",
                fontsize=5.8,
                color=PALETTE["ink"],
                zorder=5,
            )
    ax.axvspan(1.005, 1.115, color=PALETTE["band"], alpha=0.75, linewidth=0, zorder=0)
    ax.plot([0.36, 1.06], [3.5, 3.5], color=PALETTE["grid"], linewidth=0.6, zorder=1)
    ax.text(0.985, len(stages) - 0.36, "observed after the paid test", ha="right", va="center", fontsize=5.8,
            color=PALETTE["muted"], zorder=5)
    ax.set_yticks(np.arange(len(stages))[::-1], [labels[s] for s in stages])
    ax.set_ylim(-0.62, len(stages) - 0.22)
    ax.set_xlim(0.36, 1.115)
    ax.set_xticks(np.arange(0.4, 1.01, 0.1))
    ax.set_xlabel("Balanced accuracy with 95 percent pack-bootstrap interval", labelpad=1.0)
    ax.set_title("Screening performance as information accumulates", pad=10)
    handles = [
        Line2D([], [], color=PALETTE["history"], linewidth=4.0, label=TARGET_LABELS["any_second_life"]),
        Line2D([], [], color=PALETTE["early"], linewidth=4.0, label=TARGET_LABELS["survives_100_at_70pct"]),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.255),
        ncol=2,
        handlelength=1.5,
        columnspacing=1.6,
        borderaxespad=0,
    )
    panel_label(ax, "a", dx=-0.112, dy=1.018)

    # ---------------- panel b: decision errors -----------------------------
    bx = fig.add_subplot(grid[1])
    confusion = {
        "passport": (10, 17),
        "passport_plus_physical": (11, 11),
        "early_diagnostics": (7, 6),
        "full_diagnostics": (2, 1),
    }
    stages_c = list(confusion.keys())
    y = np.arange(len(stages_c))[::-1].astype(float)
    fa = np.array([confusion[s][0] for s in stages_c], dtype=float)
    fr = np.array([confusion[s][1] for s in stages_c], dtype=float)
    bx.barh(y + 0.19, fa, height=0.30, color=PALETTE["physical"], label="False accepts", zorder=3)
    bx.barh(y - 0.19, fr, height=0.30, color=PALETTE["full"], label="False rejects", zorder=3)
    for yi, va, vr in zip(y, fa, fr):
        bx.text(va + 0.55, yi + 0.19, f"{int(va)}", ha="left", va="center", fontsize=5.9, color=PALETTE["ink"], zorder=5)
        bx.text(vr + 0.55, yi - 0.19, f"{int(vr)}", ha="left", va="center", fontsize=5.9, color=PALETTE["ink"], zorder=5)
    bx.set_yticks(y, [labels[s] for s in stages_c])
    bx.set_ylim(-0.72, len(stages_c) - 0.28)
    bx.set_xlim(0, 22.5)
    bx.set_xticks(np.arange(0, 21, 5))
    bx.set_xlabel("Cells in the held-out replay (n = 100)")
    bx.set_title("Decision errors for second-life eligibility", pad=10)
    bx.legend(loc="upper right", bbox_to_anchor=(1.0, 0.34), ncol=1, handlelength=1.2)
    panel_label(bx, "b", dx=-0.112, dy=1.030)

    empty = fs.require_panel_content(fig, {"3a progression": ax, "3b decision errors": bx})
    if empty:
        raise SystemExit(f"Figure 3 has an empty panel: {empty}")
    report = save_figure(fig, "fig3_screening_performance_progression", OUT, name="Figure 3")
    print(report.render())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4. Screening resource and risk frontier
# ---------------------------------------------------------------------------
def figure_frontier(front: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(DOUBLE_COL, 78 * fs.MM))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 0.085], hspace=0.34, left=0.092, right=0.985, top=0.855, bottom=0.075
    )
    grid = outer[0].subgridspec(1, 2, width_ratios=[1.30, 1.0], wspace=0.40)
    ax = fig.add_subplot(grid[0])
    stage_style = {
        "passport": ("Battery history", PALETTE["history"], "-"),
        "passport_plus_physical": ("History and physical observations", PALETTE["physical"], (0, (5, 2))),
        "early_diagnostics": ("History, physical and early diagnostics", PALETTE["early"], "-"),
    }
    for stage, (label, colour, style) in stage_style.items():
        g = front[front.stage == stage].sort_values("fraction_test_hours_saved")
        ax.plot(
            g.fraction_test_hours_saved * 100,
            g.eligible_recall * 100,
            color=colour,
            linestyle=style,
            linewidth=1.2,
            label=label,
            zorder=3,
        )
    marks = []
    for stage, tau in (("passport", 0.50), ("early_diagnostics", 0.30)):
        row = front[(front.stage == stage) & (np.isclose(front.threshold, tau))].iloc[0]
        marks.append((stage, tau, row))
        ax.plot(
            row.fraction_test_hours_saved * 100,
            row.eligible_recall * 100,
            marker="o",
            markersize=4.6,
            markerfacecolor="white",
            markeredgecolor=stage_style[stage][1],
            markeredgewidth=1.2,
            zorder=6,
        )
    note = (
        "History at 0.50: 17 false rejects, 410 h stopped\n"
        "Early diagnostics at 0.30: 4 false rejects, 314 h stopped"
    )
    ax.text(0.012, 0.115, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.9,
            color=PALETTE["history"], zorder=7, linespacing=1.6)
    ax.text(30.0, 42.0, "Early diagnostics at 0.30\n4 false rejects", ha="center", va="center",
            fontsize=5.9, color=PALETTE["early"], zorder=7, linespacing=1.35)
    ax.plot([0.0, 100.0], [100.0, 100.0], color=PALETTE["grid"], linewidth=0.6, linestyle=(0, (3, 2.5)), zorder=1)
    ax.text(2.0, 106.0, "All tests completed", ha="left", va="top", fontsize=5.9, color=PALETTE["muted"], zorder=5)
    ax.set_xlabel("Recorded electrical-test hours stopped (percent of 569.9 h)")
    ax.set_ylabel("Recall of second-life\neligible cells (percent)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_title("Screening resource and risk frontier", pad=10)
    panel_label(ax, "a", dx=-0.108, dy=1.025)

    bx = fig.add_subplot(grid[1])
    bx.axis("off")
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 1)
    bx.set_title("Operational reading of two thresholds", pad=9)
    headers = ["Quantity", "History\nat 0.50", "Early diag.\nat 0.30"]
    xs = [0.015, 0.545, 0.815]
    head_y = 0.90
    for x, text, ha in zip(xs, headers, ["left", "center", "center"]):
        bx.text(x, head_y, text, ha=ha, va="center", fontsize=6.2, fontweight="bold", color=PALETTE["ink"])
    bx.plot([0.008, 0.992], [head_y - 0.078, head_y - 0.078], color=PALETTE["ink"], linewidth=0.6)
    rows = [
        ("Tests stopped", "71", "54"),
        ("Stopped hours", "410.1", "314.5"),
        ("Stopped hours, percent", "72.0", "55.2"),
        ("Cells retained", "29", "46"),
        ("False rejects", "17", "4"),
        ("False accepts", "10", "14"),
    ]
    row_h = 0.112
    for i, (name, a, b) in enumerate(rows):
        yv = head_y - 0.078 - (i + 0.60) * row_h
        if i % 2 == 0:
            bx.add_patch(Rectangle((0.004, yv - 0.051), 0.992, 0.102, facecolor="#F4F4F4", edgecolor="none", zorder=0))
        bx.text(xs[0], yv, name, ha="left", va="center", fontsize=6.0, color=PALETTE["ink"], zorder=2)
        bx.text(xs[1], yv, a, ha="center", va="center", fontsize=6.0, color=PALETTE["history"], zorder=2)
        bx.text(xs[2], yv, b, ha="center", va="center", fontsize=6.0, color=PALETTE["early"], zorder=2)
    note_y = head_y - 0.078 - (len(rows) + 0.75) * row_h
    bx.text(
        0.012,
        note_y,
        "Hours describe the recorded electrical channel time attached to\nstopped cells before any acquisition or verification burden.",
        ha="left",
        va="top",
        fontsize=5.8,
        color=PALETTE["muted"],
        linespacing=1.4,
    )

    legend_ax = fig.add_subplot(outer[1])
    legend_ax.axis("off")
    handles = [
        Line2D([], [], color=colour, linestyle=style, linewidth=1.4, label=label)
        for label, colour, style in stage_style.values()
    ]
    legend_ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        ncol=3,
        handlelength=2.0,
        columnspacing=1.8,
        borderaxespad=0,
        frameon=False,
    )

    empty = fs.require_panel_content(fig, {"4a frontier": ax, "4b threshold table": bx}, min_ink=0.02)
    if empty:
        raise SystemExit(f"Figure 4 has an empty panel: {empty}")
    report = save_figure(fig, "fig4_screening_resource_risk_frontier", OUT, name="Figure 4")
    print(report.render())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5. Independent cohort and cross-cohort progression
# ---------------------------------------------------------------------------
def figure_external(metrics: pd.DataFrame) -> None:
    wang_path = RESULTS / "wang_external_validation_metrics.csv"
    if not wang_path.exists():
        wang_path = ROOT / "results_reference" / "wang_external_validation_metrics.csv"
    wang = pd.read_csv(wang_path).set_index("stage")
    order = ["capacity_only", "explicit_snapshot", "implicit_single_cycle", "explicit_plus_implicit"]
    ba = metrics[metrics.metric == "balanced_accuracy"]

    fig = plt.figure(figsize=(DOUBLE_COL, 80 * fs.MM))
    grid = fig.add_gridspec(
        1, 2, width_ratios=[1.20, 1.0], wspace=0.36, left=0.215, right=0.985, top=0.875, bottom=0.215
    )
    ax = fig.add_subplot(grid[0])
    y = np.arange(len(order))[::-1]
    for yi, stage in zip(y, order):
        row = wang.loc[stage]
        highlight = stage == "explicit_snapshot"
        colour = PALETTE["wang"] if highlight else PALETTE["muted"]
        ax.errorbar(
            row.balanced_accuracy,
            yi,
            xerr=[[row.balanced_accuracy - row.bootstrap_ba_low], [row.bootstrap_ba_high - row.balanced_accuracy]],
            fmt="o",
            markersize=4.6 if highlight else 3.6,
            capsize=2.0,
            elinewidth=0.7,
            markeredgewidth=0.5,
            markeredgecolor="white",
            color=colour,
            zorder=4,
        )
        ax.text(
            row.balanced_accuracy,
            yi + 0.34,
            f"{row.balanced_accuracy:.3f}",
            ha="center",
            va="center",
            fontsize=5.8,
            color=colour,
            zorder=5,
        )
    ax.plot([0.5, 0.5], [-0.72, len(order) - 0.32], color=PALETTE["grid"], linewidth=0.7, zorder=0)
    ax.text(0.515, -0.63, "chance", ha="left", va="center", fontsize=5.8, color=PALETTE["muted"])
    ax.set_yticks(y, [WANG_LABELS[s] for s in order])
    ax.set_ylim(-0.72, len(order) - 0.14)
    ax.set_xlim(0.36, 0.99)
    ax.set_xticks(np.arange(0.4, 0.91, 0.1))
    ax.set_xlabel("Balanced accuracy with 95 percent protocol-bootstrap interval")
    ax.set_title("Independent cell cohort, protocol-held-out", pad=9)
    panel_label(ax, "a", dx=-0.315, dy=1.03)

    bx = fig.add_subplot(grid[1])
    salinas_any = ba[ba.target == "any_second_life"].set_index("stage")
    series = [
        (
            "Source collection, 16 packs",
            [
                salinas_any.loc["passport", "estimate"],
                salinas_any.loc["early_diagnostics", "estimate"],
                salinas_any.loc["full_diagnostics", "estimate"],
            ],
            PALETTE["history"],
            0.10,
        ),
        (
            "Independent cohort, 16 protocols",
            [
                wang.loc["capacity_only", "balanced_accuracy"],
                wang.loc["explicit_snapshot", "balanced_accuracy"],
                wang.loc["implicit_single_cycle", "balanced_accuracy"],
            ],
            PALETTE["wang"],
            -0.10,
        ),
    ]
    x_positions = np.array([0, 1, 2])
    for name, values, colour, dy in series:
        bx.plot(
            x_positions,
            values,
            linestyle="none",
            marker="o",
            markersize=4.4,
            color=colour,
            zorder=3,
        )
        for xv, yv in zip(x_positions, values):
            bx.text(xv, yv + (0.033 if dy > 0 else -0.033), f"{yv:.3f}", ha="center",
                    va="bottom" if dy > 0 else "top", fontsize=5.8, color=colour, zorder=5)
    handles = [
        Line2D([], [], color=PALETTE["history"], marker="o", markersize=4.4, linestyle="none", label=series[0][0]),
        Line2D([], [], color=PALETTE["wang"], marker="o", markersize=4.4, linestyle="none", label=series[1][0]),
    ]
    bx.set_xticks(x_positions, ["Battery\nhistory", "Early\ndiagnostics", "Full\ndiagnostics"])
    bx.set_xlim(-0.45, 2.45)
    bx.set_ylim(0.50, 1.07)
    bx.set_yticks(np.arange(0.5, 1.01, 0.1))
    bx.set_ylabel("Balanced accuracy")
    bx.set_xlabel("Information block available")
    bx.set_title("Cross-cohort progression", pad=9)
    bx.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.90), ncol=1, handlelength=1.2, borderaxespad=0)
    panel_label(bx, "b", dx=-0.245, dy=1.03)

    empty = fs.require_panel_content(fig, {"5a independent cohort": ax, "5b cross cohort": bx})
    if empty:
        raise SystemExit(f"Figure 5 has an empty panel: {empty}")
    report = save_figure(fig, "fig5_independent_cohort_validation", OUT, name="Figure 5")
    print(report.render())
    plt.close(fig)


def main() -> None:
    fs.apply_style()
    metrics = load_metrics()
    front = load_frontier()
    figure_timeline()
    figure_trajectories()
    figure_progression(metrics)
    figure_frontier(front)
    figure_external(metrics)
    print("\nfigures written to", OUT)


if __name__ == "__main__":
    main()
