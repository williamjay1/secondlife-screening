"""Nature-style figure helpers with a programmatic layout audit.

The audit renders each figure to a raster buffer and inspects the real text
bounding boxes, so overlap and clipping violations are detected from the
rendered geometry rather than from an assumption about the layout code.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text

MM = 1.0 / 25.4
_DEBUG = False
SINGLE_COL = 88 * MM
ONE_HALF_COL = 120 * MM
DOUBLE_COL = 180 * MM

FONT_STACK = ["Arial", "Helvetica", "DejaVu Sans"]

PALETTE = {
    "history": "#4C6EA8",
    "physical": "#5C9EAD",
    "early": "#E08A3C",
    "full": "#9E5A7A",
    "delayed": "#B8B8B8",
    "wang": "#3F7D5C",
    "ink": "#1A1A1A",
    "muted": "#5A5A5A",
    "grid": "#D6D6D6",
    "band": "#EDEDED",
}

# Colour-blind safe class colours (Okabe-Ito derived).
CLASS_COLORS = {0: "#0072B2", 1: "#D55E00", 2: "#009E73"}
CLASS_LABELS = {0: "Recycle", 1: "Repurpose", 2: "Reuse"}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.5,
            "legend.labelspacing": 0.35,
            "legend.borderpad": 0.2,
            "legend.columnspacing": 1.0,
            "axes.linewidth": 0.6,
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.minor.size": 1.4,
            "ytick.minor.size": 1.4,
            "lines.linewidth": 1.0,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.6,
            "text.color": PALETTE["ink"],
            "figure.dpi": 150,
            "savefig.dpi": 1200,
            "savefig.bbox": None,
            "pdf.fonttype": 42,
            "pdf.compression": 6,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


# --------------------------------------------------------------------------
# Layout audit
# --------------------------------------------------------------------------
@dataclass
class Violation:
    kind: str
    detail: str


@dataclass
class AuditReport:
    figure: str
    violations: list[Violation] = field(default_factory=list)
    n_texts: int = 0
    n_lines: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    def render(self) -> str:
        head = f"[{self.figure}] texts={self.n_texts} lines={self.n_lines} violations={len(self.violations)}"
        if self.ok:
            return head + "  OK"
        body = "\n".join(f"    - {v.kind}: {v.detail}" for v in self.violations)
        return head + "\n" + body


def _collect_texts(fig) -> list[tuple[Text, str, tuple[float, float, float, float]]]:
    """Return (artist, label, display bbox) for every visible text artist."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    out: list[tuple[Text, str, tuple[float, float, float, float]]] = []
    for artist in fig.findobj(match=lambda a: isinstance(a, Text)):
        if not artist.get_visible():
            continue
        label = artist.get_text()
        if not label or not label.strip():
            continue
        try:
            bb = artist.get_window_extent(renderer=renderer)
        except Exception:  # pragma: no cover - defensive
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        out.append((artist, label, (bb.x0, bb.y0, bb.x1, bb.y1)))
    return out


def _collect_lines(fig):
    """Return display-space segments for drawn data strokes, in figure pixels.

    Only genuine strokes are collected: degenerate segments, filled areas
    produced by ``fill_between`` and marker paths are skipped, because they are
    not ink that a reader can see crossing a label.
    """
    fig.canvas.draw()
    transform = fig.transFigure.inverted()

    def to_fig(ax, pts):
        return transform.transform(ax.transData.transform(pts)) * fig.bbox.size

    segments = []
    for ax in fig.get_axes():
        for line in ax.get_lines():
            if not line.get_visible():
                continue
            try:
                lw = line.get_linewidth()
            except Exception:  # pragma: no cover - defensive
                lw = 1.0
            if lw <= 0:
                continue
            data = np.asarray(line.get_xydata(), dtype=float)
            if data.ndim != 2 or len(data) < 2:
                continue
            pts = to_fig(ax, data)
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
                if not all(np.isfinite([x0, y0, x1, y1])):
                    continue
                if (x1 - x0) ** 2 + (y1 - y0) ** 2 < 4.0:
                    continue
                segments.append((x0, y0, x1, y1))
        for coll in ax.collections:
            if not coll.get_visible():
                continue
            segments.extend(_collection_segments(ax, coll, to_fig))
    return segments


def _collection_segments(ax, coll, to_fig):
    """Approximate visible strokes of a collection with its path outlines."""
    import matplotlib.collections as mcoll
    from matplotlib.path import Path as MplPath

    out = []
    try:
        paths = coll.get_paths()
    except Exception:  # pragma: no cover - defensive
        return out
    if not paths:
        return out
    offsets = getattr(coll, "get_offsets", lambda: None)()
    is_scatter = isinstance(coll, mcoll.PathCollection) and offsets is not None and len(offsets) > 0
    if is_scatter:
        # Scatter markers are small glyphs; represent each as a short stroke so
        # a label printed on top of a marker is still detected.
        pts = to_fig(ax, np.asarray(offsets, dtype=float))
        for x0, y0 in pts:
            if np.isfinite(x0) and np.isfinite(y0):
                out.append((x0, y0, x0 + 3.0, y0))
        return out
    for path in paths:
        verts = np.asarray(path.vertices, dtype=float)
        if verts.size == 0:
            continue
        codes = path.codes
        if codes is None:
            rings = [verts]
        else:
            rings, current = [], []
            for v, c in zip(verts, codes):
                if c == MplPath.MOVETO and current:
                    rings.append(np.asarray(current))
                    current = []
                current.append(v)
            if current:
                rings.append(np.asarray(current))
        for ring in rings:
            if len(ring) < 2:
                continue
            pts = to_fig(ax, ring)
            if not np.all(np.isfinite(pts)):
                continue
            for i in range(len(pts) - 1):
                out.append((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]))
    return out


def _overlap(a, b, pad: float = 0.6) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def _segment_hits_box(seg, box, pad: float = 0.4) -> bool:
    x0, y0, x1, y1 = seg
    bx0, by0, bx1, by1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    if x0 == x1 and y0 == y1:
        return bx0 <= x0 <= bx1 and by0 <= y0 <= by1
    # Liang-Barsky clip test.
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x0 - bx0),
        (dx, bx1 - x0),
        (-dy, y0 - by0),
        (dy, by1 - y0),
    ):
        if p == 0:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return False
    return True


def audit_figure(fig, name: str = "figure", check_lines: bool = True) -> AuditReport:
    """Detect text-text overlap, text-line crossing and out-of-canvas text."""
    report = AuditReport(figure=name)
    fig.canvas.draw()
    texts = _collect_texts(fig)
    report.n_texts = len(texts)
    fig_w, fig_h = fig.bbox.size

    if _DEBUG:
        for _artist, _label, _box in texts:
            print(f"    [box] {_label[:34]!r} ({_box[0]:.1f},{_box[1]:.1f},{_box[2]:.1f},{_box[3]:.1f})")

    axis_decor = set()
    for ax in fig.get_axes():
        for artist in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            axis_decor.add(id(artist))

    for artist, label, box in texts:
        if box[0] < -0.5 or box[1] < -0.5 or box[2] > fig_w + 0.5 or box[3] > fig_h + 0.5:
            report.violations.append(
                Violation("out-of-canvas", f"{label!r} bbox=({box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}) canvas=({fig_w:.0f}x{fig_h:.0f})")
            )

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i], texts[j]
            if _overlap(a[2], b[2]):
                report.violations.append(
                    Violation("text-overlap", f"{a[1]!r} vs {b[1]!r}")
                )

    if check_lines:
        segments = _collect_lines(fig)
        report.n_lines = len(segments)
        for artist, label, box in texts:
            # Tick labels sit at the axes boundary by construction; the axes
            # frame and gridlines are expected to touch them.
            if id(artist) in axis_decor:
                continue
            for seg in segments:
                if _segment_hits_box(seg, box):
                    report.violations.append(
                        Violation(
                            "text-line-crossing",
                            f"{label!r} crossed by segment ({seg[0]:.0f},{seg[1]:.0f})-({seg[2]:.0f},{seg[3]:.0f})",
                        )
                    )
                    break
    return report


def _is_reference_line(ax, line) -> bool:
    """True for a full width or full height line, which carries no data marks."""
    data = np.asarray(line.get_xydata(), dtype=float)
    if data.ndim != 2 or len(data) < 2:
        return False
    x = data[:, 0]
    y = data[:, 1]
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return False
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span_x = abs(x1 - x0)
    span_y = abs(y1 - y0)
    horizontal = np.ptp(y) <= 1e-9 * max(1.0, span_y) and np.ptp(x) >= 0.5 * span_x
    vertical = np.ptp(x) <= 1e-9 * max(1.0, span_x) and np.ptp(y) >= 0.5 * span_y
    return bool(horizontal or vertical)


def panel_ink(fig, ax, dpi: int = 150, margin: float = 0.04, exclude_reference_lines: bool = False) -> float:
    """Fraction of drawn ink inside an axes box, ignoring the frame.

    The box is shrunk by ``margin`` on every side so spines, ticks and tick
    labels are excluded, which leaves only marks that belong to the data. With
    ``exclude_reference_lines`` the full width and full height lines are hidden
    first, so a threshold marker or a chance line cannot be mistaken for data.
    """
    import io

    from PIL import Image

    hidden = []
    if exclude_reference_lines:
        for line in list(ax.get_lines()):
            if _is_reference_line(ax, line):
                hidden.append(line)
                line.set_visible(False)
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        image = Image.open(buf).convert("L")
    finally:
        for line in hidden:
            line.set_visible(True)

    width, height = image.size
    bbox = ax.get_window_extent()
    scale = dpi / fig.dpi
    x0 = bbox.x0 * scale
    y0 = (fig.bbox.height - bbox.y1) * scale
    x1 = bbox.x1 * scale
    y1 = (fig.bbox.height - bbox.y0) * scale
    dx = (x1 - x0) * margin
    dy = (y1 - y0) * margin
    box = (
        max(0, int(x0 + dx)),
        max(0, int(y0 + dy)),
        min(width, int(x1 - dx)),
        min(height, int(y1 - dy)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return 1.0
    crop = np.asarray(image.crop(box), dtype=float)
    return float((crop < 245).mean())


def require_panel_content(fig, panels: dict[str, object], min_ink: float = 0.004) -> list[str]:
    """Fail when a panel contains no drawn data marks.

    Reference lines, band fills and annotations are excluded from the measure,
    so the value reports only lines, markers and bars that come from data.
    """
    problems = []
    for name, ax in panels.items():
        ink = panel_ink(fig, ax, exclude_reference_lines=True)
        status = "ok" if ink >= min_ink else "EMPTY"
        print(f"    data content {name}: ink {ink:.4f} {status}")
        if ink < min_ink:
            problems.append(name)
    return problems


def save_figure(fig, stem: str, outdir, audit: bool = True, name: str | None = None) -> AuditReport:
    """Write a vector PDF plus a high-resolution PNG and audit the layout."""
    from pathlib import Path

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = audit_figure(fig, name or stem) if audit else AuditReport(figure=name or stem)
    fig.savefig(outdir / f"{stem}.pdf", format="pdf")
    fig.savefig(outdir / f"{stem}.png", format="png", dpi=1200)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    return report


def panel_label(ax, letter: str, dx: float = -0.16, dy: float = 1.06) -> None:
    """Bold panel letter in axes coordinates, placed outside the data area."""
    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=PALETTE["ink"],
    )


def short_stage_labels() -> dict[str, str]:
    return {
        "passport": "Battery history",
        "passport_plus_physical": "+ Physical observations",
        "early_diagnostics": "+ Early diagnostics",
        "full_diagnostics": "+ Full diagnostics",
        "full_plus_delayed_x29": "+ Delayed measurement",
    }


def wang_stage_labels() -> dict[str, str]:
    return {
        "capacity_only": "Capacity only",
        "explicit_snapshot": "Five explicit diagnostics",
        "implicit_single_cycle": "108 implicit features",
        "explicit_plus_implicit": "Explicit + implicit",
    }
