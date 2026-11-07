"""Shared figure style for the paper.

Every plotting script calls ``apply_style()`` first. Text is set in STIX, a
Times design that ships with matplotlib, so the figures match the paper's
body font on any machine. Labels and axes are black, grid lines light grey,
legends framed and panel titles centred. Colours come from seaborn's
colorblind palette: one colour per training protocol (Standard green,
Strict orange, Reset-free blue) and, where a figure colours bodies by
family, Floor vermillion, VT blue and MVT green. Markers are filled with a
dark edge, and bars are solid with black error bars.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, MaxNLocator

# ---------------------------------------------------------------------------
# Colours (seaborn "colorblind" palette).
# ---------------------------------------------------------------------------

NAVY = "#0173B2"      # Reset-free
ORANGE = "#DE8F05"    # Strict
TEAL = "#029E73"      # Standard
TEXT = "#000000"
DEEMPH = "#7F7F7F"
FAINT = "#DCDCDC"
GATE_RED = "#C0392B"
EDGE = "#262626"      # marker edges

PROTOCOL_COLOR = {
    "reset_free": NAVY,
    "strict": ORANGE,
    "episodic": TEAL,
}

PROTOCOL_LABEL = {
    "reset_free": "Reset-free",
    "strict": "Strict",
    "episodic": "Standard",
}

# Figures that colour bodies by family use these; the others use marker shape.
FAMILY_COLOR = {
    "floor": "#D55E00",
    "vt": "#0173B2",
    "mvt": "#029E73",
}

FAMILY_MARKER = {
    "floor": "o",
    "vt": "s",
    "mvt": "^",
}

FAMILY_LABEL = {"floor": "Floor", "vt": "VT", "mvt": "MVT"}

TAU_TO_PROTOCOL = {"0p00": "reset_free", "0p20": "episodic", "0p50": "strict"}

# ---------------------------------------------------------------------------
# Canvas widths (inches). The NeurIPS text width is 5.5 in; wider canvases
# are scaled down to it in the paper.
# ---------------------------------------------------------------------------

COL_WIDTH = 3.5
TEXT_WIDTH = 5.5
FULL_WIDTH = 7.0

# ---------------------------------------------------------------------------
# rcParams.
# ---------------------------------------------------------------------------

PARAMS = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10.0,
    "axes.titlesize": 10.5,
    "axes.titleweight": "normal",
    "axes.titlelocation": "center",
    "axes.titlepad": 5.0,
    "axes.labelsize": 10.0,
    "axes.labelpad": 3.0,
    "axes.labelweight": "normal",
    "axes.linewidth": 0.8,
    "axes.edgecolor": TEXT,
    "axes.labelcolor": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 2.5,
    "axes.grid": False,
    "grid.color": FAINT,
    "grid.linewidth": 0.6,
    "grid.alpha": 1.0,
    "lines.linewidth": 1.8,
    "lines.markersize": 5.0,
    "lines.markeredgewidth": 0.6,
    "patch.linewidth": 0.6,
    "patch.edgecolor": EDGE,
    "legend.fontsize": 9.0,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "#BDBDBD",
    "legend.fancybox": True,
    "legend.borderpad": 0.4,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0.4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.045,
    "text.color": TEXT,
}


def apply_style() -> dict:
    plt.rcParams.update(PARAMS)
    return PARAMS


# ---------------------------------------------------------------------------
# Helpers used by the plotting scripts.
# ---------------------------------------------------------------------------

def min_ticks(ax, n: int = 4) -> None:
    """Keep at least ``n`` ticks on axes that use matplotlib's automatic locator."""
    for axis in (ax.xaxis, ax.yaxis):
        if type(axis.get_major_locator()) is AutoLocator:
            axis.set_major_locator(MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10],
                                               min_n_ticks=n))


def tufte_axes(ax, *, offset_pt: float = 0.0) -> None:
    """Hide the top and right spines (the left and bottom ones stay attached)."""
    min_ticks(ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if offset_pt:
        for side in ("bottom", "left"):
            if side in ax.spines and ax.spines[side].get_visible():
                ax.spines[side].set_position(("outward", offset_pt))
    ax.tick_params(top=False, right=False)


def light_grid(ax, *, axis: str = "y") -> None:
    """Light grey grid lines behind the data."""
    min_ticks(ax)
    ax.grid(True, axis=axis, color=FAINT, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def style_legend(legend) -> None:
    """Give a legend the shared frame and text style."""
    if legend is None:
        return
    legend.set_frame_on(True)
    frame = legend.get_frame()
    frame.set_edgecolor("#BDBDBD")
    frame.set_alpha(1.0)
    for text in legend.get_texts():
        text.set_color(TEXT)
        text.set_fontsize(9.0)


def column_header(ax, text: str, *, dy: float = 1.04, x: float = 0.5,
                  weight: str = "normal", size: float = 9.5) -> None:
    """Centred panel title just above the axes."""
    ax.text(x, dy, text, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=size + 1.0, fontweight="normal", color=TEXT)


def pastel_bar(ax, x, h, *, color: str, width: float = 0.7,
               yerr=None, capsize: float = 2.5, label: str | None = None,
               alpha: float = 1.0, lw: float = 0.0):
    """Solid bar with black error bars."""
    return ax.bar(x, h, width=width, yerr=yerr, capsize=capsize,
                  facecolor=color, edgecolor=color, linewidth=lw,
                  error_kw={"ecolor": TEXT, "elinewidth": 0.9, "capthick": 0.9},
                  label=label, zorder=2)


def value_labels(ax, xs, hs, *, fmt: str = "{:.2f}", dy: float = 0.02,
                 size: float = 8.5, color: str | None = None) -> None:
    """Write numeric values just above each bar top."""
    color = color or TEXT
    for x, h in zip(xs, hs):
        ax.text(x, h + dy, fmt.format(h), ha="center", va="bottom",
                fontsize=size, color=color)


def open_circle(ax, x, y, *, color: str, marker: str = "o",
                size: float = 28, lw: float = 0.6, label: str | None = None):
    """Filled marker with a dark edge."""
    return ax.scatter(x, y, marker=marker, s=size, facecolors=color,
                      edgecolors=EDGE, linewidths=lw, label=label,
                      zorder=3)
