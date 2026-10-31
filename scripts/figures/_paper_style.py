"""Unified figure style for the paper (Berkeley/RAIL convention).

Single source of truth for matplotlib rcParams, the protocol/family
palettes, and the higher-level visual primitives (pastel-fill bars,
open-circle scatter, column-header subtitles, Tufte spines).
Import and call ``apply_style()`` at the top of every plotting script.

Conventions:
  - Protocol = colour. Reset-free = navy hero (#3B6FB6, the paper's
    contribution); Strict = warm orange (#E1812B); Standard = muted teal
    (#4D9DA5, the baseline). Bars/lines carry these at full saturation;
    pastel fills use alpha 0.55, outlined at 1.1 pt full saturation.
  - Family = marker shape in main plots (Floor circle, VT square, MVT
    triangle). Appendix diagnostics may use the colour-blind family palette.
  - No decorative full grids. Use faint one-axis grids only where they improve
    quantitative reading; semantic references (zero, chance, parity) remain
    explicit dashed lines.
  - Top + right spines hidden; bottom/left spines offset outward 4 pt.
  - Subplot titles replaced by 9 pt sans semibold column headers placed
    above-left of each panel.
  - Body 9.5 pt, ticks 8 pt, headers 9 pt, annotations 8.5 pt; Lato
    sans (fallback Helvetica/Arial/DejaVu); TrueType (type 42) embedded.
  - Boxed legends removed; use frameless legends above the plot or compact
    inline keys. Avoid duplicate legends across small multiples.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Colour palette (Berkeley/RAIL convention: cool hero + warm accent + cool
# tertiary; greys reserved for de-emphasised data and text only).
# ---------------------------------------------------------------------------

NAVY = "#3B6FB6"
ORANGE = "#E1812B"
TEAL = "#4D9DA5"
TEXT = "#2B2B2B"
DEEMPH = "#9C9C9C"
FAINT = "#E5E5E5"
GATE_RED = "#B33A3A"

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

# Family is a secondary visual channel, encoded by marker shape. The
# colours below are a fallback for figures that genuinely need
# colour-coded families (a few appendix figures); body figures should
# use FAMILY_MARKER instead and reserve colour for the protocol axis.
FAMILY_COLOR = {
    "floor": "#D55E00",  # Okabe-Ito vermillion
    "vt": "#0072B2",     # Okabe-Ito blue
    "mvt": "#009E73",    # Okabe-Ito green
}

FAMILY_MARKER = {
    "floor": "o",
    "vt": "s",
    "mvt": "^",
}

FAMILY_LABEL = {"floor": "Floor", "vt": "VT", "mvt": "MVT"}

TAU_TO_PROTOCOL = {"0p00": "reset_free", "0p20": "episodic", "0p50": "strict"}

# ---------------------------------------------------------------------------
# Canvas widths (inches). NeurIPS textwidth ~5.5"; \textwidth in two-column
# papers fits a 7.0" panoramic figure when used with full-page floats.
# ---------------------------------------------------------------------------

COL_WIDTH = 3.5
TEXT_WIDTH = 5.5
FULL_WIDTH = 7.0

# ---------------------------------------------------------------------------
# rcParams. Build once, apply at the top of every plotting script.
# ---------------------------------------------------------------------------

PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9.5,
    "axes.titlesize": 9.0,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 6.0,
    "axes.labelsize": 9.0,
    "axes.labelpad": 3.0,
    "axes.labelweight": "regular",
    "axes.linewidth": 0.85,
    "axes.edgecolor": TEXT,
    "axes.labelcolor": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.major.width": 0.85,
    "ytick.major.width": 0.85,
    "xtick.major.pad": 3.0,
    "ytick.major.pad": 3.0,
    "axes.grid": False,
    "grid.color": FAINT,
    "grid.linewidth": 0.45,
    "grid.alpha": 0.7,
    "lines.linewidth": 1.55,
    "lines.markersize": 4.6,
    "lines.markeredgewidth": 1.1,
    "patch.linewidth": 1.1,
    "patch.edgecolor": TEXT,
    "legend.fontsize": 8.0,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.6,
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
# Visual primitives. Keep them small and explicit so each call site shows
# what it's doing without hiding the figure logic.
# ---------------------------------------------------------------------------

def tufte_axes(ax, *, offset_pt: float = 4.0) -> None:
    """Hide top/right spines; push bottom spine outward by ``offset_pt``."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if "bottom" in ax.spines and ax.spines["bottom"].get_visible():
        ax.spines["bottom"].set_position(("outward", offset_pt))
    if "left" in ax.spines and ax.spines["left"].get_visible():
        ax.spines["left"].set_position(("outward", offset_pt))
    ax.tick_params(top=False, right=False)


def light_grid(ax, *, axis: str = "y") -> None:
    """Apply a faint quantitative grid behind data marks."""
    ax.grid(True, axis=axis, color=FAINT, linewidth=0.45, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def style_legend(legend) -> None:
    """Normalize legend typography and remove frames when a legend exists."""
    if legend is None:
        return
    legend.set_frame_on(False)
    for text in legend.get_texts():
        text.set_color(TEXT)
        text.set_fontsize(8.0)


def column_header(ax, text: str, *, dy: float = 1.06, x: float = 0.0,
                  weight: str = "semibold", size: float = 9.0) -> None:
    """Place a small left-aligned semibold header above the panel."""
    ax.text(x, dy, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=size, fontweight=weight, color=TEXT)


def pastel_bar(ax, x, h, *, color: str, width: float = 0.7,
               yerr=None, capsize: float = 3.0, label: str | None = None,
               alpha: float = 0.55, lw: float = 1.2):
    """Draw a Berkeley-style pastel-fill / dark-outline bar."""
    return ax.bar(x, h, width=width, yerr=yerr, capsize=capsize,
                  facecolor=color, alpha=alpha, edgecolor=color,
                  linewidth=lw, error_kw={"ecolor": color, "elinewidth": 1.0,
                                          "capthick": 1.0},
                  label=label, zorder=2)


def value_labels(ax, xs, hs, *, fmt: str = "{:.2f}", dy: float = 0.02,
                 size: float = 8.0, color: str | None = None) -> None:
    """Write numeric values just above each bar top."""
    color = color or TEXT
    for x, h in zip(xs, hs):
        ax.text(x, h + dy, fmt.format(h), ha="center", va="bottom",
                fontsize=size, color=color)


def open_circle(ax, x, y, *, color: str, marker: str = "o",
                size: float = 28, lw: float = 1.2, label: str | None = None):
    """Open-circle scatter point with thick coloured edge (RAIL style)."""
    return ax.scatter(x, y, marker=marker, s=size, facecolors="white",
                      edgecolors=color, linewidths=lw, label=label,
                      zorder=3)
