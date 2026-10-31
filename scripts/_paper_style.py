"""Unified figure style for the paper (Berkeley/RAIL convention).

Single source of truth for matplotlib rcParams, the protocol palette, the
family marker shapes, and the higher-level visual primitives (pastel-fill
bars, open-circle scatter, column-header subtitles, Tufte spines).
Import and call ``apply_style()`` at the top of every plotting script.
``apply_style()`` also registers the Lato TTFs (see ``register_lato``).

Conventions (camera-ready style decision, 2026-10):
  - Protocol = colour, and only protocol is coloured. Order is always
    ascending tau: Reset-free (tau = 0), Standard (tau = 0.20), Strict
    (tau = 0.50). Use ``PROTOCOL_ORDER`` for legends, panel order and axes,
    and ``PROTOCOL_LABEL_TAU`` for labels with a real Greek tau. The internal
    key for Standard is "lenient" (a draft name), and "episodic" is a
    deprecated alias of it kept so old scripts still run.
  - Protocol colours are separable in grayscale. Relative luminance Y
    (WCAG 2.x, sRGB linearised) and CIE L*:
        Reset-free  navy        #3B6FB6   Y = 0.157   L* = 46.5
        Standard     dark teal   #17504B   Y = 0.064   L* = 30.5
        Strict      orange      #E1812B   Y = 0.319   L* = 63.2
    Standard is the darkest of the three (L* gap 16 to Reset-free, 33 to
    Strict). The previous teal #4D9DA5 (L* = 60.3) was nearly identical to
    Strict in grayscale and is retired. Wherever protocols appear as series,
    add the redundant cue from ``PROTOCOL_LINESTYLE`` (lines) or
    ``PROTOCOL_HATCH`` (filled areas and bars).
  - Family = marker shape only (Floor circle, VT square, MVT triangle),
    drawn in ``FAMILY_INK`` dark grey. Never colour families anywhere in the
    paper. ``FAMILY_COLOR`` is deprecated: it still resolves for old
    scripts but emits a DeprecationWarning, and new figures must not use it.
  - Inert bodies (the 28 bodies whose Standard cutoff lies below the torso
    radius) are drawn with open markers (``INERT_FACECOLOR``). Cut-off
    bodies are drawn with a cross (``CUTOFF_MARKER``) where relevant. A body
    in both sets keeps its open marker and gets the cross on top, so every
    one of the 28 has an open marker. Filled body markers have a dark edge
    (``FILLED_EDGE``), not a white one: where two filled markers overlap, a
    white edge draws the outline of an open marker. Use
    ``body_marker_kwargs`` to get consistent scatter keyword arguments.
  - Export at final printed width (``TEXT_WIDTH`` 5.5 in, ``HALF_WIDTH``
    2.7 in) and include at scale 1, so the smallest text is at least
    ``MIN_FONT_PT`` (7 pt). Save a vector PDF plus a 200 dpi PNG
    (``save_figure``). Lato is embedded as TrueType (pdf.fonttype 42).
  - No decorative full grids. Use faint one-axis grids only where they
    improve quantitative reading; semantic references (zero, chance,
    parity) remain explicit dashed lines.
  - Top and right spines hidden; bottom and left spines offset outward 4 pt.
  - Subplot titles replaced by 9 pt sans semibold column headers placed
    above-left of each panel.
  - Body 9.5 pt, ticks 8 pt, headers 9 pt, annotations 8.5 pt; Lato sans
    (fallback Helvetica, Arial, DejaVu Sans, with a visible warning).
  - Boxed legends removed; use frameless legends above the plot or compact
    inline keys. Avoid duplicate legends across small multiples.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# Fonts. Lato is not installed system-wide on every machine, so the TTFs are
# registered from a directory at runtime. Override with PAPER_FONT_DIR.
# ---------------------------------------------------------------------------

DEFAULT_FONT_DIR = str(Path(__file__).resolve().parents[1] / "fonts")


def _lato_available() -> bool:
    try:
        path = font_manager.findfont(
            font_manager.FontProperties(family="Lato"), fallback_to_default=False
        )
    except ValueError:
        return False
    return bool(path)


def register_lato(font_dir: str | os.PathLike | None = None) -> bool:
    """Make Lato available to matplotlib; return True if it is usable.

    If Lato is already known to matplotlib nothing is done. Otherwise every
    ``*.ttf`` in ``font_dir`` (default: $PAPER_FONT_DIR, then
    ``DEFAULT_FONT_DIR``) is registered. If Lato is still missing, a
    warning is emitted and matplotlib falls back to the next family in
    ``font.sans-serif``; the fallback is never silent.
    """
    if _lato_available():
        return True
    candidates = [font_dir, os.environ.get("PAPER_FONT_DIR"), DEFAULT_FONT_DIR]
    for cand in candidates:
        if not cand:
            continue
        d = Path(cand).expanduser()
        if not d.is_dir():
            continue
        for ttf in sorted(d.glob("*.ttf")):
            font_manager.fontManager.addfont(str(ttf))
        if _lato_available():
            return True
    warnings.warn(
        "Lato not found (checked font_dir, $PAPER_FONT_DIR and "
        f"{DEFAULT_FONT_DIR}); figures will fall back to Helvetica, Arial "
        "or DejaVu Sans.",
        RuntimeWarning,
        stacklevel=2,
    )
    return False


# ---------------------------------------------------------------------------
# Colour palette. Colour encodes protocol only; greys are reserved for
# de-emphasised data, family markers and text.
# ---------------------------------------------------------------------------

NAVY = "#3B6FB6"
ORANGE = "#E1812B"
DARK_TEAL = "#17504B"
TEAL = DARK_TEAL  # deprecated name; was #4D9DA5, which is not grayscale-separable from ORANGE
TEXT = "#2B2B2B"
DEEMPH = "#9C9C9C"
FAINT = "#E5E5E5"
GATE_RED = "#B33A3A"
FAMILY_INK = "#4A4A4A"  # dark grey for family marker edges and fills

PROTOCOL_ORDER = ["reset_free", "lenient", "strict"]  # ascending tau

PROTOCOL_TAU = {"reset_free": 0.0, "lenient": 0.20, "strict": 0.50}

# Deprecated protocol keys that old scripts still use.
PROTOCOL_ALIASES = {"episodic": "lenient", "standard": "lenient"}


class _ProtocolDict(dict):
    """Dict keyed by canonical protocol names that also resolves aliases.

    Iteration, ``keys()`` and ``items()`` yield only the canonical keys in
    ``PROTOCOL_ORDER``; lookups with "episodic" or "standard" resolve to
    "lenient" so old scripts keep working.
    """

    def __missing__(self, key):
        if key in PROTOCOL_ALIASES:
            return self[PROTOCOL_ALIASES[key]]
        raise KeyError(key)

    def __contains__(self, key):
        return dict.__contains__(self, PROTOCOL_ALIASES.get(key, key))

    def get(self, key, default=None):
        return dict.get(self, PROTOCOL_ALIASES.get(key, key), default)


PROTOCOL_COLOR = _ProtocolDict({
    "reset_free": NAVY,
    "lenient": DARK_TEAL,
    "strict": ORANGE,
})

PROTOCOL_LABEL = _ProtocolDict({
    "reset_free": "Reset-free",
    "lenient": "Standard",
    "strict": "Strict",
})

# Labels with the threshold, using a real Greek tau (U+03C4, present in Lato).
PROTOCOL_LABEL_TAU = _ProtocolDict({
    "reset_free": "Reset-free (τ = 0)",
    "lenient": "Standard (τ = 0.20)",
    "strict": "Strict (τ = 0.50)",
})

# Redundant cue for lines: solid, dashed, dotted in ascending tau.
PROTOCOL_LINESTYLE = _ProtocolDict({
    "reset_free": "-",
    "lenient": (0, (5.0, 2.0)),
    "strict": (0, (1.2, 1.6)),
})

# Redundant cue for filled areas and bars (use with pastel_bar(hatch=...)).
PROTOCOL_HATCH = _ProtocolDict({
    "reset_free": None,
    "lenient": "////",
    "strict": "....",
})

TAU_TO_PROTOCOL = {"0p00": "reset_free", "0p20": "lenient", "0p50": "strict"}

# ---------------------------------------------------------------------------
# Family: marker shape only, always in FAMILY_INK. No family colours.
# ---------------------------------------------------------------------------

FAMILY_ORDER = ["floor", "vt", "mvt"]

FAMILY_MARKER = {
    "floor": "o",
    "vt": "s",
    "mvt": "^",
}

FAMILY_LABEL = {"floor": "Floor", "vt": "VT", "mvt": "MVT"}

# Deprecated. Kept only so pre-camera-ready scripts still import; accessing
# it warns. Do not use in any paper figure (families are never coloured).
_DEPRECATED_FAMILY_COLOR = {
    "floor": "#D55E00",
    "vt": "#0072B2",
    "mvt": "#009E73",
}


def __getattr__(name):
    if name == "FAMILY_COLOR":
        warnings.warn(
            "FAMILY_COLOR is deprecated: families are encoded by marker shape "
            "(FAMILY_MARKER) in FAMILY_INK, never by colour.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_FAMILY_COLOR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Body-level encodings.
#   Inert: the 28 bodies whose Standard cutoff lies below the torso radius
#   (flags in records/standard_threshold_audit.json). Open marker.
#   Cut-off: bodies removed by a cutoff, where relevant. Cross marker.
# ---------------------------------------------------------------------------

INERT_FACECOLOR = "white"   # open marker; edge keeps the series colour
INERT_EDGEWIDTH = 1.0
VALID_EDGEWIDTH = 0.6
FILLED_EDGE = "#2B2B2B"     # edge of filled body markers (never white, see above)
CUTOFF_MARKER = "x"
CUTOFF_LINEWIDTH = 1.0
INERT_LABEL = "Standard cutoff below the torso radius"  # legend text, with " (28)"
CUTOFF_LABEL = "Cut off"


def body_marker_kwargs(color: str, *, family: str | None = None,
                       inert: bool = False, cutoff: bool = False,
                       size: float = 22) -> dict:
    """Scatter kwargs for one body or group of bodies.

    ``color`` is the protocol colour (or FAMILY_INK when the panel has no
    protocol axis). ``family`` picks the marker shape. ``inert`` draws an
    open marker. ``cutoff`` alone draws a cross in place of the family
    shape. For a body that is both inert and cut off, draw the inert marker
    and then the cutoff marker at the same point.
    """
    if cutoff:
        return {"marker": CUTOFF_MARKER, "s": size, "c": color,
                "linewidths": CUTOFF_LINEWIDTH}
    marker = FAMILY_MARKER[family] if family else "o"
    if inert:
        return {"marker": marker, "s": size, "facecolors": INERT_FACECOLOR,
                "edgecolors": color, "linewidths": INERT_EDGEWIDTH}
    return {"marker": marker, "s": size, "facecolors": color,
            "edgecolors": FILLED_EDGE, "linewidths": VALID_EDGEWIDTH * 0.5}


# ---------------------------------------------------------------------------
# Canvas widths (inches). NeurIPS text width is 5.5 in. Export at the final
# printed width and include with \includegraphics at scale 1 (width=\linewidth
# for a 5.5 in figure) so that 8 pt ticks print at 8 pt.
# ---------------------------------------------------------------------------

TEXT_WIDTH = 5.5
HALF_WIDTH = 2.7
COL_WIDTH = HALF_WIDTH  # deprecated name; was 3.5 (not a NeurIPS width)
FULL_WIDTH = 7.0  # wider than the NeurIPS text block; avoid for camera-ready
MIN_FONT_PT = 7.0
PNG_DPI = 200

# ---------------------------------------------------------------------------
# rcParams. Build once, apply at the top of every plotting script.
# ---------------------------------------------------------------------------

PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Lato",
    "mathtext.it": "Lato:italic",
    "mathtext.bf": "Lato:bold",
    "mathtext.sf": "Lato",
    "mathtext.cal": "Lato:italic",
    "mathtext.tt": "DejaVu Sans Mono",
    "mathtext.fallback": "stixsans",
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
    "axes.unicode_minus": True,
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
    "hatch.linewidth": 0.6,
    "legend.fontsize": 8.0,
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "legend.handletextpad": 0.6,
    "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0.4,
    "figure.dpi": 300,
    "savefig.dpi": PNG_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.045,
    "text.color": TEXT,
}


def apply_style(font_dir: str | os.PathLike | None = None) -> dict:
    """Register Lato, then apply the paper rcParams."""
    register_lato(font_dir)
    plt.rcParams.update(PARAMS)
    return PARAMS


def save_figure(fig, stem: str | os.PathLike, *, tight: bool = False) -> list[Path]:
    """Save ``stem``.pdf (vector) and ``stem``.png (PNG_DPI).

    With ``tight=False`` (default) the page size equals ``fig`` size, so a
    figure built at TEXT_WIDTH or HALF_WIDTH prints at that width when
    included at scale 1. ``tight=True`` crops to the drawn content, which
    changes the width; check it before including at scale 1.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches=None would fall back to rcParams["savefig.bbox"] ("tight"),
    # so the uncropped case sets the rcParam to "standard" explicitly.
    out = []
    with plt.rc_context({"savefig.bbox": "tight" if tight else "standard"}):
        for ext, kw in (("pdf", {}), ("png", {"dpi": PNG_DPI})):
            path = stem.with_suffix(f".{ext}")
            fig.savefig(path, **kw)
            out.append(path)
    return out


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
               alpha: float = 0.55, lw: float = 1.2, hatch: str | None = None):
    """Draw a Berkeley-style pastel-fill / dark-outline bar.

    Pass ``hatch=PROTOCOL_HATCH[protocol]`` for the grayscale cue. The fill
    alpha is applied to the face colour only, so the outline and hatch stay
    at full saturation.
    """
    from matplotlib.colors import to_rgba
    return ax.bar(x, h, width=width, yerr=yerr, capsize=capsize,
                  facecolor=to_rgba(color, alpha), edgecolor=color,
                  linewidth=lw, hatch=hatch,
                  error_kw={"ecolor": color, "elinewidth": 1.0,
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
    """Open-marker scatter point with thick coloured edge (RAIL style)."""
    return ax.scatter(x, y, marker=marker, s=size, facecolors="white",
                      edgecolors=color, linewidths=lw, label=label,
                      zorder=3)


def text_with_subscripts(target, x, y, pieces, *, xycoords=None,
                         rotation: float = 0.0, ha: str = "left", raise_pt: float = 0.0,
                         color: str = TEXT, zorder: float = 5) -> list:
    """Draw a label whose subscripts print at no less than ``MIN_FONT_PT``.

    Mathtext draws a subscript at 70% of the base size (5.25 pt for 7.5 pt
    text, 6.3 pt for 9 pt text), below the house minimum. Here a label is a
    list of pieces ``(text, fontsize, drop_pt, text_kwargs)``, drawn one
    after another along a common baseline that starts at (x, y), each piece
    lowered by ``drop_pt`` points (use about 0.25 times the base size for a
    subscript). ``xycoords`` defaults to data coordinates for an Axes and to
    figure fraction for a Figure. ``rotation`` (degrees, screen) turns the
    whole baseline about the start point, and ``ha`` ("left", "center",
    "right") aligns the whole label. ``raise_pt`` lifts the whole label off
    its baseline (for example by the font descent, to match va="bottom").
    Offsets are in points, so the layout is the same in the PDF and the PNG.
    Returns the text artists.
    """
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.text import Annotation
    from matplotlib.transforms import IdentityTransform

    is_fig = isinstance(target, Figure)
    fig = target if is_fig else target.figure
    if xycoords is None:
        xycoords = "figure fraction" if is_fig else "data"
    renderer = fig.canvas.get_renderer()

    def width_pt(text, size, kw):
        probe = fig.text(0, 0, text, fontsize=size, **kw)
        w = probe.get_window_extent(renderer=renderer).width * 72.0 / fig.dpi
        probe.remove()
        return w

    widths = [width_pt(t, size, kw) for t, size, _, kw in pieces]
    advance = {"left": 0.0, "center": -0.5 * sum(widths), "right": -sum(widths)}[ha]
    c, s = np.cos(np.radians(rotation)), np.sin(np.radians(rotation))
    artists = []
    for (text, size, drop, kw), w in zip(pieces, widths):
        drop -= raise_pt
        opts = dict(xy=(x, y), xycoords=xycoords,
                    xytext=(advance * c + drop * s, advance * s - drop * c),
                    textcoords="offset points", rotation=rotation,
                    rotation_mode="anchor", ha="left", va="baseline", fontsize=size,
                    color=color, zorder=zorder, annotation_clip=False, **kw)
        if is_fig:
            ann = Annotation(text, **opts)
            ann.set_transform(IdentityTransform())
            fig.add_artist(ann)
        else:
            ann = target.annotate(text, **opts)
        artists.append(ann)
        advance += w
    return artists


def baseline_px(text_artist) -> float:
    """Display height (pixels) of the baseline of a drawn single-line text."""
    fig = text_artist.figure
    renderer = fig.canvas.get_renderer()
    box = text_artist.get_window_extent(renderer=renderer)
    probe = fig.text(0, 0.5, text_artist.get_text(),
                     fontproperties=text_artist.get_fontproperties(), va="baseline")
    descent = 0.5 * fig.bbox.height - probe.get_window_extent(renderer=renderer).y0
    probe.remove()
    return box.y0 + descent


def relative_luminance(hex_color: str) -> float:
    """WCAG 2.x relative luminance Y of an sRGB hex colour."""
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def cie_lightness(hex_color: str) -> float:
    """CIE L* (0 to 100) of an sRGB hex colour."""
    y = relative_luminance(hex_color)
    return 116 * y ** (1 / 3) - 16 if y > 0.008856 else 903.3 * y
