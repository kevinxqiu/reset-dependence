#!/usr/bin/env python3
"""Regenerate the figures of the paper from the released run summaries and traces.

Run from the repository root after unzipping the dataset archives and fetching the
UNIMAL body files (docs/REPRODUCE.md, sections 1 and 4); the renders label their panels with
the Lato fonts in fonts/:

    MUJOCO_GL=egl python scripts/figures/make_figures.py           # Figures 2 to 9 and 12
    MUJOCO_GL=egl python scripts/figures/make_figures.py --render  # also Figures 1, 10, 11 and 13

--render adds the four full-scene MuJoCo renders, which need an offscreen OpenGL
backend: EGL by default (set here when MUJOCO_GL is unset), or MUJOCO_GL=osmesa on a
machine without EGL. The scripts run in
one process, so they all use the same backend.

Figures are written to figures/. The scripts in this folder are the paper's plotting
code, so their output matches the printed figures; the renders can differ in shading
and in the grid lines of the floor between machines and MuJoCo builds.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # before any script imports mujoco

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# (paper figures, script in this folder, its arguments)
PLOTS = [
    ("Figure 2", "plot_termination_mechanism.py", ["figures/fig_protocol_timeline.pdf"]),
    ("Figures 3, 7, 8 and 9", "analyze_final.py", ["--out_dir", "figures", "--stats_out", "figures/final_stats.json"]),
    ("Figure 5", "render_temporal_gait.py", []),
    ("Figure 6", "plot_robustness_eval.py", []),
    ("Figure 4", "plot_rf_vs_strict_scatter.py", []),
    ("Figure 12", "render_boundary_free_forest.py", []),
]
RENDERS = [
    ("Figures 1 and 13", "render_chronophoto_3d.py", []),
    ("Figure 10", "render_morphology.py", []),
    ("Figure 11", "render_top1_row.py", []),
]


def register_fonts() -> None:
    from matplotlib import font_manager

    fonts = sorted((ROOT / "fonts").glob("*.ttf"))
    if not fonts:
        print("warning: no fonts/*.ttf found; text falls back to another font (docs/REPRODUCE.md, section 4)")
    for f in fonts:
        font_manager.fontManager.addfont(str(f))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render", action="store_true", help="also draw the four full-scene MuJoCo renders")
    args = parser.parse_args()

    (ROOT / "figures").mkdir(exist_ok=True)
    register_fonts()
    sys.path.insert(0, str(HERE))
    failed = []
    for label, script, argv in PLOTS + (RENDERS if args.render else []):
        print(f"== {label}: scripts/figures/{script}", flush=True)
        sys.argv = [script, *argv]
        try:
            runpy.run_path(str(HERE / script), run_name="__main__")
        except SystemExit as e:
            if e.code not in (0, None):
                failed.append(f"{label} ({script}, exit {e.code})")
        except Exception as e:  # report every failure at the end, then exit non-zero
            failed.append(f"{label} ({script}: {type(e).__name__}: {e})")
    if not args.render:
        print("Skipped Figures 1, 10, 11 and 13 (MuJoCo renders); rerun with --render.")
    if failed:
        print("Failed: " + "; ".join(failed))
        return 1
    print(f"Wrote the figures to {ROOT / 'figures'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
