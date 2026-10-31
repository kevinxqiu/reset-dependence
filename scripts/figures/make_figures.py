#!/usr/bin/env python3
"""Regenerate the figures of the paper from the released run summaries and traces.

Run from the repository root after unzipping the dataset archives and fetching the
UNIMAL body files (docs/REPRODUCE.md, sections 1 and 4), with the Lato fonts in fonts/:

    MUJOCO_GL=egl python scripts/figures/make_figures.py           # Figures 2 to 10 and 13
    MUJOCO_GL=egl python scripts/figures/make_figures.py --render  # also Figures 1, 11, 12 and 14

Figure 6 includes two small MuJoCo renders, and --render adds the four full-scene
renders, so every run needs an offscreen OpenGL backend: EGL by default (set here when
MUJOCO_GL is unset), or MUJOCO_GL=osmesa on a machine without EGL. The scripts run in
one process, so they all use the same backend.

Figures are written to figures/. The scripts in this folder are the paper's plotting
code, so their output matches the printed figures up to renderer shading. Figure 6 is
drawn by scripts/plot_cr_fig2_cutoffs.py, which this runner calls in a separate process.
"""
from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # before any script imports mujoco

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# (paper figures, script in this folder, its arguments)
PLOTS = [
    ("Figure 2", "plot_termination_mechanism.py", ["figures/fig_protocol_timeline.pdf"]),
    ("Figures 3, 8, 9 and 10", "analyze_final.py", ["--out_dir", "figures", "--stats_out", "figures/final_stats.json"]),
    ("Figure 5", "render_temporal_gait.py", []),
    ("Figure 7", "plot_robustness_eval.py", []),
    ("Figure 4", "plot_rf_vs_strict_scatter.py", []),
    ("Figure 13", "render_boundary_free_forest.py", []),
]
RENDERS = [
    ("Figures 1 and 14", "render_chronophoto_3d.py", []),
    ("Figure 11", "render_morphology.py", []),
    ("Figure 12", "render_top1_row.py", []),
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
    print("== Figure 6: scripts/plot_cr_fig2_cutoffs.py", flush=True)
    if subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_cr_fig2_cutoffs.py")], cwd=ROOT).returncode != 0:
        failed.append("Figure 6 (plot_cr_fig2_cutoffs.py)")
    if not args.render:
        print("Skipped Figures 1, 11, 12 and 14 (MuJoCo renders); rerun with --render.")
    if failed:
        print("Failed: " + "; ".join(failed))
        return 1
    print(f"Wrote the figures to {ROOT / 'figures'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
