#!/usr/bin/env python3
r"""Compose the bond-disorder prediction figure grid (fully vector).

Builds a 3x2 panel figure from the temperature_pruning assets:

    row 1 : cifar_resnet    accuracy_curves | critical_line
    row 2 : mnist28         accuracy_curves | critical_line
    row 3 : sklearn_digits  accuracy_curves | critical_line

To stay crisp at any zoom level the source panels are re-rendered as *vector*
PDFs (from each dataset's cached results.json) and tiled losslessly with
PyMuPDF under a LaTeX-typeset title/label frame -- no rasterisation of the
plot interiors. Run with the project venv:

    .venv/bin/python tools/compose_bond_disorder_grid.py

Output: a vector .pdf (the artifact to \includegraphics in the paper) plus a
high-dpi .png preview, under assets/temperature_pruning/.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "assets" / "temperature_pruning"
OUT_STEM = ASSETS / "bond_disorder_prediction_grid"

# (subdir, LaTeX row label) top -> bottom
ROWS = [
    ("cifar_resnet", r"CIFAR-10 (ResNet)"),
    ("mnist28", r"MNIST"),
    ("sklearn_digits", r"scikit-learn digits"),
]
# (results key / panel function, LaTeX column header) left -> right
COLS = [
    ("accuracy_curves", r"Accuracy curves"),
    ("critical_line", r"Critical line"),
]
TITLE = r"Bond disorder prediction of the critical pruning density"

PT_PER_IN = 72.0
MIN_R2 = 0.80  # matches make_all_plots() so panels reproduce the cached PNGs

# --- frame geometry (inches) ---------------------------------------------
PANEL_H = 4.6
LEFT_MARGIN = 0.75   # room for the rotated row labels
GAP_COL = 0.30
RIGHT_MARGIN = 0.25
GAP_ROW = 0.25
BOTTOM_MARGIN = 0.25
HEADER_H = 0.45      # column-header band
TITLE_H = 0.90       # title band


def regenerate_panels(stage: Path) -> dict:
    """Re-render every panel as a vector PDF from cached results.json."""
    from temperature_pruning.plotting.plots import (plot_accuracy_curves,
                                                     plot_critical_line)

    fns = {
        "accuracy_curves": plot_accuracy_curves,
        "critical_line": plot_critical_line,
    }
    paths: dict = {}
    for subdir, _ in ROWS:
        results = json.loads((ASSETS / subdir / "results.json").read_text())
        for key, _ in COLS:
            out = stage / f"{subdir}__{key}.pdf"
            fns[key](results, str(out), min_r2=MIN_R2)
            paths[(subdir, key)] = out
    return paths


def _page_aspect(pdf_path: Path) -> float:
    with fitz.open(pdf_path) as doc:
        r = doc[0].rect
    return r.width / r.height


def build_layout(panel_paths: dict) -> dict:
    """Compute page size, per-column widths and every slot rectangle (inches)."""
    # size each column to its own panel aspect so nothing is stretched
    col_asp = [
        _page_aspect(panel_paths[(ROWS[0][0], key)]) for key, _ in COLS
    ]
    col_w = [PANEL_H * a for a in col_asp]

    width = LEFT_MARGIN + col_w[0] + GAP_COL + col_w[1] + RIGHT_MARGIN
    block_h = 3 * PANEL_H + 2 * GAP_ROW
    height = BOTTOM_MARGIN + block_h + HEADER_H + TITLE_H

    x_left = [LEFT_MARGIN, LEFT_MARGIN + col_w[0] + GAP_COL]
    block_top = BOTTOM_MARGIN + block_h  # top edge of the panel block

    slots = {}
    for r in range(len(ROWS)):
        y_top = block_top - r * (PANEL_H + GAP_ROW)
        y_bottom = y_top - PANEL_H
        for c in range(len(COLS)):
            slots[(r, c)] = (x_left[c], y_bottom, col_w[c], PANEL_H)

    return {
        "width": width,
        "height": height,
        "col_w": col_w,
        "x_left": x_left,
        "block_top": block_top,
        "header_y": BOTTOM_MARGIN + block_h + HEADER_H / 2,
        "title_y": BOTTOM_MARGIN + block_h + HEADER_H + TITLE_H / 2,
        "slots": slots,
    }


def build_frame(layout: dict, frame_pdf: Path) -> None:
    """Render the LaTeX title, column headers and row labels (vector PDF)."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
    })
    W, H = layout["width"], layout["height"]
    fig = plt.figure(figsize=(W, H))

    # title (LaTeX, not bold)
    fig.text(0.5, layout["title_y"] / H, TITLE,
             ha="center", va="center", fontsize=21)

    # column headers
    for c, (_, header) in enumerate(COLS):
        cx = layout["x_left"][c] + layout["col_w"][c] / 2
        fig.text(cx / W, layout["header_y"] / H, header,
                 ha="center", va="center", fontsize=16)

    # row labels (rotated)
    for r, (_, label) in enumerate(ROWS):
        x0, y0, w, h = layout["slots"][(r, 0)]
        ry = y0 + h / 2
        fig.text((LEFT_MARGIN / 2) / W, ry / H, label,
                 ha="center", va="center", rotation=90, fontsize=15)

    fig.savefig(frame_pdf, transparent=True)
    plt.close(fig)


def compose(layout: dict, frame_pdf: Path, panel_paths: dict) -> None:
    """Overlay each vector panel onto the frame at its slot, save pdf + png."""
    W, H = layout["width"], layout["height"]
    Hpt = H * PT_PER_IN

    doc = fitz.open(frame_pdf)
    page = doc[0]
    for r, (subdir, _) in enumerate(ROWS):
        for c, (key, _) in enumerate(COLS):
            x0, y0, w, h = layout["slots"][(r, c)]
            rect = fitz.Rect(
                x0 * PT_PER_IN,
                Hpt - (y0 + h) * PT_PER_IN,
                (x0 + w) * PT_PER_IN,
                Hpt - y0 * PT_PER_IN,
            )
            with fitz.open(panel_paths[(subdir, key)]) as src:
                page.show_pdf_page(rect, src, 0)

    pdf_path = OUT_STEM.with_suffix(".pdf")
    png_path = OUT_STEM.with_suffix(".png")
    doc.save(pdf_path)
    page.get_pixmap(dpi=200).save(png_path)
    doc.close()
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        panel_paths = regenerate_panels(stage)
        layout = build_layout(panel_paths)
        frame_pdf = stage / "_frame.pdf"
        build_frame(layout, frame_pdf)
        compose(layout, frame_pdf, panel_paths)


if __name__ == "__main__":
    main()
