"""Generate the calibration table figures used in the report.

Renders the starting map, the corrected map, the change between them and the
per-cell sample counts as annotated grids, in the row-by-column layout used by
engine calibration software: engine speed down the rows, manifold pressure
across the columns.

Cells the log never reached are drawn in flat grey rather than coloured, so
that values the pipeline extrapolated are visually distinct from values it
measured.

Run from the repository root::

    python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from numpy.typing import NDArray

from vetuner.correction import select_samples
from vetuner.drive_cycle import generate_drive_cycle
from vetuner.engine_model import simulate_engine
from vetuner.gating import gate_samples
from vetuner.iteration import iterate_calibration
from vetuner.sensors import apply_sensor_model
from vetuner.surface_fit import cell_sample_counts
from vetuner.ve_surface import (
    BASE_MAP_SURFACE,
    DEFAULT_MAP_AXIS,
    DEFAULT_RPM_AXIS,
    ve_table,
)

OUTPUT_DIR = Path("outputs")
"""Directory the figures are written to."""

DPI = 200
"""Resolution of the saved figures, in dots per inch."""

TUNER_COLOURS = [
    (0.00, "#3b6fc4"),
    (0.20, "#4fb3d9"),
    (0.40, "#6fc96f"),
    (0.55, "#c3dc5a"),
    (0.70, "#f2de5a"),
    (0.85, "#f2a03c"),
    (1.00, "#e2564a"),
]
"""Blue to red ramp matching the convention used by engine calibration software."""

TUNER_CMAP = LinearSegmentedColormap.from_list("tuner", TUNER_COLOURS)
"""Sequential colour map for volumetric efficiency and sample counts."""

DIFF_CMAP = LinearSegmentedColormap.from_list(
    "tuner_diff", [(0.0, "#3b6fc4"), (0.5, "#eef0eb"), (1.0, "#e2564a")]
)
"""Diverging colour map for signed changes, centred on no change."""

VOID_COLOUR = "#d6dae0"
"""Fill for cells containing no log samples."""


def draw_table(
    ax: mpl.axes.Axes,
    values: NDArray[np.float64],
    counts: NDArray[np.int64] | None,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    title: str,
    fmt: str = "{:.1f}",
    void_style: str = "hatch",
    rpm_axis: NDArray[np.float64] = DEFAULT_RPM_AXIS,
    map_axis: NDArray[np.float64] = DEFAULT_MAP_AXIS,
) -> mpl.image.AxesImage:
    """Draw one annotated table grid onto an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    values : numpy.ndarray
        Cell values shaped ``(len(rpm_axis), len(map_axis))``.
    counts : numpy.ndarray, optional
        Samples per cell, used to mark which cells the log reached.
    cmap : matplotlib.colors.Colormap
        Colour map applied to the values.
    norm : matplotlib.colors.Normalize
        Value-to-colour mapping, shared between figures that must be compared.
    title : str
        Axis title.
    fmt : str, optional
        Format string for the cell annotations, by default ``"{:.1f}"``.
    void_style : str, optional
        How to mark cells with no samples, by default ``"hatch"``. Hatching
        overlays the coloured cell, so the table's shape stays readable while
        extrapolated values remain identifiable. ``"grey"`` fills those cells
        flat instead and replaces the value with a dash, which suits a table
        whose values are the sample counts themselves.
    rpm_axis, map_axis : numpy.ndarray, optional
        Table breakpoints, used for the tick labels.

    Returns
    -------
    matplotlib.image.AxesImage
        The drawn image, for attaching a colour bar.
    """
    blank = counts == 0 if counts is not None else np.zeros(values.shape, dtype=bool)
    greyed = void_style == "grey" and counts is not None

    shown = np.ma.masked_where(blank, values) if greyed else values
    image = ax.imshow(shown, cmap=cmap, norm=norm, aspect="auto", origin="upper")
    if greyed:
        image.cmap.set_bad(VOID_COLOUR)

    if counts is not None and not greyed:
        for i, j in zip(*np.nonzero(blank), strict=True):
            ax.add_patch(
                mpl.patches.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    hatch="////",
                    edgecolor="white",
                    linewidth=0,
                    alpha=0.5,
                )
            )

    ax.set_xticks(np.arange(len(map_axis)), [f"{m:.0f}" for m in map_axis], fontsize=7)
    ax.set_yticks(np.arange(len(rpm_axis)), [f"{r:.0f}" for r in rpm_axis], fontsize=7)
    ax.set_xlabel("manifold pressure (kPa abs)", fontsize=8)
    ax.set_ylabel("engine speed (rpm)", fontsize=8)
    ax.set_title(title, fontsize=10, pad=8, loc="left")

    ax.set_xticks(np.arange(len(map_axis) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rpm_axis) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            hidden = greyed and blank[i, j]
            ax.text(
                j,
                i,
                "–" if hidden else fmt.format(values[i, j]),
                ha="center",
                va="center",
                fontsize=5.6,
                color="#8b95a3" if hidden else "#14171c",
            )

    return image


def main() -> None:
    """Run the calibration and write every figure to the output directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    train = generate_drive_cycle()
    base = ve_table(params=BASE_MAP_SURFACE)

    sim = simulate_engine(train, base)
    log = apply_sensor_model(sim, seed=0)
    kept = select_samples(log, gate_samples(log).accepted)
    counts = cell_sample_counts(kept.rpm, kept.map_kpa)

    calibrated = iterate_calibration(
        train, base, gain=0.6, max_passes=5, method="gaussian_process"
    ).final_table

    before, after = base * 100.0, calibrated * 100.0
    change = after - before

    ve_norm = Normalize(
        vmin=np.floor(min(before.min(), after.min())),
        vmax=np.ceil(max(before.max(), after.max())),
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    img = draw_table(axes[0], before, counts, TUNER_CMAP, ve_norm, "Starting map")
    draw_table(axes[1], after, counts, TUNER_CMAP, ve_norm, "Corrected map")
    fig.colorbar(img, ax=axes, label="volumetric efficiency (%)", fraction=0.025, pad=0.02)
    fig.suptitle(
        "VE table before and after calibration — hatched cells were never visited by the log",
        fontsize=11,
        y=0.97,
    )
    fig.savefig(OUTPUT_DIR / "ve_before_after.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    limit = float(np.abs(change).max())
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    img = draw_table(
        ax,
        change,
        None,
        DIFF_CMAP,
        Normalize(vmin=-limit, vmax=limit),
        "Change applied by the calibration",
        fmt="{:+.1f}",
    )
    fig.colorbar(img, ax=ax, label="change in VE (percentage points)", fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ve_change.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    img = draw_table(
        ax,
        counts.astype(float),
        counts,
        TUNER_CMAP,
        LogNorm(vmin=1, vmax=max(counts.max(), 2)),
        f"Gated samples per cell — {(counts > 0).sum()} of {counts.size} cells reached",
        fmt="{:.0f}",
        void_style="grey",
    )
    fig.colorbar(img, ax=ax, label="samples", fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cell_counts.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"coverage: {(counts > 0).sum()} of {counts.size} cells")
    print(
        f"VE range: {before.min():.1f}-{before.max():.1f}% "
        f"-> {after.min():.1f}-{after.max():.1f}%"
    )
    print(f"largest change: {change.min():+.1f} to {change.max():+.1f} points")
    for name in ("ve_before_after", "ve_change", "cell_counts"):
        print(f"wrote {OUTPUT_DIR / (name + '.png')}")


if __name__ == "__main__":
    main()
