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
from vetuner.drive_cycle import VALIDATION_CYCLE, generate_drive_cycle
from vetuner.engine_model import read_ve_table, simulate_engine
from vetuner.gating import gate_samples
from vetuner.iteration import iterate_calibration
from vetuner.metrics import evaluate_afr
from vetuner.pulse_width import pulse_width_map
from vetuner.sensors import apply_sensor_model
from vetuner.surface_fit import cell_sample_counts
from vetuner.ve_surface import (
    BASE_MAP_SURFACE,
    DEFAULT_MAP_AXIS,
    DEFAULT_RPM_AXIS,
    TRUE_SURFACE,
    ve_table,
    volumetric_efficiency,
)

OUTPUT_DIR = Path("outputs")
"""Directory the figures are written to."""

DPI = 200
"""Resolution of the saved figures, in dots per inch."""

GAIN = 0.6
"""Damping gain of the fixed point iteration, shared by every figure."""

PASSES = 5
"""Correction passes applied, shared by every figure."""

REFINEMENTS = ((16, 12), (31, 23), (61, 45), (121, 89))
"""Grid sizes for the refinement study, each halving the breakpoint spacing."""

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


def discretisation_error(n_rpm: int, n_map: int) -> float:
    """Measure a perfect table's worst interpolation error at that resolution.

    The table is set exactly equal to the true surface at every node, then read
    back at cell midpoints, where interpolating across a curved surface is least
    accurate. The residual is the cost of representing the surface on a finite
    grid, independent of any drive cycle.

    Parameters
    ----------
    n_rpm, n_map : int
        Number of breakpoints on each axis.

    Returns
    -------
    float
        Worst relative error at the cell midpoints, as a percentage.
    """
    rpm_axis = np.linspace(DEFAULT_RPM_AXIS[0], DEFAULT_RPM_AXIS[-1], n_rpm)
    map_axis = np.linspace(DEFAULT_MAP_AXIS[0], DEFAULT_MAP_AXIS[-1], n_map)
    table = ve_table(rpm_axis=rpm_axis, map_axis=map_axis, params=TRUE_SURFACE)

    rpm_mid = 0.5 * (rpm_axis[:-1] + rpm_axis[1:])
    map_mid = 0.5 * (map_axis[:-1] + map_axis[1:])
    grid_rpm, grid_map = np.meshgrid(rpm_mid, map_mid, indexing="ij")

    read = read_ve_table(table, grid_rpm.ravel(), grid_map.ravel(), rpm_axis, map_axis)
    true = volumetric_efficiency(grid_rpm.ravel(), grid_map.ravel(), TRUE_SURFACE)
    return float(np.abs(read / true - 1.0).max() * 100.0)


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
        train, base, gain=GAIN, max_passes=PASSES, method="gaussian_process"
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
        "VE table before and after calibration, dashed cells were never visited by the log",
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

    # --- the input log -------------------------------------------------------
    channels = (
        (log.rpm, "engine speed (rpm)", "#3b6fc4", None),
        (log.map_kpa, "manifold pressure (kPa abs)", "#4fb3d9", None),
        (log.afr, "air-fuel ratio", "#e2564a", sim.afr_target),
    )

    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 5.8), sharex=True)
    for ax, (values, label, colour, reference) in zip(axes, channels, strict=True):
        if reference is not None:
            ax.plot(log.time_s, reference, "k--", linewidth=1.0, label="target")
            ax.legend(fontsize=8, loc="upper right")
        ax.plot(log.time_s, values, color=colour, linewidth=0.9)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("time (s)", fontsize=9)
    fig.suptitle(
        f"Input log",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "input_log.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # --- coverage before and after gating ------------------------------------
    raw_counts = cell_sample_counts(log.rpm, log.map_kpa)
    count_norm = LogNorm(vmin=1, vmax=max(raw_counts.max(), 2))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    img = draw_table(
        axes[0],
        raw_counts.astype(float),
        raw_counts,
        TUNER_CMAP,
        count_norm,
        f"All logged samples — {(raw_counts > 0).sum()} of {raw_counts.size} cells",
        fmt="{:.0f}",
        void_style="grey",
    )
    draw_table(
        axes[1],
        counts.astype(float),
        counts,
        TUNER_CMAP,
        count_norm,
        f"After gating — {(counts > 0).sum()} of {counts.size} cells",
        fmt="{:.0f}",
        void_style="grey",
    )
    fig.colorbar(img, ax=axes, label="samples", fraction=0.025, pad=0.02)
    fig.suptitle(
        f"Coverage of the speed-load plane — {len(log.rpm)} samples logged, "
        f"{len(kept.rpm)} accepted",
        fontsize=11,
        y=0.97,
    )
    fig.savefig(OUTPUT_DIR / "coverage.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    pulses = pulse_width_map(calibrated)
    duty_pct = pulses.duty * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    img = draw_table(
        axes[0],
        pulses.pulse_width_ms,
        counts,
        TUNER_CMAP,
        Normalize(pulses.pulse_width_ms.min(), pulses.pulse_width_ms.max()),
        "Commanded pulse width (ms)",
        fmt="{:.2f}",
    )
    fig.colorbar(img, ax=axes[0], label="pulse width (ms)", fraction=0.04, pad=0.02)
    img = draw_table(
        axes[1],
        duty_pct,
        counts,
        TUNER_CMAP,
        Normalize(duty_pct.min(), duty_pct.max()),
        "Injector duty (%)",
        fmt="{:.1f}",
    )
    fig.colorbar(img, ax=axes[1], label="duty (%)", fraction=0.04, pad=0.02)
    fig.suptitle(
        f"Injector output",
        fontsize=11,
        y=0.97,
    )
    fig.savefig(OUTPUT_DIR / "injector_output.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    test = generate_drive_cycle(cycle=VALIDATION_CYCLE)
    before_result = simulate_engine(test, base)
    after_result = simulate_engine(test, calibrated)
    before_error = evaluate_afr(base, test).mean_abs_pct
    after_error = evaluate_afr(calibrated, test).mean_abs_pct

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.0), sharex=True, sharey=True)
    panels = (
        (axes[0], before_result, before_error, "Starting map"),
        (axes[1], after_result, after_error, "Calibrated"),
    )
    for ax, result, error, label in panels:
        ax.plot(test.time_s, result.afr_target, "k--", linewidth=1.1, label="target")
        ax.plot(
            test.time_s, result.afr_actual, color="#e2564a", linewidth=1.1, label="actual"
        )
        ax.fill_between(
            test.time_s,
            result.afr_target,
            result.afr_actual,
            color="#e2564a",
            alpha=0.18,
            interpolate=True,
        )
        ax.set_title(f"{label} - mean error {error:.2f}%", fontsize=10, loc="left")
        ax.set_ylabel("air-fuel ratio", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylim(11, 16)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].set_xlabel("time (s)", fontsize=9)
    fig.suptitle("Fuelling accuracy", fontsize=11, y=0.98)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "afr_before_after.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"coverage: {(counts > 0).sum()} of {counts.size} cells")
    print(
        f"VE range: {before.min():.1f}-{before.max():.1f}% "
        f"-> {after.min():.1f}-{after.max():.1f}%"
    )
    print(f"largest change: {change.min():+.1f} to {change.max():+.1f} points")
    print(
        f"pulse width: {pulses.pulse_width_ms.min():.2f}"
        f"-{pulses.pulse_width_ms.max():.2f} ms, "
        f"duty {duty_pct.min():.1f}-{duty_pct.max():.1f}%, "
        f"{int((~pulses.feasible).sum())} infeasible"
    )
    print(f"fuelling error: {before_error:.2f}% -> {after_error:.2f}%")
    print("discretisation floor at cell midpoints:")
    previous = None
    for n_rpm, n_map in REFINEMENTS:
        worst = discretisation_error(n_rpm, n_map)
        ratio = "-" if previous is None else f"{previous / worst:.2f}"
        print(f"  {n_rpm:3d} x {n_map:<3d}  {worst:7.4f}%  ratio {ratio}")
        previous = worst
    for name in (
        "input_log",
        "coverage",
        "ve_before_after",
        "ve_change",
        "cell_counts",
        "injector_output",
        "afr_before_after",
    ):
        print(f"wrote {OUTPUT_DIR / (name + '.png')}")


if __name__ == "__main__":
    main()