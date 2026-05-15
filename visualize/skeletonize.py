"""Visualization helpers for the skeletonize pipeline.

These functions render matplotlib figures showing what the
distance-transform-based detector found, optionally with arm pairings
overlaid, and a before/after comparison of the resolver's effect on
the skeleton.

All functions return a ``matplotlib.figure.Figure`` so the caller can
either save it (``fig.savefig(...)``) or display it (``plt.show()``).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure, SubFigure
from matplotlib.axes import Axes
from numpy.typing import NDArray

from release.skeletonize.crossings import Crossing, DetectResult

# Distinct colors for the two paired strokes of a 4-arm crossing.
# Picked to be both colorblind-friendly and clearly distinguishable
# from the red used for fat pixels.
_PAIR_COLORS = ("#2E86AB", "#E07A5F")  # blue-ish and orange-ish


def _annotation_offset(
    cy: float, cx: float, shape: Tuple[int, int], radius: int = 60
) -> Tuple[float, float]:
    """Place an annotation away from the image edge.

    Tries up-right by default; flips to down-left if too close to the
    top/right edges.
    """
    H, W = shape
    dy = -radius if cy > radius else radius
    dx = radius if cx < W - radius else -radius
    return cx + dx, cy + dy


def visualize_detection(
    binary: NDArray[np.bool_],
    detection: DetectResult,
    *,
    ax: Optional[Axes] = None,
    show_skeleton: bool = True,
    show_arms: bool = True,
    show_pairing_lines: bool = True,
    title: Optional[str] = None,
) -> Figure | SubFigure:
    """Annotate every detected ribbon-collapse crossing on the binary.

    For each detection, draws:
        - the fat pixels in red,
        - a callout box with degree, pairing score, and the length of
          skeleton inside the fat region (``chromosome_skel_length``),
        - (optional) the four arm endpoints as colored squares,
          color-coded by which pair they belong to,
        - (optional) thin lines between paired endpoints showing how
          the resolver will reconnect them.

    Arguments:
        binary: the binarized drawing.
        detection: output of ``detect_crossings``.
        ax: optional matplotlib Axes to draw into; if ``None`` a new
            figure is created.
        show_skeleton: faintly underlay the full skeleton.
        show_arms: mark each crossing's 4 arm endpoints with squares
            colored by pair.
        show_pairing_lines: draw a thin line between each pair of
            arm endpoints, mirroring what the resolver will produce.
        title: optional figure title.

    Returns:
        The figure containing the rendered visualization.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure  # type: ignore[assignment]

    ax.imshow(binary, cmap="gray_r", alpha=0.5)

    if show_skeleton:
        ys, xs = np.where(detection.skel)
        ax.scatter(xs, ys, s=0.6, c="#3F51B5", alpha=0.35, linewidths=0)

    H, W = binary.shape
    for crossing in detection.crossings:
        # Fat pixels, red.
        ax.scatter(
            crossing.fat_pixels[:, 1],
            crossing.fat_pixels[:, 0],
            s=18,
            c="red",
            alpha=0.85,
            linewidths=0,
        )

        # Callout box.
        cy, cx = crossing.centroid
        ps = (
            f"{crossing.pairing_score:.2f}"
            if crossing.pairing_score is not None
            else "—"
        )
        label = (
            f"deg = {crossing.degree}\n"
            f"pair_score = {ps}\n"
            f"skel_in_fat = {crossing.chromosome_skel_length}\n"
            f"peak_dt = {crossing.peak_dt:.1f}"
        )
        tx, ty = _annotation_offset(cy, cx, binary.shape)
        ax.annotate(
            label,
            xy=(cx, cy),
            xytext=(tx, ty),
            fontsize=8,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.3", fc="lightyellow", ec="black", alpha=0.95
            ),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
        )

        # Arm endpoints (colored by pair).
        if (show_arms or show_pairing_lines) and crossing.arm_pairing is not None:
            for pair_idx, (i, j) in enumerate(crossing.arm_pairing):
                color = _PAIR_COLORS[pair_idx]
                ep_i = crossing.arm_endpoints[i]
                ep_j = crossing.arm_endpoints[j]
                if show_arms:
                    ax.scatter(
                        [ep_i[1], ep_j[1]],
                        [ep_i[0], ep_j[0]],
                        s=80,
                        c=color,
                        marker="s",
                        edgecolors="black",
                        linewidths=1.2,
                        zorder=10,
                    )
                if show_pairing_lines:
                    ax.plot(
                        [ep_i[1], ep_j[1]],
                        [ep_i[0], ep_j[0]],
                        c=color,
                        lw=2.2,
                        alpha=0.55,
                        zorder=9,
                    )

    if title is not None:
        ax.set_title(title)

    # Legend if we used arm colors.
    if show_arms and any(c.arm_pairing is not None for c in detection.crossings):
        handles = [
            mpatches.Patch(color="red", label="fat pixels"),
            mpatches.Patch(color=_PAIR_COLORS[0], label="stroke A"),
            mpatches.Patch(color=_PAIR_COLORS[1], label="stroke B"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)  # image coordinates: y grows downward
    return fig


def visualize_resolution(
    binary: NDArray[np.bool_],
    detection: DetectResult,
    skel_before: NDArray[np.bool_],
    skel_after: NDArray[np.bool_],
    *,
    crop_to_crossings: bool = True,
    crop_padding: int = 50,
    title: Optional[str] = None,
) -> Figure:
    """Side-by-side comparison of the skeleton before and after the
    resolver runs.

    Each detected crossing is highlighted in both panels (the same red
    fat pixels), but the left panel shows the original merged-segment
    skeleton running through them and the right panel shows the
    replacement segments the resolver drew between paired arm
    endpoints.

    Arguments:
        binary: the binarized drawing.
        detection: output of ``detect_crossings``.
        skel_before: the skeleton fed to ``resolve_crossings``.
        skel_after: the skeleton it returned.
        crop_to_crossings: if True and there are detections, the figure
            is cropped to the bounding box around all detected
            crossings (plus ``crop_padding`` pixels of margin) so the
            difference is visible. If False, the full image is shown.
        crop_padding: extra pixels around the detection bounding box.
        title: optional figure suptitle.

    Returns:
        The figure with two side-by-side axes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    H, W = binary.shape
    if crop_to_crossings and detection.crossings:
        all_y = np.concatenate([c.fat_pixels[:, 0] for c in detection.crossings])
        all_x = np.concatenate([c.fat_pixels[:, 1] for c in detection.crossings])
        y0 = max(0, int(all_y.min()) - crop_padding)
        y1 = min(H, int(all_y.max()) + crop_padding + 1)
        x0 = max(0, int(all_x.min()) - crop_padding)
        x1 = min(W, int(all_x.max()) + crop_padding + 1)
    else:
        y0, y1, x0, x1 = 0, H, 0, W

    for ax, sk, sub_title in [
        (axes[0], skel_before, "before resolve_crossings"),
        (axes[1], skel_after, "after resolve_crossings"),
    ]:
        ax.imshow(binary[y0:y1, x0:x1], cmap="gray_r", alpha=0.35)
        sy, sx = np.where(sk[y0:y1, x0:x1])
        ax.scatter(sx, sy, s=2.0, c="#3F51B5", alpha=0.85, linewidths=0)
        for crossing in detection.crossings:
            in_crop = (
                (crossing.fat_pixels[:, 0] >= y0)
                & (crossing.fat_pixels[:, 0] < y1)
                & (crossing.fat_pixels[:, 1] >= x0)
                & (crossing.fat_pixels[:, 1] < x1)
            )
            ax.scatter(
                crossing.fat_pixels[in_crop, 1] - x0,
                crossing.fat_pixels[in_crop, 0] - y0,
                s=14,
                c="red",
                alpha=0.7,
                linewidths=0,
            )
        ax.set_title(sub_title)
        ax.set_aspect("equal")
        ax.axis("off")

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def visualize_pipeline(
    binary: NDArray[np.bool_],
    skel_before: NDArray[np.bool_],
    detection: DetectResult,
    skel_after: NDArray[np.bool_],
    *,
    title: Optional[str] = None,
) -> Figure:
    """Four-panel pipeline overview: binary, skeleton, detections,
    resolved skeleton.

    Useful for end-to-end inspection of one drawing.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Top-left: binary.
    axes[0, 0].imshow(binary, cmap="gray_r")
    axes[0, 0].set_title("binary")
    axes[0, 0].axis("off")

    # Top-right: skeleton before.
    axes[0, 1].imshow(binary, cmap="gray_r", alpha=0.2)
    ys, xs = np.where(skel_before)
    axes[0, 1].scatter(xs, ys, s=1, c="#3F51B5", alpha=0.9, linewidths=0)
    axes[0, 1].set_title("cleaned skeleton (input to resolver)")
    axes[0, 1].axis("off")
    axes[0, 1].set_aspect("equal")

    # Bottom-left: detections.
    visualize_detection(
        binary,
        detection,
        ax=axes[1, 0],
        title=f"detection ({len(detection.crossings)} crossings)",
    )

    # Bottom-right: resolved skeleton.
    axes[1, 1].imshow(binary, cmap="gray_r", alpha=0.2)
    ys, xs = np.where(skel_after)
    axes[1, 1].scatter(xs, ys, s=1, c="#3F51B5", alpha=0.9, linewidths=0)
    for crossing in detection.crossings:
        axes[1, 1].scatter(
            crossing.fat_pixels[:, 1],
            crossing.fat_pixels[:, 0],
            s=12,
            c="red",
            alpha=0.5,
            linewidths=0,
        )
    axes[1, 1].set_title("resolved skeleton")
    axes[1, 1].axis("off")
    axes[1, 1].set_aspect("equal")

    if title is not None:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig
