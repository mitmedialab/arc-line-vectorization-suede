"""Render stroke graphs for visual inspection.

The primary check this visualization enables: are the junctions
correctly classified? Look at each marked junction (drawn as a
hollow circle so the underlying ink stays visible through it):

- RED circle: pure-terminal junction. Every participating stroke ENDS
  here. Should appear where strokes converge to a common endpoint
  (e.g. a Y where three lines meet, with no through-traffic).
- BLUE circle: pure-crossing junction. Every participating stroke
  PASSES THROUGH. Should appear where strokes cross without anything
  terminating (e.g. two whiskers crossing each other in mid-air).
- YELLOW circle: mixed junction. At least one stroke terminates + at
  least one passes through. The most common kind in real drawings:
  a cat's ear endpoint sitting on a head outline that loops past,
  or a tail rooting on a body curve.

A small legend in the top-left of the rendered image repeats the
same key, so you don't have to remember which colour is which when
scanning a batch of outputs. Each junction is also labelled with its
index into ``graph.junctions`` (e.g. "0", "1", ...), so you can refer
to specific junctions unambiguously when discussing what went wrong.

Small grey lines drawn from each junction marker show the per-
participant tangent directions. For a crossing participant you'll see
two short legs (one for ``tangent_in``, one for ``tangent_out``);
for a terminal participant just one leg pointing into the polyline.
This makes it easy to spot misclassifications: a terminal participant
whose tangent leg points OUT of an obvious endpoint location, or a
crossing participant whose two legs aren't roughly opposite, both
indicate a problem upstream.
"""

from __future__ import annotations
from typing import List, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from release.graph import Participation, Role, StrokeGraph, _is_closed_polyline

# Colours for the three junction kinds (drawn as circle outlines, not
# filled, so the underlying junction pixels remain visible through them).
_COLOR_TERMINAL = (240, 80, 80)  # red: every participant terminates here
_COLOR_CROSSING = (80, 140, 240)  # blue: every participant passes through
_COLOR_MIXED = (240, 220, 80)  # yellow: at least one of each


def visualize_graph(
    binary: NDArray[np.bool_],
    graph: StrokeGraph,
    scale: int = 4,
    junction_radius: float = 6.0,
    junction_outline_width: int = 2,
    show_tangents: bool = True,
    tangent_length_px: float = 24.0,
    show_labels: bool = True,
    label_font_size: int = 12,
    show_legend: bool = True,
    legend_font_size: int = 14,
    output_path: Optional[str] = None,
) -> Image.Image:
    """Render polylines + junctions, colour-coded by junction type.

    Junctions are drawn as hollow circles (outline only) so the pixels
    underneath them remain visible - useful when you want to check
    whether a marker is sitting exactly on top of a stroke pixel or
    off to one side. Each junction is also labelled with its index
    into ``graph.junctions`` (e.g. "0", "1", "2", ...) so you can refer
    to specific junctions unambiguously.

    Args:
        binary: original binary image, used only for the faint
            background canvas so polylines sit on a recognizable
            silhouette of the original drawing.
        graph: the stroke graph to render.
        scale: integer upscale factor (nearest-neighbour) applied
            BEFORE the markers are drawn, so the junction circles and
            tangent legs stay crisp instead of being interpolated.
        junction_radius: marker circle radius in scaled-image pixels.
        junction_outline_width: stroke width of the marker circle in
            scaled-image pixels. 1 is fine but harder to spot on dense
            backgrounds; 2-3 is more legible.
        show_tangents: draw short grey legs from each junction
            indicating its participants' tangent directions.
        tangent_length_px: leg length in scaled-image pixels.
        show_labels: draw the junction index next to each marker.
        label_font_size: point size of junction index labels.
        show_legend: draw a small key in the top-left of the rendered
            image showing what each colour means.
        legend_font_size: point size of the legend labels.
        output_path: optional PNG output path.
    """
    # Reuse the segment-visualizer helpers so polyline colours match
    # across pipeline stages (segments -> fused -> graph).
    from visualize.segment import _make_canvas, _paint_segments

    canvas = _make_canvas(binary)
    _paint_segments(canvas, graph.polylines)

    H, W = binary.shape
    img = Image.fromarray(canvas)
    if scale > 1:
        img = img.resize((W * scale, H * scale), Image.Resampling.NEAREST)

    draw = ImageDraw.Draw(img)

    # Draw tangent legs first so the marker circles sit on top of them
    # rather than getting half-covered by the leg endpoints.
    if show_tangents:
        for j in graph.junctions:
            jx = float(j.location[0]) * scale
            jy = float(j.location[1]) * scale
            for p in j.participants:
                _draw_tangent_legs(draw, jx, jy, p, tangent_length_px)

    label_font = _load_legend_font(label_font_size) if show_labels else None

    for j_idx, j in enumerate(graph.junctions):
        jx = float(j.location[0]) * scale
        jy = float(j.location[1]) * scale
        if j.is_terminal and j.has_crossing:
            color = _COLOR_MIXED
        elif j.is_terminal:
            color = _COLOR_TERMINAL
        else:
            color = _COLOR_CROSSING
        _draw_circle_outline(
            draw, jx, jy, junction_radius, color, junction_outline_width
        )

        if show_labels and label_font is not None:
            # Place the label up-and-right of the marker so it doesn't
            # land on top of the right-pointing tangent leg (most legs
            # are roughly axis-aligned with the polylines they came
            # from, so up-right is a safe bet on average).
            label_x = jx + junction_radius + 3
            label_y = jy - junction_radius - label_font_size - 1
            draw.text((label_x, label_y), str(j_idx), fill=color, font=label_font)

    if show_legend:
        _draw_legend(
            draw,
            junction_radius=junction_radius,
            outline_width=junction_outline_width,
            font_size=legend_font_size,
        )

    if output_path is not None:
        img.save(output_path)
    return img


# ---------------------------------------------------------------------------


def _draw_circle_outline(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    r: float,
    color: Tuple[int, int, int],
    width: int,
) -> None:
    draw.ellipse(
        [(x - r, y - r), (x + r, y + r)],
        outline=color,
        width=width,
    )


def _load_legend_font(size: int):
    """Best-effort font loader.

    ``ImageFont.load_default(size=...)`` accepts a size argument on
    Pillow 10.1+; older Pillow ignores it and returns a fixed-size
    bitmap font. We try the modern API first, fall back to the bitmap
    default if that's all that's available - the legend stays readable
    either way, just smaller on old Pillow.
    """
    try:
        return ImageFont.load_default(size=size)
    except (AttributeError, TypeError):
        return ImageFont.load_default()


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    junction_radius: float,
    outline_width: int,
    font_size: int,
    pad: float = 14.0,
) -> None:
    """Draw a small key in the top-left of the canvas."""
    entries = [
        (_COLOR_TERMINAL, "terminal  (all strokes end here)"),
        (_COLOR_CROSSING, "crossing  (all strokes pass through)"),
        (_COLOR_MIXED, "mixed     (at least one of each)"),
    ]
    font = _load_legend_font(font_size)
    line_height = max(junction_radius * 2 + 6, float(font_size) + 6)
    for i, (color, label) in enumerate(entries):
        cx = pad + junction_radius
        cy = pad + junction_radius + i * line_height
        _draw_circle_outline(draw, cx, cy, junction_radius, color, outline_width)
        text_x = cx + junction_radius + 10
        # Vertically centre the text against the circle. PIL anchors
        # text at the top-left by default, so back off by half the
        # nominal font height.
        text_y = cy - font_size / 2
        draw.text((text_x, text_y), label, fill=(220, 220, 220), font=font)


def _draw_tangent_legs(
    draw: ImageDraw.ImageDraw,
    jx: float,
    jy: float,
    participation: Participation,
    length: float,
) -> None:
    def stroke(tan: NDArray[np.float64], color: Tuple[int, int, int]) -> None:
        ex = jx + float(tan[0]) * length
        ey = jy + float(tan[1]) * length
        draw.line([(jx, jy), (ex, ey)], fill=color, width=1)

    if participation.role == Role.CROSSING and participation.tangent_out is not None:
        stroke(participation.tangent_in, (170, 170, 170))
        stroke(participation.tangent_out, (170, 170, 170))
    else:
        # Terminal: single direction into the polyline body. Brighter so
        # it's distinguishable from a crossing leg at a glance.
        stroke(participation.tangent_in, (220, 220, 220))


def describe(
    self: StrokeGraph,
    junction_indices: List[int],
    neighborhood_radius: int = 5,
) -> str:
    """Verbose textual dump of specific junctions, for diagnosing
    classifications that look wrong in the visualization.

    For each junction prints, per participant: the participating
    polyline (with its length and whether it's closed), the chosen
    ``point_index`` and whether it's an endpoint or interior, the
    assigned role, the two tangents (or just one for TERMINAL), the
    deflection angle (which determines CUSP vs CROSSING), and the
    polyline's coordinates in a small neighborhood around the rep
    so geometry issues like "the rep is 3 points off the actual
    cusp tip" become obvious.

    Args:
        junction_indices: which junctions to describe.
        neighborhood_radius: how many polyline points on each side
            of the rep to print. 0 disables the neighborhood dump.
    """
    out: List[str] = []
    for jidx in junction_indices:
        if not 0 <= jidx < len(self.junctions):
            out.append(f"Junction {jidx}: out of range")
            continue
        j = self.junctions[jidx]
        out.append(
            f"Junction {jidx} @ ({j.location[0]:.2f}, {j.location[1]:.2f})  "
            f"is_anchored={j.is_anchored} has_crossing={j.has_crossing}"
        )
        for k, p in enumerate(j.participants):
            poly = self.polylines[p.polyline_index]
            n = len(poly)
            closed = _is_closed_polyline(poly)
            is_ep = p.point_index == 0 or p.point_index == n - 1
            pt = poly[p.point_index]
            out.append(
                f"  [{k}] poly {p.polyline_index} (len={n}, closed={closed})  "
                f"idx {p.point_index} ({'ENDPOINT' if is_ep else 'interior'})  "
                f"pt=({pt[0]:.2f}, {pt[1]:.2f})"
            )
            tin = p.tangent_in
            out.append(
                f"        role={p.role.name}  "
                f"t_arc={p.arc_length_t:.3f}  "
                f"t_in=({tin[0]:+.3f}, {tin[1]:+.3f})"
            )
            if p.tangent_out is not None:
                tout = p.tangent_out
                incoming = -tin
                cos = float(np.clip(np.dot(incoming, tout), -1.0, 1.0))
                defl = float(np.degrees(np.arccos(cos)))
                out.append(
                    f"        t_out=({tout[0]:+.3f}, {tout[1]:+.3f})  "
                    f"deflection={defl:.1f}°  "
                    f"(cusp if >= cusp_angle_threshold_deg)"
                )
            if neighborhood_radius > 0:
                lo = max(0, p.point_index - neighborhood_radius)
                hi = min(n - 1, p.point_index + neighborhood_radius)
                out.append(f"        neighborhood (idx {lo}..{hi}):")
                for offset in range(hi - lo + 1):
                    idx = lo + offset
                    x, y = poly[idx]
                    marker = "  <-- rep" if idx == p.point_index else ""
                    out.append(f"          [{idx:>4}] ({x:.2f}, {y:.2f}){marker}")
    return "\n".join(out)
