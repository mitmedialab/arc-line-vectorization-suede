"""Render a DrawingCommand list to SVG.

We simulate the robot to recover the drawn primitives (line segments and
arcs) in order, then write an SVG with each *drawn* primitive coloured by
its index in the draw sequence (rainbow / hue ramp). Pen-up traversals are
shown as faint dashed grey lines so you can sanity-check ordering.
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .commands import DrawingCommand


def _hsl(hue_deg: float, sat: float = 80.0, light: float = 50.0) -> str:
    return f"hsl({hue_deg:.1f}, {sat:.0f}%, {light:.0f}%)"


def _simulate(
    commands: Sequence[DrawingCommand],
    start_pos: Tuple[float, float],
    start_heading: float,
):
    """Replay commands; collect drawn segments, pen-up segments, and bounds."""
    pos = np.array(start_pos, dtype=float)
    heading = float(start_heading)
    drawn = []  # list of dicts {kind, ...} in draw order
    pen_up = []  # list of (p0, p1) tuples
    xs, ys = [pos[0]], [pos[1]]

    def update_bounds(px, py):
        xs.append(px)
        ys.append(py)

    for cmd in commands:
        if cmd["kind"] == "spin":
            heading += math.radians(cmd["degrees"])
        elif cmd["kind"] == "line":
            new_pos = pos + cmd["distance"] * np.array(
                [math.cos(heading), math.sin(heading)]
            )
            if cmd["penDown"]:
                drawn.append(
                    {
                        "kind": "line",
                        "p0": pos.copy(),
                        "p1": new_pos.copy(),
                    }
                )
            else:
                pen_up.append((pos.copy(), new_pos.copy()))
            update_bounds(new_pos[0], new_pos[1])
            pos = new_pos
        elif cmd["kind"] == "arc":
            r = float(cmd["radius"])
            sweep = math.radians(cmd["degrees"])
            ccw = sweep > 0
            # Centre is 90deg to the left of heading for CCW, right for CW.
            normal_angle = heading + (math.pi / 2 if ccw else -math.pi / 2)
            center = pos + r * np.array(
                [math.cos(normal_angle), math.sin(normal_angle)]
            )
            start_a = math.atan2(pos[1] - center[1], pos[0] - center[0])
            end_a = start_a + sweep
            new_pos = center + r * np.array([math.cos(end_a), math.sin(end_a)])
            drawn.append(
                {
                    "kind": "arc",
                    "p0": pos.copy(),
                    "p1": new_pos.copy(),
                    "center": center.copy(),
                    "radius": r,
                    "sweep": sweep,
                }
            )
            # Sample arc for bounds
            n_samp = max(2, int(abs(sweep) * 8))
            for k in range(n_samp + 1):
                t = k / n_samp
                a = start_a + t * sweep
                update_bounds(
                    center[0] + r * math.cos(a),
                    center[1] + r * math.sin(a),
                )
            pos = new_pos
            heading += sweep
        else:
            raise ValueError(f"Unknown command kind: {cmd!r}")

    return drawn, pen_up, (min(xs), min(ys), max(xs), max(ys))


def _render_drawing_parts(
    drawn,
    pen_up,
    stroke_width,
    pen_up_stroke_width,
    show_pen_up,
):
    """SVG fragments for one drawing -- pen-up dashes, drawn primitives
    rainbow-colored by execution order, start dot and end ring -- in the
    drawing's native coordinate system. Caller is responsible for the
    outer <svg>, the background <rect>, and any wrapping <g transform>
    that places the drawing in the final layout.
    """
    parts = []

    if show_pen_up:
        parts.append('<g stroke="#bbb" stroke-dasharray="2,2" fill="none">')
        for p0, p1 in pen_up:
            parts.append(
                f'  <line x1="{p0[0]:.2f}" y1="{p0[1]:.2f}" '
                f'x2="{p1[0]:.2f}" y2="{p1[1]:.2f}" '
                f'stroke-width="{pen_up_stroke_width}" />'
            )
        parts.append("</g>")

    n = len(drawn)
    parts.append('<g fill="none" stroke-linecap="round">')
    for i, d in enumerate(drawn):
        hue = 360.0 * i / max(n, 1)
        color = _hsl(hue)
        if d["kind"] == "line":
            p0, p1 = d["p0"], d["p1"]
            parts.append(
                f'  <line x1="{p0[0]:.2f}" y1="{p0[1]:.2f}" '
                f'x2="{p1[0]:.2f}" y2="{p1[1]:.2f}" '
                f'stroke="{color}" stroke-width="{stroke_width}" />'
            )
        else:  # arc
            p0, p1 = d["p0"], d["p1"]
            r = d["radius"]
            sweep = d["sweep"]
            if abs(sweep) >= 2 * math.pi - 1e-3:
                center = d["center"]
                start_a = math.atan2(p0[1] - center[1], p0[0] - center[0])
                mid_a = start_a + sweep / 2.0
                pmx = center[0] + r * math.cos(mid_a)
                pmy = center[1] + r * math.sin(mid_a)
                sweep_flag = 1 if sweep > 0 else 0
                parts.append(
                    f'  <path d="M {p0[0]:.2f} {p0[1]:.2f} '
                    f"A {r:.2f} {r:.2f} 0 0 {sweep_flag} {pmx:.2f} {pmy:.2f} "
                    f'A {r:.2f} {r:.2f} 0 0 {sweep_flag} {p1[0]:.2f} {p1[1]:.2f}" '
                    f'stroke="{color}" stroke-width="{stroke_width}" />'
                )
            else:
                large_arc = 1 if abs(sweep) > math.pi else 0
                sweep_flag = 1 if sweep > 0 else 0
                parts.append(
                    f'  <path d="M {p0[0]:.2f} {p0[1]:.2f} '
                    f"A {r:.2f} {r:.2f} 0 {large_arc} {sweep_flag} "
                    f'{p1[0]:.2f} {p1[1]:.2f}" '
                    f'stroke="{color}" stroke-width="{stroke_width}" />'
                )
    parts.append("</g>")

    if drawn:
        first_p0 = drawn[0]["p0"]
        last_p1 = drawn[-1]["p1"]
        parts.append(
            f'<circle cx="{first_p0[0]:.2f}" cy="{first_p0[1]:.2f}" '
            f'r="2" fill="black" />'
        )
        parts.append(
            f'<circle cx="{last_p1[0]:.2f}" cy="{last_p1[1]:.2f}" '
            f'r="3" fill="none" stroke="black" stroke-width="1" />'
        )

    return parts


def commands_to_svg(
    commands,
    output_path: Optional[str] = None,
    start_pos=(0.0, 0.0),
    start_heading=0.0,
    stroke_width=1.5,
    pen_up_stroke_width=0.5,
    padding=8.0,
    show_pen_up=True,
):
    """Render a command list to an SVG file. Returns the SVG string."""
    drawn, pen_up, (minx, miny, maxx, maxy) = _simulate(
        commands, start_pos, start_heading
    )
    minx -= padding
    miny -= padding
    maxx += padding
    maxy += padding
    width = maxx - minx
    height = maxy - miny

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{minx:.2f} {miny:.2f} {width:.2f} {height:.2f}" '
        f'width="{width:.0f}" height="{height:.0f}">',
        '<rect width="100%" height="100%" fill="white" '
        f'x="{minx:.2f}" y="{miny:.2f}" />',
    ]
    parts.extend(
        _render_drawing_parts(
            drawn,
            pen_up,
            stroke_width,
            pen_up_stroke_width,
            show_pen_up,
        )
    )
    parts.append("</svg>")

    svg = "\n".join(parts)
    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(svg)
    return svg


def commands_to_svg_compare(
    commands_a: Sequence[DrawingCommand],
    commands_b: Sequence[DrawingCommand],
    output_path: Optional[str] = None,
    label_a="A",
    label_b="B",
    start_pos=(0.0, 0.0),
    start_heading=0.0,
    stroke_width=1.5,
    pen_up_stroke_width=0.5,
    padding=8.0,
    panel_gap=24.0,
    label_height=28.0,
    show_pen_up=True,
):
    """Render two command lists side by side at the same scale.

    Both panels share a unified bounding box (the union of each
    drawing's padded bbox), so a primitive at drawing-space (X, Y) in
    `commands_a` appears at exactly the same panel-relative position
    as a primitive at (X, Y) in `commands_b`. That equivalence is what
    makes "spot the difference" actually work -- if you rendered each
    panel to its own bbox, drawings of slightly different extent would
    end up at different scales and the visual diff would be muddled.
    Each panel still gets its own rainbow over its own primitives.
    """
    drawn_a, pen_up_a, bbox_a = _simulate(commands_a, start_pos, start_heading)
    drawn_b, pen_up_b, bbox_b = _simulate(commands_b, start_pos, start_heading)

    minx = min(bbox_a[0], bbox_b[0]) - padding
    miny = min(bbox_a[1], bbox_b[1]) - padding
    maxx = max(bbox_a[2], bbox_b[2]) + padding
    maxy = max(bbox_a[3], bbox_b[3]) + padding
    panel_w = maxx - minx
    panel_h = maxy - miny

    total_w = 2 * panel_w + panel_gap
    total_h = panel_h + label_height
    label_baseline = label_height * 0.7
    font_size = label_height * 0.5

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {total_w:.2f} {total_h:.2f}" '
        f'width="{total_w:.0f}" height="{total_h:.0f}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<line x1="{panel_w + panel_gap / 2:.2f}" '
        f'y1="{label_height:.2f}" '
        f'x2="{panel_w + panel_gap / 2:.2f}" '
        f'y2="{total_h:.2f}" '
        f'stroke="#e0e0e0" stroke-width="0.5" />',
        f'<text x="{panel_w / 2:.2f}" y="{label_baseline:.2f}" '
        f'text-anchor="middle" font-family="sans-serif" '
        f'font-size="{font_size:.2f}" fill="#333">{label_a}</text>',
        f'<text x="{panel_w + panel_gap + panel_w / 2:.2f}" '
        f'y="{label_baseline:.2f}" '
        f'text-anchor="middle" font-family="sans-serif" '
        f'font-size="{font_size:.2f}" fill="#333">{label_b}</text>',
    ]

    parts.append(f'<g transform="translate({-minx:.2f}, {label_height - miny:.2f})">')
    parts.extend(
        _render_drawing_parts(
            drawn_a,
            pen_up_a,
            stroke_width,
            pen_up_stroke_width,
            show_pen_up,
        )
    )
    parts.append("</g>")

    parts.append(
        f'<g transform="translate({panel_w + panel_gap - minx:.2f}, '
        f'{label_height - miny:.2f})">'
    )
    parts.extend(
        _render_drawing_parts(
            drawn_b,
            pen_up_b,
            stroke_width,
            pen_up_stroke_width,
            show_pen_up,
        )
    )
    parts.append("</g>")

    parts.append("</svg>")

    svg = "\n".join(parts)
    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(svg)
    return svg


def _primitive_length(primitive: dict) -> float:
    if primitive["kind"] == "line":
        return float(np.linalg.norm(primitive["p1"] - primitive["p0"]))
    return float(abs(primitive["sweep"]) * primitive["radius"])


def _allocate_frames_by_length(
    lengths: List[float],
    total_frames: int,
) -> List[int]:
    if not lengths:
        return []
    if total_frames <= 0:
        total_frames = len(lengths)
    sum_len = float(sum(lengths))
    if sum_len <= 1e-9:
        return [max(1, total_frames // len(lengths))] * len(lengths)

    alloc = [max(1, int(round(total_frames * (L / sum_len)))) for L in lengths]
    cur = sum(alloc)
    if cur == total_frames:
        return alloc

    # Adjust allocation to hit the target frame count exactly.
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    if cur < total_frames:
        k = 0
        while cur < total_frames:
            alloc[order[k % len(order)]] += 1
            cur += 1
            k += 1
    else:
        k = 0
        while cur > total_frames:
            idx = order[k % len(order)]
            if alloc[idx] > 1:
                alloc[idx] -= 1
                cur -= 1
            k += 1
    return alloc


def _draw_partial_primitive(
    draw: ImageDraw.ImageDraw,
    primitive: dict,
    t: float,
    color: str,
    stroke_width: int,
    map_pt,
) -> None:
    t = float(max(0.0, min(1.0, t)))
    if primitive["kind"] == "line":
        p0 = primitive["p0"]
        p1 = primitive["p1"]
        p = p0 + t * (p1 - p0)
        draw.line([map_pt(p0), map_pt(p)], fill=color, width=stroke_width)
        return

    center = primitive["center"]
    radius = float(primitive["radius"])
    sweep = float(primitive["sweep"]) * t
    p0 = primitive["p0"]
    start_a = math.atan2(p0[1] - center[1], p0[0] - center[0])
    n_samp = max(2, int(abs(sweep) * radius * 0.8))
    pts = []
    for k in range(n_samp + 1):
        a = start_a + sweep * (k / n_samp)
        p = np.array(
            [center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)]
        )
        pts.append(map_pt(p))
    draw.line(pts, fill=color, width=stroke_width)


def commands_to_svg_gif(
    commands: Sequence[DrawingCommand],
    output_path: str,
    start_pos: Tuple[float, float] = (0.0, 0.0),
    start_heading: float = 0.0,
    stroke_width: int = 2,
    padding: float = 8.0,
    scale: float = 4.0,
    fps: int = 24,
    duration_s: Optional[float] = None,
    units_per_second: float = 60.0,
    max_total_frames: int = 240,
    max_pixels_per_frame: int = 400_000,
    show_pen_up: bool = False,
    pen_up_stroke_width: int = 1,
) -> str:
    """Create an animated GIF of the drawing process in primitive order.

    The geometry and ordering match the SVG simulator: each line/arc is
    animated progressively, then the next primitive starts.
    """
    drawn, pen_up, (minx, miny, maxx, maxy) = _simulate(
        commands, start_pos, start_heading
    )

    minx -= padding
    miny -= padding
    maxx += padding
    maxy += padding
    width_px = max(1, int(math.ceil((maxx - minx) * scale)))
    height_px = max(1, int(math.ceil((maxy - miny) * scale)))
    px_count = width_px * height_px
    if px_count > max_pixels_per_frame > 0:
        shrink = math.sqrt(max_pixels_per_frame / float(px_count))
        scale *= shrink
        width_px = max(1, int(math.ceil((maxx - minx) * scale)))
        height_px = max(1, int(math.ceil((maxy - miny) * scale)))

    def map_pt(p: np.ndarray) -> Tuple[float, float]:
        return ((float(p[0]) - minx) * scale, (float(p[1]) - miny) * scale)

    lengths = [_primitive_length(d) for d in drawn]
    total_length = float(sum(lengths))
    if duration_s is None:
        duration_s = max(1.0, total_length / max(1e-6, units_per_second))
    total_frames = max(1, int(round(duration_s * max(1, fps))))
    if max_total_frames > 0:
        total_frames = min(total_frames, max_total_frames)
    frames_per_primitive = _allocate_frames_by_length(lengths, total_frames)

    base = Image.new("RGB", (width_px, height_px), "white")
    base_draw = ImageDraw.Draw(base)
    if show_pen_up:
        for p0, p1 in pen_up:
            base_draw.line(
                [map_pt(p0), map_pt(p1)],
                fill=(190, 190, 190),
                width=pen_up_stroke_width,
            )

    frames: List[Image.Image] = []
    n = len(drawn)
    for i, primitive in enumerate(drawn):
        hue = 360.0 * i / max(n, 1)
        color = _hsl(hue)
        n_frames = frames_per_primitive[i] if i < len(frames_per_primitive) else 1
        for k in range(1, n_frames + 1):
            frame = base.copy()
            draw = ImageDraw.Draw(frame)
            _draw_partial_primitive(
                draw,
                primitive,
                t=k / n_frames,
                color=color,
                stroke_width=stroke_width,
                map_pt=map_pt,
            )
            frames.append(
                frame.convert(
                    "P",
                    palette=Image.Palette.ADAPTIVE,
                    colors=256,
                    dither=Image.Dither.NONE,
                )
            )

        _draw_partial_primitive(
            base_draw,
            primitive,
            t=1.0,
            color=color,
            stroke_width=stroke_width,
            map_pt=map_pt,
        )

    if not frames:
        frames = [
            base.convert(
                "P",
                palette=Image.Palette.ADAPTIVE,
                colors=256,
                dither=Image.Dither.NONE,
            )
        ]

    frame_ms = max(1, int(round(1000 / max(1, fps))))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=frame_ms,
        loop=0,
    )
    return output_path


# ---------------------------------------------------------------------------
# Heat map: estimated firmware time per spatial cell
# ---------------------------------------------------------------------------

# Inferno-style colormap, evaluated at five keypoints and linearly
# interpolated between them. Inlined so this module stays matplotlib-free
# (matplotlib is a dev-only dependency).
_HEAT_COLORMAP_KEYS: List[Tuple[float, Tuple[int, int, int]]] = [
    (0.00, (0, 0, 4)),
    (0.25, (87, 16, 110)),
    (0.50, (188, 55, 84)),
    (0.75, (249, 142, 9)),
    (1.00, (252, 255, 164)),
]


def _heat_color(t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    keys = _HEAT_COLORMAP_KEYS
    for i in range(len(keys) - 1):
        t0, c0 = keys[i]
        t1, c1 = keys[i + 1]
        if t <= t1:
            f = 0.0 if t1 - t0 < 1e-9 else (t - t0) / (t1 - t0)
            return (
                int(round(c0[0] + f * (c1[0] - c0[0]))),
                int(round(c0[1] + f * (c1[1] - c0[1]))),
                int(round(c0[2] + f * (c1[2] - c0[2]))),
            )
    return keys[-1][1]


def _simulate_with_time(
    commands: Sequence[DrawingCommand],
    start_pos: Tuple[float, float],
    start_heading: float,
    pixels_per_inch: float,
    sample_step_px: float,
):
    """Walk ``commands`` and produce a list of ``(xy, dt, kind)`` samples,
    where each sample covers an equal slice of the parent command's
    firmware-model time and lives at the pen's spatial position during
    that slice.

    * Spins put all their time at the current pen position (the pen
      doesn't move while spinning, so the time accumulates in one cell).
    * Lines and arcs sample uniformly along the path, with at least 2
      samples and an upper bound determined by ``sample_step_px``.
    * ``kind`` is one of ``"draw"``, ``"penup"``, ``"spin"`` so callers
      can distinguish productive time from overhead.

    Also returns ``(min_x, min_y, max_x, max_y)`` covering every
    sampled position so the caller can size a grid that contains the
    full trajectory (including pen-up jumps).
    """
    # Local imports keep ``release.visualize`` matplotlib-free at import
    # time. ``optimize`` is part of the same package, so importing it
    # here is just resolving a sibling module on demand.
    from .optimize import (
        estimate_arc_time,
        estimate_line_time,
        estimate_spin_time,
        _INCHES_PER_MICROSTEP,
    )

    pos = np.array(start_pos, dtype=float)
    heading = float(start_heading)
    samples: List[Tuple[NDArrayLike, float, str]] = []
    xs = [pos[0]]
    ys = [pos[1]]

    def _bump(x: float, y: float) -> None:
        xs.append(x)
        ys.append(y)

    for cmd in commands:
        if cmd["kind"] == "spin":
            t = estimate_spin_time(cmd["degrees"])
            samples.append((pos.copy(), float(t), "spin"))
            heading += math.radians(cmd["degrees"])
        elif cmd["kind"] == "line":
            distance_px = float(cmd["distance"])
            distance_inches = distance_px / pixels_per_inch
            distance_microsteps = distance_inches / _INCHES_PER_MICROSTEP
            t = estimate_line_time(distance_microsteps)
            direction = np.array([math.cos(heading), math.sin(heading)])
            new_pos = pos + distance_px * direction
            n = max(2, int(distance_px / max(sample_step_px, 1e-6)) + 1)
            dt = t / n
            kind = "draw" if cmd["penDown"] else "penup"
            for k in range(n):
                f = (k + 0.5) / n
                xy = pos + f * distance_px * direction
                samples.append((xy, dt, kind))
                _bump(xy[0], xy[1])
            pos = new_pos
        elif cmd["kind"] == "arc":
            r = float(cmd["radius"])
            sweep = math.radians(cmd["degrees"])
            radius_inches = r / pixels_per_inch
            t = estimate_arc_time(radius_inches, cmd["degrees"])
            ccw = sweep > 0.0
            normal_angle = heading + (math.pi / 2.0 if ccw else -math.pi / 2.0)
            center = pos + r * np.array(
                [math.cos(normal_angle), math.sin(normal_angle)]
            )
            start_a = math.atan2(pos[1] - center[1], pos[0] - center[0])
            arc_len_px = abs(sweep) * r
            n = max(2, int(arc_len_px / max(sample_step_px, 1e-6)) + 1)
            dt = t / n
            for k in range(n):
                f = (k + 0.5) / n
                a = start_a + f * sweep
                xy = center + r * np.array([math.cos(a), math.sin(a)])
                samples.append((xy, dt, "draw"))
                _bump(xy[0], xy[1])
            end_a = start_a + sweep
            pos = center + r * np.array([math.cos(end_a), math.sin(end_a)])
            heading += sweep
        else:
            raise ValueError(f"Unknown command kind: {cmd!r}")

    if not samples:
        return [], (float(pos[0]), float(pos[1]), float(pos[0]), float(pos[1]))
    return samples, (min(xs), min(ys), max(xs), max(ys))


# NDArray alias kept loose because we don't import the typing helper here.
NDArrayLike = np.ndarray


def commands_to_heatmap(
    commands: Sequence[DrawingCommand],
    output_path: Optional[str] = None,
    start_pos: Tuple[float, float] = (0.0, 0.0),
    start_heading: float = 0.0,
    pixels_per_inch: float = 1.0,
    cell_size: float = 4.0,
    padding: float = 16.0,
    include_pen_up: bool = True,
    include_spin: bool = True,
    overlay_drawing: bool = True,
    overlay_color: Tuple[int, int, int, int] = (255, 255, 255, 110),
    upscale: int = 2,
    gamma: float = 0.5,
    saturation_percentile: float = 99.0,
):
    """Render a heat map of estimated firmware drawing time per spatial
    cell, returned as a PIL ``Image`` and optionally written to
    ``output_path``.

    Each command's time (computed by the same estimator that
    ``OptimizeRoute`` uses, see ``release.optimize``) is binned into a
    2D grid by its spatial trajectory:

    * **Lines / arcs** distribute their time uniformly across samples
      taken at ``sample_step = cell_size / 2`` along the path.
    * **Spins** dump all of their time at the current pen position,
      because the pen doesn't move while spinning. A run of "small
      line + spin + small line + spin" therefore lights up a single
      cell with the cumulative cost of the spins on top of the
      cumulative line draw time, which is exactly the pattern we
      want to surface.
    * **Pen-up moves** are included by default and shaded the same way
      lines are, so wasted backtracks (e.g., a 3 px reposition between
      two chain ends that should have coincided) appear as a bright
      streak between the disjoint pieces.

    Each command's time (computed by the same estimator that
    ``OptimizeRoute`` uses, see ``release.optimize``) is binned into a
    2D grid by its spatial trajectory:

    Args:
        commands: a sequence of ``DrawingCommand`` (typically
            ``LowGeometryVectorize(...).consolidated`` or
            ``OptimizeRoute(...).commands``).
        output_path: optional path to save a PNG.
        start_pos / start_heading: robot's starting pose; MUST match
            the pose used when generating ``commands`` so the
            simulator replays the same trajectory.
        pixels_per_inch: passed through to the time estimator. Same
            value you'd use with ``OptimizeRoute``.
        cell_size: heat-map cell size in *drawing pixels*. Smaller =
            sharper localization, larger = more thermal smoothing.
        padding: extra drawing-pixel margin around the trajectory's
            bounding box.
        include_pen_up / include_spin: include those command kinds in
            the heat (default True; set False to focus on pen-down
            drawing only).
        overlay_drawing: superimpose a thin translucent outline of the
            pen-down strokes so it's clear what each hot region maps
            to. Spins/pen-ups deliberately don't appear in the overlay
            — they're the things we're trying to *find*.
        overlay_color: RGBA for the overlay outline.
        upscale: integer scale factor for the output image; the heat
            grid is built at ``cell_size`` resolution and nearest-
            neighbor upscaled by this factor so the cells remain
            crisp at viewing size.
        gamma: brightness curve applied after normalization. The raw
            time distribution is dominated by a small number of very
            hot cells (a spin that dumps 4 s into one pixel sits next
            to thousands of pen-down cells with 0.01 s each), so a
            linear ramp pushes most of the drawing into the darkest
            color. ``gamma = 0.5`` (sqrt) lifts the mid-tones into a
            visible range while still letting hot spots saturate.
        saturation_percentile: per-cell time at this percentile maps
            to the brightest color; anything brighter saturates. 99
            means the top 1% of cells (typically spins and overlapping
            samples at corners) sit at the colormap's top end and the
            rest of the drawing reads against the contrast.
    """
    sample_step_px = max(cell_size * 0.5, 0.5)
    samples, (minx, miny, maxx, maxy) = _simulate_with_time(
        commands,
        start_pos,
        start_heading,
        pixels_per_inch,
        sample_step_px,
    )
    minx -= padding
    miny -= padding
    maxx += padding
    maxy += padding
    width_px = max(1.0, maxx - minx)
    height_px = max(1.0, maxy - miny)
    nx = max(1, int(math.ceil(width_px / cell_size)))
    ny = max(1, int(math.ceil(height_px / cell_size)))

    grid = np.zeros((ny, nx), dtype=np.float64)
    for xy, dt, kind in samples:
        if kind == "penup" and not include_pen_up:
            continue
        if kind == "spin" and not include_spin:
            continue
        cx = int((xy[0] - minx) / cell_size)
        cy = int((xy[1] - miny) / cell_size)
        if 0 <= cx < nx and 0 <= cy < ny:
            grid[cy, cx] += dt

    # Percentile-based saturation: use the high percentile of nonzero
    # cells (rather than the absolute max) so a single spin pixel
    # doesn't waste the entire dynamic range. Anything above the
    # percentile saturates to the brightest color.
    nonzero = grid[grid > 0.0]
    if nonzero.size == 0:
        # All zero — render a blank image and bail.
        img = Image.new("RGB", (nx * upscale, ny * upscale), color=(0, 0, 4))
        if output_path is not None:
            img.save(output_path)
        return img
    vmax = float(np.percentile(nonzero, saturation_percentile))
    if vmax <= 0.0:
        vmax = float(nonzero.max())

    # Build the RGB heat image. Gamma lifts mid-tones; clip saturates
    # the top end at the chosen percentile.
    normalized = np.clip(grid / vmax, 0.0, 1.0)
    if gamma != 1.0:
        normalized = np.power(normalized, float(gamma))
    rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
    # Sample colormap on a 256-entry lookup, then index in.
    lut = np.array(
        [_heat_color(i / 255.0) for i in range(256)], dtype=np.uint8
    )
    idx = np.clip((normalized * 255.0).astype(np.int32), 0, 255)
    rgb[:] = lut[idx]

    img = Image.fromarray(rgb, mode="RGB")
    if upscale > 1:
        img = img.resize(
            (nx * upscale, ny * upscale), resample=Image.Resampling.NEAREST
        )

    if overlay_drawing:
        # Replay just the pen-down primitives at the final image's
        # pixel resolution and stroke them in a translucent color so
        # they read as a faint outline behind the heat.
        drawn, _pen_up, _ = _simulate(commands, start_pos, start_heading)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)

        def _to_image(p) -> Tuple[float, float]:
            px = (p[0] - minx) / cell_size * upscale
            py = (p[1] - miny) / cell_size * upscale
            return (float(px), float(py))

        for d in drawn:
            if d["kind"] == "line":
                odraw.line(
                    [_to_image(d["p0"]), _to_image(d["p1"])],
                    fill=overlay_color,
                    width=max(1, upscale // 2),
                )
            else:
                center = d["center"]
                r = float(d["radius"])
                sweep = float(d["sweep"])
                start_a = math.atan2(
                    d["p0"][1] - center[1], d["p0"][0] - center[0]
                )
                n_samp = max(8, int(abs(sweep) * r * 0.5))
                pts = []
                for k in range(n_samp + 1):
                    a = start_a + sweep * (k / n_samp)
                    p = np.array(
                        [
                            center[0] + r * math.cos(a),
                            center[1] + r * math.sin(a),
                        ]
                    )
                    pts.append(_to_image(p))
                odraw.line(pts, fill=overlay_color, width=max(1, upscale // 2))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    if output_path is not None:
        img.save(output_path)
    return img
