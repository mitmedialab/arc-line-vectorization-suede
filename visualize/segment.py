from typing import List, Tuple

import colorsys
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.signal import convolve2d

from visualize.utils import stitch

from release.segment.fusion import FusionCandidate


def distinct_colors(
    n: int,
    sat: float = 0.85,
    light: float = 0.55,
    seed_hue: float = 0.05,
) -> List[Tuple[int, int, int]]:
    """Generate n visually-distinct RGB colours via golden-ratio hue spacing.

    Adjacent indices get well-separated hues regardless of `n`, which
    matters here because adjacent segments at a branch are then easy
    to tell apart.
    """
    if n <= 0:
        return []
    phi = 0.618033988749895  # golden-ratio conjugate
    out: List[Tuple[int, int, int]] = []
    h = seed_hue % 1.0
    for _ in range(n):
        r, g, b = colorsys.hls_to_rgb(h, light, sat)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
        h = (h + phi) % 1.0
    return out


def node_pixels(skel: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Return a boolean mask of node pixels (endpoints and branches).

    Useful as a visualization aid or as a starting point for building
    your own graph structure on top of the segments.
    """
    skel = skel.astype(bool)
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    deg = convolve2d(
        skel.astype(int), kernel, mode="same", boundary="fill", fillvalue=0
    )
    return skel & ((deg == 1) | (deg >= 3))


def _save_canvas(
    canvas: NDArray[np.uint8], scale: int, output_path: str | None = None
) -> Image.Image:
    H, W, _ = canvas.shape
    img = Image.fromarray(canvas)
    if scale > 1:
        img = img.resize((W * scale, H * scale), Image.Resampling.NEAREST)
    if output_path is not None:
        img.save(output_path)
    return img


def _segment_gradient_colors(
    base_rgb: Tuple[int, int, int],
    n: int,
    start_light: float = 0.10,
    end_light: float = 0.90,
) -> NDArray[np.uint8]:
    """Generate an RGB gradient for one segment while preserving hue.

    The segment's base color defines hue/saturation; only lightness changes
    from start to end to encode point ordering.
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if n == 1:
        return np.array([base_rgb], dtype=np.uint8)

    r, g, b = (base_rgb[0] / 255.0, base_rgb[1] / 255.0, base_rgb[2] / 255.0)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    # Keep saturation from the base color but clamp lightness to a
    # high-contrast range so order is clearly visible.
    lo = max(0.05, min(start_light, end_light))
    hi = min(0.95, max(start_light, end_light))
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)

    out = np.zeros((n, 3), dtype=np.uint8)
    for i, ti in enumerate(t):
        li = lo + (hi - lo) * float(ti)
        rr, gg, bb = colorsys.hls_to_rgb(h, li, s)
        out[i] = (int(round(rr * 255)), int(round(gg * 255)), int(round(bb * 255)))
    return out


def visualize_segments(
    skel: NDArray[np.bool_],
    segments: List[NDArray[np.float64]],
    scale: int = 4,
    background: Tuple[int, int, int] = (0, 0, 0),
    show_unsegmented: bool = False,
    show_node_pixels: bool = True,
    node_color: Tuple[int, int, int] = (255, 255, 255),
    output_path: str | None = None,
) -> Image.Image:
    """Render each segment in a distinct colour gradient and save as PNG.

    Args:
        skel: original boolean skeleton (used for canvas size and to
            locate node pixels).
        segments: list of polylines as returned by `segment_skeleton`.
        output_path: PNG output path.
        scale: integer upscale factor (nearest-neighbour). 1 = native
            resolution; bump up for visibility on small skeletons.
        background: RGB triple for empty pixels.
        show_unsegmented: if True, skeleton pixels not covered by any
            segment are drawn in dim grey. Should be empty in normal
            operation; useful for catching tracing bugs.
        show_node_pixels: if True, draw branch / endpoint pixels in
            `node_color` on top of the segment colours, so branches
            stand out.
        node_color: colour for the node-pixel overlay.

    Returns the rendered RGB array (also written to `output_path`).
    """
    skel = skel.astype(bool)
    H, W = skel.shape
    canvas = np.full((H, W, 3), background, dtype=np.uint8)

    if show_unsegmented:
        canvas[skel] = (60, 60, 60)

    colors = distinct_colors(len(segments))
    for color, seg in zip(colors, segments):
        if len(seg) == 0:
            continue
        # `seg` is (N, 2) of (x, y); convert to integer pixel indices.
        ix = np.clip(np.round(seg[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.round(seg[:, 1]).astype(int), 0, H - 1)
        grad = _segment_gradient_colors(color, len(seg))
        for x, y, col in zip(ix, iy, grad):
            canvas[y, x] = col

    if show_node_pixels:
        canvas[node_pixels(skel)] = node_color

    return _save_canvas(canvas, scale, output_path)


def _score_color(score: float) -> Tuple[int, int, int]:
    """Map a tangent score in [-1, 1] to an RGB triple.
    Green for high (good), grey for neutral, red for low (bad)."""
    s = max(-1.0, min(1.0, score))
    if s >= 0:
        # 0 -> grey(120,120,120); 1 -> green(0, 255, 0)
        return (int(120 * (1 - s)), int(120 + 135 * s), int(120 * (1 - s)))
    t = -s
    # 0 -> grey; 1 -> red
    return (int(120 + 135 * t), int(120 * (1 - t)), int(120 * (1 - t)))


def _make_canvas(
    binary: NDArray[np.bool_], ink_color: Tuple[int, int, int] = (40, 40, 40)
) -> NDArray[np.uint8]:
    H, W = binary.shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[binary] = ink_color
    return canvas


def _paint_segments(
    canvas: NDArray[np.uint8], segments: List[NDArray[np.float64]]
) -> None:
    H, W, _ = canvas.shape
    for color, seg in zip(distinct_colors(len(segments)), segments):
        if len(seg) == 0:
            continue
        ix = np.clip(np.round(seg[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.round(seg[:, 1]).astype(int), 0, H - 1)
        grad = _segment_gradient_colors(color, len(seg))
        for x, y, col in zip(ix, iy, grad):
            canvas[y, x] = col


def visualize_fusion_candidates(
    binary: NDArray[np.bool_],
    segments: List[NDArray[np.float64]],
    candidates: List[FusionCandidate],
    scale: int = 4,
    output_path: str | None = None,
) -> Image.Image:
    """Render every candidate connection.

    - Original binary is drawn as a faint grey background.
    - Each segment is drawn in a distinct color.
    - Each candidate's connecting path is drawn in a colour reflecting
      its tangent score (green = aligned, grey = neutral, red = opposite).
    - Endpoints involved in any candidate get a white dot.
    """
    canvas = _make_canvas(binary)
    H, W, _ = canvas.shape
    _paint_segments(canvas, segments)
    # Candidate paths: only paint over background
    for c in candidates:
        col = _score_color(c.tangent_score)
        path = c.connecting_path
        ix = np.clip(np.round(path[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.round(path[:, 1]).astype(int), 0, H - 1)
        for x, y in zip(ix, iy):
            cur = canvas[y, x]
            if cur[0] <= 50 and cur[1] <= 50 and cur[2] <= 50:
                canvas[y, x] = col
    # Endpoints
    eps = set()
    for c in candidates:
        ay, ax = (
            segments[c.seg_a][-1 if c.end_a == 1 else 0][1],
            segments[c.seg_a][-1 if c.end_a == 1 else 0][0],
        )
        by, bx = (
            segments[c.seg_b][-1 if c.end_b == 1 else 0][1],
            segments[c.seg_b][-1 if c.end_b == 1 else 0][0],
        )
        eps.add((int(round(ay)), int(round(ax))))
        eps.add((int(round(by)), int(round(bx))))
    for y, x in eps:
        if 0 <= y < H and 0 <= x < W:
            canvas[y, x] = (255, 255, 255)

    return _save_canvas(canvas, scale, output_path)


def visualize_fusion_decisions(
    binary: NDArray[np.bool_],
    segments: List[NDArray[np.float64]],
    accepted: List[FusionCandidate],
    all_candidates: List[FusionCandidate],
    scale: int = 4,
    output_path: str | None = None,
) -> Image.Image:
    """Render which fusion candidates were accepted vs rejected.

    - Background and segments as in `visualize_fusion_candidates`.
    - Rejected candidates: dim red dotted path (every other pixel).
    - Accepted candidates: bright green solid path.
    """
    canvas = _make_canvas(binary)
    H, W, _ = canvas.shape
    _paint_segments(canvas, segments)
    accepted_keys = {(c.seg_a, c.end_a, c.seg_b, c.end_b) for c in accepted}
    # Rejected first, faintly
    for c in all_candidates:
        if (c.seg_a, c.end_a, c.seg_b, c.end_b) in accepted_keys:
            continue
        path = c.connecting_path
        for k in range(0, len(path), 2):  # every other pixel
            x, y = path[k]
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                cur = canvas[yi, xi]
                if cur[0] <= 50 and cur[1] <= 50 and cur[2] <= 50:
                    canvas[yi, xi] = (130, 50, 50)
    # Accepted on top, bright
    for c in accepted:
        path = c.connecting_path
        ix = np.clip(np.round(path[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.round(path[:, 1]).astype(int), 0, H - 1)
        canvas[iy, ix] = (60, 255, 60)
    return _save_canvas(canvas, scale, output_path)


def visualize_fused(
    binary: NDArray[np.bool_],
    fused_segments: List[NDArray[np.float64]],
    scale: int = 4,
    output_path: str | None = None,
) -> Image.Image:
    """Render the fused segments. Each in a distinct color, on the binary
    background. Equivalent to `visualize_segments` but uses the binary
    image as the canvas, which keeps the original ink visible."""
    canvas = _make_canvas(binary)
    _paint_segments(canvas, fused_segments)
    return _save_canvas(canvas, scale, output_path)


def _moving_average(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    if window <= 1 or len(values) < 3:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def _segment_derivative_and_curvature(
    segment: NDArray[np.float64],
    curvature_window: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute per-point dx, dy and curvature on an (N,2) polyline."""
    if len(segment) == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    if len(segment) == 1:
        z = np.zeros(1, dtype=np.float64)
        return z, z, z

    x = segment[:, 0]
    y = segment[:, 1]

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denom = np.power(dx * dx + dy * dy, 1.5)
    denom = np.maximum(denom, 1e-6)
    curvature = np.abs(dx * ddy - dy * ddx) / denom
    curvature = _moving_average(curvature, curvature_window)
    return dx, dy, curvature


def _segment_derivative_stack_and_curvature(
    segment: NDArray[np.float64],
    curvature_window: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Compute first/second/third x/y derivatives and curvature."""
    if len(segment) == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty, empty, empty, empty, empty
    if len(segment) == 1:
        z = np.zeros(1, dtype=np.float64)
        return z, z, z, z, z, z, z

    x = segment[:, 0]
    y = segment[:, 1]

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    dddx = np.gradient(ddx)
    dddy = np.gradient(ddy)

    denom = np.power(dx * dx + dy * dy, 1.5)
    denom = np.maximum(denom, 1e-6)
    curvature = np.abs(dx * ddy - dy * ddx) / denom
    curvature = _moving_average(curvature, curvature_window)
    return dx, dy, ddx, ddy, dddx, dddy, curvature


def _robust_positive_max(values: list[NDArray[np.float64]]) -> float:
    if not values:
        return 1.0
    arr = np.concatenate(values)
    if len(arr) == 0:
        return 1.0
    vmax = float(np.percentile(arr, 95))
    return vmax if vmax > 1e-9 else 1.0


def _robust_signed_max(
    x_values: list[NDArray[np.float64]], y_values: list[NDArray[np.float64]]
) -> float:
    mags: list[NDArray[np.float64]] = []
    for xv, yv in zip(x_values, y_values):
        if len(xv) == 0:
            continue
        mags.append(np.sqrt(xv * xv + yv * yv))
    return _robust_positive_max(mags)


def _render_vector_panel(
    H: int,
    W: int,
    data: list[
        tuple[
            NDArray[np.int_], NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]
        ]
    ],
    signed_max: float,
    magnitude_max: float,
) -> NDArray[np.uint8]:
    accum = np.zeros((H, W, 3), dtype=np.float64)
    counts = np.zeros((H, W), dtype=np.float64)
    for ix, iy, xv, yv in data:
        r = _encode_signed_channel(xv, signed_max).astype(np.float64)
        g = _encode_signed_channel(yv, signed_max).astype(np.float64)
        mag = np.sqrt(xv * xv + yv * yv)
        b = _encode_positive_channel(mag, magnitude_max).astype(np.float64)
        np.add.at(accum[:, :, 0], (iy, ix), r)
        np.add.at(accum[:, :, 1], (iy, ix), g)
        np.add.at(accum[:, :, 2], (iy, ix), b)
        np.add.at(counts, (iy, ix), 1.0)

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    mask = counts > 0
    if np.any(mask):
        canvas[mask] = np.round(accum[mask] / counts[mask, None]).astype(np.uint8)
    return canvas


def _render_scalar_panel(
    H: int,
    W: int,
    data: list[tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.float64]]],
    max_value: float,
) -> NDArray[np.uint8]:
    accum = np.zeros((H, W), dtype=np.float64)
    counts = np.zeros((H, W), dtype=np.float64)
    for ix, iy, values in data:
        encoded = _encode_positive_channel(values, max_value).astype(np.float64)
        np.add.at(accum, (iy, ix), encoded)
        np.add.at(counts, (iy, ix), 1.0)

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    mask = counts > 0
    if np.any(mask):
        avg = np.round(accum[mask] / counts[mask]).astype(np.uint8)
        canvas[mask, 0] = avg
        canvas[mask, 1] = avg
        canvas[mask, 2] = avg
    return canvas


def _encode_signed_channel(
    values: NDArray[np.float64], max_abs: float
) -> NDArray[np.uint8]:
    if max_abs <= 0:
        max_abs = 1.0
    clipped = np.clip(values / max_abs, -1.0, 1.0)
    return np.round((clipped + 1.0) * 127.5).astype(np.uint8)


def _encode_positive_channel(
    values: NDArray[np.float64],
    max_value: float,
) -> NDArray[np.uint8]:
    if max_value <= 0:
        max_value = 1.0
    scaled = np.clip(values / max_value, 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def visualize_fused_geometry_channels(
    binary: NDArray[np.bool_],
    fused_segments: List[NDArray[np.float64]],
    scale: int = 4,
    derivative_max_abs: float = 1.5,
    curvature_window: int = 5,
    output_path_prefix: str | None = None,
) -> List[Image.Image]:
    """Render one RGB image per fused segment with channels:
    - R: signed dX (x derivative)
    - G: signed dY (y derivative)
    - B: local curvature magnitude

    Curvature is smoothed with a moving average so local noise does not
    dominate tiny pixel-level curvature estimates.
    """
    H, W = binary.shape
    outputs: List[Image.Image] = []
    for idx, seg in enumerate(fused_segments):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        if len(seg) == 0:
            img = _save_canvas(canvas, scale)
            outputs.append(img)
            if output_path_prefix is not None:
                img.save(f"{output_path_prefix}.seg{idx:03d}.png")
            continue

        dx, dy, curvature = _segment_derivative_and_curvature(seg, curvature_window)
        # Robust curvature normalization by upper percentile so one outlier
        # does not flatten the rest of the segment's variation.
        curvature_max = float(np.percentile(curvature, 95)) if len(curvature) else 1.0
        if curvature_max <= 1e-9:
            curvature_max = 1.0

        r = _encode_signed_channel(dx, derivative_max_abs)
        g = _encode_signed_channel(dy, derivative_max_abs)
        b = _encode_positive_channel(curvature, curvature_max)

        ix = np.clip(np.round(seg[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.round(seg[:, 1]).astype(int), 0, H - 1)
        canvas[iy, ix, 0] = r
        canvas[iy, ix, 1] = g
        canvas[iy, ix, 2] = b

        img = _save_canvas(canvas, scale)
        outputs.append(img)
        if output_path_prefix is not None:
            img.save(f"{output_path_prefix}.seg{idx:03d}.png")

    return outputs


def visualize_fused_geometry_overlay(
    binary: NDArray[np.bool_],
    fused_segments: List[NDArray[np.float64]],
    scale: int = 1,
    derivative_max_abs: float = 1.5,
    curvature_window: int = 5,
    output_path: str | None = None,
) -> Image.Image:
    """Render a single full-frame geometry image for all fused segments.

    Channels:
    - R: signed dX
    - G: signed dY
    - B: local curvature magnitude

    Every painted pixel stays at its original image coordinate. If multiple
    segments hit the same pixel, channel values are averaged.
    """
    H, W = binary.shape
    accum = np.zeros((H, W, 3), dtype=np.float64)
    counts = np.zeros((H, W), dtype=np.float64)

    segment_data: List[
        tuple[
            NDArray[np.int_],
            NDArray[np.int_],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]
    ] = []
    all_curvature: List[NDArray[np.float64]] = []

    for seg in fused_segments:
        if len(seg) == 0:
            continue
        dx, dy, curvature = _segment_derivative_and_curvature(seg, curvature_window)
        ix = np.clip(np.round(seg[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.round(seg[:, 1]).astype(int), 0, H - 1)
        segment_data.append((ix, iy, dx, dy, curvature))
        if len(curvature):
            all_curvature.append(curvature)

    if all_curvature:
        curvature_concat = np.concatenate(all_curvature)
        curvature_max = float(np.percentile(curvature_concat, 95))
    else:
        curvature_max = 1.0
    if curvature_max <= 1e-9:
        curvature_max = 1.0

    for ix, iy, dx, dy, curvature in segment_data:
        r = _encode_signed_channel(dx, derivative_max_abs).astype(np.float64)
        g = _encode_signed_channel(dy, derivative_max_abs).astype(np.float64)
        b = _encode_positive_channel(curvature, curvature_max).astype(np.float64)
        for x, y, rv, gv, bv in zip(ix, iy, r, g, b):
            accum[y, x, 0] += rv
            accum[y, x, 1] += gv
            accum[y, x, 2] += bv
            counts[y, x] += 1.0

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    mask = counts > 0
    if np.any(mask):
        canvas[mask] = np.round(accum[mask] / counts[mask, None]).astype(np.uint8)

    return _save_canvas(canvas, scale, output_path)


def visualize_fused_geometry_panels(
    binary: NDArray[np.bool_],
    fused_segments: List[NDArray[np.float64]],
    scale: int = 1,
    curvature_window: int = 5,
    output_path: str | None = None,
) -> Image.Image:
    """Create a stitched 4-panel visualization:
    1) first derivative, 2) second derivative, 3) third derivative,
    4) curvature.

    Each panel keeps segment pixels at original image coordinates.
    """
    H, W = binary.shape

    first_data: list[
        tuple[
            NDArray[np.int_], NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]
        ]
    ] = []
    second_data: list[
        tuple[
            NDArray[np.int_], NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]
        ]
    ] = []
    third_data: list[
        tuple[
            NDArray[np.int_], NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]
        ]
    ] = []
    curvature_data: list[
        tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.float64]]
    ] = []

    first_x: list[NDArray[np.float64]] = []
    first_y: list[NDArray[np.float64]] = []
    second_x: list[NDArray[np.float64]] = []
    second_y: list[NDArray[np.float64]] = []
    third_x: list[NDArray[np.float64]] = []
    third_y: list[NDArray[np.float64]] = []
    first_mag: list[NDArray[np.float64]] = []
    second_mag: list[NDArray[np.float64]] = []
    third_mag: list[NDArray[np.float64]] = []
    curvature_vals: list[NDArray[np.float64]] = []

    for seg in fused_segments:
        if len(seg) == 0:
            continue
        dx, dy, ddx, ddy, dddx, dddy, curvature = (
            _segment_derivative_stack_and_curvature(seg, curvature_window)
        )
        ix = np.clip(np.round(seg[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.round(seg[:, 1]).astype(int), 0, H - 1)

        first_data.append((ix, iy, dx, dy))
        second_data.append((ix, iy, ddx, ddy))
        third_data.append((ix, iy, dddx, dddy))
        curvature_data.append((ix, iy, curvature))

        first_x.append(dx)
        first_y.append(dy)
        second_x.append(ddx)
        second_y.append(ddy)
        third_x.append(dddx)
        third_y.append(dddy)
        first_mag.append(np.sqrt(dx * dx + dy * dy))
        second_mag.append(np.sqrt(ddx * ddx + ddy * ddy))
        third_mag.append(np.sqrt(dddx * dddx + dddy * dddy))
        curvature_vals.append(curvature)

    first_signed_max = _robust_signed_max(first_x, first_y)
    second_signed_max = _robust_signed_max(second_x, second_y)
    third_signed_max = _robust_signed_max(third_x, third_y)
    first_mag_max = _robust_positive_max(first_mag)
    second_mag_max = _robust_positive_max(second_mag)
    third_mag_max = _robust_positive_max(third_mag)
    curvature_max = _robust_positive_max(curvature_vals)

    first_panel = _render_vector_panel(
        H, W, first_data, first_signed_max, first_mag_max
    )
    second_panel = _render_vector_panel(
        H, W, second_data, second_signed_max, second_mag_max
    )
    third_panel = _render_vector_panel(
        H, W, third_data, third_signed_max, third_mag_max
    )
    curvature_panel = _render_scalar_panel(H, W, curvature_data, curvature_max)

    panel_images = [
        _save_canvas(first_panel, scale),
        _save_canvas(second_panel, scale),
        _save_canvas(third_panel, scale),
        _save_canvas(curvature_panel, scale),
    ]
    stitched = stitch(panel_images)
    if output_path is not None:
        stitched.save(output_path)
    return stitched
