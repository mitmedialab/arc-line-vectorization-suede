"""Sanity tests for the vectorize module. These verify the assumptions
behind the primitive geometry and the simulator replay.

Run with:  python3 -m vectorize.tests
"""

from __future__ import annotations
import math
import sys
import traceback
from typing import List, Sequence, Tuple, cast
import numpy as np

from release.commands import DrawingCommand
from release.vectorize.low_geometry.primitives import Arc, Line, Circle, tangent_at_end
from release.visualize import _simulate
from release.vectorize.low_geometry.fitting import (
    fit_arc,
    fit_circle,
    fit_full_circle,
    fit_polyline,
    is_closed_polyline,
)


def _approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) < tol


def _close(a, b, tol=1e-6):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b)) < tol


# ---------------------------------------------------------------------------
# Test 1: Arc(p0, p1, bulge) geometry must be self-consistent.
#
# For known cases we can compute the center and radius analytically. The
# Arc primitive's .center(), .radius(), .sweep() must agree with the
# textbook formula:   radius * (1 - cos(sweep/2)) = sagitta
# and the points p0, p1 must both lie exactly on the resulting circle.


def test_arc_quarter_circle_minor():
    """Quarter circle from (10,0) to (0,10) bulging out through (~7.07, 7.07).
    With bulge = tan(45°/2) = tan(22.5°), sweep = 90°, radius = 10, center = (0,0).
    """
    bulge = math.tan(math.pi / 8)  # tan(22.5°)
    a = Arc(np.array([10.0, 0.0]), np.array([0.0, 10.0]), bulge)
    assert _close(a.center(), [0, 0], 1e-9), f"center wrong: {a.center()}"
    assert _approx(a.radius(), 10.0), f"radius wrong: {a.radius()}"
    assert _approx(
        math.degrees(a.sweep()), 90.0
    ), f"sweep wrong: {math.degrees(a.sweep())}"
    # p0, p1 both at distance r from center
    assert _approx(np.linalg.norm(a.p0 - a.center()), 10.0)
    assert _approx(np.linalg.norm(a.p1 - a.center()), 10.0)
    # Midpoint of arc should be roughly at (7.07, 7.07)
    mid = a.point_at(0.5)
    expected_mid = np.array([10 * math.cos(math.pi / 4), 10 * math.sin(math.pi / 4)])
    assert _close(
        mid, expected_mid, 1e-6
    ), f"midpoint wrong: {mid}, expected {expected_mid}"


def test_arc_tangents_at_endpoints():
    """For a quarter-arc from (10,0) to (0,10) centered at origin (CCW sweep
    in standard math = CW on screen for y-down), the tangent at p0 should
    be perpendicular to (p0 - center) = (10, 0), so tangent is along
    (0, 1) — and pointing in the direction of motion.
    """
    bulge = math.tan(math.pi / 8)
    a = Arc(np.array([10.0, 0.0]), np.array([0.0, 10.0]), bulge)
    t0 = tangent_at_end(a, "start")
    t1 = tangent_at_end(a, "end")
    # Tangent at start: perpendicular to (p0 - center)=(10,0), pointing
    # toward p1=(0,10). So should be (0, 1).
    assert _close(t0, [0, 1], 1e-9), f"start tangent wrong: {t0}"
    # Tangent at end: perpendicular to (p1 - center)=(0,10), pointing
    # *away* from p0 along the direction of travel. p0 was at (10,0).
    # After a 90° rotation, traveling CCW, end tangent at (0,10) is (-1, 0).
    assert _close(t1, [-1, 0], 1e-9), f"end tangent wrong: {t1}"


def test_arc_major_arc():
    """An arc with |sweep| > 180° (major arc). With bulge = tan(270°/4) =
    tan(67.5°) ≈ +2.414, the arc traces 270° going from (10,0) the long
    way around to (0,10). The bulge convention puts the center on the
    SAME side of the chord as the arc midpoint (not the opposite side
    like for a minor arc). For this case the center comes out at
    (10, 10) with radius 10 — also valid since both (10,0) and (0,10)
    are exactly 10 away from (10,10)."""
    bulge = math.tan(math.radians(67.5))
    a = Arc(np.array([10.0, 0.0]), np.array([0.0, 10.0]), bulge)
    assert _approx(math.degrees(a.sweep()), 270.0), f"sweep: {math.degrees(a.sweep())}"
    assert _approx(a.radius(), 10.0, tol=1e-4), f"r: {a.radius()}"
    # Both endpoints must lie exactly on the circle the arc parameterizes.
    c = a.center()
    assert _approx(np.linalg.norm(a.p0 - c), a.radius(), 1e-4)
    assert _approx(np.linalg.norm(a.p1 - c), a.radius(), 1e-4)
    # The arc midpoint must also lie on the circle.
    mid = a.point_at(0.5)
    assert _approx(np.linalg.norm(mid - c), a.radius(), 1e-4)


def test_arc_negative_sweep():
    """Same endpoints but opposite direction: sweep = -90°. The arc goes
    CW in image coords (through (7.07, -7.07) approximately, i.e. below
    the x-axis). Bulge = tan(-22.5°) < 0.
    """
    bulge = -math.tan(math.pi / 8)
    a = Arc(np.array([10.0, 0.0]), np.array([0.0, 10.0]), bulge)
    assert _approx(math.degrees(a.sweep()), -90.0), f"sweep: {math.degrees(a.sweep())}"
    # Center is on the OPPOSITE side from the +sweep case
    # For a -90° arc from (10,0) to (0,10), center is at (10, 10)
    assert _close(a.center(), [10, 10], 1e-6), f"center: {a.center()}"


# ---------------------------------------------------------------------------
# Test 2: fit_arc on a known sampled arc should recover the same arc.


def test_fit_arc_recovers_known_geometry():
    """Sample 100 points along a known arc (60° sweep, r=50, center=(100,100))
    and verify fit_arc gives back the same parameters."""
    cx, cy, r = 100.0, 100.0, 50.0
    theta0 = math.radians(30.0)
    sweep = math.radians(60.0)
    n = 100
    pts = np.array(
        [
            [
                cx + r * math.cos(theta0 + sweep * t / (n - 1)),
                cy + r * math.sin(theta0 + sweep * t / (n - 1)),
            ]
            for t in range(n)
        ]
    )
    arc, rms = fit_arc(pts)
    assert arc is not None, "fit_arc returned None"
    assert _approx(arc.radius(), r, tol=0.1), f"radius: {arc.radius()}"
    assert _close(arc.center(), [cx, cy], tol=0.1), f"center: {arc.center()}"
    assert _approx(
        math.degrees(arc.sweep()), 60.0, tol=1.0
    ), f"sweep: {math.degrees(arc.sweep())}"


def test_fit_full_circle_on_clean_circle():
    """Sample 200 points on a perfect circle. fit_full_circle should
    give negligible rms and the exact center/radius."""
    cx, cy, r = 50.0, 80.0, 30.0
    pts = np.array(
        [
            [
                cx + r * math.cos(2 * math.pi * i / 200),
                cy + r * math.sin(2 * math.pi * i / 200),
            ]
            for i in range(201)  # closed loop
        ]
    )
    circle, rms = fit_full_circle(pts)
    assert circle is not None
    assert _close(circle.center, [cx, cy], 0.1), f"center: {circle.center}"
    assert _approx(circle.radius, r, 0.1), f"r: {circle.radius}"
    assert rms < 0.1, f"rms: {rms}"


def test_fit_full_circle_on_noisy_circle():
    """A noisy hand-drawn-like circle. fit_full_circle should still find a
    reasonable center/radius even with several pixels of noise."""
    rng = np.random.default_rng(0)
    cx, cy, r = 50.0, 80.0, 30.0
    n = 200
    angles = np.linspace(0, 2 * math.pi, n)
    pts = np.column_stack(
        [
            cx + r * np.cos(angles) + rng.normal(0, 2.0, n),
            cy + r * np.sin(angles) + rng.normal(0, 2.0, n),
        ]
    )
    # Make it closed
    pts = np.vstack([pts, pts[0:1]])
    circle, rms = fit_full_circle(pts)
    assert circle is not None
    err_c = np.linalg.norm(circle.center - np.array([cx, cy]))
    err_r = abs(circle.radius - r)
    assert err_c < 1.0, f"center off by {err_c}"
    assert err_r < 1.0, f"r off by {err_r}"
    # rms should be ~ noise stddev ≈ 2 px
    assert 1.0 < rms < 4.0, f"rms: {rms}"


def test_fit_polyline_recognizes_closed_circle():
    """A clean closed circular polyline should yield a single Circle
    primitive in the chain."""
    cx, cy, r = 50.0, 80.0, 30.0
    pts = np.array(
        [
            [
                cx + r * math.cos(2 * math.pi * i / 100),
                cy + r * math.sin(2 * math.pi * i / 100),
            ]
            for i in range(101)
        ]
    )
    chain = fit_polyline(pts, use_dp=False)
    assert len(chain) == 1, f"expected 1 piece, got {len(chain)}"
    assert isinstance(chain[0].primitive, Circle), f"got {type(chain[0].primitive)}"
    assert _approx(chain[0].primitive.radius, r, 0.5)


def test_fit_polyline_recognizes_noisy_closed_loop_as_circle():
    """A NOISY closed loop should ALSO get a single Circle (the whole
    point of the loose closed-loop tolerance). This is what hand-drawn
    wheels look like."""
    rng = np.random.default_rng(1)
    cx, cy, r = 50.0, 80.0, 30.0
    n = 200
    angles = np.linspace(0, 2 * math.pi, n)
    pts = np.column_stack(
        [
            cx + r * np.cos(angles) + rng.normal(0, 1.0, n),
            cy + r * np.sin(angles) + rng.normal(0, 1.0, n),
        ]
    )
    pts = np.vstack([pts, pts[0:1]])
    chain = fit_polyline(pts, use_dp=False)
    assert (
        len(chain) == 1
    ), f"noisy circle should be 1 piece, got {len(chain)}: {[type(c.primitive).__name__ for c in chain]}"
    assert isinstance(chain[0].primitive, Circle)


def test_fit_polyline_near_closed_wheel_becomes_circle():
    """A polyline tracing 350° of a wheel with a small endpoint gap
    (the stroke didn't quite meet itself) should be recognized as a
    Circle. fit_arc would produce a near-360° single Arc which is
    geometrically nonsense — the bulge magnitude approaches infinity
    as the chord shrinks."""
    cx, cy, r = 50.0, 80.0, 30.0
    n = 200
    pts = np.array(
        [
            [
                cx + r * math.cos(math.radians(5) + math.radians(350) * i / (n - 1)),
                cy + r * math.sin(math.radians(5) + math.radians(350) * i / (n - 1)),
            ]
            for i in range(n)
        ]
    )
    chain = fit_polyline(pts, use_dp=False)
    assert (
        len(chain) == 1
    ), f"got {len(chain)} pieces: {[type(c.primitive).__name__ for c in chain]}"
    assert isinstance(chain[0].primitive, Circle), f"got {type(chain[0].primitive)}"
    c = chain[0].primitive
    assert _approx(c.radius, r, 0.5)
    assert _close(c.center, [cx, cy], 0.5)


def test_fit_arc_rejects_near_full_circle():
    """fit_arc must refuse to fit a 340° arc as a single Arc."""
    cx, cy, r = 50.0, 80.0, 30.0
    n = 100
    sweep_deg = 340.0
    pts = np.array(
        [
            [
                cx + r * math.cos(math.radians(sweep_deg) * i / (n - 1)),
                cy + r * math.sin(math.radians(sweep_deg) * i / (n - 1)),
            ]
            for i in range(n)
        ]
    )
    arc, rms = fit_arc(pts)
    assert arc is None, f"fit_arc should return None for 340° sweep, got {arc}"


def test_count_corners_pure_circle():
    """A clean circle polyline has NO corners."""
    rng = np.random.default_rng(0)
    cx, cy, r = 100.0, 100.0, 50.0
    n = 300
    angles = np.linspace(0, 2 * math.pi, n)
    pts = np.column_stack(
        [
            cx + r * np.cos(angles) + rng.normal(0, 0.5, n),
            cy + r * np.sin(angles) + rng.normal(0, 0.5, n),
        ]
    )
    from release.vectorize.low_geometry.fitting import count_corners

    nc = count_corners(pts)
    assert nc == 0, f"clean circle should have 0 corners, got {nc}"


def _synthetic_cat_head_with_ears(cx=100.0, cy=100.0, r=50.0):
    """Helper: build a closed polyline tracing a cat head with two
    triangular ears, similar to what hand-drawn cats look like.
    Each ear is a triangle with multiple points along each side."""
    pts: List[Tuple[float, float]] = []
    # bottom half (in y-down image coords, sin > 0 is the bottom)
    for t in np.linspace(0.0, math.pi, 80):
        pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    # left side up to where left ear starts (going from theta=pi to theta=pi+0.4)
    for t in np.linspace(math.pi, math.pi + 0.4, 15):
        pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    # left ear: triangle pointing outward at angle ~pi+0.7
    ear_dir = (math.cos(math.pi + 0.7), math.sin(math.pi + 0.7))
    base_a = (cx + r * math.cos(math.pi + 0.4), cy + r * math.sin(math.pi + 0.4))
    apex_l = (cx + 1.6 * r * ear_dir[0], cy + 1.6 * r * ear_dir[1])
    base_b = (cx + r * math.cos(math.pi + 1.0), cy + r * math.sin(math.pi + 1.0))
    for i in range(1, 10):
        a = i / 10.0
        pts.append(
            (
                base_a[0] + a * (apex_l[0] - base_a[0]),
                base_a[1] + a * (apex_l[1] - base_a[1]),
            )
        )
    pts.append(apex_l)
    for i in range(1, 10):
        a = i / 10.0
        pts.append(
            (
                apex_l[0] + a * (base_b[0] - apex_l[0]),
                apex_l[1] + a * (base_b[1] - apex_l[1]),
            )
        )
    # top between ears
    for t in np.linspace(math.pi + 1.0, math.pi + 1.7, 20):
        pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    # right ear
    ear_dir = (math.cos(math.pi + 2.0), math.sin(math.pi + 2.0))
    base_a = (cx + r * math.cos(math.pi + 1.7), cy + r * math.sin(math.pi + 1.7))
    apex_r = (cx + 1.6 * r * ear_dir[0], cy + 1.6 * r * ear_dir[1])
    base_b = (cx + r * math.cos(math.pi + 2.3), cy + r * math.sin(math.pi + 2.3))
    for i in range(1, 10):
        a = i / 10.0
        pts.append(
            (
                base_a[0] + a * (apex_r[0] - base_a[0]),
                base_a[1] + a * (apex_r[1] - base_a[1]),
            )
        )
    pts.append(apex_r)
    for i in range(1, 10):
        a = i / 10.0
        pts.append(
            (
                apex_r[0] + a * (base_b[0] - apex_r[0]),
                apex_r[1] + a * (base_b[1] - apex_r[1]),
            )
        )
    # right side back to the start
    for t in np.linspace(math.pi + 2.3, 2 * math.pi, 25):
        pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return np.array(pts)


def test_count_corners_cat_head_with_ears():
    """A closed polyline tracing a cat-head outline with two
    triangular ears should be detected as having ≥2 corners (the ear
    tips). This is what must reject the Circle shortcut."""
    pts = _synthetic_cat_head_with_ears()
    from release.vectorize.low_geometry.fitting import count_corners

    nc = count_corners(pts)
    assert nc >= 2, f"cat-head-with-2-ears should have ≥2 corners, got {nc}"


def test_fit_polyline_cat_head_with_ears_not_a_circle():
    """A closed polyline with 2 ear-tip corners must NOT be returned
    as a single Circle primitive."""
    pts = _synthetic_cat_head_with_ears()
    chain = fit_polyline(pts, use_dp=False)
    if len(chain) == 1:
        assert not isinstance(
            chain[0].primitive, Circle
        ), f"cat-with-ears must not collapse to a Circle"


def test_fit_polyline_open_triangular_ear_has_sharp_apex():
    """An open polyline tracing one triangular cat ear (up one side,
    across the apex, down the other side) should fit cleanly as a
    small chain whose joint at the ear apex is a sharp angle, NOT a
    smooth curve. This regression-tests the corner-split path."""
    # Base at (0, 50), apex at (50, 0), base at (100, 50).
    # Each side is 71 px, the apex angle is 90°.
    n_side = 40
    side_up = np.column_stack(
        [
            np.linspace(0, 50, n_side),
            np.linspace(50, 0, n_side),
        ]
    )
    side_down = np.column_stack(
        [
            np.linspace(50, 100, n_side)[1:],
            np.linspace(0, 50, n_side)[1:],
        ]
    )
    pts = np.vstack([side_up, side_down])

    from release.vectorize.low_geometry.fitting import find_corners

    cs = find_corners(pts)
    assert len(cs) >= 1, f"should find at least 1 corner, got {cs}"

    chain = fit_polyline(pts, use_dp=False)
    # Two pieces, end of first ≈ apex (50, 0), start of second ≈ apex
    assert len(chain) == 2, (
        f"ear should fit as 2 pieces, got {len(chain)}: "
        f"{[type(c.primitive).__name__ for c in chain]}"
    )
    p1 = chain[0].primitive
    p2 = chain[1].primitive
    if isinstance(p1, Line):
        end1 = p1.p1
    else:
        end1 = cast(Arc, p1).p1
    apex_distance = np.linalg.norm(np.asarray(end1) - np.array([50.0, 0.0]))
    assert (
        apex_distance < 3.0
    ), f"first piece end {end1} should be at apex (50, 0), off by {apex_distance:.1f}px"


def test_merge_arc_pairs_combines_two_half_circles():
    """Two arcs whose centers and radii nearly match and whose
    endpoints close up should merge into a single Circle. This is the
    bird-head case (upstream segmenter splits a closed circle into
    two open ~180° polylines)."""
    from release.vectorize.low_geometry.beautify import merge_arc_pairs

    cx, cy, r = 100.0, 100.0, 50.0
    # Upper half: from (150, 100) CCW to (50, 100). Sweep = 180°,
    # bulge = tan(45°) = 1.0
    upper = Arc(np.array([cx + r, cy]), np.array([cx - r, cy]), bulge=1.0)
    # Lower half: from (50, 100) CCW to (150, 100). Sweep = 180°, bulge = 1.0
    lower = Arc(np.array([cx - r, cy]), np.array([cx + r, cy]), bulge=1.0)
    # Add small noise to centers/radii to simulate "almost same circle"
    # Note: we can't directly perturb center/radius — Arc derives them
    # from endpoints+bulge. So slightly perturb the endpoints.
    upper2 = Arc(
        np.array([cx + r + 0.5, cy + 0.5]),
        np.array([cx - r - 0.3, cy - 0.2]),
        bulge=1.02,
    )
    lower2 = Arc(
        np.array([cx - r - 0.3, cy - 0.2]),
        np.array([cx + r + 0.5, cy + 0.5]),
        bulge=0.98,
    )

    new_prims, pairs = merge_arc_pairs([upper2, lower2])
    assert len(pairs) == 1, f"should merge 1 pair, got {len(pairs)}"
    assert (
        len(new_prims) == 1
    ), f"should have 1 primitive after merge, got {len(new_prims)}"
    assert isinstance(
        new_prims[0], Circle
    ), f"merged should be Circle, got {type(new_prims[0])}"
    c = new_prims[0]
    assert abs(c.radius - r) < 2.0, f"radius off: {c.radius} vs {r}"
    assert np.linalg.norm(c.center - np.array([cx, cy])) < 2.0


def test_merge_arc_pairs_skips_unrelated_arcs():
    """Two arcs with different centers should not be merged."""
    from release.vectorize.low_geometry.beautify import merge_arc_pairs

    a = Arc(np.array([0.0, 0.0]), np.array([10.0, 0.0]), bulge=0.5)
    b = Arc(np.array([100.0, 100.0]), np.array([110.0, 100.0]), bulge=0.5)
    _, pairs = merge_arc_pairs([a, b])
    assert pairs == [], f"unrelated arcs should not merge, got {pairs}"


def test_merge_arc_pairs_skips_crescent_moon():
    """Two arcs forming a CRESCENT — same endpoints, opposite bulges,
    but materially different radii — must NOT be merged into a Circle.
    A crescent moon's outer arc is wider than its inner arc by 15-30%.

    This regression-tests the personmoon case where r=158 (outer arc)
    and r=133 (inner arc) made a crescent that briefly got
    false-positive-merged when the radius tolerance was too loose.
    """
    from release.vectorize.low_geometry.beautify import merge_arc_pairs

    # Two arcs with same endpoints, opposite bulges, ~17% radius diff.
    # Endpoints far apart; both sweep > 180° to clear the 320° gate.
    outer = Arc(
        np.array([100.0, 100.0]),
        np.array([100.0, 300.0]),
        bulge=2.0,  # sweep = 4*atan(2) ≈ 252°
    )
    inner = Arc(
        np.array([100.0, 100.0]),
        np.array([100.0, 300.0]),
        bulge=-1.6,  # sweep = -4*atan(1.6) ≈ -232°
    )
    r_outer = outer.radius()
    r_inner = inner.radius()
    rel = abs(r_outer - r_inner) / max(r_outer, r_inner)
    assert (
        rel > 0.10
    ), f"this test only makes sense if the radii differ by >10%, got {rel:.2f}"
    _, pairs = merge_arc_pairs([outer, inner])
    assert pairs == [], (
        f"crescent (radii {r_outer:.0f} vs {r_inner:.0f}) "
        f"should not merge, got {pairs}"
    )


def test_merge_arc_pairs_skips_eggmoon_style_crescent():
    """The real eggmoon crescent has radii within 5% (passes the
    radius gate!) but its outer-edge arc sweeps ~224° and its inner-
    edge arc sweeps ~150° — a 74° sweep difference. A true circle
    traced as two halves has both arcs sweeping ~180° (within a few
    degrees of each other). Reject the pair when sweep magnitudes
    differ by more than ~30-40°.

    This regression-tests the failure where the eggmoon crescent
    incorrectly merged into a full Circle despite the test_skips_
    crescent_moon test passing (the synthetic test had 17% radius
    diff, real data has only 4%, but the sweep difference is the
    real signal)."""
    from release.vectorize.low_geometry.beautify import merge_arc_pairs

    # Approximate the real eggmoon crescent's geometry: two arcs
    # with the same endpoints, similar radii, but very different
    # sweeps. Use the chord (283,73)→(317,261) ≈ 191 px chord.
    # For an arc of sweep θ with chord c, radius = c / (2*sin(θ/2))
    # For sweep 224°: r = 191 / (2*sin(112°)) ≈ 103
    # For sweep 150°: r = 191 / (2*sin(75°)) ≈ 99
    # bulge = tan(sweep/4): sweep 224° → bulge tan(56°) ≈ 1.48
    # sweep -150° → bulge tan(-37.5°) ≈ -0.77
    outer = Arc(
        np.array([283.0, 73.0]),
        np.array([317.0, 261.0]),
        bulge=1.48,
    )
    inner = Arc(
        np.array([283.0, 73.0]),
        np.array([317.0, 261.0]),
        bulge=-0.77,
    )
    r_outer = outer.radius()
    r_inner = inner.radius()
    rel = abs(r_outer - r_inner) / max(r_outer, r_inner)
    # Verify our test scenario: radii ARE within 10% (so the radius
    # gate wouldn't reject), but the sweep difference is the saving
    # discriminator.
    assert rel < 0.10, (
        f"test premise broken: radii should be within 10% to exercise "
        f"the sweep check, got {rel:.2f}"
    )
    _, pairs = merge_arc_pairs([outer, inner])
    assert pairs == [], (
        f"eggmoon-style crescent (radii close, sweeps very different) "
        f"must not merge, got {pairs}"
    )


def test_simulator_circle_lands_at_correct_position():
    """A Circle primitive at center (100, 200) with radius 30, when
    emitted as commands and re-simulated, should produce an arc whose
    center matches (100, 200) — not displaced by routing's stale
    heading. Regression test for the bikelove wheel offset bug."""
    from release.vectorize.low_geometry.primitives import Circle
    from release.vectorize.low_geometry.routing import to_commands, order_primitives
    from release.visualize import _simulate

    circle = Circle(center=np.array([100.0, 200.0]), radius=30.0)
    prims = [circle]
    start_pos = np.zeros(2)
    tour = order_primitives(prims, start_pos)
    cmds = to_commands(prims, tour, start_pos, start_heading=0.0)
    drawn, _, _ = _simulate(cmds, (start_pos[0], start_pos[1]), 0.0)

    # Find the full-circle arc in the drawn ops
    circles_drawn = [
        d
        for d in drawn
        if d["kind"] == "arc" and abs(math.degrees(d["sweep"])) >= 350.0
    ]
    assert (
        len(circles_drawn) == 1
    ), f"expected 1 full-circle replay, got {len(circles_drawn)}"
    drawn_center = circles_drawn[0]["center"]
    err = np.linalg.norm(drawn_center - np.array([100.0, 200.0]))
    assert (
        err < 1.0
    ), f"replayed circle center {drawn_center} != primitive center (100, 200), off by {err:.1f}px"


def test_fit_polyline_half_heart_does_not_collapse_to_single_arc():
    """A polyline tracing one half of a heart (one lobe + the cusp +
    the bottom point) must NOT be returned as a single Arc primitive.
    Half-hearts have systematic curvature variation that an arc
    averages away (turning the heart into a semi-oval). This is the
    bikelove heart regression test.

    The polyline traces a half-heart: smooth lobe-bulge on one side,
    sharper turn near the top cusp, smooth descent to bottom point.
    """
    n = 200
    # Parametric half-heart: t ∈ [0, π] gives the right half
    t = np.linspace(0.0, math.pi, n)
    pts = np.column_stack(
        [
            16 * np.sin(t) ** 3,
            -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)),
        ]
    )
    chain = fit_polyline(pts, use_dp=False)
    # A half-heart fitted as a single Arc loses the cusp; ensure we
    # got more than one piece OR a non-Arc piece.
    if len(chain) == 1:
        from release.vectorize.low_geometry.primitives import Arc as _Arc

        assert not isinstance(
            chain[0].primitive, _Arc
        ), "half-heart must not collapse to a single Arc"


def test_fit_polyline_extracts_closed_subloop():
    """A polyline that traces a near-closed loop (e.g., wheel rim)
    and then continues into separate strokes should have the loop
    extracted as a Circle, not fit as a chain of arcs that don't
    close. This is the bikelove right-wheel pattern: poly[14]
    traces the wheel rim for ~650 indices, then continues to the
    bottom squiggle."""
    # Build a synthetic polyline: 200 pts on a circle, then a
    # straight line away
    cx, cy, r = 100.0, 100.0, 50.0
    n_loop = 200
    angles = np.linspace(0, 2 * math.pi, n_loop)
    loop_pts = np.column_stack(
        [
            cx + r * np.cos(angles),
            cy + r * np.sin(angles),
        ]
    )
    # End point of the loop is close to start point (the loop closes).
    # Now extend with a line going right
    tail_n = 50
    tail_pts = np.column_stack(
        [
            np.linspace(loop_pts[-1, 0], loop_pts[-1, 0] + 200, tail_n),
            np.linspace(loop_pts[-1, 1], loop_pts[-1, 1] + 5, tail_n),
        ]
    )
    pts = np.vstack([loop_pts, tail_pts[1:]])  # don't duplicate the joining point

    from release.vectorize.low_geometry.fitting import find_closed_subloop

    subloop = find_closed_subloop(pts)
    assert subloop is not None, "should find the embedded near-closed loop"
    i, j = subloop
    # The sub-loop should be approximately at the start of the polyline
    assert i < 20, f"sub-loop should start near index 0, got i={i}"
    assert (
        j > 150 and j < n_loop + 10
    ), f"sub-loop should end near index {n_loop}, got j={j}"

    # Now check that fit_polyline extracts a Circle for the loop portion.
    from release.vectorize.low_geometry.fitting import fit_polyline
    from release.vectorize.low_geometry.primitives import Circle as _C

    chain = fit_polyline(pts)
    # There should be a Circle somewhere in the chain
    has_circle = any(isinstance(c.primitive, _C) for c in chain)
    assert has_circle, (
        f"chain should contain a Circle for the closed sub-loop, "
        f"got primitives: {[type(c.primitive).__name__ for c in chain]}"
    )


def test_fit_polyline_oval_is_not_a_circle():
    """A closed oval polyline should NOT be reduced to a single
    Circle, even when the circle fit's RMS is below the usual gate.
    The cat-head outline in catcar (bbox 444x339, aspect 1.31) fits
    a circle with rms 3.5% but rendering it as a perfect circle
    overshoots in the vertical axis by ~25 px, pushing the head
    down into the car body. The polyline should chain-subdivide
    into multiple arcs that follow the oval instead.
    """
    # Synthesize a closed oval polyline (axes a=200, b=140).
    n_pts = 300
    angles = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
    a, b = 200.0, 140.0
    pts = np.column_stack(
        [
            500.0 + a * np.cos(angles),
            400.0 + b * np.sin(angles),
        ]
    )
    pts = np.vstack([pts, pts[0:1]])  # close the loop

    from release.vectorize.low_geometry.fitting import fit_polyline, fit_full_circle
    from release.vectorize.low_geometry.primitives import Circle as _C

    # Sanity: the oval IS well-fit by a circle in RMS terms.
    circle, rms = fit_full_circle(pts)
    extent = math.hypot(
        pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min()
    )
    assert circle is not None
    assert rms / extent < 0.06, (
        f"test premise broken: oval should pass the 6% RMS gate, "
        f"got {rms/extent:.3f}"
    )

    # But the chain should NOT be a single Circle.
    chain = fit_polyline(pts)
    if len(chain) == 1 and isinstance(chain[0].primitive, _C):
        assert False, "oval (aspect 1.43) collapsed to a single Circle"
    # Chain should have multiple pieces tracing the oval.
    assert len(chain) >= 2, (
        f"oval should chain-subdivide into multiple primitives, got "
        f"{len(chain)} pieces: {[type(c.primitive).__name__ for c in chain]}"
    )


# ---------------------------------------------------------------------------
# Test 3: simulator replay should reach the primitive's endpoint.


def test_simulator_arc_lands_at_p1():
    """If we emit the right (spin, arc) commands for a known arc, the
    simulator must land exactly at the arc's p1."""
    bulge = math.tan(math.pi / 8)  # 90° sweep
    arc = Arc(np.array([10.0, 0.0]), np.array([0.0, 10.0]), bulge)
    t0 = tangent_at_end(arc, "start")
    start_heading = math.atan2(t0[1], t0[0])
    r = arc.radius()
    sweep_deg = math.degrees(arc.sweep())
    cmds: Sequence[DrawingCommand] = [
        {"kind": "spin", "degrees": math.degrees(start_heading)},
        {"kind": "arc", "radius": r, "degrees": sweep_deg},
    ]
    ops = _simulate(cmds, (10.0, 0.0), 0.0)
    # The single arc op should end at (0, 10)
    arc_op = [op for op in ops if op[0] == "arc"][0]
    assert _close(arc_op[1]["p1"], [0.0, 10.0], 1e-6), f"sim p1: {arc_op[1]['p1']}"


def test_simulator_arc_lands_at_p1_for_major_arc():
    """Same test but for a major (270°) arc."""
    bulge = math.tan(math.radians(67.5))  # 270° sweep
    arc = Arc(np.array([10.0, 0.0]), np.array([0.0, 10.0]), bulge)
    t0 = tangent_at_end(arc, "start")
    start_heading = math.atan2(t0[1], t0[0])
    r = arc.radius()
    sweep_deg = math.degrees(arc.sweep())
    cmds: Sequence[DrawingCommand] = [
        {"kind": "spin", "degrees": math.degrees(start_heading)},
        {"kind": "arc", "radius": r, "degrees": sweep_deg},
    ]
    ops = _simulate(cmds, (10.0, 0.0), 0.0)
    arc_op = [op for op in ops if op[0] == "arc"][0]
    assert _close(arc_op[1]["p1"], [0.0, 10.0], 1e-5), f"sim p1: {arc_op[1]['p1']}"


# ---------------------------------------------------------------------------


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            failures.append((t.__name__, tb))
    print()
    if failures:
        print(f"=== {len(failures)} FAILURES ===")
        for name, tb in failures:
            print(f"\n--- {name} ---")
            print(tb)
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
