import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager

from release import (
    Skeletonize,
    Segment,
    StrokeGraph,
    LowGeometryVectorize,
    HighGeometryVectorize,
    OptimizeRoute,
)
from release.visualize import (
    commands_to_svg,
    commands_to_svg_compare,
    commands_to_svg_gif,
)

from visualize.skeletonize import visualize_pipeline
from visualize.segment import visualize_segments, visualize_fused
from visualize.graph import visualize_graph, describe
from visualize.utils import stitch


import json
import numpy as np
from numpy.typing import NDArray
from PIL import Image

examples = [
    "hooreye",
    "alien",
    "angel",
    "angryhashtag",
    "beachnugget",
    "cheese",
    "cute",
    "flowers",
    "ghost",
    "ghostclock",
    "towelbat",
    "tree",
    "smile",
    "scribble",
    "bikelove",
    "birdlove",
    "candelsun",
    "catballoon",
    "catcar",
    "catflame",
    "catheart",
    "cathug",
    "eggmoon",
    "eyecloud",
    "heartman",
    "housesun",
    "personmoon",
    "treecar1",
    "treecar2",
    "treestar",
    "vasesun",
]


class Visualize:
    @classmethod
    def clear(cls):
        extensions = ["png", "svg", "json", "gif"]
        suffixes = [
            "skeleton",
            "segments",
            "fused_geometry",
            "vectorized",
            "graph",
            "commands",
            "optimized",
            "low.vectorized",
            "high.vectorized",
        ]
        for example in examples:
            for suffix in suffixes:
                for ext in extensions:
                    path = f"examples/{example}.{suffix}.{ext}"
                    if os.path.exists(path):
                        os.remove(path)

    @classmethod
    def skeleton(cls, skeleton: Skeletonize, name: str):
        fig = visualize_pipeline(
            skeleton.binary,
            skeleton.collapsed,
            skeleton.detection,
            skeleton.uncrossed,
        )
        fig.savefig(f"examples/{name}.skeleton.png")

    @classmethod
    def segments(cls, skeleton: Skeletonize, segment: Segment, name: str):
        stitch(
            [
                visualize_segments(
                    skeleton.collapsed,
                    segment.segments,
                    show_node_pixels=False,
                ),
                visualize_fused(
                    skeleton.binary,
                    segment.fused_pre_repair,
                ),
                visualize_fused(
                    skeleton.binary,
                    segment.repaired,
                ),
                visualize_fused(
                    skeleton.binary,
                    segment.fused_post_repair,
                ),
            ]
        ).save(f"examples/{name}.segments.png")

    @classmethod
    def vectorized(
        cls,
        low: LowGeometryVectorize,
        high: HighGeometryVectorize,
        name: str,
        start_pos: NDArray,
        start_heading: float,
    ):
        commands_to_svg_compare(
            low.consolidated,
            high.commands,
            f"examples/{name}.vectorized.svg",
            start_pos=start_pos,
            start_heading=start_heading,
        )
        commands_to_svg_gif(
            low.consolidated,
            f"examples/{name}.low.vectorized.gif",
            start_pos=(start_pos[0], start_pos[1]),
            start_heading=start_heading,
        )
        commands_to_svg_gif(
            high.commands,
            f"examples/{name}.high.vectorized.gif",
            start_pos=(start_pos[0], start_pos[1]),
            start_heading=start_heading,
        )

    @classmethod
    def optimized(
        cls,
        low: LowGeometryVectorize,
        optimized: OptimizeRoute,
        name: str,
        start_pos: NDArray,
        start_heading: float,
    ):
        commands_to_svg_compare(
            low.consolidated,
            optimized.commands,
            f"examples/{name}.optimized.svg",
            label_a=f"before ({optimized.estimated_time_before:.2f}s)",
            label_b=f"after ({optimized.estimated_time_after:.2f}s)",
            start_pos=start_pos,
            start_heading=start_heading,
        )
        commands_to_svg_gif(
            optimized.commands,
            f"examples/{name}.optimized.gif",
            start_pos=(start_pos[0], start_pos[1]),
            start_heading=start_heading,
        )


@contextmanager
def step(timings, name):
    t0 = time.perf_counter()
    yield
    timings.append((name, time.perf_counter() - t0))


def _format_timings(timings):
    total = sum(dt for _, dt in timings)
    width = max(len(n) for n, _ in timings)
    lines = [f"  {n:<{width}} {dt * 1000:>8.1f} ms" for n, dt in timings]
    lines.append(f"  {'TOTAL':<{width}} {total * 1000:>8.1f} ms")
    return "\n".join(lines)


def process_example(example: str) -> str:
    """Run the full pipeline for a single example. Returns a formatted
    log string so the parent can print results as workers complete
    without interleaving partial output across processes.
    """
    timings = []

    with step(timings, "skeletonize"):
        skeleton = Skeletonize(
            f"examples/{example}.png",
            Skeletonize.Config.Binarize(threshold=0.5),
            Skeletonize.Config.Skeletonize(method="zhang"),
            Skeletonize.Config.Collapse(
                skeletonize_method="lee",
                max_hole_area=10,
                max_thin_thickness=3.0,
                reskeletonize=True,
            ),
            detect_config={
                "local_tau_radius": 40,
                "fat_ratio": 1.3,
                "min_fat_area": 8,
                "group_dilate": 15,
                "skel_ring_dilate": 5,
                "pairing_tangent_steps": 8,
                "pairing_threshold": 1.2,
                "min_chromosome_skel_length": 15,
            },
        )

    with step(timings, "viz.skeleton"):
        Visualize.skeleton(skeleton, example)

    junction_tol = 2.5
    tangent_sample = 10

    with step(timings, "segment"):
        segment = Segment(
            skeleton.uncrossed,
            skeleton.binary,
            Segment.Config.Segment(min_length=10.0),
            Segment.Config.Fuse(
                max_path_length=20,
                lookback=10,
                min_tangent_score=0.5,
                gap_penalty=0.05,
                curvature_penalty=3.0,
            ),
            Segment.Config.Repair(
                junction_tol=junction_tol,
                stable_skip=2,
                stable_sample=6,
                max_junction_region_length=20,
                min_output_polyline_length=2,
                min_tangent_spread_deg=15.0,
                interp_max_spacing=1.0,
                min_curvature_spike_ratio=2.0,
                curvature_context_window=8,
            ),
            Segment.Config.PostRepairFuse(
                junction_tol=junction_tol,
                tangent_skip=2,
                tangent_sample=tangent_sample,
                min_tangent_score=0.6,
                curvature_penalty=1.0,
            ),
        )

    with step(timings, "viz.segments"):
        Visualize.segments(skeleton, segment, example)

    with step(timings, "graph"):
        graph = StrokeGraph(
            segment.fused_post_repair,
            StrokeGraph.Config.Build(
                junction_tol=junction_tol,
                terminal_tangent_window=10,
                crossing_tangent_skip=2,
                crossing_tangent_half_window=6,
                cusp_angle_threshold_deg=50.0,
                cluster_merge_centroid_distance=10.0,
                cluster_merge_index_gap=10,
            ),
        )

    with step(timings, "viz.graph"):
        visualize_graph(
            skeleton.binary,
            graph,
            scale=1,
            output_path=f"examples/{example}.graph.png",
        )

    start_pos = np.array([0.0, 0.0])
    start_heading = 0.0

    with step(timings, "low_geometry"):
        low_geometry = LowGeometryVectorize(
            graph,
            start_pos=start_pos,
            start_heading=start_heading,
        )

    with step(timings, "high_geometry"):
        high_geometry = HighGeometryVectorize(
            segment.fused_post_repair,
            start_pos=start_pos,
            start_heading=start_heading,
            commands=HighGeometryVectorize.Config.ToCommands(
                sigma=2.0,
                corner_threshold=0.25,
                max_fit_residual=5.0,
            ),
            consolidate=HighGeometryVectorize.Config.Consolidate(
                center_tol_rel=0.25,
                radius_tol_rel=0.25,
                center_tol_abs=3.0,
                radius_tol_abs=3.0,
                max_endpoint_snap_rel=0.15,
                max_endpoint_snap_abs=6.0,
                proximity_min_radius_ratio=0.4,
                line_angle_tol_deg=6.0,
                line_offset_tol_abs=5.0,
                min_line_length=5.0,
                max_line_endpoint_snap_abs=5.0,
                junction_epsilon=3.0,
                merge_arcs=True,
                merge_lines=True,
                return_report=False,
            ),
        )

    with step(timings, "viz.vectorized"):
        Visualize.vectorized(
            low_geometry,
            high_geometry,
            example,
            start_pos=start_pos,
            start_heading=start_heading,
        )

    with step(timings, "optimize"):
        optimized = OptimizeRoute(
            low_geometry.consolidated,
            start_pos=start_pos,
            start_heading=start_heading,
            cfg=OptimizeRoute.Config.Optimize(
                pixels_per_inch=1.0,
                pen_up_join_tol=0.5,
                two_opt_passes=16,
                or_opt_passes=8,
                or_opt_max_segment_len=3,
            ),
        )

    with step(timings, "viz.optimized"):
        Visualize.optimized(
            low_geometry,
            optimized,
            example,
            start_pos=start_pos,
            start_heading=start_heading,
        )

    return f"\n{example}: {optimized.stats()}\n{_format_timings(timings)}"


def process(only=None, max_workers=None):
    """Run every example in parallel via ProcessPoolExecutor and print
    each worker's log as it completes (so output streams in roughly the
    order finished, not the order submitted).
    """
    Visualize.clear()

    selected = [e for e in examples if not only or e in only]
    if not selected:
        return

    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_example, e): e for e in selected}
        for fut in as_completed(futures):
            example = futures[fut]
            try:
                print(fut.result())
            except Exception as exc:
                print(f"\n{example}: FAILED with {type(exc).__name__}: {exc}")
    wall = time.perf_counter() - t_start
    print(f"\nWall clock: {wall:.2f}s for {len(selected)} examples")


if __name__ == "__main__":
    process()
