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
from release.auto_config import derive_configs
from release.visualize import (
    commands_to_heatmap,
    commands_to_overlay,
    commands_to_svg,
    commands_to_svg_compare_n,
    commands_to_svg_gif_compare_n,
)
from release.optimize import estimate_total_time

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
            "graph",
            "vectorized",
            "heatmap",
            "overlay",
            "overlay.clean",
            # Suffixes we used to emit before the cleanup. Kept here
            # so re-running test.sh against an old examples/ dir
            # wipes the leftover files.
            "fused_geometry",
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
        optimized: OptimizeRoute,
        name: str,
        start_pos: NDArray,
        start_heading: float,
    ):
        # One static SVG and one animated GIF, each as a 3-panel
        # side-by-side comparison of high vs low vs optimized
        # geometry. The label above each panel encodes the panel's
        # name and its estimated firmware drawing time.
        high_time = estimate_total_time(high.commands)
        panels = [
            (high.commands, f"high ({high_time:.2f}s)"),
            (low.consolidated, f"low ({optimized.estimated_time_before:.2f}s)"),
            (optimized.commands, f"optimized ({optimized.estimated_time_after:.2f}s)"),
        ]
        commands_to_svg_compare_n(
            panels,
            f"examples/{name}.vectorized.svg",
            start_pos=start_pos,
            start_heading=start_heading,
        )
        commands_to_svg_gif_compare_n(
            panels,
            f"examples/{name}.vectorized.gif",
            start_pos=(start_pos[0], start_pos[1]),
            start_heading=start_heading,
        )

    @classmethod
    def heatmap(
        cls,
        optimized: OptimizeRoute,
        name: str,
        start_pos: NDArray,
        start_heading: float,
    ):
        commands_to_heatmap(
            optimized.commands,
            f"examples/{name}.heatmap.png",
            start_pos=(start_pos[0], start_pos[1]),
            start_heading=start_heading,
        )

    @classmethod
    def overlay(
        cls,
        optimized: OptimizeRoute,
        name: str,
        start_pos: NDArray,
        start_heading: float,
    ):
        # Produce two overlays:
        #   .overlay.png        — labels + leader lines + Drive/Line/Spin/Arc text.
        #                         The diagnostic-rich view for understanding what
        #                         the robot does step by step.
        #   .overlay.clean.png  — no labels. Just the faded source with the
        #                         red pen-down strokes and dotted transit lines
        #                         on top. The "how faithful is the vectorization"
        #                         view at a glance.
        commands_to_overlay(
            optimized.commands,
            f"examples/{name}.png",
            f"examples/{name}.overlay.png",
            start_pos=(start_pos[0], start_pos[1]),
            start_heading=start_heading,
            show_labels=True,
        )
        commands_to_overlay(
            optimized.commands,
            f"examples/{name}.png",
            f"examples/{name}.overlay.clean.png",
            start_pos=(start_pos[0], start_pos[1]),
            start_heading=start_heading,
            show_labels=False,
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

    # Derive every numerical tolerance from the input image's stroke
    # width so the pipeline doesn't need a per-image tuning pass — see
    # release/auto_config.py for the scaling rules.
    image_path = f"examples/{example}.png"
    cfg = derive_configs(image_path)

    with step(timings, "skeletonize"):
        skeleton = Skeletonize(
            image_path,
            Skeletonize.Config.Binarize(**cfg["binarize"]),
            Skeletonize.Config.Skeletonize(**cfg["skeletonize"]),
            Skeletonize.Config.Collapse(**cfg["collapse"]),
            detect_config=cfg["detect"],
        )

    with step(timings, "viz.skeleton"):
        Visualize.skeleton(skeleton, example)

    with step(timings, "segment"):
        segment = Segment(
            skeleton.uncrossed,
            skeleton.binary,
            Segment.Config.Segment(**cfg["segment"]),
            Segment.Config.Fuse(**cfg["fuse"]),
            Segment.Config.Repair(**cfg["repair"]),
            Segment.Config.PostRepairFuse(**cfg["post_repair_fuse"]),
        )

    with step(timings, "viz.segments"):
        Visualize.segments(skeleton, segment, example)

    with step(timings, "graph"):
        graph = StrokeGraph(
            segment.fused_post_repair,
            StrokeGraph.Config.Build(**cfg["graph_build"]),
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
            commands=HighGeometryVectorize.Config.ToCommands(**cfg["high_geometry_commands"]),
            consolidate=HighGeometryVectorize.Config.Consolidate(**cfg["high_geometry_consolidate"]),
        )

    with step(timings, "optimize"):
        optimized = OptimizeRoute(
            low_geometry.consolidated,
            start_pos=start_pos,
            start_heading=start_heading,
            cfg=OptimizeRoute.Config.Optimize(**cfg["optimize_route"]),
        )

    with step(timings, "viz.vectorized"):
        # 3-panel comparison (high / low / optimized) in both SVG
        # (static, per-panel labelled with drawing time) and GIF
        # (synchronised side-by-side animation). This single step
        # replaces the previous per-mode .vectorized.gif /
        # .optimized.svg / .optimized.gif outputs.
        Visualize.vectorized(
            low_geometry,
            high_geometry,
            optimized,
            example,
            start_pos=start_pos,
            start_heading=start_heading,
        )

    with step(timings, "viz.heatmap"):
        Visualize.heatmap(
            optimized,
            example,
            start_pos=start_pos,
            start_heading=start_heading,
        )

    with step(timings, "viz.overlay"):
        Visualize.overlay(
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
