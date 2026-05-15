import os

from release import (
    Skeletonize,
    Segment,
    StrokeGraph,
    LowGeometryVectorize,
    HighGeometryVectorize,
)

from visualize.skeletonize import visualize_pipeline
from visualize.segment import visualize_segments, visualize_fused
from visualize.graph import visualize_graph, describe

# from vectorize import Vectorize
from release.visualize import (
    commands_to_svg,
    commands_to_svg_compare,
    commands_to_svg_gif,
)


import json
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from visualize.utils import stitch

examples = [
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
            "low.vectorized",
            "high.vectorized",
            "graph",
            "commands",
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
            low.commands,
            high.commands,
            f"examples/{name}.vectorized.svg",
            start_pos=start_pos,
            start_heading=start_heading,
        )
        commands_to_svg_gif(
            low.commands,
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


def process():
    Visualize.clear()

    only = None

    for example in examples:
        if only and len(only) > 0 and example not in only:
            continue
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
        Visualize.skeleton(skeleton, example)

        junction_tol = 2.5
        tangent_sample = 10

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
        Visualize.segments(skeleton, segment, example)

        graph = StrokeGraph(
            segment.fused_post_repair,
            StrokeGraph.Config.Build(
                junction_tol=junction_tol,  # match Repair.junction_tol
                terminal_tangent_window=10,  # should match fuse.lookback
                crossing_tangent_skip=2,  # baseline; dynamic walk handles arbitrary bridges
                crossing_tangent_half_window=6,
                cusp_angle_threshold_deg=50.0,  # raise to ~50 to handle bikelove's cusp-like junction
                cluster_merge_centroid_distance=10.0,
                cluster_merge_index_gap=10,
            ),
        )
        visualize_graph(
            skeleton.binary, graph, scale=1, output_path=f"examples/{example}.graph.png"
        )

        start_pos = np.array([0.0, 0.0])
        start_heading = 0.0
        vectorized = {
            "low_geometry": LowGeometryVectorize(
                graph,
                start_pos=start_pos,
                start_heading=start_heading,
            ),
            "high_geometry": HighGeometryVectorize(
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
            ),
        }

        Visualize.vectorized(
            vectorized["low_geometry"],
            vectorized["high_geometry"],
            example,
            start_pos=start_pos,
            start_heading=start_heading,
        )


process()
