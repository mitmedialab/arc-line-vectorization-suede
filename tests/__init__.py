import os

from release import (
    default_pipeline,
    Skeletonize,
    Segment,
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

        skeleton, segment, graph, low_geometry, high_geometry = default_pipeline(
            f"examples/{example}.png"
        )

        Visualize.skeleton(skeleton, example)
        Visualize.segments(skeleton, segment, example)
        visualize_graph(
            skeleton.binary, graph, scale=1, output_path=f"examples/{example}.graph.png"
        )
        Visualize.vectorized(
            low_geometry,
            high_geometry,
            example,
            start_pos=np.array([0.0, 0.0]),
            start_heading=0.0,
        )


process()
