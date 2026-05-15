import numpy as np
from numpy.typing import NDArray
from skimage import io
from skimage.morphology import skeletonize as _skeletonize

from .cleanup import collapse_small_holes, CollapseConfig
from .crossings import (
    DetectConfig,
    DetectResult,
    detect_crossings,
    resolve_crossings,
)

from typing import Literal, TypedDict, NamedTuple


class BinarizeConfig(TypedDict):
    threshold: float  # 0.0 to 1.0


def to_binary(path: str, config: BinarizeConfig) -> NDArray[np.bool_]:
    img: np.ndarray = io.imread(path, as_gray=True)
    if img.dtype != np.float64 and img.dtype != np.float32:
        img = img / 255.0
    return img < config["threshold"]


class SkeletonizeConfig(TypedDict):
    method: Literal["lee", "zhang"]


def skeletonize(
    mask: NDArray[np.bool_], config: SkeletonizeConfig
) -> NDArray[np.bool_]:
    return _skeletonize(mask, method=config["method"])


class Skeletonize:
    """End-to-end skeletonization pipeline.

    Stages:
      1. binarize  -> self.binary
      2. skeletonize -> self.skeletonized
      3. collapse_small_holes -> self.collapsed
            (fixes thin double-pixel artifacts from Lee thinning)
      4. detect_crossings -> self.detection
            (identifies ribbon-collapse regions in the binary; uses
            the cleaned skeleton from stage 3 so arm endpoints land
            on actual cleaned-skeleton pixels for downstream resolution)
      5. resolve_crossings -> self.uncrossed
            (rewrites each detected ribbon collapse: erases the merged
            skeleton segment and replaces it with two straight lines
            between paired arm endpoints, restoring two non-intersecting
            paths through the crossing)

    The final output for downstream consumption is `self.uncrossed`.
    Earlier stages are kept on the instance so visualization / debug
    can compare them.
    """

    class Config:
        class Binarize(BinarizeConfig):
            pass

        class Skeletonize(SkeletonizeConfig):
            pass

        class Collapse(CollapseConfig):
            pass

        class Detect(DetectConfig):
            pass

    class Output(NamedTuple):
        binary: NDArray[np.bool_]
        skeletonized: NDArray[np.bool_]
        collapsed: NDArray[np.bool_]
        detection: DetectResult
        uncrossed: NDArray[np.bool_]

    def __init__(
        self,
        path: str,
        binarize_config: Config.Binarize,
        skeletonize_config: Config.Skeletonize,
        collapse_config: Config.Collapse,
        detect_config: Config.Detect,
    ):
        self.binary = to_binary(path, binarize_config)
        self.skeletonized = skeletonize(self.binary, skeletonize_config)
        self.collapsed = collapse_small_holes(self.skeletonized, collapse_config)
        # Detection uses the BINARY for its distance-transform analysis
        # but is given the CLEANED skeleton so arm endpoints land on
        # pixels that will survive the resolver's edits.
        self.detection = detect_crossings(
            self.binary, detect_config, skel=self.collapsed
        )
        # Resolution rewrites the cleaned skeleton in place at every
        # detected crossing.
        self.uncrossed = resolve_crossings(self.collapsed, self.detection)

        self.output = self.Output(
            binary=self.binary,
            skeletonized=self.skeletonized,
            collapsed=self.collapsed,
            detection=self.detection,
            uncrossed=self.uncrossed,
        )
