"""
Segmentation module for VLA-Attacker project.

Uses SAM3 (Segment Anything Model 3) for text prompt segmentation,
or SAM2 for automatic mask generation (backwards compatible).

SAM3 Features:
- Text prompt segmentation ("robot gripper", "wooden block", etc.)
- Video tracking with text prompts
- Promptable Concept Segmentation (PCS)

Example:
    from segmentation import SAMSegmenter, SegmentationConfig

    # SAM3 text prompt segmentation
    segmenter = SAMSegmenter(model_name="sam3")
    result = segmenter.segment_frame_with_text(
        image,
        text_prompts=["robot gripper", "target object"]
    )
"""

from .config import SegmentationConfig
from .libero_loader import LiberoDataLoader, DemoInfo
from .sam_segmenter import (
    SAMSegmenter,
    SAM3Segmenter,
    SegmentationResult,
    SAM3_AVAILABLE,
    SAM2_AVAILABLE,
)
from .mask_storage import MaskStorage

__all__ = [
    # Main classes
    "SegmentationConfig",
    "LiberoDataLoader",
    "DemoInfo",
    "SAMSegmenter",
    "SAM3Segmenter",
    "SegmentationResult",
    "MaskStorage",
    # Availability flags
    "SAM3_AVAILABLE",
    "SAM2_AVAILABLE",
]
