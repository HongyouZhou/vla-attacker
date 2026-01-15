"""
Segmentation module for VLA-Attacker project.
Uses SAM2/SAM3 to segment objects in LIBERO dataset images.
"""

from .config import SegmentationConfig
from .libero_loader import LiberoDataLoader
from .sam_segmenter import SAMSegmenter
from .mask_storage import MaskStorage

__all__ = [
    "SegmentationConfig",
    "LiberoDataLoader", 
    "SAMSegmenter",
    "MaskStorage",
]
