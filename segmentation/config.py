"""
Configuration for segmentation module.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SegmentationConfig:
    """Configuration for SAM segmentation pipeline."""
    
    # SAM Model settings
    # Options: sam2.1_hiera_tiny, sam2.1_hiera_small, sam2.1_hiera_base_plus, sam2.1_hiera_large
    sam_model_name: str = "sam2.1_hiera_tiny"  # Use tiny for AMD integrated GPU
    sam_checkpoint_dir: str = "checkpoints/sam2"
    
    # Device settings
    device: str = "cuda"  # or "cpu" for testing without GPU
    
    # LIBERO dataset paths
    libero_root: Path = field(default_factory=lambda: Path("LIBERO"))
    libero_datasets_dir: Path = field(default_factory=lambda: Path("LIBERO/datasets"))
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("LIBERO/segmentation_data"))
    
    # Which views to process
    views: List[str] = field(default_factory=lambda: ["agentview_rgb", "eye_in_hand_rgb"])
    
    # SAM Automatic Mask Generator settings
    points_per_side: int = 32  # Number of points sampled along one side of the image
    pred_iou_thresh: float = 0.86  # Filter masks by predicted IoU
    stability_score_thresh: float = 0.92  # Filter masks by stability score
    crop_n_layers: int = 0  # Number of layers to crop (0 for faster processing)
    min_mask_region_area: int = 100  # Minimum mask area in pixels
    
    # Video tracking settings (for consistent object IDs across frames)
    enable_tracking: bool = True
    tracking_memory_frames: int = 7  # Number of frames to keep in memory
    
    # Processing settings
    batch_size: int = 1  # Frames to process in parallel (keep low for integrated GPU)
    num_workers: int = 2  # DataLoader workers
    
    # Which task suites to process
    task_suites: List[str] = field(default_factory=lambda: [
        "libero_spatial",
        "libero_object", 
        "libero_goal",
        "libero_100"
    ])
    
    def __post_init__(self):
        """Convert paths to Path objects if they are strings."""
        if isinstance(self.libero_root, str):
            self.libero_root = Path(self.libero_root)
        if isinstance(self.libero_datasets_dir, str):
            self.libero_datasets_dir = Path(self.libero_datasets_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
    
    @classmethod
    def for_testing(cls) -> "SegmentationConfig":
        """Create a config suitable for quick local testing on integrated GPU."""
        return cls(
            sam_model_name="sam2.1_hiera_tiny",
            device="cuda",
            points_per_side=16,  # Fewer points for speed
            crop_n_layers=0,
            min_mask_region_area=200,
            enable_tracking=True,
            batch_size=1,
        )
    
    @classmethod
    def for_production(cls) -> "SegmentationConfig":
        """Create a config for full dataset processing with better GPU."""
        return cls(
            sam_model_name="sam2.1_hiera_large",
            device="cuda",
            points_per_side=32,
            crop_n_layers=1,
            min_mask_region_area=100,
            enable_tracking=True,
            batch_size=4,
        )
