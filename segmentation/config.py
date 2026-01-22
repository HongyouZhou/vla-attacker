"""
Configuration for segmentation module.

Supports both SAM2 (automatic mask generation) and SAM3 (text prompt segmentation).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SegmentationConfig:
    """Configuration for SAM segmentation pipeline."""

    # Model settings
    # Options:
    #   - "sam3" (recommended): SAM3 with text prompt segmentation
    #   - "sam2.1_hiera_tiny", "sam2.1_hiera_small", etc.: SAM2 automatic segmentation
    sam_model_name: str = "sam3"
    sam_checkpoint_dir: str = "checkpoints/sam2"  # Only used for SAM2

    # Device settings
    device: str = "cuda"  # "cuda" or "cpu"

    # LIBERO dataset paths
    libero_root: Path = field(default_factory=lambda: Path("LIBERO"))
    libero_datasets_dir: Path = field(
        default_factory=lambda: Path("LIBERO/libero/datasets")
    )

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("LIBERO/segmentation_data"))

    # Which views to process
    views: List[str] = field(
        default_factory=lambda: ["agentview_rgb", "eye_in_hand_rgb"]
    )

    # ===================
    # SAM3 Text Prompts
    # ===================
    # Text prompts for objects to segment in LIBERO scenes
    # These can be customized per task or use general prompts
    text_prompts: List[str] = field(
        default_factory=lambda: [
            # Material-based prompts that SAM3 responds to better
            "metal",  # Detects metallic objects like robot arm, bowls
            "ceramic",  # Plates, bowls
            "plastic",  # Various objects
            "glass",  # Transparent objects
            # Simple object descriptors
            "container",
            "tool",
            "handle",
            # Colors can help
            "red object",
            "blue object",
            "black object",
            "white object",
        ]
    )

    # Task-specific prompts (override default prompts for specific tasks)
    # Using material and color-based prompts that SAM3 responds to
    task_prompts: dict = field(
        default_factory=lambda: {
            # Spatial reasoning tasks - focus on key objects
            "libero_spatial": [
                "metal",  # Robot arm, metallic objects like bowls
            ],
            # Object manipulation tasks
            "libero_object": [
                "metal",
            ],
            # Goal-based tasks
            "libero_goal": [
                "metal",
            ],
        }
    )

    # ===================
    # SAM2 Settings (for backwards compatibility)
    # ===================
    # SAM Automatic Mask Generator settings (SAM2 only)
    points_per_side: int = 32
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92
    crop_n_layers: int = 0
    min_mask_region_area: int = 100

    # Video tracking settings
    enable_tracking: bool = True
    tracking_memory_frames: int = 7

    # Processing settings
    batch_size: int = 1
    num_workers: int = 2

    # Which task suites to process
    task_suites: List[str] = field(
        default_factory=lambda: [
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_100",
        ]
    )

    def __post_init__(self):
        """Convert paths to Path objects if they are strings."""
        if isinstance(self.libero_root, str):
            self.libero_root = Path(self.libero_root)
        if isinstance(self.libero_datasets_dir, str):
            self.libero_datasets_dir = Path(self.libero_datasets_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    def get_prompts_for_task(self, task_suite: str) -> List[str]:
        """Get text prompts for a specific task suite.

        Args:
            task_suite: Task suite name (e.g., "libero_spatial")

        Returns:
            List of text prompts to use
        """
        if task_suite in self.task_prompts:
            return self.task_prompts[task_suite]
        return self.text_prompts

    @property
    def use_sam3(self) -> bool:
        """Check if using SAM3 (text prompt mode)."""
        return self.sam_model_name.lower() == "sam3"

    @classmethod
    def for_testing(cls) -> "SegmentationConfig":
        """Create a config suitable for quick local testing."""
        return cls(
            sam_model_name="sam3",
            device="cuda",
            enable_tracking=True,
            batch_size=1,
            # Use fewer prompts for testing
            text_prompts=[
                "robot gripper",
                "wooden block",
            ],
        )

    @classmethod
    def for_sam2_testing(cls) -> "SegmentationConfig":
        """Create a config for SAM2 testing (automatic segmentation)."""
        return cls(
            sam_model_name="sam2.1_hiera_tiny",
            device="cuda",
            points_per_side=16,
            crop_n_layers=0,
            min_mask_region_area=200,
            enable_tracking=True,
            batch_size=1,
        )

    @classmethod
    def for_production(cls) -> "SegmentationConfig":
        """Create a config for full dataset processing."""
        return cls(
            sam_model_name="sam3",
            device="cuda",
            enable_tracking=True,
            batch_size=4,
            # Full set of prompts for comprehensive segmentation
            text_prompts=[
                "robot gripper",
                "robot arm",
                "wooden block",
                "red object",
                "blue object",
                "green object",
                "bowl",
                "plate",
                "mug",
                "cup",
                "drawer",
                "cabinet door",
                "button",
                "container",
                "target location",
            ],
        )
