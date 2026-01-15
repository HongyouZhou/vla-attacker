"""
SAM2/SAM3 segmenter wrapper for automatic mask generation and video tracking.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# SAM2 imports (will be installed separately)
try:
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM2 not installed. Run 'pip install git+https://github.com/facebookresearch/sam2.git'")

import torch


@dataclass  
class SegmentationResult:
    """Result from segmenting a single frame."""
    frame_idx: int
    masks: np.ndarray       # (N, H, W) boolean masks
    object_ids: np.ndarray  # (N,) unique object IDs
    bboxes: np.ndarray      # (N, 4) bounding boxes [x1, y1, x2, y2]
    scores: np.ndarray      # (N,) confidence scores
    areas: np.ndarray       # (N,) mask areas in pixels


class SAMSegmenter:
    """
    Wrapper for SAM2 automatic mask generation and video tracking.
    
    For VLA attack preparation:
    1. Use automatic mask generator to segment first frame
    2. Use video predictor to track objects through the video
    """
    
    # SAM2 model configurations
    MODEL_CONFIGS = {
        "sam2.1_hiera_tiny": {
            "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "checkpoint": "sam2.1_hiera_tiny.pt"
        },
        "sam2.1_hiera_small": {
            "config": "configs/sam2.1/sam2.1_hiera_s.yaml", 
            "checkpoint": "sam2.1_hiera_small.pt"
        },
        "sam2.1_hiera_base_plus": {
            "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "checkpoint": "sam2.1_hiera_base_plus.pt"
        },
        "sam2.1_hiera_large": {
            "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "checkpoint": "sam2.1_hiera_large.pt"
        },
    }
    
    def __init__(
        self,
        model_name: str = "sam2.1_hiera_tiny",
        checkpoint_dir: str = "checkpoints/sam2",
        device: str = "cuda",
        # Automatic mask generator settings
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
        crop_n_layers: int = 0,
        min_mask_region_area: int = 100,
    ):
        """
        Initialize SAM segmenter.
        
        Args:
            model_name: SAM2 model variant (see MODEL_CONFIGS)
            checkpoint_dir: Directory containing SAM2 checkpoints
            device: Device to run inference on
            points_per_side: Grid size for automatic mask generation
            pred_iou_thresh: Predicted IoU threshold for filtering
            stability_score_thresh: Stability score threshold for filtering
            crop_n_layers: Number of crop layers for hierarchical segmentation
            min_mask_region_area: Minimum mask area in pixels
        """
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        
        # Store settings
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.crop_n_layers = crop_n_layers
        self.min_mask_region_area = min_mask_region_area
        
        # Models (lazy loaded)
        self._sam_model = None
        self._mask_generator = None
        self._video_predictor = None
        self._video_state = None
        
    def _get_model_config(self) -> Tuple[str, str]:
        """Get config and checkpoint paths for the model."""
        if self.model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {self.model_name}. Available: {list(self.MODEL_CONFIGS.keys())}")
        
        config = self.MODEL_CONFIGS[self.model_name]
        checkpoint_path = self.checkpoint_dir / config["checkpoint"]
        
        return config["config"], str(checkpoint_path)
    
    def _load_mask_generator(self):
        """Load the automatic mask generator."""
        if not SAM2_AVAILABLE:
            raise RuntimeError("SAM2 is not installed")
            
        if self._mask_generator is not None:
            return
        
        config_file, checkpoint_path = self._get_model_config()
        
        print(f"Loading SAM2 model: {self.model_name}")
        self._sam_model = build_sam2(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=self.device,
        )
        
        self._mask_generator = SAM2AutomaticMaskGenerator(
            model=self._sam_model,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            crop_n_layers=self.crop_n_layers,
            min_mask_region_area=self.min_mask_region_area,
        )
        print("SAM2 automatic mask generator ready")
    
    def _load_video_predictor(self):
        """Load the video predictor for tracking."""
        if not SAM2_AVAILABLE:
            raise RuntimeError("SAM2 is not installed")
            
        if self._video_predictor is not None:
            return
        
        config_file, checkpoint_path = self._get_model_config()
        
        print(f"Loading SAM2 video predictor: {self.model_name}")
        self._video_predictor = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=self.device,
        )
        print("SAM2 video predictor ready")
    
    def segment_frame(self, image: np.ndarray, frame_idx: int = 0) -> SegmentationResult:
        """
        Segment a single frame using automatic mask generation.
        
        Args:
            image: RGB image array of shape (H, W, 3)
            frame_idx: Frame index for tracking
            
        Returns:
            SegmentationResult with masks and metadata
        """
        self._load_mask_generator()
        
        # Generate masks
        masks_data = self._mask_generator.generate(image)
        
        if not masks_data:
            # No masks found
            h, w = image.shape[:2]
            return SegmentationResult(
                frame_idx=frame_idx,
                masks=np.zeros((0, h, w), dtype=bool),
                object_ids=np.array([], dtype=np.int32),
                bboxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.array([], dtype=np.float32),
                areas=np.array([], dtype=np.int32),
            )
        
        # Extract masks and metadata
        n_masks = len(masks_data)
        
        masks = np.stack([m['segmentation'] for m in masks_data], axis=0)
        object_ids = np.arange(n_masks, dtype=np.int32)
        bboxes = np.array([m['bbox'] for m in masks_data], dtype=np.float32)  # [x, y, w, h]
        # Convert to [x1, y1, x2, y2]
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        scores = np.array([m['predicted_iou'] for m in masks_data], dtype=np.float32)
        areas = np.array([m['area'] for m in masks_data], dtype=np.int32)
        
        return SegmentationResult(
            frame_idx=frame_idx,
            masks=masks,
            object_ids=object_ids,
            bboxes=bboxes,
            scores=scores,
            areas=areas,
        )
    
    def init_video_tracking(self, video_frames: np.ndarray):
        """
        Initialize video tracking with all frames.
        
        Args:
            video_frames: Array of shape (T, H, W, 3) with RGB frames
        """
        self._load_video_predictor()
        
        # Initialize inference state
        self._video_state = self._video_predictor.init_state(video_path=video_frames)
        
    def add_tracking_masks(
        self,
        frame_idx: int,
        masks: np.ndarray,
        object_ids: np.ndarray,
    ):
        """
        Add masks to track from a specific frame.
        
        Args:
            frame_idx: Frame index where masks are from
            masks: Boolean masks of shape (N, H, W)
            object_ids: Object IDs of shape (N,)
        """
        if self._video_predictor is None or self._video_state is None:
            raise RuntimeError("Video tracking not initialized. Call init_video_tracking first.")
        
        for mask, obj_id in zip(masks, object_ids):
            # Add each mask to the video predictor
            self._video_predictor.add_new_mask(
                inference_state=self._video_state,
                frame_idx=frame_idx,
                obj_id=int(obj_id),
                mask=mask,
            )
    
    def propagate_tracking(self) -> Dict[int, SegmentationResult]:
        """
        Propagate tracking through all video frames.
        
        Returns:
            Dictionary mapping frame_idx to SegmentationResult
        """
        if self._video_predictor is None or self._video_state is None:
            raise RuntimeError("Video tracking not initialized")
        
        results = {}
        
        # Propagate through video
        for frame_idx, obj_ids, masks in self._video_predictor.propagate_in_video(self._video_state):
            # Convert to numpy
            masks_np = (masks > 0.5).cpu().numpy()
            obj_ids_np = np.array(obj_ids, dtype=np.int32)
            
            # Compute bboxes from masks
            n_masks, h, w = masks_np.shape
            bboxes = np.zeros((n_masks, 4), dtype=np.float32)
            areas = np.zeros(n_masks, dtype=np.int32)
            
            for i, mask in enumerate(masks_np):
                if mask.any():
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    y1, y2 = np.where(rows)[0][[0, -1]]
                    x1, x2 = np.where(cols)[0][[0, -1]]
                    bboxes[i] = [x1, y1, x2, y2]
                    areas[i] = mask.sum()
            
            results[frame_idx] = SegmentationResult(
                frame_idx=frame_idx,
                masks=masks_np,
                object_ids=obj_ids_np,
                bboxes=bboxes,
                scores=np.ones(n_masks, dtype=np.float32),  # Tracking doesn't have scores
                areas=areas,
            )
        
        return results
    
    def reset_tracking(self):
        """Reset video tracking state."""
        if self._video_predictor is not None and self._video_state is not None:
            self._video_predictor.reset_state(self._video_state)
        self._video_state = None
    
    def segment_video_with_tracking(
        self, 
        frames: np.ndarray,
        init_frame_idx: int = 0,
    ) -> Dict[int, SegmentationResult]:
        """
        Segment a video with consistent object tracking.
        
        1. Segment the first frame using automatic mask generator
        2. Track objects through remaining frames
        
        Args:
            frames: Video frames of shape (T, H, W, 3) in RGB
            init_frame_idx: Which frame to use for initial segmentation
            
        Returns:
            Dictionary mapping frame_idx to SegmentationResult
        """
        results = {}
        
        # Step 1: Segment initial frame
        print(f"Segmenting initial frame {init_frame_idx}...")
        init_result = self.segment_frame(frames[init_frame_idx], init_frame_idx)
        results[init_frame_idx] = init_result
        
        if len(init_result.masks) == 0:
            print("No objects found in initial frame")
            return results
        
        print(f"Found {len(init_result.masks)} objects in initial frame")
        
        # Step 2: Initialize video tracking
        print("Initializing video tracking...")
        self.init_video_tracking(frames)
        
        # Step 3: Add initial masks
        self.add_tracking_masks(init_frame_idx, init_result.masks, init_result.object_ids)
        
        # Step 4: Propagate through video
        print("Propagating masks through video...")
        tracked_results = self.propagate_tracking()
        results.update(tracked_results)
        
        # Cleanup
        self.reset_tracking()
        
        return results


if __name__ == "__main__":
    # Test basic functionality (without SAM2 installed)
    print("SAM2 available:", SAM2_AVAILABLE)
    
    # Create dummy test
    segmenter = SAMSegmenter(
        model_name="sam2.1_hiera_tiny",
        device="cpu",
    )
    print("Segmenter created")
    print(f"Model config: {segmenter._get_model_config()}")
