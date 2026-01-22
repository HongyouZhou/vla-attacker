"""
SAM3 segmenter wrapper for text prompt segmentation and video tracking.

SAM3 (Segment Anything Model 3) supports:
- Text prompt segmentation (Promptable Concept Segmentation)
- Visual prompts (boxes, points)
- Video tracking with text prompts

Requirements:
- Python 3.12+
- PyTorch 2.7+
- CUDA 12.6+
- SAM3 from: https://github.com/facebookresearch/sam3
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import tempfile
import shutil

import torch

# SAM3 imports
try:
    from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
    from sam3.model.sam3_image_processor import Sam3Processor

    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    print("Warning: SAM3 not installed. Install with:")
    print("  git clone https://github.com/facebookresearch/sam3.git")
    print("  cd sam3 && pip install -e .")

# Fallback to SAM2 if SAM3 not available
try:
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


@dataclass
class SegmentationResult:
    """Result from segmenting a single frame."""

    frame_idx: int
    masks: np.ndarray  # (N, H, W) boolean masks
    object_ids: np.ndarray  # (N,) unique object IDs
    bboxes: np.ndarray  # (N, 4) bounding boxes [x1, y1, x2, y2]
    scores: np.ndarray  # (N,) confidence scores
    areas: np.ndarray  # (N,) mask areas in pixels
    prompts: Optional[List[str]] = None  # Text prompts used (SAM3 only)


class SAM3Segmenter:
    """
    Wrapper for SAM3 text prompt segmentation and video tracking.

    SAM3 allows segmenting objects using natural language descriptions,
    such as "red cup", "robot arm", "wooden block", etc.

    For VLA attack preparation:
    1. Use text prompts to segment specific objects of interest
    2. Use video predictor to track objects through the video
    """

    def __init__(
        self,
        device: str = "cuda",
    ):
        """
        Initialize SAM3 segmenter.

        Args:
            device: Device to run inference on
        """
        if not SAM3_AVAILABLE:
            raise RuntimeError(
                "SAM3 is not installed. Install with:\n"
                "  git clone https://github.com/facebookresearch/sam3.git\n"
                "  cd sam3 && pip install -e ."
            )

        self.device = device

        # Models (lazy loaded)
        self._image_model = None
        self._processor = None
        self._video_predictor = None
        self._current_session_id = None

    def _load_image_model(self):
        """Load the SAM3 image model and processor."""
        if self._processor is not None:
            return

        print("Loading SAM3 image model...")
        self._image_model = build_sam3_image_model()
        self._processor = Sam3Processor(self._image_model)
        print("SAM3 image model ready")

    def _load_video_predictor(self):
        """Load the SAM3 video predictor."""
        if self._video_predictor is not None:
            return

        print("Loading SAM3 video predictor...")
        self._video_predictor = build_sam3_video_predictor()
        print("SAM3 video predictor ready")

    def segment_frame_with_text(
        self,
        image: np.ndarray,
        text_prompts: List[str],
        frame_idx: int = 0,
    ) -> SegmentationResult:
        """
        Segment objects in a frame using text prompts.

        Args:
            image: RGB image array of shape (H, W, 3)
            text_prompts: List of text descriptions for objects to segment
                         e.g., ["red cup", "robot gripper", "wooden block"]
            frame_idx: Frame index for tracking

        Returns:
            SegmentationResult with masks and metadata
        """
        self._load_image_model()

        # Convert numpy array to PIL Image
        from PIL import Image as PILImage

        if isinstance(image, np.ndarray):
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image

        h, w = (
            image.shape[:2]
            if isinstance(image, np.ndarray)
            else (pil_image.height, pil_image.width)
        )

        all_masks = []
        all_bboxes = []
        all_scores = []
        all_object_ids = []
        all_prompts = []

        object_id_counter = 0

        # Process each text prompt
        for prompt in text_prompts:
            # Set image for inference
            inference_state = self._processor.set_image(pil_image)

            # Get segmentation with text prompt
            output = self._processor.set_text_prompt(
                state=inference_state, prompt=prompt
            )

            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]

            # Convert to numpy if needed
            if torch.is_tensor(masks):
                masks = masks.cpu().numpy()
            if torch.is_tensor(boxes):
                boxes = boxes.cpu().numpy()
            if torch.is_tensor(scores):
                scores = scores.cpu().numpy()

            # Add results for each detected instance
            n_instances = len(masks) if masks is not None and len(masks) > 0 else 0

            for i in range(n_instances):
                mask = masks[i]
                # Ensure mask is 2D and boolean
                if mask.ndim == 3:
                    mask = mask.squeeze()
                mask = mask > 0.5

                all_masks.append(mask)
                all_bboxes.append(
                    boxes[i] if boxes is not None else self._mask_to_bbox(mask)
                )
                all_scores.append(scores[i] if scores is not None else 1.0)
                all_object_ids.append(object_id_counter)
                all_prompts.append(prompt)
                object_id_counter += 1

        if len(all_masks) == 0:
            return SegmentationResult(
                frame_idx=frame_idx,
                masks=np.zeros((0, h, w), dtype=bool),
                object_ids=np.array([], dtype=np.int32),
                bboxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.array([], dtype=np.float32),
                areas=np.array([], dtype=np.int32),
                prompts=[],
            )

        masks_array = np.stack(all_masks, axis=0)
        areas = np.array([m.sum() for m in all_masks], dtype=np.int32)

        return SegmentationResult(
            frame_idx=frame_idx,
            masks=masks_array,
            object_ids=np.array(all_object_ids, dtype=np.int32),
            bboxes=np.array(all_bboxes, dtype=np.float32),
            scores=np.array(all_scores, dtype=np.float32),
            areas=areas,
            prompts=all_prompts,
        )

    def _mask_to_bbox(self, mask: np.ndarray) -> np.ndarray:
        """Convert a binary mask to bounding box [x1, y1, x2, y2]."""
        if not mask.any():
            return np.array([0, 0, 0, 0], dtype=np.float32)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def segment_video_with_text(
        self,
        frames: np.ndarray,
        text_prompts: List[str],
        init_frame_idx: int = 0,
    ) -> Dict[int, SegmentationResult]:
        """
        Segment and track objects through a video using text prompts.

        Args:
            frames: Video frames of shape (T, H, W, 3) in RGB
            text_prompts: List of text descriptions for objects to segment
            init_frame_idx: Which frame to use for initial prompt

        Returns:
            Dictionary mapping frame_idx to SegmentationResult
        """
        self._load_video_predictor()

        results = {}
        T, H, W, C = frames.shape

        # Save frames to temporary directory as JPEG files
        # (SAM3 video predictor expects a folder path)
        temp_dir = tempfile.mkdtemp(prefix="sam3_video_")
        try:
            from PIL import Image as PILImage

            print(f"Preparing {T} frames for video tracking...")
            for i, frame in enumerate(frames):
                img = PILImage.fromarray(frame)
                img.save(f"{temp_dir}/{i:05d}.jpg")

            # Start video session
            print("Starting SAM3 video session...")
            response = self._video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=temp_dir,
                )
            )
            session_id = response["session_id"]
            self._current_session_id = session_id

            # Add text prompts at the initial frame
            object_id_counter = 0
            prompt_to_object_ids = {}

            for prompt in text_prompts:
                print(f"  Adding prompt: '{prompt}' at frame {init_frame_idx}")
                response = self._video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=init_frame_idx,
                        text=prompt,
                    )
                )

                # Track which object IDs correspond to which prompt
                if "outputs" in response:
                    outputs = response["outputs"]
                    if outputs is not None and "obj_ids" in outputs:
                        for obj_id in outputs["obj_ids"]:
                            prompt_to_object_ids[obj_id] = prompt

            # Propagate through video using stream request (generator)
            print("Propagating masks through video...")
            for frame_output in self._video_predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    propagation_direction="both",
                )
            ):
                frame_idx = frame_output.get("frame_index", 0)
                outputs = frame_output.get("outputs", {})

                if outputs is None:
                    continue

                # Extract masks from outputs - SAM3 uses different key names
                # Try both possible key names for compatibility
                masks_tensor = outputs.get(
                    "out_binary_masks", outputs.get("pred_masks", None)
                )
                object_ids = outputs.get("out_obj_ids", outputs.get("obj_ids", []))
                scores_array = outputs.get("out_probs", None)
                boxes_array = outputs.get("out_boxes_xywh", None)

                # Convert to list if numpy array
                if isinstance(object_ids, np.ndarray):
                    object_ids = object_ids.tolist()

                if masks_tensor is None or len(object_ids) == 0:
                    results[frame_idx] = SegmentationResult(
                        frame_idx=frame_idx,
                        masks=np.zeros((0, H, W), dtype=bool),
                        object_ids=np.array([], dtype=np.int32),
                        bboxes=np.zeros((0, 4), dtype=np.float32),
                        scores=np.array([], dtype=np.float32),
                        areas=np.array([], dtype=np.int32),
                        prompts=[],
                    )
                    continue

                # Convert masks to numpy
                if torch.is_tensor(masks_tensor):
                    masks = (masks_tensor > 0.5).cpu().numpy()
                elif isinstance(masks_tensor, np.ndarray):
                    masks = (
                        masks_tensor.astype(bool)
                        if masks_tensor.dtype != bool
                        else masks_tensor
                    )
                else:
                    masks = np.array(masks_tensor) > 0.5

                # Handle batch dimension if present
                if masks.ndim == 4:  # (N, 1, H, W)
                    masks = masks.squeeze(1)

                # Compute bboxes and areas
                n_masks = len(masks)
                if boxes_array is not None and len(boxes_array) == n_masks:
                    # Convert xywh to x1y1x2y2
                    bboxes = np.array(boxes_array, dtype=np.float32)
                    if bboxes.shape[1] == 4:
                        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # x2 = x + w
                        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # y2 = y + h
                else:
                    bboxes = np.zeros((n_masks, 4), dtype=np.float32)
                    for i, mask in enumerate(masks):
                        bboxes[i] = self._mask_to_bbox(mask)

                areas = np.zeros(n_masks, dtype=np.int32)
                prompts = []

                for i, (mask, obj_id) in enumerate(zip(masks, object_ids)):
                    if mask.ndim == 3:
                        mask = mask.squeeze()
                    areas[i] = mask.sum()
                    prompts.append(prompt_to_object_ids.get(obj_id, "unknown"))

                # Use provided scores if available
                if scores_array is not None and len(scores_array) == n_masks:
                    scores = np.array(scores_array, dtype=np.float32)
                else:
                    scores = np.ones(n_masks, dtype=np.float32)

                results[frame_idx] = SegmentationResult(
                    frame_idx=frame_idx,
                    masks=masks,
                    object_ids=np.array(object_ids, dtype=np.int32),
                    bboxes=bboxes,
                    scores=scores,
                    areas=areas,
                    prompts=prompts,
                )

            # Close session
            self._video_predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )
            self._current_session_id = None

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

        return results

    def reset(self):
        """Reset video tracking state."""
        if self._video_predictor is not None and self._current_session_id is not None:
            try:
                self._video_predictor.handle_request(
                    request=dict(
                        type="end_session",
                        session_id=self._current_session_id,
                    )
                )
            except Exception:
                pass
        self._current_session_id = None


class SAMSegmenter:
    """
    Unified wrapper for SAM2/SAM3 segmentation.

    Provides backwards compatibility with SAM2 while supporting SAM3's
    new text prompt capabilities.

    For VLA attack preparation:
    1. Use automatic mask generator to segment first frame (SAM2)
    2. OR use text prompts to segment specific objects (SAM3)
    3. Use video predictor to track objects through the video
    """

    # SAM2 model configurations (for backwards compatibility)
    MODEL_CONFIGS = {
        "sam2.1_hiera_tiny": {
            "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "checkpoint": "sam2.1_hiera_tiny.pt",
        },
        "sam2.1_hiera_small": {
            "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "checkpoint": "sam2.1_hiera_small.pt",
        },
        "sam2.1_hiera_base_plus": {
            "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "checkpoint": "sam2.1_hiera_base_plus.pt",
        },
        "sam2.1_hiera_large": {
            "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "checkpoint": "sam2.1_hiera_large.pt",
        },
    }

    def __init__(
        self,
        model_name: str = "sam3",  # Default to SAM3
        checkpoint_dir: str = "checkpoints/sam2",
        device: str = "cuda",
        # Automatic mask generator settings (SAM2 only)
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
        crop_n_layers: int = 0,
        min_mask_region_area: int = 100,
    ):
        """
        Initialize SAM segmenter.

        Args:
            model_name: "sam3" for SAM3, or SAM2 model variant (see MODEL_CONFIGS)
            checkpoint_dir: Directory containing SAM2 checkpoints (SAM2 only)
            device: Device to run inference on
            points_per_side: Grid size for automatic mask generation (SAM2 only)
            pred_iou_thresh: Predicted IoU threshold for filtering (SAM2 only)
            stability_score_thresh: Stability score threshold for filtering (SAM2 only)
            crop_n_layers: Number of crop layers for hierarchical segmentation (SAM2 only)
            min_mask_region_area: Minimum mask area in pixels
        """
        self.model_name = model_name
        self.device = device
        self.use_sam3 = model_name.lower() == "sam3"

        if self.use_sam3:
            if not SAM3_AVAILABLE:
                raise RuntimeError(
                    "SAM3 is not installed. Install with:\n"
                    "  git clone https://github.com/facebookresearch/sam3.git\n"
                    "  cd sam3 && pip install -e ."
                )
            self._sam3_segmenter = SAM3Segmenter(device=device)
        else:
            if not SAM2_AVAILABLE:
                raise RuntimeError("SAM2 is not installed")

            # SAM2 settings
            self.checkpoint_dir = Path(checkpoint_dir)
            self.points_per_side = points_per_side
            self.pred_iou_thresh = pred_iou_thresh
            self.stability_score_thresh = stability_score_thresh
            self.crop_n_layers = crop_n_layers
            self.min_mask_region_area = min_mask_region_area

            # SAM2 models (lazy loaded)
            self._sam_model = None
            self._mask_generator = None
            self._video_predictor = None
            self._video_state = None

    def _get_model_config(self) -> Tuple[str, str]:
        """Get config and checkpoint paths for SAM2 model."""
        if self.model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {self.model_name}. Available: {list(self.MODEL_CONFIGS.keys())}"
            )

        config = self.MODEL_CONFIGS[self.model_name]
        checkpoint_path = self.checkpoint_dir / config["checkpoint"]

        return config["config"], str(checkpoint_path)

    def _load_mask_generator(self):
        """Load SAM2 automatic mask generator."""
        if self.use_sam3:
            raise RuntimeError(
                "Automatic mask generation is not available with SAM3. Use text prompts instead."
            )

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
        """Load SAM2 video predictor."""
        if self.use_sam3:
            raise RuntimeError("Use segment_video_with_text for SAM3 video tracking")

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

    # ========================
    # SAM3 Text Prompt Methods
    # ========================

    def segment_frame_with_text(
        self,
        image: np.ndarray,
        text_prompts: List[str],
        frame_idx: int = 0,
    ) -> SegmentationResult:
        """
        Segment objects using text prompts (SAM3 only).

        Args:
            image: RGB image array of shape (H, W, 3)
            text_prompts: List of text descriptions, e.g., ["red cup", "robot arm"]
            frame_idx: Frame index for tracking

        Returns:
            SegmentationResult with masks and metadata
        """
        if not self.use_sam3:
            raise RuntimeError(
                "Text prompt segmentation requires SAM3. Initialize with model_name='sam3'"
            )

        return self._sam3_segmenter.segment_frame_with_text(
            image, text_prompts, frame_idx
        )

    def segment_video_with_text(
        self,
        frames: np.ndarray,
        text_prompts: List[str],
        init_frame_idx: int = 0,
    ) -> Dict[int, SegmentationResult]:
        """
        Segment and track objects through video using text prompts (SAM3 only).

        Args:
            frames: Video frames of shape (T, H, W, 3) in RGB
            text_prompts: List of text descriptions
            init_frame_idx: Which frame to use for initial prompt

        Returns:
            Dictionary mapping frame_idx to SegmentationResult
        """
        if not self.use_sam3:
            raise RuntimeError(
                "Text prompt video segmentation requires SAM3. Initialize with model_name='sam3'"
            )

        return self._sam3_segmenter.segment_video_with_text(
            frames, text_prompts, init_frame_idx
        )

    # ========================
    # SAM2 Automatic Methods
    # ========================

    def segment_frame(
        self, image: np.ndarray, frame_idx: int = 0
    ) -> SegmentationResult:
        """
        Segment a single frame using automatic mask generation (SAM2 only).

        Args:
            image: RGB image array of shape (H, W, 3)
            frame_idx: Frame index for tracking

        Returns:
            SegmentationResult with masks and metadata
        """
        if self.use_sam3:
            raise RuntimeError(
                "Automatic mask generation is not available with SAM3. "
                "Use segment_frame_with_text() with text prompts instead."
            )

        self._load_mask_generator()

        # Generate masks
        masks_data = self._mask_generator.generate(image)

        if not masks_data:
            h, w = image.shape[:2]
            return SegmentationResult(
                frame_idx=frame_idx,
                masks=np.zeros((0, h, w), dtype=bool),
                object_ids=np.array([], dtype=np.int32),
                bboxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.array([], dtype=np.float32),
                areas=np.array([], dtype=np.int32),
            )

        n_masks = len(masks_data)

        masks = np.stack([m["segmentation"] for m in masks_data], axis=0)
        object_ids = np.arange(n_masks, dtype=np.int32)
        bboxes = np.array([m["bbox"] for m in masks_data], dtype=np.float32)
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        scores = np.array([m["predicted_iou"] for m in masks_data], dtype=np.float32)
        areas = np.array([m["area"] for m in masks_data], dtype=np.int32)

        return SegmentationResult(
            frame_idx=frame_idx,
            masks=masks,
            object_ids=object_ids,
            bboxes=bboxes,
            scores=scores,
            areas=areas,
        )

    def init_video_tracking(self, video_frames: np.ndarray):
        """Initialize SAM2 video tracking with all frames."""
        if self.use_sam3:
            raise RuntimeError("Use segment_video_with_text for SAM3")

        self._load_video_predictor()
        self._video_state = self._video_predictor.init_state(video_path=video_frames)

    def add_tracking_masks(
        self,
        frame_idx: int,
        masks: np.ndarray,
        object_ids: np.ndarray,
    ):
        """Add masks to track from a specific frame (SAM2 only)."""
        if self.use_sam3:
            raise RuntimeError("Use segment_video_with_text for SAM3")

        if self._video_predictor is None or self._video_state is None:
            raise RuntimeError("Video tracking not initialized")

        for mask, obj_id in zip(masks, object_ids):
            self._video_predictor.add_new_mask(
                inference_state=self._video_state,
                frame_idx=frame_idx,
                obj_id=int(obj_id),
                mask=mask,
            )

    def propagate_tracking(self) -> Dict[int, SegmentationResult]:
        """Propagate SAM2 tracking through all video frames."""
        if self.use_sam3:
            raise RuntimeError("Use segment_video_with_text for SAM3")

        if self._video_predictor is None or self._video_state is None:
            raise RuntimeError("Video tracking not initialized")

        results = {}

        for frame_idx, obj_ids, masks in self._video_predictor.propagate_in_video(
            self._video_state
        ):
            masks_np = (masks > 0.5).cpu().numpy()
            obj_ids_np = np.array(obj_ids, dtype=np.int32)

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
                scores=np.ones(n_masks, dtype=np.float32),
                areas=areas,
            )

        return results

    def reset_tracking(self):
        """Reset video tracking state."""
        if self.use_sam3:
            self._sam3_segmenter.reset()
        else:
            if self._video_predictor is not None and self._video_state is not None:
                self._video_predictor.reset_state(self._video_state)
            self._video_state = None

    def segment_video_with_tracking(
        self,
        frames: np.ndarray,
        init_frame_idx: int = 0,
    ) -> Dict[int, SegmentationResult]:
        """
        Segment a video with consistent object tracking (SAM2 only).

        For SAM3, use segment_video_with_text() instead.
        """
        if self.use_sam3:
            raise RuntimeError(
                "Automatic video segmentation is not available with SAM3. "
                "Use segment_video_with_text() with text prompts instead."
            )

        results = {}

        print(f"Segmenting initial frame {init_frame_idx}...")
        init_result = self.segment_frame(frames[init_frame_idx], init_frame_idx)
        results[init_frame_idx] = init_result

        if len(init_result.masks) == 0:
            print("No objects found in initial frame")
            return results

        print(f"Found {len(init_result.masks)} objects in initial frame")

        print("Initializing video tracking...")
        self.init_video_tracking(frames)

        self.add_tracking_masks(
            init_frame_idx, init_result.masks, init_result.object_ids
        )

        print("Propagating masks through video...")
        tracked_results = self.propagate_tracking()
        results.update(tracked_results)

        self.reset_tracking()

        return results


if __name__ == "__main__":
    print(f"SAM3 available: {SAM3_AVAILABLE}")
    print(f"SAM2 available: {SAM2_AVAILABLE}")

    if SAM3_AVAILABLE:
        print("\nTesting SAM3 segmenter...")
        segmenter = SAMSegmenter(model_name="sam3", device="cuda")
        print("SAM3 segmenter created successfully")
    elif SAM2_AVAILABLE:
        print("\nTesting SAM2 segmenter...")
        segmenter = SAMSegmenter(
            model_name="sam2.1_hiera_tiny",
            device="cpu",
        )
        print("SAM2 segmenter created")
        print(f"Model config: {segmenter._get_model_config()}")
