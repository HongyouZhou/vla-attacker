"""
HDF5 storage for segmentation masks.
Efficient storage and retrieval of masks for attack preparation.
"""

import h5py
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import asdict

from .sam_segmenter import SegmentationResult
from .libero_loader import DemoInfo


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class MaskStorage:
    """
    Efficient HDF5 storage for segmentation masks.

    Storage structure:
    - output_dir/
      └── task_suite/
          └── task_name/
              └── demo_X/
                  ├── view_name_masks.hdf5
                  └── metadata.json
    """

    def __init__(self, output_dir: Path):
        """
        Initialize mask storage.

        Args:
            output_dir: Root directory for segmentation output
        """
        self.output_dir = Path(output_dir)

    def get_demo_dir(self, demo_info: DemoInfo) -> Path:
        """Get the output directory for a specific demo."""
        return (
            self.output_dir
            / demo_info.task_suite
            / demo_info.task_name
            / f"demo_{demo_info.demo_id}"
        )

    def get_masks_path(self, demo_info: DemoInfo, view: str) -> Path:
        """Get the HDF5 file path for a specific view's masks."""
        return self.get_demo_dir(demo_info) / f"{view}_masks.hdf5"

    def get_metadata_path(self, demo_info: DemoInfo) -> Path:
        """Get the metadata JSON file path."""
        return self.get_demo_dir(demo_info) / "metadata.json"

    def save_results(
        self,
        demo_info: DemoInfo,
        view: str,
        results: Dict[int, SegmentationResult],
        config: dict,
    ):
        """
        Save segmentation results to HDF5 file.

        Args:
            demo_info: Demo information
            view: View name (e.g., "agentview_rgb")
            results: Dictionary mapping frame_idx to SegmentationResult
            config: Segmentation configuration used
        """
        demo_dir = self.get_demo_dir(demo_info)
        demo_dir.mkdir(parents=True, exist_ok=True)

        masks_path = self.get_masks_path(demo_info, view)

        with h5py.File(masks_path, "w") as f:
            f.attrs["num_frames"] = len(results)
            f.attrs["image_height"] = demo_info.image_size[0]
            f.attrs["image_width"] = demo_info.image_size[1]
            f.attrs["view"] = view

            # Store each frame's results
            for frame_idx, result in results.items():
                frame_grp = f.create_group(f"frame_{frame_idx:04d}")

                # Use compression for masks (they're binary, very compressible)
                if len(result.masks) > 0:
                    frame_grp.create_dataset(
                        "masks",
                        data=result.masks.astype(np.uint8),
                        compression="gzip",
                        compression_opts=4,
                    )
                    frame_grp.create_dataset("object_ids", data=result.object_ids)
                    frame_grp.create_dataset("bboxes", data=result.bboxes)
                    frame_grp.create_dataset("scores", data=result.scores)
                    frame_grp.create_dataset("areas", data=result.areas)
                else:
                    # Empty frame
                    h, w = demo_info.image_size
                    frame_grp.create_dataset(
                        "masks", data=np.zeros((0, h, w), dtype=np.uint8)
                    )
                    frame_grp.create_dataset(
                        "object_ids", data=np.array([], dtype=np.int32)
                    )
                    frame_grp.create_dataset(
                        "bboxes", data=np.zeros((0, 4), dtype=np.float32)
                    )
                    frame_grp.create_dataset(
                        "scores", data=np.array([], dtype=np.float32)
                    )
                    frame_grp.create_dataset("areas", data=np.array([], dtype=np.int32))

                frame_grp.attrs["num_objects"] = len(result.masks)

        print(f"Saved masks to {masks_path}")

        # Update metadata
        self._update_metadata(demo_info, view, results, config)

    def _update_metadata(
        self,
        demo_info: DemoInfo,
        view: str,
        results: Dict[int, SegmentationResult],
        config: dict,
    ):
        """Update or create metadata JSON file."""
        metadata_path = self.get_metadata_path(demo_info)

        # Load existing metadata or create new
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {
                "task_suite": demo_info.task_suite,
                "task_name": demo_info.task_name,
                "demo_id": demo_info.demo_id,
                "num_frames": demo_info.num_frames,
                "image_size": list(demo_info.image_size),
                "views": {},
                "created_at": datetime.now().isoformat(),
            }

        # Compute object statistics
        all_object_ids = set()
        object_stats = {}

        for frame_idx, result in results.items():
            frame_idx = int(frame_idx)  # Ensure native Python int for JSON
            for obj_id, area in zip(result.object_ids, result.areas):
                obj_id = int(obj_id)
                all_object_ids.add(obj_id)

                if obj_id not in object_stats:
                    object_stats[obj_id] = {
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "total_area": 0,
                        "frame_count": 0,
                    }

                stats = object_stats[obj_id]
                stats["last_frame"] = int(max(stats["last_frame"], frame_idx))
                stats["total_area"] += int(area)
                stats["frame_count"] += 1

        # Compute average area
        for obj_id, stats in object_stats.items():
            stats["avg_area"] = stats["total_area"] // max(1, stats["frame_count"])
            del stats["total_area"]
            del stats["frame_count"]

        # Update view info
        metadata["views"][view] = {
            "num_frames_processed": len(results),
            "num_objects": len(all_object_ids),
            "objects": object_stats,
            "segmentation_config": config,
            "processed_at": datetime.now().isoformat(),
        }

        metadata["updated_at"] = datetime.now().isoformat()

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)

        print(f"Updated metadata at {metadata_path}")

    def load_frame_masks(
        self,
        demo_info: DemoInfo,
        view: str,
        frame_idx: int,
    ) -> Optional[SegmentationResult]:
        """
        Load masks for a specific frame.

        Args:
            demo_info: Demo information
            view: View name
            frame_idx: Frame index

        Returns:
            SegmentationResult or None if not found
        """
        masks_path = self.get_masks_path(demo_info, view)

        if not masks_path.exists():
            return None

        with h5py.File(masks_path, "r") as f:
            frame_key = f"frame_{frame_idx:04d}"

            if frame_key not in f:
                return None

            frame_grp = f[frame_key]

            return SegmentationResult(
                frame_idx=frame_idx,
                masks=frame_grp["masks"][:].astype(bool),
                object_ids=frame_grp["object_ids"][:],
                bboxes=frame_grp["bboxes"][:],
                scores=frame_grp["scores"][:],
                areas=frame_grp["areas"][:],
            )

    def load_all_masks(
        self,
        demo_info: DemoInfo,
        view: str,
    ) -> Dict[int, SegmentationResult]:
        """
        Load all masks for a demo/view.

        Args:
            demo_info: Demo information
            view: View name

        Returns:
            Dictionary mapping frame_idx to SegmentationResult
        """
        masks_path = self.get_masks_path(demo_info, view)

        if not masks_path.exists():
            return {}

        results = {}

        with h5py.File(masks_path, "r") as f:
            for key in f.keys():
                if key.startswith("frame_"):
                    frame_idx = int(key.split("_")[1])
                    frame_grp = f[key]

                    results[frame_idx] = SegmentationResult(
                        frame_idx=frame_idx,
                        masks=frame_grp["masks"][:].astype(bool),
                        object_ids=frame_grp["object_ids"][:],
                        bboxes=frame_grp["bboxes"][:],
                        scores=frame_grp["scores"][:],
                        areas=frame_grp["areas"][:],
                    )

        return results

    def load_metadata(self, demo_info: DemoInfo) -> Optional[dict]:
        """Load metadata for a demo."""
        metadata_path = self.get_metadata_path(demo_info)

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            return json.load(f)

    def is_processed(self, demo_info: DemoInfo, view: str) -> bool:
        """Check if a demo/view has already been processed."""
        masks_path = self.get_masks_path(demo_info, view)
        return masks_path.exists()

    def get_object_mask_at_frame(
        self,
        demo_info: DemoInfo,
        view: str,
        frame_idx: int,
        object_id: int,
    ) -> Optional[np.ndarray]:
        """
        Get a specific object's mask at a specific frame.
        Useful for attack preparation.

        Args:
            demo_info: Demo information
            view: View name
            frame_idx: Frame index
            object_id: Object ID to retrieve

        Returns:
            Boolean mask of shape (H, W) or None if not found
        """
        result = self.load_frame_masks(demo_info, view, frame_idx)

        if result is None:
            return None

        # Find the object
        idx = np.where(result.object_ids == object_id)[0]

        if len(idx) == 0:
            return None

        return result.masks[idx[0]]


if __name__ == "__main__":
    # Test the storage
    from .libero_loader import DemoInfo

    storage = MaskStorage(Path("test_output"))

    # Create dummy demo info
    demo = DemoInfo(
        task_suite="libero_spatial",
        task_name="test_task",
        demo_id=0,
        hdf5_path=Path("dummy.hdf5"),
        num_frames=10,
        image_size=(128, 128),
        available_views=["agentview_rgb"],
    )

    print(f"Demo dir: {storage.get_demo_dir(demo)}")
    print(f"Masks path: {storage.get_masks_path(demo, 'agentview_rgb')}")
    print(f"Metadata path: {storage.get_metadata_path(demo)}")
