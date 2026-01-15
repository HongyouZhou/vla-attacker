"""
LIBERO dataset loader for segmentation pipeline.
Loads demonstration data from HDF5 files.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass


@dataclass
class DemoInfo:
    """Information about a demonstration."""
    task_suite: str
    task_name: str
    demo_id: int
    hdf5_path: Path
    num_frames: int
    image_size: Tuple[int, int]
    available_views: List[str]


class LiberoDataLoader:
    """Load LIBERO demonstration data from HDF5 files."""
    
    def __init__(self, datasets_dir: Path, views: List[str] = None):
        """
        Initialize the loader.
        
        Args:
            datasets_dir: Path to LIBERO datasets directory
            views: List of view names to load (e.g., ["agentview_rgb", "eye_in_hand_rgb"])
        """
        self.datasets_dir = Path(datasets_dir)
        self.views = views or ["agentview_rgb", "eye_in_hand_rgb"]
        
    def discover_demos(self, task_suites: List[str] = None) -> List[DemoInfo]:
        """
        Discover all available demonstrations.
        
        Args:
            task_suites: Optional list of task suites to filter (e.g., ["libero_spatial"])
        
        Returns:
            List of DemoInfo objects
        """
        demos = []
        
        if task_suites is None:
            task_suites = ["libero_spatial", "libero_object", "libero_goal", "libero_100"]
        
        for suite in task_suites:
            suite_dir = self.datasets_dir / suite
            if not suite_dir.exists():
                print(f"Warning: Task suite directory not found: {suite_dir}")
                continue
            
            # Find all HDF5 files
            for hdf5_path in suite_dir.glob("*.hdf5"):
                task_name = hdf5_path.stem
                
                # Open file and discover demos
                try:
                    with h5py.File(hdf5_path, 'r') as f:
                        if 'data' not in f:
                            continue
                        
                        data_group = f['data']
                        demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])
                        
                        for demo_key in demo_keys:
                            demo_id = int(demo_key.split('_')[1])
                            demo_group = data_group[demo_key]
                            
                            # Get number of frames
                            num_frames = demo_group.attrs.get('num_samples', 0)
                            
                            # Check available views
                            obs_group = demo_group.get('obs', None)
                            if obs_group is None:
                                continue
                            
                            available_views = []
                            image_size = (128, 128)  # Default
                            
                            for view in self.views:
                                if view in obs_group:
                                    available_views.append(view)
                                    # Get image size from first frame
                                    img_data = obs_group[view]
                                    if len(img_data.shape) >= 3:
                                        image_size = (img_data.shape[1], img_data.shape[2])
                            
                            if available_views:
                                demos.append(DemoInfo(
                                    task_suite=suite,
                                    task_name=task_name,
                                    demo_id=demo_id,
                                    hdf5_path=hdf5_path,
                                    num_frames=num_frames,
                                    image_size=image_size,
                                    available_views=available_views,
                                ))
                except Exception as e:
                    print(f"Error loading {hdf5_path}: {e}")
                    continue
        
        return demos
    
    def load_frames(
        self, 
        demo_info: DemoInfo, 
        view: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Load frames from a demonstration as a generator.
        
        Args:
            demo_info: DemoInfo object
            view: View name (e.g., "agentview_rgb")
            start_frame: Starting frame index
            end_frame: Ending frame index (exclusive), None for all frames
            
        Yields:
            Tuple of (frame_index, image_array)
        """
        with h5py.File(demo_info.hdf5_path, 'r') as f:
            demo_key = f"demo_{demo_info.demo_id}"
            obs_group = f['data'][demo_key]['obs']
            
            if view not in obs_group:
                raise ValueError(f"View {view} not found in demo")
            
            images = obs_group[view]
            num_frames = images.shape[0]
            
            if end_frame is None:
                end_frame = num_frames
            
            for i in range(start_frame, min(end_frame, num_frames)):
                # Images are stored as (T, H, W, C) in BGR format, convert to RGB
                img = images[i][:]
                if img.shape[-1] == 3:
                    img = img[..., ::-1].copy()  # BGR to RGB
                yield i, img
    
    def load_all_frames(
        self,
        demo_info: DemoInfo,
        view: str,
    ) -> np.ndarray:
        """
        Load all frames at once (use with caution for memory).
        
        Args:
            demo_info: DemoInfo object
            view: View name
            
        Returns:
            Array of shape (T, H, W, C) in RGB format
        """
        frames = []
        for _, frame in self.load_frames(demo_info, view):
            frames.append(frame)
        return np.stack(frames, axis=0)


if __name__ == "__main__":
    # Test the loader
    loader = LiberoDataLoader(Path("LIBERO/datasets"))
    demos = loader.discover_demos(["libero_spatial"])
    
    print(f"Found {len(demos)} demonstrations")
    if demos:
        demo = demos[0]
        print(f"First demo: {demo.task_name} (demo {demo.demo_id})")
        print(f"  Frames: {demo.num_frames}")
        print(f"  Image size: {demo.image_size}")
        print(f"  Views: {demo.available_views}")
        
        # Load first frame
        for idx, frame in loader.load_frames(demo, demo.available_views[0], end_frame=1):
            print(f"  Frame {idx} shape: {frame.shape}")
