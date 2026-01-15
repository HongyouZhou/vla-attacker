#!/usr/bin/env python3
"""
Visualize segmentation results.
Generates images and videos showing masks, bounding boxes, and object IDs.

Usage:
    # Visualize specific demo
    python scripts/visualize_masks.py --suite libero_spatial --task TASK_NAME --demo 0 --video
    
    # Visualize first found processed demo
    python scripts/visualize_masks.py --auto --video
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from segmentation.config import SegmentationConfig
from segmentation.libero_loader import LiberoDataLoader, DemoInfo
from segmentation.mask_storage import MaskStorage

try:
    import cv2
except ImportError:
    print("Error: OpenCV is required for visualization.")
    print("Install with: pip install opencv-python")
    sys.exit(1)


def generate_color_palette(n_colors: int) -> np.ndarray:
    """Generate distinct colors for visualization."""
    np.random.seed(42)  # Reproducible colors
    # Generate random colors in HSV then convert to RGB for better distinctness
    # For now, simple random RGB is fine but we avoid dark colors
    colors = np.random.randint(100, 255, size=(n_colors, 3), dtype=np.uint8)
    return colors


def draw_segmentation(
    image: np.ndarray,
    masks: np.ndarray,
    object_ids: np.ndarray,
    bboxes: np.ndarray = None,
    alpha: float = 0.5,
    draw_boxes: bool = True,
) -> np.ndarray:
    """
    Draw masks and bounding boxes on image.
    
    Args:
        image: RGB image of shape (H, W, 3)
        masks: Boolean masks of shape (N, H, W)
        object_ids: Object IDs of shape (N,)
        bboxes: Bounding boxes of shape (N, 4) in [x, y, w, h] format
        alpha: Transparency of mask overlay
        draw_boxes: Whether to draw bounding boxes
        
    Returns:
        Image with visualization
    """
    result = image.copy()
    overlay = image.copy()
    
    # Generate colors based on ALL possible IDs to keep consistency
    # (Assuming max ID isn't huge, or we hash the ID)
    max_id = 100  # Default palette size
    if len(object_ids) > 0:
        max_id = max(max_id, max(object_ids) + 1)
    colors = generate_color_palette(max_id)
    
    for i, (mask, obj_id) in enumerate(zip(masks, object_ids)):
        if mask.sum() == 0:
            continue
            
        color = colors[obj_id % len(colors)].tolist()
        
        # Draw mask
        # Create colored mask
        colored_mask = np.zeros_like(image)
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]
            
        # Apply strict overlay where mask is True
        # Method: blend overlay onto result
        overlay[mask] = colored_mask[mask]
        
        # Find contours for border
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color, 1)

        # Draw Bounding Box
        if draw_boxes and bboxes is not None and len(bboxes) > i:
            x, y, w, h = map(int, bboxes[i])
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 1)
            
            # Put ID text
            label = f"ID:{obj_id}"
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(result, (x, y - h_text - 4), (x + w_text, y), color, -1)
            cv2.putText(
                result, label, (x, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
            )

    # Blend overlay
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    
    return result


def create_video(
    demo: DemoInfo,
    view: str,
    loader: LiberoDataLoader,
    storage: MaskStorage,
    output_path: Path,
    fps: int = 20,
):
    """Create a video visualization of the segmentation."""
    print(f"Generating video: {output_path}")
    
    # Load all masks first
    all_masks = storage.load_all_masks(demo, view)
    if not all_masks:
        print("  No masks found!")
        return

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = demo.image_size
    # Side-by-side: Original | Segmented
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w * 2, h))
    
    frames_gen = loader.load_frames(demo, view)
    
    for frame_idx, frame in tqdm(frames_gen, total=demo.num_frames, desc="  Rendering frames"):
        if frame_idx in all_masks:
            res = all_masks[frame_idx]
            vis_frame = draw_segmentation(
                frame, res.masks, res.object_ids, res.bboxes
            )
        else:
            vis_frame = frame.copy()
            cv2.putText(vis_frame, "No Mask", (10, 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Create side-by-side
        combined = np.hstack((frame, vis_frame))
        
        # Add frame info
        cv2.putText(
            combined, f"Frame: {frame_idx}", (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        # Convert RGB to BGR for OpenCV
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        video_writer.write(combined_bgr)
        
    video_writer.release()
    print(f"  Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize segmentation results")
    
    parser.add_argument("--auto", action="store_true", help="Automatically find a processed demo")
    parser.add_argument("--suite", type=str, help="Task suite name")
    parser.add_argument("--task", type=str, help="Task name")
    parser.add_argument("--demo", type=int, default=0, help="Demo ID")
    parser.add_argument("--view", type=str, default="agentview_rgb", help="View name")
    parser.add_argument("--video", action="store_true", help="Generate video instead of images")
    parser.add_argument("--output", type=str, default="visualization_results", help="Output directory")
    
    args = parser.parse_args()
    
    config = SegmentationConfig()
    loader = LiberoDataLoader(config.libero_datasets_dir, views=[args.view])
    storage = MaskStorage(config.output_dir)
    
    demo_to_proc = None
    
    if args.auto:
        # Find any demo that has been processed
        print("Searching for processed demos...")
        all_demos = loader.discover_demos(config.task_suites)
        for d in all_demos:
            if storage.is_processed(d, args.view):
                demo_to_proc = d
                print(f"Found processed demo: {d.task_name} (ID: {d.demo_id})")
                break
        if demo_to_proc is None:
            print("No processed demos found! Run segment_dataset.py first.")
            sys.exit(1)
            
    elif args.suite:
        # User specified demo
        demos = loader.discover_demos([args.suite])
        if args.task:
            demos = [d for d in demos if d.task_name == args.task and d.demo_id == args.demo]
        else:
            demos = [d for d in demos if d.demo_id == args.demo]
            
        if not demos:
            print("Demo not found.")
            sys.exit(1)
        demo_to_proc = demos[0]
        
    else:
        print("Please specify --auto or provide --suite, --task arguments.")
        parser.print_help()
        sys.exit(1)

    # Setup output
    out_dir = Path(args.output) / demo_to_proc.task_suite / demo_to_proc.task_name / f"demo_{demo_to_proc.demo_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.video:
        vid_path = out_dir / f"{args.view}_vis.mp4"
        create_video(demo_to_proc, args.view, loader, storage, vid_path)
    else:
        # Image mode: save few frames
        print("Generating sample frames...")
        frames_gen = loader.load_frames(demo_to_proc, args.view)
        # Just grab first, middle, last
        indices = [0, demo_to_proc.num_frames // 2, demo_to_proc.num_frames - 1]
        
        all_masks = storage.load_all_masks(demo_to_proc, args.view)
        
        for idx, frame in frames_gen:
            if idx in indices:
                if idx in all_masks:
                    res = all_masks[idx]
                    vis = draw_segmentation(frame, res.masks, res.object_ids, res.bboxes)
                else:
                    vis = frame
                
                out_path = out_dir / f"{args.view}_frame_{idx:04d}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                print(f"Saved {out_path}")
