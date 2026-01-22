#!/usr/bin/env python3
"""
Visualize segmentation results as overlays on original images.
Helps debug SAM3/SAM2 segmentation quality.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import h5py
from PIL import Image
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation import SegmentationConfig, LiberoDataLoader, MaskStorage


def random_color():
    """Generate a random color for mask visualization."""
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def overlay_masks_on_image(
    image: np.ndarray, masks: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay masks on an image with random colors.

    Args:
        image: RGB image (H, W, 3)
        masks: Boolean masks (N, H, W)
        alpha: Transparency for masks

    Returns:
        Image with overlaid masks
    """
    overlay = image.copy().astype(np.float32)

    if len(masks) == 0:
        return image

    # Generate colors for each mask
    colors = [random_color() for _ in range(len(masks))]

    for mask, color in zip(masks, colors):
        # Create colored mask
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c],
            )

        # Draw mask contour
        # Find edges
        from scipy import ndimage

        edges = ndimage.binary_dilation(mask) ^ mask
        for c in range(3):
            overlay[:, :, c] = np.where(edges, color[c], overlay[:, :, c])

    return overlay.astype(np.uint8)


def create_visualization_grid(images: list, titles: list, cols: int = 2) -> Image.Image:
    """Create a grid of images with titles."""
    from PIL import ImageDraw, ImageFont

    if len(images) == 0:
        return Image.new("RGB", (400, 100), color="white")

    # Get image dimensions
    w, h = images[0].size

    # Calculate grid dimensions
    rows = (len(images) + cols - 1) // cols

    # Create grid image
    grid_w = cols * w + (cols + 1) * 10
    grid_h = rows * (h + 30) + 10
    grid = Image.new("RGB", (grid_w, grid_h), color="white")

    draw = ImageDraw.Draw(grid)

    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // cols
        col = i % cols

        x = col * w + (col + 1) * 10
        y = row * (h + 30) + 10

        # Paste image
        grid.paste(img, (x, y))

        # Draw title
        draw.text((x, y + h + 2), title[:40], fill="black")

    return grid


def visualize_demo(
    loader: LiberoDataLoader,
    storage: MaskStorage,
    demo_info,
    view: str,
    frame_indices: list = None,
    output_path: Path = None,
):
    """Visualize segmentation results for a demo."""
    import json

    # Load original frames
    with h5py.File(demo_info.hdf5_path, "r") as f:
        frames = f[f"data/demo_{demo_info.demo_id}/obs/{view}"][:]
        # Flip vertically + BGR to RGB (LIBERO stores images upside-down)
        frames = frames[:, ::-1, :, ::-1].copy()

    print(f"Loaded {len(frames)} frames from {view}")

    # Load segmentation results
    masks_path = storage.get_masks_path(demo_info, view)
    if not masks_path.exists():
        print(f"No segmentation results found at {masks_path}")
        return None

    # Load metadata to get prompts
    metadata_path = masks_path.parent / "metadata.json"
    prompts = []
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            view_info = metadata.get("views", {}).get(view, {})
            seg_config = view_info.get("segmentation_config", {})
            prompts = seg_config.get("text_prompts", [])
            print(f"Prompts used: {prompts}")

    # Select frame indices
    if frame_indices is None:
        # Sample evenly spaced frames
        n_samples = min(8, len(frames))
        frame_indices = np.linspace(0, len(frames) - 1, n_samples, dtype=int).tolist()

    images = []
    titles = []

    with h5py.File(masks_path, "r") as f:
        for frame_idx in frame_indices:
            frame = frames[frame_idx]

            frame_key = f"frame_{frame_idx:04d}"
            if frame_key in f:
                masks = f[frame_key]["masks"][:]
                object_ids = f[frame_key]["object_ids"][:]
                n_objects = len(masks)
            else:
                masks = np.zeros((0, frame.shape[0], frame.shape[1]), dtype=bool)
                n_objects = 0

            # Create original and overlay images
            original = Image.fromarray(frame)

            if n_objects > 0:
                overlay = overlay_masks_on_image(frame, masks)
                overlay_img = Image.fromarray(overlay)
                title = f"Frame {frame_idx}: {n_objects} objects"
            else:
                overlay_img = original.copy()
                title = f"Frame {frame_idx}: NO MASKS"

            # Upscale for better visibility
            scale = 2
            original = original.resize(
                (original.width * scale, original.height * scale), Image.NEAREST
            )
            overlay_img = overlay_img.resize(
                (overlay_img.width * scale, overlay_img.height * scale), Image.NEAREST
            )

            images.append(original)
            titles.append(f"Original {frame_idx}")
            images.append(overlay_img)
            titles.append(title)

    # Create grid with prompts
    grid = create_visualization_grid_with_prompts(images, titles, prompts, cols=4)

    if output_path:
        grid.save(output_path)
        print(f"Saved visualization to {output_path}")

    return grid


def create_visualization_grid_with_prompts(
    images: list, titles: list, prompts: list, cols: int = 4
) -> Image.Image:
    """Create a grid of images with titles and prompts footer."""
    from PIL import ImageDraw, ImageFont

    if len(images) == 0:
        return Image.new("RGB", (400, 100), color="white")

    # Get image dimensions
    w, h = images[0].size

    # Calculate grid dimensions
    rows = (len(images) + cols - 1) // cols

    # Add extra space for prompts footer
    prompts_text = f"Prompts: {', '.join(prompts)}" if prompts else "No prompts"
    footer_height = 40

    # Create grid image
    grid_w = cols * w + (cols + 1) * 10
    grid_h = rows * (h + 30) + 10 + footer_height
    grid = Image.new("RGB", (grid_w, grid_h), color="white")

    draw = ImageDraw.Draw(grid)

    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // cols
        col = i % cols

        x = col * w + (col + 1) * 10
        y = row * (h + 30) + 10

        # Paste image
        grid.paste(img, (x, y))

        # Draw title
        draw.text((x, y + h + 2), title[:40], fill="black")

    # Draw prompts footer
    footer_y = rows * (h + 30) + 15
    draw.rectangle(
        [(5, footer_y - 5), (grid_w - 5, grid_h - 5)], fill="#f0f0f0", outline="#cccccc"
    )
    draw.text((10, footer_y), prompts_text, fill="darkblue")

    return grid


def visualize_raw_sam3_debug(
    demo_info, view: str, prompts: list, output_path: Path = None
):
    """
    Debug visualization: Run SAM3 on a single frame and show what it detects.
    This bypasses the full pipeline to debug SAM3 directly.
    """
    import tempfile
    import shutil

    print(f"\n{'=' * 60}")
    print("DEBUG: Running SAM3 directly on a single frame")
    print(f"{'=' * 60}")

    # Load a frame
    with h5py.File(demo_info.hdf5_path, "r") as f:
        frame = f[f"data/demo_{demo_info.demo_id}/obs/{view}"][0]
        # Flip vertically + BGR to RGB (LIBERO stores images upside-down)
        frame = frame[::-1, :, ::-1].copy()

    print(f"Frame shape: {frame.shape}")

    # Save frame to temp directory
    temp_dir = tempfile.mkdtemp(prefix="sam3_debug_")
    try:
        img = Image.fromarray(frame)
        img.save(f"{temp_dir}/00000.jpg")

        # Also save an upscaled version to test
        upscaled = img.resize((512, 512), Image.LANCZOS)
        upscaled_dir = tempfile.mkdtemp(prefix="sam3_upscaled_")
        upscaled.save(f"{upscaled_dir}/00000.jpg")

        from sam3.model_builder import build_sam3_video_predictor

        print("\nLoading SAM3 video predictor...")
        predictor = build_sam3_video_predictor()

        results_128 = test_prompts(predictor, temp_dir, prompts, "128x128")
        results_512 = test_prompts(predictor, upscaled_dir, prompts, "512x512")

        # Create visualization
        images = []
        titles = []

        # Original
        images.append(img.resize((256, 256), Image.NEAREST))
        titles.append("Original 128x128")

        images.append(upscaled.resize((256, 256), Image.LANCZOS))
        titles.append("Upscaled 512x512")

        # Results summary
        print(f"\n{'=' * 60}")
        print("RESULTS SUMMARY")
        print(f"{'=' * 60}")
        print(f"128x128: {results_128}")
        print(f"512x512: {results_512}")

        grid = create_visualization_grid(images, titles, cols=2)

        if output_path:
            grid.save(output_path)
            print(f"\nSaved debug visualization to {output_path}")

        # Cleanup
        shutil.rmtree(upscaled_dir)

    finally:
        shutil.rmtree(temp_dir)

    return results_128, results_512


def test_prompts(predictor, frame_dir: str, prompts: list, label: str):
    """Test prompts on a single frame."""
    results = {}

    # Start session
    response = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": frame_dir,
        }
    )
    session_id = response["session_id"]

    try:
        for prompt in prompts:
            response = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt,
                }
            )

            outputs = response.get("outputs", {})
            n_objects = len(outputs.get("out_obj_ids", []))
            results[prompt] = n_objects
            print(f"  [{label}] '{prompt}': {n_objects} objects detected")

    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Visualize segmentation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--demo", type=int, default=0, help="Demo index to visualize")
    parser.add_argument(
        "--view",
        default="agentview_rgb",
        help="View to visualize (agentview_rgb or eye_in_hand_rgb)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="segmentation_viz.png",
        help="Output path for visualization",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=None,
        help="Specific frame indices to visualize",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run SAM3 debug mode on a single frame"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["robot arm", "bowl", "drawer", "cabinet", "plate", "object"],
        help="Prompts to test in debug mode",
    )

    args = parser.parse_args()

    config = SegmentationConfig()
    loader = LiberoDataLoader(config.libero_datasets_dir)
    storage = MaskStorage(config.output_dir)

    # Find a demo
    demos = loader.discover_demos(["libero_spatial"])
    if not demos:
        print("No demos found!")
        return

    if args.demo >= len(demos):
        print(f"Demo index {args.demo} out of range. Found {len(demos)} demos.")
        args.demo = 0

    demo = demos[args.demo]
    print(f"\nUsing demo: {demo.task_name}/demo_{demo.demo_id}")

    output_path = Path(args.output)

    if args.debug:
        # Debug mode: test SAM3 directly
        visualize_raw_sam3_debug(demo, args.view, args.prompts, output_path)
    else:
        # Normal mode: visualize existing segmentation results
        visualize_demo(loader, storage, demo, args.view, args.frames, output_path)


if __name__ == "__main__":
    main()
