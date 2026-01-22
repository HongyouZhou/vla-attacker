#!/usr/bin/env python3
"""
Main script to segment LIBERO dataset using SAM3 (text prompts) or SAM2 (automatic).

SAM3 Usage (recommended):
    # Test with text prompts
    python scripts/segment_dataset.py --test --prompts "robot gripper" "wooden block"

    # Process with custom prompts
    python scripts/segment_dataset.py --suite libero_spatial --prompts "robot arm" "target object"

SAM2 Usage (automatic segmentation):
    # Test with SAM2 automatic segmentation
    python scripts/segment_dataset.py --test --model sam2.1_hiera_tiny

Common options:
    # Process all demos
    python scripts/segment_dataset.py --all

    # Process specific task suite
    python scripts/segment_dataset.py --suite libero_spatial
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from segmentation.config import SegmentationConfig
from segmentation.libero_loader import LiberoDataLoader, DemoInfo
from segmentation.sam_segmenter import SAMSegmenter, SAM3_AVAILABLE, SAM2_AVAILABLE
from segmentation.mask_storage import MaskStorage


def process_demo_sam3(
    demo: DemoInfo,
    view: str,
    loader: LiberoDataLoader,
    segmenter: SAMSegmenter,
    storage: MaskStorage,
    config: SegmentationConfig,
    force: bool = False,
):
    """
    Process a single demo's view using SAM3 text prompts.
    """
    # Check if already processed
    if not force and storage.is_processed(demo, view):
        print(f"  Skipping {view} (already processed)")
        return

    print(f"  Processing {view} with SAM3 text prompts...")

    # Get prompts for this task suite
    prompts = config.get_prompts_for_task(demo.task_suite)
    print(f"    Using prompts: {prompts}")

    # Load all frames for this view
    frames = loader.load_all_frames(demo, view)
    print(f"    Loaded {len(frames)} frames of shape {frames.shape[1:]}")

    if config.enable_tracking:
        # Use video tracking with text prompts
        results = segmenter.segment_video_with_text(
            frames=frames,
            text_prompts=prompts,
            init_frame_idx=0,
        )
    else:
        # Segment each frame independently with text prompts
        results = {}
        for i, frame in enumerate(tqdm(frames, desc="    Segmenting")):
            results[i] = segmenter.segment_frame_with_text(frame, prompts, i)

    # Save results
    config_dict = {
        "sam_model_name": config.sam_model_name,
        "text_prompts": prompts,
        "enable_tracking": config.enable_tracking,
    }

    storage.save_results(demo, view, results, config_dict)

    # Print stats
    total_objects = sum(len(r.masks) for r in results.values())
    unique_objects = len(
        set(obj_id for r in results.values() for obj_id in r.object_ids)
    )

    print(
        f"    Done! Total mask instances: {total_objects}, Unique objects: {unique_objects}"
    )


def process_demo_sam2(
    demo: DemoInfo,
    view: str,
    loader: LiberoDataLoader,
    segmenter: SAMSegmenter,
    storage: MaskStorage,
    config: SegmentationConfig,
    force: bool = False,
):
    """
    Process a single demo's view using SAM2 automatic segmentation.
    """
    # Check if already processed
    if not force and storage.is_processed(demo, view):
        print(f"  Skipping {view} (already processed)")
        return

    print(f"  Processing {view} with SAM2 automatic segmentation...")

    # Load all frames for this view
    frames = loader.load_all_frames(demo, view)
    print(f"    Loaded {len(frames)} frames of shape {frames.shape[1:]}")

    if config.enable_tracking:
        # Use video tracking for consistent IDs
        results = segmenter.segment_video_with_tracking(frames, init_frame_idx=0)
    else:
        # Segment each frame independently
        results = {}
        for i, frame in enumerate(tqdm(frames, desc="    Segmenting")):
            results[i] = segmenter.segment_frame(frame, i)

    # Save results
    config_dict = {
        "sam_model_name": config.sam_model_name,
        "points_per_side": config.points_per_side,
        "pred_iou_thresh": config.pred_iou_thresh,
        "stability_score_thresh": config.stability_score_thresh,
        "min_mask_region_area": config.min_mask_region_area,
        "enable_tracking": config.enable_tracking,
    }

    storage.save_results(demo, view, results, config_dict)

    # Print stats
    total_objects = sum(len(r.masks) for r in results.values())
    unique_objects = len(
        set(obj_id for r in results.values() for obj_id in r.object_ids)
    )

    print(
        f"    Done! Total mask instances: {total_objects}, Unique objects: {unique_objects}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Segment LIBERO dataset with SAM3 (text prompts) or SAM2 (automatic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SAM3 with text prompts (recommended)
  python scripts/segment_dataset.py --test --prompts "robot gripper" "wooden block"
  
  # SAM2 automatic segmentation
  python scripts/segment_dataset.py --test --model sam2.1_hiera_tiny
  
  # Process specific task suite
  python scripts/segment_dataset.py --suite libero_spatial
        """,
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (single demo, fast settings)",
    )
    parser.add_argument("--all", action="store_true", help="Process all demos")
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Process specific task suite (e.g., libero_spatial)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force reprocessing even if already done"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sam3",
        help="Model to use: 'sam3' (text prompts) or SAM2 variant (e.g., sam2.1_hiera_tiny)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Text prompts for SAM3 (e.g., --prompts 'robot gripper' 'wooden block')",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of demos to process"
    )

    args = parser.parse_args()

    # Check model availability
    use_sam3 = args.model.lower() == "sam3"

    if use_sam3 and not SAM3_AVAILABLE:
        print("Error: SAM3 is not installed.")
        print("Install with:")
        print("  git clone https://github.com/facebookresearch/sam3.git")
        print("  cd sam3 && pip install -e .")
        print("\nAlternatively, use SAM2 with --model sam2.1_hiera_tiny")
        sys.exit(1)

    if not use_sam3 and not SAM2_AVAILABLE:
        print("Error: SAM2 is not installed.")
        print(
            "Install with: pip install git+https://github.com/facebookresearch/sam2.git"
        )
        sys.exit(1)

    # Create config
    if args.test:
        if use_sam3:
            config = SegmentationConfig.for_testing()
        else:
            config = SegmentationConfig.for_sam2_testing()
        config.device = args.device
        print(f"Running in TEST mode with {args.model}")
    else:
        config = SegmentationConfig(
            sam_model_name=args.model,
            device=args.device,
        )

    # Override prompts if provided
    if args.prompts and use_sam3:
        config.text_prompts = args.prompts

    # Determine which suites to process
    if args.suite:
        task_suites = [args.suite]
    elif args.test:
        task_suites = ["libero_spatial"]  # Smallest suite for testing
    else:
        task_suites = config.task_suites

    print(f"\n{'=' * 60}")
    print(
        f"LIBERO Segmentation with {'SAM3 Text Prompts' if use_sam3 else 'SAM2 Automatic'}"
    )
    print(f"{'=' * 60}")
    print(f"  Model: {config.sam_model_name}")
    print(f"  Device: {config.device}")
    print(f"  Task suites: {task_suites}")
    print(f"  Tracking: {config.enable_tracking}")
    if use_sam3:
        print(f"  Text prompts: {config.text_prompts}")
    print()

    # Initialize components
    loader = LiberoDataLoader(
        datasets_dir=config.libero_datasets_dir,
        views=config.views,
    )

    if use_sam3:
        segmenter = SAMSegmenter(
            model_name="sam3",
            device=config.device,
        )
        process_demo = process_demo_sam3
    else:
        segmenter = SAMSegmenter(
            model_name=config.sam_model_name,
            checkpoint_dir=config.sam_checkpoint_dir,
            device=config.device,
            points_per_side=config.points_per_side,
            pred_iou_thresh=config.pred_iou_thresh,
            stability_score_thresh=config.stability_score_thresh,
            min_mask_region_area=config.min_mask_region_area,
        )
        process_demo = process_demo_sam2

    storage = MaskStorage(output_dir=config.output_dir)

    # Discover demos
    print("Discovering demos...")
    demos = loader.discover_demos(task_suites)
    print(f"Found {len(demos)} demos\n")

    if not demos:
        print("No demos found. Make sure LIBERO datasets are downloaded.")
        print("Run: python LIBERO/benchmark_scripts/download_libero_datasets.py")
        sys.exit(1)

    # Limit if requested
    if args.test and args.limit is None:
        args.limit = 1

    if args.limit:
        demos = demos[: args.limit]
        print(f"Processing {len(demos)} demos (limited)\n")

    # Process demos
    for demo in demos:
        print(f"\nDemo: {demo.task_suite}/{demo.task_name}/demo_{demo.demo_id}")
        print(f"  Frames: {demo.num_frames}, Views: {demo.available_views}")

        for view in demo.available_views:
            if view not in config.views:
                continue

            try:
                process_demo(
                    demo=demo,
                    view=view,
                    loader=loader,
                    segmenter=segmenter,
                    storage=storage,
                    config=config,
                    force=args.force,
                )
            except Exception as e:
                print(f"  Error processing {view}: {e}")
                import traceback

                traceback.print_exc()
                continue

    print("\n" + "=" * 60)
    print("Segmentation complete!")
    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
