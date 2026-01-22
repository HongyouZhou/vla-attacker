# Segmentation Module for VLA-Attacker

This module uses **SAM3** (Segment Anything Model 3) to segment objects in LIBERO dataset images using **text prompts**, preparing them for adversarial attacks.

## Key Features

- **SAM3 Text Prompt Segmentation**: Segment objects by describing them (e.g., "robot gripper", "wooden block")
- **Video Tracking**: Track segmented objects across video frames
- **Backwards Compatible**: Also supports SAM2 automatic segmentation

## Setup

### 1. Install SAM3 (Recommended)

**Prerequisites:**

- Python 3.12+
- PyTorch 2.7+
- CUDA 12.6+

```bash
# Create a new conda environment
conda create -n sam3 python=3.12
conda activate sam3

# Install PyTorch with CUDA
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Clone and install SAM3
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# Authenticate with Hugging Face (required for model download)
# Request access at: https://huggingface.co/facebook/sam3
pip install huggingface_hub
huggingface-cli login
```

### 2. (Alternative) Install SAM2

If you can't use SAM3 (e.g., older Python/PyTorch version), you can use SAM2:

```bash
# Install SAM2
pip install git+https://github.com/facebookresearch/sam2.git

# Download checkpoints
mkdir -p checkpoints/sam2
cd checkpoints/sam2
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

### 3. Download LIBERO Dataset

```bash
cd LIBERO
python benchmark_scripts/download_libero_datasets.py
```

## Usage

### SAM3 Text Prompt Segmentation (Recommended)

```bash
# Test with specific text prompts
python scripts/segment_dataset.py --test --prompts "robot gripper" "wooden block"

# Process a task suite with custom prompts
python scripts/segment_dataset.py --suite libero_spatial \
    --prompts "robot arm" "target object" "bowl"

# Process all data with default prompts
python scripts/segment_dataset.py --all
```

### SAM2 Automatic Segmentation

```bash
# Use SAM2 automatic mask generation
python scripts/segment_dataset.py --test --model sam2.1_hiera_tiny

# Process with SAM2
python scripts/segment_dataset.py --suite libero_spatial --model sam2.1_hiera_tiny
```

### Python API

```python
from segmentation import SegmentationConfig, LiberoDataLoader, SAMSegmenter, MaskStorage

# Initialize with SAM3
segmenter = SAMSegmenter(model_name="sam3", device="cuda")

# Load an image
import numpy as np
from PIL import Image
image = np.array(Image.open("path/to/image.jpg"))

# Segment with text prompts
result = segmenter.segment_frame_with_text(
    image=image,
    text_prompts=["robot gripper", "wooden block", "bowl"],
    frame_idx=0,
)

# Access results
print(f"Found {len(result.masks)} objects")
for i, (mask, prompt) in enumerate(zip(result.masks, result.prompts)):
    print(f"  Object {i}: '{prompt}' - area: {mask.sum()} pixels")
```

### Video Segmentation with Tracking

```python
# Load video frames (T, H, W, 3)
frames = loader.load_all_frames(demo_info, view="agentview_rgb")

# Segment and track through video
results = segmenter.segment_video_with_text(
    frames=frames,
    text_prompts=["robot gripper", "target object"],
    init_frame_idx=0,
)

# Results is a dict: {frame_idx: SegmentationResult}
for frame_idx, result in results.items():
    print(f"Frame {frame_idx}: {len(result.masks)} objects tracked")
```

## Output Format

Segmentation results are saved to `LIBERO/segmentation_data/`:

```
LIBERO/segmentation_data/
├── libero_spatial/
│   └── task_name/
│       └── demo_0/
│           ├── agentview_rgb_masks.hdf5
│           ├── eye_in_hand_rgb_masks.hdf5
│           └── metadata.json
└── ...
```

### HDF5 Structure

Each `*_masks.hdf5` file contains:

- `frame_XXXX/masks`: Boolean masks, shape (N, H, W)
- `frame_XXXX/object_ids`: Unique object IDs that persist across frames
- `frame_XXXX/bboxes`: Bounding boxes [x1, y1, x2, y2]
- `frame_XXXX/scores`: Confidence scores
- `frame_XXXX/areas`: Mask areas in pixels

### Metadata JSON

```json
{
  "task_suite": "libero_spatial",
  "task_name": "...",
  "demo_id": 0,
  "num_frames": 300,
  "image_size": [128, 128],
  "views": {
    "agentview_rgb": {
      "num_objects": 5,
      "objects": {
        "0": {"first_frame": 0, "last_frame": 299, "avg_area": 1234},
        ...
      },
      "segmentation_config": {
        "sam_model_name": "sam3",
        "text_prompts": ["robot gripper", "wooden block"]
      }
    }
  }
}
```

## Configuration Options

See `segmentation/config.py` for all options:

### SAM3 Options

- `text_prompts`: List of text descriptions for objects to segment
- `task_prompts`: Task-specific prompt overrides

### SAM2 Options (if not using SAM3)

- `sam_model_name`: Model variant (tiny/small/base_plus/large)
- `points_per_side`: Grid density for automatic segmentation
- `enable_tracking`: Use video tracking for consistent IDs

## Text Prompt Tips for LIBERO

Good text prompts for LIBERO scenes:

- **Robot parts**: "robot gripper", "robot arm", "robot end effector"
- **Objects**: "wooden block", "red cube", "blue cube", "bowl", "mug", "plate"
- **Furniture**: "drawer", "cabinet door", "shelf"
- **Targets**: "target location", "goal area"

Be specific! "red wooden block" works better than just "block".

## API for Attacks

```python
from segmentation import SegmentationConfig, LiberoDataLoader, MaskStorage
from segmentation.libero_loader import DemoInfo

# Load segmentation results
config = SegmentationConfig()
storage = MaskStorage(config.output_dir)
loader = LiberoDataLoader(config.libero_datasets_dir)

# Find a demo
demos = loader.discover_demos(["libero_spatial"])
demo = demos[0]

# Get mask for a specific object at a specific frame
mask = storage.get_object_mask_at_frame(
    demo_info=demo,
    view="agentview_rgb",
    frame_idx=100,
    object_id=0,
)

# mask is a boolean array of shape (H, W)
# Use this for your adversarial attack!
```
