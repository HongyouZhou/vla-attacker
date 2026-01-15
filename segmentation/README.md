# Segmentation Module for VLA-Attacker

This module uses SAM2 (Segment Anything Model 2) to segment objects in LIBERO dataset images, preparing them for adversarial attacks.

## Setup

### 1. Install SAM2

```bash
# Install SAM2 from Facebook Research
pip install git+https://github.com/facebookresearch/sam2.git

# Or if you have issues, clone and install manually:
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

### 2. Download SAM2 Checkpoints

```bash
# Create checkpoint directory
mkdir -p checkpoints/sam2

# Download the tiny model (recommended for AMD integrated GPU)
cd checkpoints/sam2
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

# For better GPU, you can also download:
# wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
# wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
# wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### 3. Download LIBERO Dataset

```bash
cd LIBERO
python benchmark_scripts/download_libero_datasets.py
```

## Usage

### Quick Test (AMD Integrated GPU)

```bash
# Test with a single demo using lightweight settings
python scripts/segment_dataset.py --test --device cuda
```

### Process Specific Task Suite

```bash
python scripts/segment_dataset.py --suite libero_spatial
```

### Process All Data

```bash
python scripts/segment_dataset.py --all
```

### Visualize Results

```bash
# Generate video for a specific demo
python scripts/visualize_masks.py --suite libero_spatial --demo 0 --video

# Automatically visualize a processed demo
python scripts/visualize_masks.py --auto --video
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
      }
    }
  }
}
```

## API Usage

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
# Use this for your attack!
```

## Configuration Options

See `segmentation/config.py` for all options:

- `sam_model_name`: Model variant (tiny/small/base_plus/large)
- `points_per_side`: Grid density for automatic segmentation
- `enable_tracking`: Use video tracking for consistent IDs
- And more...
