---
library_name: transformers
tags:
- robotics
- vla
- image-text-to-text
- multimodal
- pretraining
license: mit
language:
- en
pipeline_tag: image-text-to-text
---

# OpenVLA 7B Fine-Tuned on LIBERO-Spatial

This model was produced by fine-tuning the [OpenVLA 7B model](https://huggingface.co/openvla/openvla-7b) via
LoRA (r=32) on the LIBERO-Spatial dataset from the [LIBERO simulation benchmark](https://libero-project.github.io/main.html).
We made a few modifications to the training dataset to improve final performance (see the
[OpenVLA paper](https://arxiv.org/abs/2406.09246) for details).

Below are the hyperparameters we used for all LIBERO experiments:

- Hardware: 8 x A100 GPUs with 80GB memory
- Fine-tuned with LoRA: `use_lora == True`, `lora_rank == 32`, `lora_dropout == 0.0`
- Learning rate: 5e-4
- Batch size: 128 (8 GPUs x 16 samples each)
- Number of training gradient steps: 50K
- No quantization at train or test time
- No gradient accumulation (i.e. `grad_accumulation_steps == 1`)
- `shuffle_buffer_size == 100_000`
- Image augmentations: Random crop, color jitter (see training code for details)

## Usage Instructions

See the [OpenVLA GitHub README](https://github.com/openvla/openvla/blob/main/README.md) for instructions on how to
run and evaluate this model in the LIBERO simulator.

## Citation

**BibTeX:**

```bibtex
@article{kim24openvla,
    title={OpenVLA: An Open-Source Vision-Language-Action Model},
    author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
    journal = {arXiv preprint arXiv:2406.09246},
    year={2024}
} 
```
