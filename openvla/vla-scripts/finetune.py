"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import random
import draccus
import torch
import torch.distributed as dist
import torch.utils.checkpoint
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

#import wandb
import swanlab
import sys
sys.path.append("/data/yihong.ji/RobustVLA-283D/openvla")
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.util.robust_augmentation.ucb_augmentation_balancer import UCBAugmentationBalancer

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on

from dataclasses import dataclass, field

@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("/data/yihong.ji/RobustVLA-283D/LIBERO/libero/datasets")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    dataset_names: list[str] = field(
        default_factory=lambda: ["libero_spatial_no_noops"]
    )  # Support for multiple datasets
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 32                                            # Fine-tuning batch size
    max_steps: int = 30_000                                         # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = False                       # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    adv_training_image: bool = True                                 # Whether to use image-space adversarial training
    use_data_augmentation: bool = True                            # Whether to use data augmentation during training
    adv_training_action_pretokenization: bool = True              # Whether to use action-space adversarial training

    # Loss Weights
    clean_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    img_loss_weight: float = 1.0

    # Action Space Adversarial Parameters (L-inf Norm)
    adv_epsilon_action: float = 0.03
    pgd_steps_action: int = 3
    pgd_alpha_action: float = 0.01

    # Image Space Adversarial Parameters (L-inf Norm)
    adv_epsilon_image: float = 8 / 255.0
    pgd_steps_image: int = 3
    pgd_alpha_image: float = 2 / 255.0

    # Data Augmentation Parameters
    augmentation_prob: float = 0.4

    # UCB Augmentation Balancer Parameters
    enable_ucb_balancing: bool = False                               # Whether to enable UCB augmentation balancing
    ucb_exploration_coeff: float = 1.0                              # UCB exploration coefficient
    ucb_window_size: int = 100                                      # Sliding window size
    ucb_ema_decay: float = 0.9                                     # EMA decay factor
    ucb_modality_floor_prob: float = 0.01                          # Minimum probability for each modality
    ucb_epsilon: float = 1e-6                                      # Numerical stability constant
    language_augmentation_dir: str = "./prompts" # Language augmentation directory

    # ==========================================================
    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on

def _apply_data_augmentation(pixel_values: torch.Tensor, cfg: FinetuneConfig) -> torch.Tensor:
    if not cfg.use_data_augmentation or torch.rand(1).item() >= cfg.augmentation_prob:
        return pixel_values

    images = pixel_values
    # Simple augmentations (brightness, noise)
    if torch.rand(1).item() < 0.5:
        brightness_factor = torch.rand(1).item() * 0.4 + 0.8  # [0.8, 1.2]
        images = torch.clamp(images * brightness_factor, -1, 1)
    if torch.rand(1).item() < 0.3:
        noise = torch.randn_like(images) * 0.01
        images = torch.clamp(images + noise, -1, 1)
    return images

# Adversarial Training Functions
def _compute_image_adversarial_loss(
    vla: DDP, cfg: FinetuneConfig, batch: Dict[str, torch.Tensor], device_id: int,
):
    """Image adversarial training"""
    input_ids = batch["input_ids"].to(device_id)
    attention_mask = batch["attention_mask"].to(device_id)
    pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device_id)
    labels = batch["labels"].to(device_id)

    #print(pixel_values)
    #breakpoint()

    # Initialize perturbation
    delta = torch.empty_like(pixel_values, device=device_id).uniform_(
        -cfg.adv_epsilon_image, cfg.adv_epsilon_image
    )
    delta = torch.clamp(pixel_values + delta, -1, 1) - pixel_values

    for step in range(cfg.pgd_steps_image):
        delta.requires_grad_(True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_dict = vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values + delta,
                labels=labels,
                use_cache=False,
            )
            loss = output_dict.loss

        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()

        with torch.no_grad():
            # Gradient ascent step
            delta = delta.detach() + cfg.pgd_alpha_image * delta.grad.sign()
            # L∞ projection
            delta = torch.clamp(delta, -cfg.adv_epsilon_image, cfg.adv_epsilon_image)
            delta = torch.clamp(pixel_values + delta, -1, 1) - pixel_values

    # Calculate final adversarial loss
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        adv_output_dict = vla(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values + delta,
            labels=labels,
            use_cache=False,
        )
        adv_loss_image = adv_output_dict.loss
        perturb_norm = delta.abs().mean().item()

    metrics = {
        "loss_adv_image": adv_loss_image.item(),
        "image_perturb_norm": perturb_norm
    }
    return adv_loss_image, metrics

def _compute_action_adversarial_loss_pretokenization(
    vla: DDP, cfg: FinetuneConfig, batch: Dict[str, torch.Tensor],
    batch_transform: RLDSBatchTransform, device_id: int,
):
    """
    Perform PGD attack on actions before tokenization to ensure only action-related tokens are changed.
    """

    required_fields = ["original_action", "prompt_template", "image"]
    missing_fields = [field for field in required_fields if field not in batch]

    if missing_fields:
        print(f"Missing required fields for adversarial training: {missing_fields}")
        return torch.tensor(0.0, device=device_id), {
            "loss_adv_action": 0.0,
            "eta_norm": 0.0
        }

    # Extract batch information
    original_actions = batch["original_action"].to(device_id)  # [batch_size, action_dim]
    prompt_templates = batch["prompt_template"]
    images = batch["image"]
    batch_size = original_actions.shape[0]

    # Initialize eta perturbation - random sign initialization
    eta = torch.randint(0, 2, original_actions.shape, device=device_id, dtype=torch.float32) * 2 - 1  # {-1, 1}
    eta = eta * cfg.adv_epsilon_action
    eta.requires_grad_(True)

    # Get tokenizer's pad_token_id
    try:
        pad_token_id = batch_transform.base_tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = batch_transform.base_tokenizer.eos_token_id
    except:
        pad_token_id = 0

    # PGD loop
    for step in range(cfg.pgd_steps_action):
        eta.requires_grad_(True)

        # Re-tokenize each sample
        perturbed_batch = {
            "input_ids": [],
            "labels": [],
            "pixel_values": []
        }

        for i in range(batch_size):
            # Add perturbation to the original action
            perturbed_action = original_actions[i] + eta[i]

            # Re-tokenize
            try:
                tokenized = batch_transform.tokenize_action_sequence(
                    perturbed_action,
                    prompt_templates[i] if isinstance(prompt_templates, list) else prompt_templates,
                    images[i] if isinstance(images, list) else images
                )

                perturbed_batch["input_ids"].append(tokenized["input_ids"].to(device_id))
                perturbed_batch["labels"].append(tokenized["labels"].to(device_id))
                perturbed_batch["pixel_values"].append(tokenized["pixel_values"].to(device_id))

            except Exception as e:
                print(f"Error in tokenization for sample {i}: {e}")
                return torch.tensor(0.0, device=device_id), {
                    "loss_adv_action": 0.0,
                    "eta_norm": 0.0
                }

        # Check if all samples were successfully tokenized
        if len(perturbed_batch["input_ids"]) != batch_size:
            return torch.tensor(0.0, device=device_id), {
                "loss_adv_action": 0.0,
                "eta_norm": 0.0
            }

        # Get maximum length and pad the sequence
        max_len = max(len(seq) for seq in perturbed_batch["input_ids"])

        # Pad input_ids and labels
        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for i in range(batch_size):
            input_ids = perturbed_batch["input_ids"][i]
            labels = perturbed_batch["labels"][i]

            # Calculate the required padding length
            pad_len = max_len - len(input_ids)

            if pad_len > 0:
                # Right padding - ensure all tensors are on the same device
                padded_input_ids.append(torch.cat([
                    input_ids,
                    torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype, device=device_id)
                ]))

                padded_labels.append(torch.cat([
                    labels,
                    torch.full((pad_len,), -100, dtype=labels.dtype, device=device_id)  # IGNORE_INDEX
                ]))
            else:
                # No padding needed
                padded_input_ids.append(input_ids)
                padded_labels.append(labels)

            # Create attention mask
            attention_mask = torch.ones(max_len, dtype=torch.long, device=device_id)
            if pad_len > 0:
                attention_mask[len(input_ids):] = 0
            attention_masks.append(attention_mask)

        # Convert to tensor and ensure it's on the correct device
        perturbed_batch_tensors = {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(attention_masks),
            "pixel_values": torch.stack(perturbed_batch["pixel_values"])
        }

        # Forward pass to compute loss
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = vla(
                    input_ids=perturbed_batch_tensors["input_ids"],
                    attention_mask=perturbed_batch_tensors["attention_mask"],
                    pixel_values=perturbed_batch_tensors["pixel_values"],
                    labels=perturbed_batch_tensors["labels"],
                    use_cache=False,
                )
                loss = outputs.loss
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return torch.tensor(0.0, device=device_id), {
                "loss_adv_action": 0.0,
                "eta_norm": 0.0
            }

        # Backward pass
        if eta.grad is not None:
            eta.grad.zero_()
        loss.backward()

        # Update eta
        with torch.no_grad():
            if eta.grad is not None:
                # Sign gradient update
                eta_grad = eta.grad.sign()
                eta = eta + cfg.pgd_alpha_action * eta_grad

                # L∞ projection
                eta = torch.clamp(eta, -cfg.adv_epsilon_action, cfg.adv_epsilon_action)
                eta.requires_grad_(True)

    # Calculate final adversarial loss
    with torch.no_grad():
        final_perturbed_batch = {
            "input_ids": [],
            "labels": [],
            "pixel_values": []
        }

        for i in range(batch_size):
            # Add final perturbation to the original action
            final_perturbed_action = original_actions[i] + eta[i].detach()

            # Re-tokenize
            try:
                tokenized = batch_transform.tokenize_action_sequence(
                    final_perturbed_action,
                    prompt_templates[i] if isinstance(prompt_templates, list) else prompt_templates,
                    images[i] if isinstance(images, list) else images
                )

                # Move to the correct device
                final_perturbed_batch["input_ids"].append(tokenized["input_ids"].to(device_id))
                final_perturbed_batch["labels"].append(tokenized["labels"].to(device_id))
                final_perturbed_batch["pixel_values"].append(tokenized["pixel_values"].to(device_id))

            except Exception as e:
                print(f"Error in final tokenization: {e}")
                return torch.tensor(0.0, device=device_id), {
                    "loss_adv_action": 0.0,
                    "eta_norm": eta.detach().abs().mean().item()
                }

        # Re-pad
        max_len = max(len(seq) for seq in final_perturbed_batch["input_ids"])

        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for i in range(batch_size):
            input_ids = final_perturbed_batch["input_ids"][i]
            labels = final_perturbed_batch["labels"][i]

            pad_len = max_len - len(input_ids)

            if pad_len > 0:
                padded_input_ids.append(torch.cat([
                    input_ids,
                    torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype, device=device_id)
                ]))

                padded_labels.append(torch.cat([
                    labels,
                    torch.full((pad_len,), -100, dtype=labels.dtype, device=device_id)
                ]))
            else:
                padded_input_ids.append(input_ids)
                padded_labels.append(labels)

            attention_mask = torch.ones(max_len, dtype=torch.long, device=device_id)
            if pad_len > 0:
                attention_mask[len(input_ids):] = 0
            attention_masks.append(attention_mask)

        final_batch_tensors = {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(attention_masks),
            "pixel_values": torch.stack(final_perturbed_batch["pixel_values"])
        }

        # Calculate final loss
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                final_outputs = vla(
                    input_ids=final_batch_tensors["input_ids"],
                    attention_mask=final_batch_tensors["attention_mask"],
                    pixel_values=final_batch_tensors["pixel_values"],
                    labels=final_batch_tensors["labels"],
                    use_cache=False,
                )
                final_loss = final_outputs.loss
        except Exception as e:
            print(f"Error in final forward pass: {e}")
            return torch.tensor(0.0, device=device_id), {
                "loss_adv_action": 0.0,
                "eta_norm": eta.detach().abs().mean().item()
            }

    metrics = {
        "loss_adv_action": final_loss.item(),
        "eta_norm": eta.detach().abs().mean().item()
    }

    return final_loss, metrics

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_names}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"
    if cfg.enable_ucb_balancing:
        exp_id += "--ucb-balancing"
    if cfg.adv_training_image:
        exp_id += "--adv-image"
    if cfg.adv_training_action_pretokenization:
        exp_id += "--adv-preTokenAction"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir , cfg.adapter_tmp_dir
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=False)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    base_model = vla.get_base_model() if cfg.use_lora else vla
    base_model.action_token_begin_idx = action_tokenizer.action_token_begin_idx

    vla = DDP(vla, device_ids=[device_id] if not cfg.use_quantization else None, find_unused_parameters=True, gradient_as_bucket_view=True)


    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        adversarial_training=cfg.adv_training_action_pretokenization
    )
    '''
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder
    )
    '''

    # Load multiple datasets
    vla_datasets = [
        RLDSDataset(
            cfg.data_root_dir,
            dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            image_aug=cfg.image_aug,
        )
        for dataset_name in cfg.dataset_names
    ]

    class MixedIterableDataset(IterableDataset):
        def __init__(self, datasets, sampling_weights=None):
            self.datasets = datasets
            self.sampling_weights = sampling_weights or [1.0] * len(datasets)
            self.iterators = None

        def __iter__(self):
            self.iterators = [iter(ds) for ds in self.datasets]
            while True:
                # Randomly select a dataset based on weights
                idx = random.choices(range(len(self.datasets)), weights=self.sampling_weights, k=1)[0]
                try:
                    yield next(self.iterators[idx])
                except StopIteration:
                    # If a dataset is exhausted, reset its iterator
                    self.iterators[idx] = iter(self.datasets[idx])
                    yield next(self.iterators[idx])
    # Dynamically set sampling weights based on the number of datasets
    if len(vla_datasets) == 1:
        sampling_weights = [1.0]
    elif len(vla_datasets) == 3:
        sampling_weights = [0.4, 0.3, 0.3]
    else:
        # For other numbers of datasets, use uniform weights
        sampling_weights = [1.0 / len(vla_datasets)] * len(vla_datasets)

    vla_dataset = MixedIterableDataset(vla_datasets, sampling_weights=sampling_weights)

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_datasets[0].dataset_statistics, run_dir)

    # Create Collator and DataLoader

    class AdversarialCollatorWrapper:
        def __init__(self, base_collator):
            self.base_collator = base_collator

        def __call__(self, batch):
            # Separate standard fields and extra fields
            standard_batch = []
            extra_fields = {}

            for item in batch:
                standard_item = {}
                for key, value in item.items():
                    if key in ["pixel_values", "input_ids", "labels", "attention_mask"]:
                        standard_item[key] = value
                    else:
                        if key not in extra_fields:
                            extra_fields[key] = []
                        extra_fields[key].append(value)
                standard_batch.append(standard_item)

            # Process standard fields with the base collator
            result = self.base_collator(standard_batch)

            # Add extra fields
            for key, values in extra_fields.items():
                if key == "original_action":
                    result[key] = torch.stack(values)
                else:
                    result[key] = values

            return result

    # Use the wrapped collator
    base_collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    collator = AdversarialCollatorWrapper(base_collator)

    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        swanlab.init(
            project="robust_openvla",
            workspace="robust_vla",
            experiment_name=f"ft+{exp_id}",
            config=vars(cfg)
        )
        swanlab.sync_wandb()
        # wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    ucb_balancer = None
    if cfg.enable_ucb_balancing:
        ucb_balancer = UCBAugmentationBalancer(
            window_size=cfg.ucb_window_size,
            exploration_coeff=cfg.ucb_exploration_coeff,
            ema_decay=cfg.ucb_ema_decay,
            modality_floor_prob=cfg.ucb_modality_floor_prob,
            epsilon=cfg.ucb_epsilon,
            prompt_augmentation_dir=cfg.language_augmentation_dir,
            device=f'cuda:{device_id}',
            batch_transform=batch_transform
        )
        print(f"[INFO] Initialized UCB Augmentation Balancer with {ucb_balancer.num_arms} arms")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        grad_norm = 0.0
        param_norm = 0.0
        for batch_idx, batch in enumerate(dataloader):
            ucb_metrics = {}
            if cfg.enable_ucb_balancing and ucb_balancer is not None:
                # 1. Select augmentation strategy
                arm_idx, augmentation_name = ucb_balancer.select_augmentation()

                # 2. Apply augmentation to the batch
                batch = ucb_balancer.apply_augmentation(batch, (arm_idx, augmentation_name))

                # 3. Calculate baseline loss
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    baseline_output = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                    )
                    baseline_loss = baseline_output.loss.item()

                ucb_metrics.update({
                    "ucb_selected_arm": arm_idx,
                    "ucb_baseline_loss": baseline_loss
                })
            else:
                batch["pixel_values"] = _apply_data_augmentation(batch["pixel_values"], cfg)
            all_adv_metrics = {}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                clean_output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                clean_loss = clean_output.loss
                total_loss = clean_loss * cfg.clean_loss_weight

                # Update UCB Balancer with clean loss if using UCB
                if cfg.enable_ucb_balancing and ucb_balancer is not None:
                    ucb_balancer.update_statistics(arm_idx, clean_loss.item())
                    ucb_stats = ucb_balancer.get_statistics()
                    ucb_metrics.update(ucb_stats)

                if cfg.adv_training_action_pretokenization:
                    adv_loss_pretok, metrics_pretok = _compute_action_adversarial_loss_pretokenization(
                        vla, cfg, batch, batch_transform, device_id
                    )
                    total_loss += adv_loss_pretok * cfg.action_loss_weight
                    all_adv_metrics.update(metrics_pretok)

                if cfg.adv_training_image:
                    adv_loss_image, metrics = _compute_image_adversarial_loss(vla, cfg, batch, device_id)
                    total_loss += adv_loss_image * cfg.img_loss_weight
                    all_adv_metrics.update(metrics)

            # Normalize loss to account for gradient accumulation
            normalized_loss = total_loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            action_logits = clean_output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(total_loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
                optimizer.step()
                param_norm = 0.0
                for p in trainable_params:
                    if p.requires_grad:
                        param_norm += p.data.norm().item() ** 2
                param_norm = param_norm ** 0.5
                optimizer.zero_grad()
                progress.update()

            # Push Metrics to W&B (every 100 gradient steps)
            if gradient_step_idx > 0 and distributed_state.is_main_process and gradient_step_idx % 100 == 0:
                metrics_dict = {
                    "train_loss": smoothened_loss,
                    "action_accuracy": smoothened_action_accuracy,
                    "l1_loss": smoothened_l1_loss,
                    "loss_clean": clean_loss.item(),
                    "grad_norm": grad_norm,
                    "param_norm": param_norm,
                }
                metrics_dict.update(all_adv_metrics)
                if cfg.enable_ucb_balancing:
                    metrics_dict.update(ucb_metrics)
                swanlab.log(metrics_dict, step=gradient_step_idx)

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
                if cfg.use_lora:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=False
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            # Overwrite latest checkpoint
                            merged_vla.save_pretrained(run_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            # Prepare to save checkpoint in new directory
                            checkpoint_dir = Path(str(run_dir) + f"/{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            # Save dataset statistics to new directory
                            save_dataset_statistics(vla_datasets[0].dataset_statistics, checkpoint_dir)

                            # Save processor and model weights to new directory
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                # Block on Main Process Checkpointing
                dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
