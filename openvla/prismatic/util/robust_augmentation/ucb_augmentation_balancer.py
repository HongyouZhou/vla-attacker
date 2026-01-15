import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn.functional as F
import numpy as np
from .img_util import apply_visual_augmentation

logger = logging.getLogger("openvla")


class UCBAugmentationBalancer:
    """RobustVLA-SW-UCB: Scale-Invariant, Improvement-Aware, and Non-Stationary Bandits for Automatic Perturbation Balancing

    This class implements the UCB algorithm described in the paper for balancing visual and linguistic augmentations
    during VLA training. It maintains statistics for each augmentation type and selects augmentations based on
    their learnable improvement potential.

    PyTorch version migrated from JAX implementation.
    """

    def __init__(
        self,
        window_size: int = 100,
        exploration_coeff: float = 1.0,
        ema_decay: float = 0.9,
        modality_floor_prob: float = 0.01,
        epsilon: float = 1e-6,
        prompt_augmentation_dir: str = "./prompts",
        device: str = 'cuda',
        batch_transform = None  # For re-tokenizing the batch after language augmentation
    ):
        """
        Args:
            window_size: Sliding window size for non-stationary handling
            exploration_coeff: Exploration coefficient alpha in UCB formula
            ema_decay: Exponential moving average decay rate beta
            modality_floor_prob: Minimum probability for each modality
            epsilon: Numerical stability constant
            prompt_augmentation_dir: Directory containing language augmentation JSON files
            device: PyTorch device ('cpu' or 'cuda')
        """
        self.window_size = window_size
        self.exploration_coeff = exploration_coeff
        self.ema_decay = ema_decay
        self.modality_floor_prob = modality_floor_prob
        self.epsilon = epsilon
        self.device = device
        self.batch_transform = batch_transform  # For re-tokenizing

        # Define augmentation types
        self.visual_augmentations = [
            'light_intensity_adjustment',
            'simulate_light_direction',
            'add_shadow',
            'simulate_blur',
            'add_scratch_and_stain',
            'image_shift',
            'image_rotation'
        ]

        # Language augmentation types
        self.language_augmentations = ['adv', 'sentence', 'word']

        # Total arms: 7 visual + 3 language + 1 no_augmentation = 11 arms
        self.all_augmentations = self.visual_augmentations + self.language_augmentations + ['no_augmentation']
        self.num_arms = len(self.all_augmentations)

        # Load language augmentation data
        self.language_aug_data = self._load_language_augmentations(prompt_augmentation_dir)

        # Initialize UCB statistics
        self._reset_statistics()

        # Step counter
        self.step_count = 0

        logger.info(f"Initialized UCB Augmentation Balancer with {self.num_arms} arms: {self.all_augmentations}")

    def _load_language_augmentations(self, prompt_dir: str) -> Dict[str, Dict]:
        """Load language augmentation data from JSON files"""
        prompt_path = Path(prompt_dir)
        aug_data = {}

        for aug_type in self.language_augmentations:
            json_file = prompt_path / f"{aug_type}.json"
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        aug_data[aug_type] = json.load(f)
                    logger.info(f"Loaded {aug_type} augmentations from {json_file}")
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")
                    aug_data[aug_type] = {}
            else:
                logger.warning(f"Language augmentation file not found: {json_file}")
                aug_data[aug_type] = {}

        return aug_data

    def _reset_statistics(self):
        """Reset all UCB statistics"""
        # EMA statistics (converted to PyTorch tensors)
        self.mu = torch.zeros(self.num_arms, device=self.device)      # Mean loss for each arm
        self.sigma = torch.ones(self.num_arms, device=self.device)    # Standard deviation for each arm

        # Sliding window statistics
        self.window_counts = torch.zeros(self.num_arms, dtype=torch.int32, device=self.device)
        self.window_rewards = torch.zeros(self.num_arms, device=self.device)

        # Sliding window history buffer (list on CPU to avoid GPU memory waste)
        self.history_buffer = []

        self.step_count = 0
        logger.info("Reset UCB statistics")

    def _compute_standardized_improvement_reward(self, arm_idx: int, current_loss: float) -> float:
        """Compute standardized improvement reward as defined in Equation (3)

        r_n(i) = clip((μ_{n-1}(i) - L_n(i)) / (σ_{n-1}(i) + ε), 0, 1)
        """
        prev_mean = self.mu[arm_idx]
        prev_std = self.sigma[arm_idx]

        # Calculate improvement = previous mean loss - current loss
        improvement = prev_mean - current_loss
        standardized_improvement = improvement / (prev_std + self.epsilon)

        # Clip to [0, 1] range
        reward = torch.clamp(standardized_improvement, 0.0, 1.0)

        return reward.item()

    def _update_ema_statistics(self, arm_idx: int, current_loss: float):
        """Update exponential moving average statistics"""
        current_loss_tensor = torch.tensor(current_loss, device=self.device)

        # Update mean: new_mu = (1-β) * mu_prev + β * current_loss
        new_mu = (1 - self.ema_decay) * self.mu[arm_idx] + self.ema_decay * current_loss_tensor

        # Update variance: new_sigma = sqrt((1-β) * sigma_prev^2 + β * (current_loss - new_mu)^2)
        variance_term = (current_loss_tensor - new_mu) ** 2
        new_variance = (1 - self.ema_decay) * (self.sigma[arm_idx] ** 2) + self.ema_decay * variance_term
        new_sigma = torch.sqrt(torch.clamp(new_variance, min=self.epsilon))

        # PyTorch supports in-place updates, simpler than JAX's .at[].set()
        self.mu[arm_idx] = new_mu
        self.sigma[arm_idx] = new_sigma

    def _update_sliding_window(self, arm_idx: int, reward: float):
        """Update sliding window statistics"""
        # Add new record to the history buffer
        self.history_buffer.append((arm_idx, reward))

        # If the window size is exceeded, remove the oldest record
        if len(self.history_buffer) > self.window_size:
            old_arm, old_reward = self.history_buffer.pop(0)
            # Subtract the statistics of the old record
            self.window_counts[old_arm] -= 1
            self.window_rewards[old_arm] -= old_reward

        # Add the statistics of the new record
        self.window_counts[arm_idx] += 1
        self.window_rewards[arm_idx] += reward

    def _compute_ucb_indices(self) -> torch.Tensor:
        """Compute UCB indices as defined in Equation (4)

        UCB_n(i) = Q̂_n^(w)(i) + α * sqrt(log(Σ_j N_n^(w)(j)) / max(1, N_n^(w)(i)))
        """
        # Calculate empirical mean (exploitation term)
        # Avoid division by zero: only calculate mean for arms that have been selected
        valid_mask = self.window_counts > 0
        empirical_means = torch.zeros_like(self.window_rewards)
        empirical_means[valid_mask] = self.window_rewards[valid_mask] / self.window_counts[valid_mask].float()

        # Clean up NaNs in UCB calculation for means
        nan_mask_means = torch.isnan(empirical_means) | torch.isinf(empirical_means)
        if nan_mask_means.any():
            logger.warning(f"Step {self.step_count}: Detected NaN/Inf in empirical_means: {empirical_means}")
            empirical_means = torch.where(nan_mask_means, torch.zeros_like(empirical_means), empirical_means)

        # Calculate exploration bonus (exploration term)
        total_counts = torch.sum(self.window_counts).float()
        log_total = torch.log(torch.clamp(total_counts, min=1.0 + self.epsilon))  # Avoid log(0), add epsilon

        # exploration_bonus = alpha * sqrt(log(total_counts) / max(1, window_counts[i]))
        exploration_bonus = self.exploration_coeff * torch.sqrt(
            log_total / torch.clamp(self.window_counts.float(), min=1.0)
        )

        # NaN detection for exploration bonus
        nan_mask_bonus = torch.isnan(exploration_bonus) | torch.isinf(exploration_bonus)
        if nan_mask_bonus.any():
            logger.warning(f"Step {self.step_count}: Detected NaN/Inf in exploration_bonus: {exploration_bonus}")
            # Replace NaN exploration bonus with a large value to ensure these arms still have a chance to be selected
            exploration_bonus = torch.where(nan_mask_bonus,
                                          torch.full_like(exploration_bonus, self.exploration_coeff * 10.0),
                                          exploration_bonus)

        # UCB index = empirical mean + exploration bonus
        ucb_indices = empirical_means + exploration_bonus

        # Assign infinite score to untried arms to ensure they are selected
        ucb_indices[self.window_counts == 0] = float('inf')

        return ucb_indices

    def select_augmentation(self) -> Tuple[int, str]:
        """Select augmentation using UCB algorithm with modality floor

        Returns:
            Tuple of (arm_index, augmentation_name)
        """
        self.step_count += 1

        # Compute UCB indices
        ucb_indices = self._compute_ucb_indices()

        # NaN cleanup at the UCB level
        nan_mask = torch.isnan(ucb_indices) | torch.isinf(ucb_indices)
        if nan_mask.any():
            logger.warning(f"Step {self.step_count}: Detected NaN/Inf in UCB indices: {ucb_indices}")
            # Replace NaN/Inf with a large value to ensure these arms can still be selected
            ucb_indices = torch.where(nan_mask, torch.tensor(float('inf'), device=self.device), ucb_indices)
            logger.info(f"Step {self.step_count}: Cleaned UCB indices: {ucb_indices}")

        # Modality floor logic: with a small probability, force selection of a modality to ensure exploration
        if torch.rand(1).item() < (2 * self.modality_floor_prob):
            # Randomly choose visual (0) or language (1) modality
            modality_choice = int(torch.randint(0, 2, (1,)).item())
            if modality_choice == 0:  # Visual augmentation
                num_visual = len(self.visual_augmentations)
                arm_idx = int(torch.randint(0, num_visual, (1,)).item())
            else:  # Language augmentation
                lang_start_idx = len(self.visual_augmentations)
                num_lang = len(self.language_augmentations)
                arm_idx = lang_start_idx + int(torch.randint(0, num_lang, (1,)).item())
        else:
            # Standard UCB selection: choose the arm with the highest UCB index
            arm_idx = int(torch.argmax(ucb_indices).item())

        augmentation_name = self.all_augmentations[arm_idx]

        logger.debug(f"Step {self.step_count}: selected arm {arm_idx} ({augmentation_name})")

        return arm_idx, augmentation_name

    def apply_augmentation(self, batch: Dict[str, Any], augmentation_info: Tuple[int, str]) -> Dict[str, Any]:
        """Apply the selected augmentation to the training batch

        Args:
            batch: OpenVLA training batch, expected format:
                   {"pixel_values": Tensor, "input_ids": Tensor, "labels": Tensor, ...}
            augmentation_info: (arm_idx, augmentation_name)

        Returns:
            Augmented batch
        """
        arm_idx, augmentation_name = augmentation_info

        if augmentation_name == "no_augmentation":
            return batch

        elif augmentation_name in self.visual_augmentations:
            # Visual augmentation: modify pixel_values
            original_pixel_values = batch["pixel_values"]
            augmented_pixel_values = self._apply_visual_augmentation(original_pixel_values, augmentation_name)
            batch["pixel_values"] = augmented_pixel_values

        elif augmentation_name in self.language_augmentations:
            # Language augmentation:
            if "language_instruction" in batch and self.batch_transform is not None:
                lang_instruction = batch["language_instruction"]

                # Handle list format
                if isinstance(lang_instruction, list):
                    original_lang = lang_instruction
                else:
                    original_lang = [str(lang_instruction)]

                # Apply language augmentation
                augmented_lang_list = self._apply_language_augmentation(original_lang, augmentation_name)
                augmented_lang = augmented_lang_list[0] if augmented_lang_list else original_lang[0]

                # Re-tokenize
                self._retokenize_with_augmented_language(
                    batch, augmented_lang
                )

        return batch

    def _apply_visual_augmentation(self, pixel_values: torch.Tensor, augmentation_name: str) -> torch.Tensor:
        """
        Apply visual augmentation to image tensor using the migrated functions
        """
        try:
            original_shape = pixel_values.shape
            need_convert_back = False

            # Detect and convert format
            if pixel_values.dim() == 4:  # (batch, ?, ?, ?)
                if pixel_values.shape[1] == 3:  # (batch, 3, height, width)
                    pixel_values = pixel_values.permute(0, 2, 3, 1)  # -> (batch, height, width, 3)
                    need_convert_back = True
                    logger.debug(f"Converted from (B,C,H,W) to (B,H,W,C): {original_shape} -> {pixel_values.shape}")
            elif pixel_values.dim() == 3:  # (?, ?, ?)
                if pixel_values.shape[0] == 3:  # (3, height, width)
                    pixel_values = pixel_values.permute(1, 2, 0)  # -> (height, width, 3)
                    need_convert_back = True
                    logger.debug(f"Converted from (C,H,W) to (H,W,C): {original_shape} -> {pixel_values.shape}")

            # Ensure values are in the [0, 1] range
            if pixel_values.min() < -0.5:
                pixel_values = (pixel_values + 1.0) / 2.0
                need_rescale = True
            else:
                need_rescale = False

            logger.debug(f"Before augmentation: shape={pixel_values.shape}, min={pixel_values.min():.3f}, max={pixel_values.max():.3f}")

            # Apply visual augmentation
            augmented = apply_visual_augmentation(pixel_values, augmentation_name)

            # If rescaling was done, convert back
            if need_rescale:
                augmented = augmented * 2.0 - 1.0

            # Convert back to original format
            if need_convert_back:
                if len(original_shape) == 4 and original_shape[1] == 3:  # Convert back to (B,C,H,W)
                    augmented = augmented.permute(0, 3, 1, 2)
                elif len(original_shape) == 3 and original_shape[0] == 3:  # Convert back to (C,H,W)
                    augmented = augmented.permute(2, 0, 1)

            logger.debug(f"After augmentation: shape={augmented.shape}, min={augmented.min():.3f}, max={augmented.max():.3f}")

            return augmented

        except Exception as e:
            logger.warning(f"Visual augmentation {augmentation_name} failed: {e}")
            logger.debug(f"Input tensor shape: {pixel_values.shape}")
            return pixel_values

    def _apply_language_augmentation(self, text_list: List[str], augmentation_name: str) -> List[str]:
        """Apply language augmentation to text list

        Args:
            text_list: List of text strings to augment
            augmentation_name: Type of language augmentation ('adv', 'sentence', 'word')

        Returns:
            List of augmented text strings
        """
        if augmentation_name not in self.language_aug_data:
            logger.warning(f"No augmentation data found for {augmentation_name}")
            return text_list

        aug_data = self.language_aug_data[augmentation_name]
        if not aug_data:
            logger.warning(f"Empty augmentation data for {augmentation_name}")
            return text_list

        augmented_texts = []

        for i, text in enumerate(text_list):
            # Handle case where input might be a list
            if isinstance(text, list):
                # If text is a list, use the first element as the actual text
                actual_text = str(text[0]) if text else ""
                logger.debug(f"Input text is a list, using first element: {actual_text}")
            else:
                actual_text = str(text)

            if augmentation_name == 'adv':
                # Adversarial augmentation: use adversarial prompts from JSON
                if 'libero_object' in aug_data:
                    libero_data = aug_data['libero_object']
                    if libero_data:
                        # Find matching original prompt or use random one
                        matching_item = None
                        for item in libero_data:
                            original_prompt = item.get('original_prompt', '')

                            # Ensure original_prompt is a string
                            if isinstance(original_prompt, list):
                                original_prompt = str(original_prompt[0]) if original_prompt else ''
                            else:
                                original_prompt = str(original_prompt)

                            if original_prompt and original_prompt.lower() in actual_text.lower():
                                matching_item = item
                                break

                        if matching_item:
                            new_prompt = matching_item.get('new_prompt', actual_text)
                        else:
                            # Use random adversarial prompt if no match found
                            item_idx = torch.randint(0, len(libero_data), (1,)).item()
                            selected_item = libero_data[item_idx]
                            new_prompt = selected_item.get('new_prompt', actual_text)

                        # Ensure new_prompt is a string
                        if isinstance(new_prompt, list):
                            augmented_text = str(new_prompt[0]) if new_prompt else actual_text
                        else:
                            augmented_text = str(new_prompt)

                        augmented_texts.append(augmented_text)
                    else:
                        augmented_texts.append(actual_text)
                else:
                    augmented_texts.append(actual_text)

            elif augmentation_name == 'sentence':
                # Sentence-level augmentation: replace with paraphrased sentences
                if 'libero_object' in aug_data:
                    libero_data = aug_data['libero_object']
                    if libero_data:
                        # Find matching original prompt or use random one
                        matching_item = None
                        for item in libero_data:
                            original_prompt = item.get('original_prompt', '')

                            # Ensure original_prompt is a string
                            if isinstance(original_prompt, list):
                                original_prompt = str(original_prompt[0]) if original_prompt else ''
                            else:
                                original_prompt = str(original_prompt)

                            if original_prompt and original_prompt.lower() in actual_text.lower():
                                matching_item = item
                                break

                        if matching_item:
                            new_prompt = matching_item.get('new_prompt', actual_text)
                        else:
                            # Use random paraphrased sentence if no match found
                            item_idx = torch.randint(0, len(libero_data), (1,)).item()
                            selected_item = libero_data[item_idx]
                            new_prompt = selected_item.get('new_prompt', actual_text)

                        # Ensure new_prompt is a string
                        if isinstance(new_prompt, list):
                            augmented_text = str(new_prompt[0]) if new_prompt else actual_text
                        else:
                            augmented_text = str(new_prompt)

                        augmented_texts.append(augmented_text)
                    else:
                        augmented_texts.append(actual_text)
                else:
                    augmented_texts.append(actual_text)

            elif augmentation_name == 'word':
                # Word-level augmentation: replace with synonym-based variations
                if 'libero_object' in aug_data:
                    libero_data = aug_data['libero_object']
                    if libero_data:
                        # Find matching original prompt or use random one
                        matching_item = None
                        for item in libero_data:
                            original_prompt = item.get('original_prompt', '')

                            # Ensure original_prompt is a string
                            if isinstance(original_prompt, list):
                                original_prompt = str(original_prompt[0]) if original_prompt else ''
                            else:
                                original_prompt = str(original_prompt)

                            if original_prompt and original_prompt.lower() in actual_text.lower():
                                matching_item = item
                                break

                        if matching_item:
                            new_prompt = matching_item.get('new_prompt', actual_text)
                        else:
                            # Use random word-level variation if no match found
                            item_idx = torch.randint(0, len(libero_data), (1,)).item()
                            selected_item = libero_data[item_idx]
                            new_prompt = selected_item.get('new_prompt', actual_text)

                        # Ensure new_prompt is a string
                        if isinstance(new_prompt, list):
                            augmented_text = str(new_prompt[0]) if new_prompt else actual_text
                        else:
                            augmented_text = str(new_prompt)

                        augmented_texts.append(augmented_text)
                    else:
                        augmented_texts.append(actual_text)
                else:
                    augmented_texts.append(actual_text)
            else:
                # Unknown augmentation type
                augmented_texts.append(actual_text)

        return augmented_texts

    def _retokenize_with_augmented_language(self, original_batch: Dict[str, Any],
                                    augmented_lang: str, augmented_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Retokenize the batch with the augmented language instruction."""

        # Check for necessary components
        if self.batch_transform is None:
            logger.warning("batch_transform not available, cannot retokenize")
            return original_batch

        try:
            # If augmented_prompt is not provided, build a default one
            if augmented_prompt is None:
                augmented_prompt = f"What action should the robot take to {augmented_lang}?"

            torch.cuda.empty_cache()

            # Check if it is batch data (usually original_action is a tensor)
            if isinstance(original_batch.get("original_action"), torch.Tensor):
                if original_batch["original_action"].dim() > 1:
                    # This is a batch, needs to be processed one by one
                    batch_size = original_batch["original_action"].shape[0]

                    new_input_ids = []
                    new_labels = []
                    new_pixel_values = []

                    for i in range(batch_size):
                        # Extract data for a single sample
                        single_action = original_batch["original_action"][i]
                        single_image = original_batch["image"][i] if isinstance(original_batch["image"], list) else original_batch["image"]

                        # Retokenize a single sample
                        single_retokenized = self.batch_transform.tokenize_action_sequence(
                            single_action,
                            augmented_prompt,
                            single_image
                        )

                        new_input_ids.append(single_retokenized["input_ids"])
                        new_labels.append(single_retokenized["labels"])
                        new_pixel_values.append(single_retokenized["pixel_values"])

                    # Find the maximum length
                    max_len = max(len(seq) for seq in new_input_ids)
                    pad_token_id = self.batch_transform.base_tokenizer.pad_token_id or self.batch_transform.base_tokenizer.eos_token_id

                    device = original_batch["input_ids"].device
                    dtype_int = original_batch["input_ids"].dtype
                    dtype_float = original_batch["pixel_values"].dtype

                    # Pre-allocate tensor space
                    if (original_batch["input_ids"].shape[0] == batch_size and
                        original_batch["input_ids"].shape[1] >= max_len):
                        padded_input_ids = original_batch["input_ids"][:, :max_len].clone()
                        padded_labels = original_batch["labels"][:, :max_len].clone()
                        attention_masks = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
                    else:
                        # Create new tensor
                        padded_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=dtype_int, device=device)
                        padded_labels = torch.full((batch_size, max_len), -100, dtype=dtype_int, device=device)
                        attention_masks = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

                    # Fill data
                    for i in range(batch_size):
                        input_ids = new_input_ids[i]
                        labels = new_labels[i]
                        seq_len = len(input_ids)
                        padded_input_ids[i, :seq_len] = input_ids
                        padded_labels[i, :seq_len] = labels
                        attention_masks[i, :seq_len] = 1

                    # Process pixel_values
                    stacked_pixel_values = torch.stack(new_pixel_values).to(dtype_float)

                    original_batch["input_ids"] = padded_input_ids
                    original_batch["labels"] = padded_labels
                    original_batch["attention_mask"] = attention_masks
                    original_batch["pixel_values"] = stacked_pixel_values
                    original_batch["language_instruction"] = augmented_lang
                    original_batch["prompt_template"] = augmented_prompt

                else:
                    # Single sample
                    retokenized = self.batch_transform.tokenize_action_sequence(
                        original_batch["original_action"],
                        augmented_prompt,
                        original_batch["image"]
                    )

                    original_batch["input_ids"] = retokenized["input_ids"]
                    original_batch["labels"] = retokenized["labels"]
                    original_batch["pixel_values"] = retokenized["pixel_values"]
                    original_batch["language_instruction"] = augmented_lang
                    original_batch["prompt_template"] = augmented_prompt

            else:
                logger.warning("original_action format not recognized")

            torch.cuda.empty_cache()
            return original_batch

        except Exception as e:
            torch.cuda.empty_cache()
            logger.warning(f"Failed to retokenize with augmented language: {e}")
            logger.debug(f"Error details: {type(e).__name__}: {str(e)}")
            return original_batch

    def update_statistics(self, arm_idx: int, current_loss: float):
        """Update UCB statistics after observing loss"""
        if hasattr(self, '_last_update_step') and self._last_update_step == self.step_count:
            return  # Prevent duplicate updates
        # Input loss NaN check
        if math.isnan(current_loss) or math.isinf(current_loss):
            logger.warning(f"Step {self.step_count}: Detected NaN/Inf loss for arm {arm_idx} ({self.all_augmentations[arm_idx]}): {current_loss}")
            # Replace NaN loss with the mean of the current arm to avoid data contamination
            if not torch.isnan(self.mu[arm_idx]) and not torch.isinf(self.mu[arm_idx]):
                current_loss = self.mu[arm_idx].item()
                logger.info(f"Step {self.step_count}: Replaced NaN loss with arm mean: {current_loss}")
            else:
                # If even the mean is NaN, use the global mean
                valid_means = self.mu[~(torch.isnan(self.mu) | torch.isinf(self.mu))]
                if len(valid_means) > 0:
                    current_loss = valid_means.mean().item()
                    logger.info(f"Step {self.step_count}: Used global mean as fallback: {current_loss}")
                else:
                    # Use a fixed value
                    current_loss = 1.0
                    logger.info(f"Step {self.step_count}: Used fallback value: {current_loss}")

        # Calculate reward
        reward = self._compute_standardized_improvement_reward(arm_idx, current_loss)

        # NaN check for reward value
        if math.isnan(reward) or math.isinf(reward):
            logger.warning(f"Step {self.step_count}: Detected NaN/Inf reward for arm {arm_idx}: {reward}")
            reward = 0.0  # Set NaN reward to 0, indicating no improvement
            logger.info(f"Step {self.step_count}: Reset reward to 0.0")

        # Update EMA statistics
        self._update_ema_statistics(arm_idx, current_loss)

        # Update sliding window statistics
        self._update_sliding_window(arm_idx, reward)

        # Increment step counter
        self.step = self.step_count

        self._last_update_step = self.step_count

        # Output log every 100 steps
        if self.step_count % 100 == 0:
            logger.info(
                f"UCB Step {self.step_count}: arm={arm_idx} ({self.all_augmentations[arm_idx]}), "
                f"loss={current_loss:.4f}, reward={reward:.4f}, "
                f"mu={self.mu[arm_idx]:.4f}, sigma={self.sigma[arm_idx]:.4f}"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get current UCB statistics for logging"""
        ucb_indices = self._compute_ucb_indices()

        # Convert tensors to Python lists/numbers for JSON serialization
        stats = {
            'ucb_step': self.step_count,
            'window_counts_sum': int(self.window_counts.sum().item()),
            'window_rewards_sum': float(self.window_rewards.sum().item()),
        }

        # Add separate metrics for each arm
        for i, aug_name in enumerate(self.all_augmentations):
            stats[f'ucb_{aug_name}_index'] = float(ucb_indices[i])
            stats[f'ucb_{aug_name}_mean'] = float(self.mu[i])
            stats[f'ucb_{aug_name}_std'] = float(self.sigma[i])
            stats[f'ucb_{aug_name}_count'] = int(self.window_counts[i])
            stats[f'ucb_{aug_name}_reward'] = float(self.window_rewards[i])

        return stats