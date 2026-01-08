import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import torch
sys.path.append("/data/yihong.ji/RobustVLA-283D/LIBERO")
from libero.libero import benchmark

import wandb
from datetime import datetime
import imageio
# Append current directory so that interpreter can find experiments.robot
from libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from openvla.experiments.robot.openvla_utils import get_processor
from openvla.experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from policies.action_noise_utils import (
    generate_uniform_action_noise,
    apply_action_noise
)

from policies.visual_noise_utils import (
    apply_visual_noise,
    generate_fixed_noise_params
)

import json

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging

    logging_dir: str = "./results_action_uniform_noise_openvla"          # Directory for eval logs
    

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    
    # fmt: on
    resize_size: int = 224
    replan_steps: int = 5

    action_noise: float = 0
    action_noise_type: str = ""  # noise type: uniform, gaussian, constant, salt_pepper, impulse
    salt_pepper_probability: float = 0.1
    impulse_probability: float = 0.05

    obs_noise: int = 0                  #gaussian, set 70
    obs_noise_type: str = "image_rotation"  # noise type: gaussian, salt_pepper, blur, image_shift, image_rotation, enhanced_color_jitter
    obs_salt_pepper_probability: float = 0.1
    obs_blur_kernel_size: int = 5
    obs_blur_sigma: float = 1.0
    obs_image_shift_ratio: float = 0.15    #0.1
    obs_image_rotation_angle: float = 20.0     #30
    obs_enhanced_color_jitter_factor: float = 0.4            #3.0

    force_basic_magnitude: float = 0.0
    force_basic_direction: str = ""
    force_basic_duration: int = 2
    force_random: bool = True
    force_freq: int = 0
    force_freq_range: int = 0
    force_magnitude_range: float = 0.5
    force_duration_range: int = 1

    prompt_file: str = "original"
    
    light_intensity: float = 1.0 #float = 1.0
    light_intensity_range: float = 0.0
    light_angle: float = 0.785
    light_change_freq: int = 0
    light_freq_range: float = 1
    light_random: bool = True


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    if cfg.task_suite_name == "libero_object_object":
        cfg.unnorm_key = "libero_object"
    elif cfg.task_suite_name == "libero_spatial_object":
        cfg.unnorm_key = "libero_spatial"
    elif cfg.task_suite_name == "libero_goal_object":
        cfg.unnorm_key = "libero_goal"
    else:
        cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    local_log_dir = os.path.join(cfg.logging_dir, "logs")        # Local directory for eval logs
    os.makedirs(local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    if cfg.prompt_file != "original":
        with open(cfg.prompt_file, "r") as f:
            prompts = json.load(f)
    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        
        if cfg.prompt_file != "original":
            if cfg.task_suite_name in prompts and task_id < len(prompts[cfg.task_suite_name]):
                modified_task_description = prompts[cfg.task_suite_name][task_id]["new_prompt"]
                if not modified_task_description:
                    print(f"Error: Task {task_id} has an empty new_prompt, using original task description.")
                    log_file.write(f"Warning: Task {task_id} has an empty new_prompt, using original task description.\n")
                else:
                    print(f"Using modified task description: \n\t{task_description}->\n\t{modified_task_description}")
                    log_file.write(f"Using modified task description: \n\t{task_description}->\n\t{modified_task_description}\n")
                    task_description = modified_task_description
            else:
                print(f"Error: Suite {cfg.task_suite_name} task {task_id} has no prompt in {cfg.prompt_file}, using original task description.")
                
                
        save_path = os.path.join(cfg.logging_dir, task_description)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\nTime: {formatted_time}")
            print(f"Task: {task_description}")
            log_file.write(f"\nTime: {formatted_time}\n")
            log_file.write(f"Task: {task_description}\n")

            # Reset environment
            env.reset()
            if cfg.light_change_freq != 0:
                env.set_lighting(intensity=cfg.light_intensity, angle=cfg.light_angle)


            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            t_last_force = 0

            t_last_light = 0
            t_next_light = max(cfg.light_change_freq, cfg.num_steps_wait+1)
            
            replay_images = []
            if cfg.task_suite_name == "libero_spatial" or cfg.task_suite_name == "libero_spatial_object":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object" or cfg.task_suite_name == "libero_object_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal" or cfg.task_suite_name == "libero_goal_object":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            t_next_force = np.random.randint(40, 50)

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            
            # Pre-generate fixed parameters for noise types that need them
            fixed_noise_params = {}
            if cfg.obs_noise > 0:
                fixed_param_noise_types = ["image_shift", "image_rotation", "enhanced_color_jitter", "blur", "mask_noise", "image_occlusion"]
                
                if cfg.obs_noise_type in fixed_param_noise_types:
                    image_shape = (resize_size, resize_size, 3)
                    episode_fixed_params = generate_fixed_noise_params(
                        cfg.obs_noise_type,
                        image_shape,
                        max_shift_ratio=cfg.obs_image_shift_ratio,
                        max_angle=cfg.obs_image_rotation_angle,
                        max_factor=cfg.obs_enhanced_color_jitter_factor
                    )
                    fixed_noise_params.update(episode_fixed_params)
            
            if cfg.action_noise_type == "uniform":
                action_noise = generate_uniform_action_noise(
                    cfg.action_noise,
                    np.array(get_libero_dummy_action(cfg.model_family)).shape
                )
                #print(action_noise)
                #breakpoint()
            else:
                action_noise = None
            
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue
                    
                    if cfg.light_change_freq != 0 and t == t_next_light:
                        light_interval = np.random.randint(
                            cfg.light_change_freq - cfg.light_freq_range,
                            cfg.light_change_freq + cfg.light_freq_range
                        )
                        t_next_light += light_interval
                        
                        if cfg.light_random:
                            # Use Gamma distribution for light intensity variation
                            expected_value = 1.0
                            variance = cfg.light_intensity_range
                            scale = variance / expected_value
                            shape = expected_value / scale
                            gamma_sample = np.random.gamma(shape=shape, scale=scale)
                            new_intensity = cfg.light_intensity * gamma_sample
                            new_intensity = max(0.01, new_intensity)
                            
                            new_angle = np.random.uniform(0, 2*np.pi) 
                            
                        else:
                            new_intensity = cfg.light_intensity
                            new_angle = cfg.light_angle + t/max_steps * np.pi
                        env.set_lighting(intensity=new_intensity, angle=new_angle)
                        print(f"Changed lighting at step {t}: intensity={new_intensity}, angle={new_angle}")
                        log_file.write(f"Changed lighting at step {t}: intensity={new_intensity}, angle={new_angle}\n")
                        
                        
                    # Apply external force
                    if cfg.force_basic_magnitude > 0 and t == t_next_force:
                        if cfg.force_random:
                            force_mag = np.random.uniform(
                                cfg.force_basic_magnitude * (1 - cfg.force_magnitude_range),
                                cfg.force_basic_magnitude * (1 + cfg.force_magnitude_range)
                            )
                            force_dir = None
                            force_duration = np.random.randint(
                                cfg.force_basic_duration - cfg.force_duration_range,
                                cfg.force_basic_duration + cfg.force_duration_range
                            )
                            force_duration = max(1, force_duration)
                        else:
                            force_mag = cfg.force_basic_magnitude
                            force_dir = cfg.force_basic_direction
                            force_duration = cfg.force_basic_duration
                        
                        env.apply_external_force(
                            force_magnitude=force_mag,
                            force_direction=force_dir,
                            force_duration=force_duration
                        )
                        print(f"Applied external force at step {t}: magnitude={force_mag}, direction={force_dir}, duration={force_duration}")
                        log_file.write(f"Applied external force at step {t}: magnitude={force_mag}, direction={force_dir}, duration={force_duration}\n")
                    
                    
                    if cfg.force_basic_magnitude > 0:
                        env.update_external_force()
                    
                    
                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Apply visual noise
                    if cfg.obs_noise > 0:
                        img = apply_visual_noise(
                            img, 
                            cfg.obs_noise_type, 
                            cfg.obs_noise,
                            salt_pepper_probability=cfg.obs_salt_pepper_probability,
                            kernel_size=cfg.obs_blur_kernel_size,
                            sigma=cfg.obs_blur_sigma,
                            max_shift_ratio=cfg.obs_image_shift_ratio,
                            max_angle=cfg.obs_image_rotation_angle,
                            max_factor=cfg.obs_enhanced_color_jitter_factor,

                            fixed_params=fixed_noise_params
                        )                        
                        

                    

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action
                    action = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                    )
                    
                    # Apply action noise
                    if cfg.action_noise > 0:
                        if cfg.action_noise_type == "uniform" and action_noise is not None:
                            action = action + action_noise
                        else:
                            action = apply_action_noise(
                                action,
                                noise_type=cfg.action_noise_type,
                                action_noise_magnitude=cfg.action_noise,
                                salt_pepper_probability=cfg.salt_pepper_probability,
                                impulse_probability=cfg.impulse_probability
                            )

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        env.clear_external_force()
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    env.clear_external_force()
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1
            success = "success" if done else "failure"
            # Save a replay video of the episode
            save_rollout_video(
                replay_images, task_episodes, success=success, save_path=save_path, log_file=log_file
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

            result_path = os.path.join(save_path, "task_result.txt")
            with open(result_path, "w") as file:
                file.write(f"{int(task_episodes)} {int(task_successes)} {float(task_successes) / float(task_episodes)}\n")

            total_result_path = os.path.join(cfg.logging_dir, "total_result.txt")
            with open(total_result_path, "w") as file:
                file.write(f"{int(total_episodes)} {int(total_successes)} {float(total_successes) / float(total_episodes)}\n")
        
        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)



def pgd_attack_image_raw_pixel(
    model,
    processor,
    img,              # uint8, HWC, [0,255]
    state_vec,
    task_description,
    cfg: GenerateConfig,
) -> None:
    device = next(model.parameters()).device

    # === raw pixel tensor ===
    pixel_values = (
        torch.from_numpy(img)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)                     # [1,3,H,W], [0,255]
    )

    # clean action
    with torch.no_grad():
        obs_clean = {
            "full_image": img,
            "state": state_vec,
        }
        a_clean = torch.tensor(
            get_action(cfg, model, obs_clean, task_description, processor),
            device=device
        )

    # === init delta (pixel scale) ===
    delta = torch.empty_like(pixel_values).uniform_(
        -cfg.pgd_eps_pixel, cfg.pgd_eps_pixel
    )
    delta = torch.clamp(pixel_values + delta, 0, 255) - pixel_values

    for _ in range(cfg.pgd_steps):
        delta.requires_grad_(True)

        img_adv = (
            (pixel_values + delta)
            .clamp(0, 255)
            .byte()
            .squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        obs_adv = {
            "full_image": img_adv,
            "state": state_vec,
        }

        a_adv = torch.tensor(
            get_action(cfg, model, obs_adv, task_description, processor),
            device=device
        )

        # ===== action-based adversarial loss =====
        loss = torch.nn.functional.mse_loss(a_adv, a_clean)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            # PGD ascent
            delta = delta + cfg.pgd_alpha_pixel * delta.grad.sign()

            # L∞ projection
            delta = torch.clamp(
                delta, -cfg.pgd_eps_pixel, cfg.pgd_eps_pixel
            )
            delta = torch.clamp(pixel_values + delta, 0, 255) - pixel_values

    return img_adv


def forward_vision_features(model, pixel_values):
    """
    pixel_values: torch.Tensor [B, 3, H, W]
                  output of vision_backbone.image_transform
    return:
        patch_features: [B, N, D]
    """
    return model.vision_backbone(pixel_values)









if __name__ == "__main__":
    eval_libero()
