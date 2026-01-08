import collections

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

from openpi_client import image_tools

import wandb
from datetime import datetime
import json

# Append current directory so that interpreter can find experiments.robot

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

from libero_utils import (
    get_libero_env,
    quat2axisangle,
    save_rollout_video,
    set_seed_everywhere,
)
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

from pi_utils import (
    create_policy,
)

from policies.action_noise_utils import (
    generate_uniform_action_noise,
    apply_action_noise
)

from policies.visual_noise_utils import (
    apply_visual_noise,
    generate_fixed_noise_params
)

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openpi"                    # Model family
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

    logging_dir: str = "./results"          # Directory for eval logs
    

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    
    # fmt: on
    resize_size: int = 224
    replan_steps: int = 5

    action_noise: float = 0
    action_noise_type: str = "uniform"  # noise type: uniform, gaussian, constant, salt_pepper, impulse
    salt_pepper_probability: float = 0.1
    impulse_probability: float = 0.05

    obs_noise: int = 0
    obs_noise_type: str = "gaussian"  # noise type: gaussian, salt_pepper, blur, image_shift, image_rotation, enhanced_color_jitter
    obs_salt_pepper_probability: float = 0.1
    obs_blur_kernel_size: int = 5
    obs_blur_sigma: float = 1.0
    obs_image_shift_ratio: float = 0.1
    obs_image_rotation_angle: float = 30.0
    obs_enhanced_color_jitter_factor: float = 3.0
    
    force_basic_magnitude: float = 0.0
    force_basic_direction: str = ""
    force_basic_duration: int = 2
    force_random: bool = True
    force_freq: int = 0
    force_freq_range: int = 0
    force_magnitude_range: float = 0.5
    force_duration_range: int = 1

    prompt_file: str = "original"
    
    light_intensity: float = 1.0
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

 
    # Load policy
    policy = create_policy(cfg.pretrained_checkpoint)

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

            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
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
                    image_shape = (cfg.resize_size, cfg.resize_size, 3)
                    episode_fixed_params = generate_fixed_noise_params(
                        cfg.obs_noise_type,
                        image_shape,
                        max_shift_ratio=cfg.obs_image_shift_ratio,
                        max_angle=cfg.obs_image_rotation_angle,
                        max_factor=cfg.obs_enhanced_color_jitter_factor
                    )
                    fixed_noise_params.update(episode_fixed_params)
            # Generate uniform noise for backward compatibility
            if cfg.action_noise_type == "uniform":
                action_noise = generate_uniform_action_noise(
                    cfg.action_noise,
                    np.array(LIBERO_DUMMY_ACTION).shape
                )
            else:
                action_noise = None
            
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
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
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, cfg.resize_size, cfg.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, cfg.resize_size, cfg.resize_size)
                    )
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
                        wrist_img = apply_visual_noise(
                            wrist_img, 
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

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = policy.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= cfg.replan_steps
                        ), f"We want to replan every {cfg.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: cfg.replan_steps])

                    action = action_plan.popleft()
                    
                    if cfg.action_noise > 0:
                        if cfg.action_noise_type == "uniform" and action_noise is not None:
                            action = action + action_noise
                        else:
                            action = apply_action_noise(
                                action, 
                                cfg.action_noise_type, 
                                cfg.action_noise,
                                salt_pepper_probability=cfg.salt_pepper_probability,
                                impulse_probability=cfg.impulse_probability
                            )

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


if __name__ == "__main__":
    eval_libero()
