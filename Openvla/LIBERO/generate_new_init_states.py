#!/usr/bin/env python3
"""
Script to generate new initial states for the new task suites with distractor objects.
"""

import os
import numpy as np
import sys
import json
import torch
from pathlib import Path

# Add current directory to path
sys.path.insert(0, './LIBERO/libero/libero')
from benchmark import get_benchmark_dict
from envs import TASK_MAPPING
# Import all problems to register them
import envs.problems.libero_tabletop_manipulation
import envs.problems.libero_coffee_table_manipulation
import envs.problems.libero_floor_manipulation
import envs.problems.libero_study_tabletop_manipulation
import envs.problems.libero_living_room_tabletop_manipulation
import envs.problems.libero_kitchen_tabletop_manipulation

# Import get_libero_path directly
def get_libero_path(query_key):
    import yaml
    config_file = os.path.expanduser("~/.libero/config.yaml")
    with open(config_file, "r") as f:
        config = dict(yaml.load(f.read(), Loader=yaml.FullLoader))
    return config[query_key]

def generate_init_states_for_task(suite_name, task_id):
    """
    Generate new initial states for a specific task.
    """
    print(f"Generating initial states for {suite_name} task {task_id}...")
    
    try:
        # Use the original libero_spatial suite for generating initial states
        # since the new suites have the same structure
        benchmark_dict = get_benchmark_dict()
        original_suite = benchmark_dict["libero_spatial"]()
        
        # Get the task from the new suite for naming purposes
        new_suite = benchmark_dict[suite_name]()
        task = new_suite.get_task(task_id)
        # Use the task name directly from the task object
        task_name = task.name
        
        print(f"  Task: {task_name}")
        
        # Use the first task from the original suite as a template
        template_task = original_suite.get_task(0)
        
        # Get paths
        init_states_path = get_libero_path("init_states")
        bddl_files_path = get_libero_path("bddl_files")
        
        # Create directory for new init states
        new_init_dir = os.path.join(init_states_path, suite_name)
        os.makedirs(new_init_dir, exist_ok=True)
        
        # Use the BDDL file from the new task, not the template
        bddl_file = os.path.join(bddl_files_path, task.problem_folder, task.bddl_file)
        
        if not os.path.exists(bddl_file):
            print(f"    Warning: BDDL file {bddl_file} does not exist")
            return False
            
        # Create environment using template task configuration
        env_kwargs = {
            "bddl_file_name": bddl_file,
            "camera_heights": 128,
            "camera_widths": 128,
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "ignore_done": True,
            "use_camera_obs": False,
            "reward_shaping": True,
            "control_freq": 20,
        }
        
        # Create environment using OffScreenRenderEnv
        from envs.env_wrapper import OffScreenRenderEnv
        env = OffScreenRenderEnv(**env_kwargs)
        
        # Generate initial states
        init_states = []
        num_init_states = 50
        
        print(f"    Generating {num_init_states} initial states...")
        
        for i in range(num_init_states):
            reset_success = False
            attempts = 0
            max_attempts = 100  # 增加最大重试次数
            
            while not reset_success and attempts < max_attempts:
                try:
                    # 添加随机种子变化来增加成功概率
                    if attempts > 0:
                        np.random.seed(np.random.randint(0, 10000))
                    env.reset()
                    reset_success = True
                except Exception as e:
                    attempts += 1
                    # 每20次尝试打印一次进度
                    if attempts % 20 == 0:
                        print(f"    State {i+1}: Attempt {attempts}/{max_attempts}, Error: {str(e)[:50]}...")
                    
                    # 如果是"Cannot place all objects"错误，继续重试
                    if "Cannot place all objects" in str(e):
                        continue
                    
                    # 其他错误也继续重试，但记录
                    if attempts >= max_attempts:
                        print(f"    Failed to reset environment after {max_attempts} attempts for state {i+1}: {e}")
                        break
            
            if reset_success:
                state = env.sim.get_state().flatten()
                init_states.append(state)
                
                if (i + 1) % 10 == 0:
                    print(f"    Generated {i + 1}/{num_init_states} states")
        
        env.close()
        
        if len(init_states) > 0:
            init_states = np.array(init_states)
            
            # Save .init file using torch.save (compatible with benchmark loading)
            init_file = os.path.join(new_init_dir, f"{task_name}.init")
            torch.save(init_states, init_file)
            
            # Save .pruned_init file (use all 50 states instead of every 5th)
            pruned_states = init_states  # Use all states instead of sampling
            pruned_init_file = os.path.join(new_init_dir, f"{task_name}.pruned_init")
            torch.save(pruned_states, pruned_init_file)
            
            print(f"    Saved {len(init_states)} states to {init_file}")
            print(f"    Saved {len(pruned_states)} pruned states to {pruned_init_file}")
            return True
        else:
            print(f"    Failed to generate any initial states")
            return False
            
    except Exception as e:
        print(f"    Error: {e}")
        return False

def main():
    # Generate initial states for the three new task suites
    # suites = ["libero_spatial_object", "libero_object_object", "libero_goal_object"]
    suites = ["libero_goal_object"]
    
    for suite_name in suites:
        print(f"\nProcessing suite: {suite_name}")
        
        try:
            # Get the benchmark
            benchmark_dict = get_benchmark_dict()
            suite = benchmark_dict[suite_name]()
            num_tasks = suite.get_num_tasks()
            
            print(f"Found {num_tasks} tasks in {suite_name}")
            
            success_count = 0
            for task_id in range(num_tasks):
                if generate_init_states_for_task(suite_name, task_id):
                    success_count += 1
            
            print(f"Successfully generated initial states for {success_count}/{num_tasks} tasks in {suite_name}")
            
        except Exception as e:
            print(f"Failed to process suite {suite_name}: {e}")

if __name__ == "__main__":
    main()