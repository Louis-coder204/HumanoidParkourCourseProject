# Humanoid Parkour Course Project

G1 humanoid robot locomotion on rough terrain using PPO.  
Isaac Lab 2.3.0 + Isaac Sim 5.1.0 + RSL-RL.

## Results Summary

All models trained 3000+ iterations, 2048 envs, on official ROUGH_TERRAINS_CFG.
Evaluated with 50 episodes on Easy (rows 0-2), Medium (0-5), Hard (0-8) terrain.

| Model | Curriculum Strategy | Easy | Medium | Hard | Track Reward |
|-------|---------------------|:---:|:---:|:---:|:---:|
| Baseline | Official terrain_levels_vel | 58% | 60% | 64% | 0.41 |
| Replay   | +20% uniform low-level replay | 72% | 88% | 74% | 0.55 |
| Adaptive | +fall-detection + 10% replay | 88% | 88% | 86% | 0.80 |

### Key Findings

1. **Catastrophic Forgetting**: Baseline Level 0 drops from 90% (iter 2000) to 20% (iter 3000) — curriculum pushes all envs to harder rows
2. **Anti-forgetting Replay**: recovers Level 0 to 75%, Level 4 improves to 95%
3. **Adaptive Curriculum**: best overall — 88% success across all difficulties, 31m distance, 0.80 track reward

## Setup

conda create -n isaaclab python=3.11 -y && conda activate isaaclab
pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab
./isaaclab.sh --install

# Copy project files into IsaacLab
cp -r config/g1_parkour source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/
cp curriculums.py source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/
cp -r tools/* IsaacLab/tools/

## Project Structure

g1_parkour/                    # Custom task registration
  parkour_env_cfg.py            # Env config (+ all ablation variants)
  agents/
    rsl_rl_ppo_cfg.py           # PPO hyperparameters
  __init__.py                   # Gym task registrations

curriculums.py                  # 3 curriculum functions
  terrain_levels_vel             # Original (Baseline)
  terrain_levels_vel_replay      # +20% uniform replay
  terrain_levels_vel_adaptive    # +fall-drive + 10% replay

tools/
  eval_g1_parkour.py            # Stratified evaluation script

## Training

conda activate isaaclab && cd IsaacLab
./isaaclab.sh --install rsl_rl

# Baseline
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py   --task Isaac-Velocity-Parkour-G1-v0 --headless   --num_envs 2048 --max_iterations 3000 --run_name baseline

# Ablation: No Curriculum
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py   --task Isaac-Velocity-Parkour-NoCurriculum-G1-v0 --headless   --num_envs 2048 --max_iterations 3000 --run_name nocurriculum

# Ablation: No Height Scan
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py   --task Isaac-Velocity-Parkour-NoHeightScan-G1-v0 --headless   --num_envs 2048 --max_iterations 3000 --run_name noheight

# Replay Curriculum
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py   --task Isaac-Velocity-Parkour-Replay-G1-v0 --headless   --num_envs 2048 --max_iterations 3000 --run_name replay

# Adaptive Curriculum
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py   --task Isaac-Velocity-Parkour-Adaptive-G1-v0 --headless   --num_envs 2048 --max_iterations 3000 --run_name adaptive

## Resuming Training

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py   --task Isaac-Velocity-Parkour-Adaptive-G1-v0 --headless   --num_envs 2048 --max_iterations 3000 --resume   --load_run <run_folder> --checkpoint model_950.pt --run_name adaptive

## Evaluation

# Stratified eval (Easy/Medium/Hard)
./isaaclab.sh -p tools/eval_g1_parkour.py   --task Isaac-Velocity-Parkour-Adaptive-G1-Play-v0   --num_envs 8 --num_trials 50 --terrain_level easy   --load_run <run_folder> --checkpoint model_3000.pt --headless

# Fixed-level eval (single terrain row)
./isaaclab.sh -p tools/eval_g1_parkour.py   --task Isaac-Velocity-Parkour-Adaptive-G1-Play-v0   --num_envs 8 --num_trials 20 --eval_level 0   --load_run <run_folder> --checkpoint model_3000.pt --headless

# Record video
./isaaclab.sh -p tools/eval_g1_parkour.py   --task Isaac-Velocity-Parkour-Adaptive-G1-Play-v0   --num_envs 4 --num_trials 10 --terrain_level easy   --load_run <run_folder> --checkpoint model_3000.pt --headless --video

## Metrics Tracked

Success rate, fall rate, distance traveled, time survived,
velocity tracking error, terrain level reached, episode return.

## Requirements

- Ubuntu 22.04+, NVIDIA GPU >= RTX 3090 (24GB)
- Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.0
- GLIBC 2.35+
