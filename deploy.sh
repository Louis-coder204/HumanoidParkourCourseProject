#!/usr/bin/env bash
# Deploy humanoid_parkour project files to IsaacLab for training/eval
# Usage: ./deploy.sh [isaaclab_root]
#   Default isaaclab_root: /home/isaac/Tingx/IsaacLab_2.3.0
set -e

ISAACLAB_ROOT="${1:-/home/isaac/Tingx/IsaacLab_2.3.0}"
DEST="${ISAACLAB_ROOT}/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"

echo "Deploying to: $ISAACLAB_ROOT"

# 1. Parkour config + hiking_mdp -> g1_parkour/
cp humanoid_parkour/__init__.py          "$DEST/config/g1_parkour/"
cp humanoid_parkour/parkour_env_cfg.py   "$DEST/config/g1_parkour/"
cp humanoid_parkour/hiking_mdp.py        "$DEST/config/g1_parkour/"
cp humanoid_parkour/agents/__init__.py   "$DEST/config/g1_parkour/agents/"
cp humanoid_parkour/agents/rsl_rl_ppo_cfg.py "$DEST/config/g1_parkour/agents/"

# 2. Curriculum functions -> mdp/
cp humanoid_parkour/curriculums.py       "$DEST/mdp/"

# 3. Eval script -> tools/
cp humanoid_parkour/eval_g1_parkour.py   "$ISAACLAB_ROOT/tools/"

echo "Done. Deployed to $ISAACLAB_ROOT"
