"""Humanoid Parkour - G1 Locomotion Tasks with Adaptive Curriculum.

Importing this package registers all custom gym environments
for the Isaac Lab task registry. Task IDs include:

    Isaac-Velocity-Parkour-G1-v0
    Isaac-Velocity-Parkour-NoCurriculum-G1-v0
    Isaac-Velocity-Parkour-NoHeightScan-G1-v0
    Isaac-Velocity-Parkour-Replay-G1-v0
    Isaac-Velocity-Parkour-Adaptive-G1-v0
    Isaac-Velocity-Hiking-Adaptive-G1-v0
    (plus corresponding Play variants)
"""

import gymnasium as gym

from .agents import rsl_rl_ppo_cfg as agents

PACKAGE_NAME = __name__

gym.register(
    id="Isaac-Velocity-Parkour-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1ParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1ParkourPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Velocity-Parkour-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1ParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1ParkourPPORunnerCfg",
    },
)

# No Curriculum
gym.register(
    id="Isaac-Velocity-Parkour-NoCurriculum-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1ParkourNoCurriculumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1ParkourNoCurriculumPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Velocity-Parkour-NoCurriculum-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1ParkourNoCurriculumEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1ParkourNoCurriculumPPORunnerCfg",
    },
)

# No Height Scan
gym.register(
    id="Isaac-Velocity-Parkour-NoHeightScan-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1ParkourNoHeightScanEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1ParkourNoHeightScanPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Velocity-Parkour-NoHeightScan-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1ParkourNoHeightScanEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1ParkourNoHeightScanPPORunnerCfg",
    },
)

# No Shaping (original - same as baseline rewards, kept for reference)
gym.register(
    id="Isaac-Velocity-Parkour-NoShaping-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1ParkourNoShapingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1ParkourNoShapingPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Velocity-Parkour-NoShaping-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1ParkourNoShapingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1ParkourNoShapingPPORunnerCfg",
    },
)

# Replay
gym.register(
    id="Isaac-Velocity-Parkour-Replay-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1ParkourReplayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1ParkourReplayPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Velocity-Parkour-Replay-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1ParkourReplayEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1ParkourReplayPPORunnerCfg",
    },
)

# Adaptive
gym.register(
    id="Isaac-Velocity-Parkour-Adaptive-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1ParkourAdaptiveEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1ParkourAdaptivePPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Velocity-Parkour-Adaptive-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1ParkourAdaptiveEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1ParkourAdaptivePPORunnerCfg",
    },
)

# Zero Shaping
gym.register(
    id="Isaac-Velocity-Parkour-ZeroShaping-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1ParkourZeroShapingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1ParkourZeroShapingPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Velocity-Parkour-ZeroShaping-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1ParkourZeroShapingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1ParkourZeroShapingPPORunnerCfg",
    },
)

# Hiking Adaptive
gym.register(
    id="Isaac-Velocity-Hiking-Adaptive-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1HikingAdaptiveEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1HikingAdaptivePPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Velocity-Hiking-Adaptive-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1HikingAdaptiveEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1HikingAdaptivePPORunnerCfg",
    },
)

# Hiking TerrainOnly
gym.register(
    id="Isaac-Velocity-HikingTerrainOnly-Adaptive-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1HikingTerrainOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1HikingTerrainOnlyPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Velocity-HikingTerrainOnly-Adaptive-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1HikingTerrainOnlyEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1HikingTerrainOnlyPPORunnerCfg",
    },
)

# Hiking Ablation: No Touchdown Rewards
gym.register(
    id="Isaac-Velocity-Hiking-Adaptive-NoTouchdown-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1HikingNoTouchdownEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1HikingNoTouchdownPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Velocity-Hiking-Adaptive-NoTouchdown-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1HikingNoTouchdownEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1HikingNoTouchdownPPORunnerCfg",
    },
)

# Hiking Ablation: No Clearance / Stance Edge
gym.register(
    id="Isaac-Velocity-Hiking-Adaptive-NoClearance-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1HikingNoClearanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1HikingNoClearancePPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Velocity-Hiking-Adaptive-NoClearance-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{PACKAGE_NAME}.parkour_env_cfg:G1HikingNoClearanceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1HikingNoClearancePPORunnerCfg",
    },
)
