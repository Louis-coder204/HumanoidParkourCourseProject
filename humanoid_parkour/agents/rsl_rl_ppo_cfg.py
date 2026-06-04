# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1ParkourPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_parkour"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1ParkourNoCurriculumPPORunnerCfg(G1ParkourPPORunnerCfg):
    experiment_name = "g1_parkour_nocurriculum"


@configclass
class G1ParkourNoHeightScanPPORunnerCfg(G1ParkourPPORunnerCfg):
    experiment_name = "g1_parkour_noheightscan"


@configclass
class G1ParkourNoShapingPPORunnerCfg(G1ParkourPPORunnerCfg):
    experiment_name = "g1_parkour_noshaping"

@configclass
class G1ParkourZeroShapingPPORunnerCfg(G1ParkourPPORunnerCfg):
    experiment_name = "g1_parkour_zeroshaping"


@configclass
class G1ParkourReplayPPORunnerCfg(G1ParkourPPORunnerCfg):
    experiment_name = "g1_parkour_replay"


@configclass
class G1ParkourAdaptivePPORunnerCfg(G1ParkourPPORunnerCfg):
    experiment_name = "g1_parkour_adaptive"
