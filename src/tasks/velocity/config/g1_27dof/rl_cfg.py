"""RL configuration for Unitree G1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_27dof_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_27dof_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )

def unitree_g1_27dof_hands_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Same PPO settings as velocity, separate log folder."""
  cfg = unitree_g1_27dof_ppo_runner_cfg()
  cfg.experiment_name = "g1_27dof_hands"
  return cfg


def unitree_g1_27dof_hands_pose_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """v1 hands pose task: same PPO settings, own log folder, longer training."""
  cfg = unitree_g1_27dof_ppo_runner_cfg()
  cfg.experiment_name = "g1_27dof_hands_pose"
  # Orientation is harder to learn than position: train 15k iterations (~6 h)
  # instead of 10k. The extra 1 makes the last checkpoint model_15000.pt.
  cfg.max_iterations = 15001
  return cfg
