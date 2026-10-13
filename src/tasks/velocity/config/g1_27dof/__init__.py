from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_27dof_flat_env_cfg,
  unitree_g1_27dof_rough_env_cfg,
)
from .rl_cfg import unitree_g1_27dof_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-27Dof-Rough",
  env_cfg=unitree_g1_27dof_rough_env_cfg(),
  play_env_cfg=unitree_g1_27dof_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_27dof_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-27Dof-Flat",
  env_cfg=unitree_g1_27dof_flat_env_cfg(),
  play_env_cfg=unitree_g1_27dof_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_27dof_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
