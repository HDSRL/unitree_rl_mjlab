"""Evaluate a velocity-policy rollout and generate a WBC-AGILE-style report."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import time

import mjlab
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import tyro
from plotly.subplots import make_subplots

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

COMMAND_SEGMENTS = [
  # duration (s), [vx (m/s), vy (m/s), yaw rate (rad/s)]
  (5.0, [0.0, 0.0, 0.0]),

  # Forward x test.
  (5.0, [0.2, 0.0, 0.0]),
  (5.0, [-0.2, 0.0, 0.0]),
  (5.0, [0.5, 0.0, 0.0]),
  (5.0, [-0.5, 0.0, 0.0]),
  (3.0, [0.0, 0.0, 0.0]),

  # Lateral y test: positive, then negative.
  (5.0, [0.0, 0.2, 0.0]),
  (5.0, [0.0, -0.2, 0.0]),
  (5.0, [0.0, 0.5, 0.0]),
  (5.0, [0.0, -0.5, 0.0]),
  (3.0, [0.0, 0.0, 0.0]),

  # Yaw test: left, then right.
  (5.0, [0.0, 0.0, 0.3]),
  (5.0, [0.0, 0.0, -0.3]),
  (5.0, [0.0, 0.0, 0.6]),
  (5.0, [0.0, 0.0, -0.6]),
  (3.0, [0.0, 0.0, 0.0]),
]

@dataclass(frozen=True)
class EvaluateConfig:
  checkpoint_file: str
  output_dir: str | None = None
  device: str | None = None
  seed: int = 42
  plateau_seconds: float = 5.0
  viewer: bool = False

def evaluation_command(time_s: float) -> np.ndarray:
  elapsed = 0.0
  for duration_s, command in COMMAND_SEGMENTS:
    elapsed += duration_s
    if time_s < elapsed:
      return np.asarray(command, dtype=np.float32)

  return np.zeros(3, dtype=np.float32)

def joint_groups(joint_names: list[str]) -> dict[str, list[int]]:
  groups = {
    "Lower body": [],
    "Waist": [],
    "Arms": [],
    "Other": [],
  }

  for index, name in enumerate(joint_names):
    lower_name = name.lower()
    if any(token in lower_name for token in ("hip", "knee", "ankle")):
      groups["Lower body"].append(index)
    elif "waist" in lower_name:
      groups["Waist"].append(index)
    elif any(token in lower_name for token in ("shoulder", "elbow", "wrist")):
      groups["Arms"].append(index)
    else:
      groups["Other"].append(index)

  return {name: indices for name, indices in groups.items() if indices}


def tracking_figure(data: dict[str, np.ndarray]) -> go.Figure:
  labels = [
    ("Linear velocity x", "m/s", 0),
    ("Linear velocity y", "m/s", 1),
    ("Yaw velocity", "rad/s", 2),
  ]
  fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=[label[0] for label in labels],
    vertical_spacing=0.08,
  )

  for row, (_, unit, axis) in enumerate(labels, start=1):
    actual = (
      data["root_com_lin_vel_b"][:, axis]
      if axis < 2
      else data["root_com_ang_vel_b"][:, 2]
    )
    fig.add_trace(
      go.Scatter(
        x=data["timestep"],
        y=actual,
        name="Actual",
        line={"color": "#1f77b4", "width": 2},
        showlegend=row == 1,
      ),
      row=row,
      col=1,
    )
    fig.add_trace(
      go.Scatter(
        x=data["timestep"],
        y=data["commands"][:, axis],
        name="Command",
        line={"color": "#d62728", "dash": "dash", "width": 2},
        showlegend=row == 1,
      ),
      row=row,
      col=1,
    )
    fig.update_yaxes(title_text=unit, row=row, col=1)

  fig.update_xaxes(title_text="Time (s)", row=3, col=1)
  fig.update_layout(
    title="Velocity Tracking",
    height=750,
    hovermode="x unified",
  )
  return fig


def com_figure(data: dict[str, np.ndarray]) -> go.Figure:
  fig = make_subplots(
    rows=2,
    cols=3,
    shared_xaxes=True,
    subplot_titles=[
      "COM position x",
      "COM position y",
      "COM position z",
      "COM velocity x",
      "COM velocity y",
      "COM velocity z",
    ],
  )
  axes = ["x", "y", "z"]

  for axis in range(3):
    fig.add_trace(
      go.Scatter(
        x=data["timestep"],
        y=data["root_com_pos_w"][:, axis],
        name=f"Position {axes[axis]}",
        showlegend=False,
      ),
      row=1,
      col=axis + 1,
    )
    fig.add_trace(
      go.Scatter(
        x=data["timestep"],
        y=data["root_com_lin_vel_w"][:, axis],
        name=f"Velocity {axes[axis]}",
        showlegend=False,
      ),
      row=2,
      col=axis + 1,
    )
    fig.update_yaxes(title_text="m", row=1, col=axis + 1)
    fig.update_yaxes(title_text="m/s", row=2, col=axis + 1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=axis + 1)

  fig.update_layout(
    title="Center of Mass Diagnostics",
    height=600,
    hovermode="x unified",
  )
  return fig


def joint_figure(
  group_name: str,
  indices: list[int],
  joint_names: list[str],
  data: dict[str, np.ndarray],
  pos_limits: np.ndarray,
  vel_limits: np.ndarray,
) -> go.Figure:
  fig = make_subplots(
    rows=len(indices),
    cols=3,
    shared_xaxes=True,
    vertical_spacing=0.02,
    horizontal_spacing=0.08,
    column_titles=["Position (rad)", "Velocity (rad/s)", "Actuator torque (Nm)"],
  )

  time_s = data["timestep"]

  for row, joint_index in enumerate(indices, start=1):
    name = joint_names[joint_index]

    fig.add_trace(
      go.Scatter(x=time_s, y=data["joint_pos"][:, joint_index], name=name, showlegend=False),
      row=row,
      col=1,
    )
    fig.add_trace(
      go.Scatter(x=time_s, y=data["joint_vel"][:, joint_index], name=name, showlegend=False),
      row=row,
      col=2,
    )
    fig.add_trace(
      go.Scatter(
        x=time_s,
        y=data["actuator_force"][:, joint_index],
        name=name,
        showlegend=False,
      ),
      row=row,
      col=3,
    )

    # Soft position and velocity limits.
    x_limits = [time_s[0], time_s[-1]]
    for limit in pos_limits[joint_index]:
      fig.add_trace(
        go.Scatter(
          x=x_limits,
          y=[limit, limit],
          mode="lines",
          line={"color": "red", "dash": "dash", "width": 1},
          hoverinfo="skip",
          showlegend=False,
        ),
        row=row,
        col=1,
      )

    if vel_limits is not None:
        velocity_limit = vel_limits[joint_index]
        for limit in (-velocity_limit, velocity_limit):
            fig.add_trace(
                go.Scatter(
                x=x_limits,
                y=[limit, limit],
                mode="lines",
                line={"color": "red", "dash": "dash", "width": 1},
                hoverinfo="skip",
                showlegend=False,
                ),
                row=row,
                col=2,
            )

    fig.update_yaxes(title_text=name, row=row, col=1)

  fig.update_xaxes(title_text="Time (s)", row=len(indices), col=1)
  fig.update_xaxes(title_text="Time (s)", row=len(indices), col=2)
  fig.update_xaxes(title_text="Time (s)", row=len(indices), col=3)
  fig.update_layout(
    title=f"{group_name}: Joint Position, Velocity, and Simulated Actuator Torque",
    height=max(500, 190 * len(indices)),
    hovermode="x unified",
  )
  return fig


def save_outputs(
  output_dir: Path,
  data: dict[str, np.ndarray],
  joint_names: list[str],
  pos_limits: np.ndarray,
  vel_limits: np.ndarray,
  metadata: dict,
) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)

  np.savez_compressed(
    output_dir / "rollout.npz",
    **data,
    joint_names=np.asarray(joint_names),
  )

  dataframe: dict[str, np.ndarray] = {
    "timestep": data["timestep"],
    "command_vx": data["commands"][:, 0],
    "command_vy": data["commands"][:, 1],
    "command_yaw_rate": data["commands"][:, 2],
    "root_com_pos_w_x": data["root_com_pos_w"][:, 0],
    "root_com_pos_w_y": data["root_com_pos_w"][:, 1],
    "root_com_pos_w_z": data["root_com_pos_w"][:, 2],
    "root_com_lin_vel_b_x": data["root_com_lin_vel_b"][:, 0],
    "root_com_lin_vel_b_y": data["root_com_lin_vel_b"][:, 1],
    "root_com_ang_vel_b_z": data["root_com_ang_vel_b"][:, 2],
  }

  for index in range(len(joint_names)):
    dataframe[f"joint_pos_{index}"] = data["joint_pos"][:, index]
    dataframe[f"joint_vel_{index}"] = data["joint_vel"][:, index]
    dataframe[f"actuator_force_{index}"] = data["actuator_force"][:, index]

  pd.DataFrame(dataframe).to_parquet(output_dir / "rollout.parquet", index=False)

  with open(output_dir / "metadata.json", "w") as file:
    json.dump(metadata, file, indent=2)

  figures = [
    tracking_figure(data),
    com_figure(data),
    *[
      joint_figure(name, indices, joint_names, data, pos_limits, vel_limits)
      for name, indices in joint_groups(joint_names).items()
    ],
  ]

  html = [
    "<!DOCTYPE html><html><head><meta charset='utf-8'>",
    "<title>G1 Policy Evaluation</title></head><body>",
    "<h1>G1 Policy Evaluation</h1>",
    f"<p><b>Checkpoint:</b> {metadata['checkpoint_file']}</p>",
    "<p><b>Torque note:</b> plotted values are MuJoCo simulated actuator "
    "forces/torques, not hardware torque estimates.</p>",
  ]

  for index, figure in enumerate(figures):
    html.append(
      figure.to_html(
        full_html=False,
        include_plotlyjs=True if index == 0 else False,
      )
    )

  html.append("</body></html>")
  (output_dir / "report.html").write_text("\n".join(html))
  print(f"[INFO] Saved evaluation to: {output_dir}")
  print(f"[INFO] Open report: {output_dir / 'report.html'}")


def run_evaluation(task_id: str, cfg: EvaluateConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  checkpoint = Path(cfg.checkpoint_file).expanduser().resolve()
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

  env_cfg = load_env_cfg(task_id, play=True)

  # Evaluation uses direct [vx, vy, yaw_rate] commands.
  # Disable the training-time heading controller, which otherwise overwrites yaw.
  twist_cfg = env_cfg.commands["twist"]
  twist_cfg.heading_command = False
  twist_cfg.ranges.heading = None
  twist_cfg.rel_heading_envs = 0.0
  twist_cfg.rel_standing_envs = 0.0
  twist_cfg.resampling_time_range = (1e9, 1e9)
  twist_cfg.init_velocity_prob = 0.0

  env_cfg.scene.num_envs = 1
  env_cfg.seed = cfg.seed

  agent_cfg = load_rl_cfg(task_id)
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = raw_env.scene["robot"]
  command_term = raw_env.command_manager.get_term("twist")
  if not hasattr(command_term, "vel_command_b"):
    raise RuntimeError("The selected task does not expose a body-frame twist command.")

  joint_names = list(robot.joint_names)
  pos_limits = robot.data.soft_joint_pos_limits[0].detach().cpu().numpy()
  # MjLab 1.2 exposes no per-joint velocity-limit tensor.
  vel_limits = None

  if robot.data.actuator_force.shape[1] != len(joint_names):
    raise RuntimeError(
      "Actuator-force ordering does not match joint ordering; stop and add "
      "an explicit actuator-to-joint mapping before plotting torque."
    )

  total_seconds = sum(duration_s for duration_s, _ in COMMAND_SEGMENTS)
  total_steps = round(total_seconds / raw_env.step_dt)

  logged: dict[str, list[np.ndarray | float]] = {
    "timestep": [],
    "commands": [],
    "root_com_pos_w": [],
    "root_com_lin_vel_w": [],
    "root_com_lin_vel_b": [],
    "root_com_ang_vel_b": [],
    "joint_pos": [],
    "joint_vel": [],
    "actuator_force": [],
    "actions": [],
  }

  terminated_early = False
  termination_time_s: float | None = None

  native_viewer = None
  if cfg.viewer:
      import mujoco
      import mujoco.viewer
      native_viewer = mujoco.viewer.launch_passive(
      raw_env.sim.mj_model,
      raw_env.sim.mj_data,
      show_left_ui=False,
      show_right_ui=False,
      )
      print("[INFO] Native viewer enabled. Close its window to stop evaluation.")

  try:
    env.reset()

    for step in range(total_steps):
      if native_viewer is not None and not native_viewer.is_running():
          print("[INFO] Viewer closed; stopping evaluation.")
          break
      step_start = time.perf_counter()
      time_s = step * raw_env.step_dt
      command = evaluation_command(time_s)

      command_tensor = torch.as_tensor(
        command,
        device=raw_env.device,
        dtype=command_term.vel_command_b.dtype,
      )
      command_term.vel_command_b[:] = command_tensor

      # The normal wrapper returns a cached observation. Recompute it after
      # changing the command so the policy sees this exact [vx, vy, yaw_rate].
      raw_env.obs_buf = raw_env.observation_manager.compute(update_history=True)
      observations = env.get_observations()

      policy_command = raw_env.command_manager.get_command("twist")[0]
      if not torch.allclose(policy_command, command_tensor, atol=1e-6):
        raise RuntimeError(
          f"Policy command mismatch: requested={command_tensor.tolist()}, "
          f"received={policy_command.tolist()}"
        )

      actions = policy(observations)

      _, _, dones, _ = env.step(actions)
      if bool(dones[0].item()):
        terminated_early = True
        termination_time_s = (step + 1) * raw_env.step_dt
        print(f"[WARN] Episode terminated at {termination_time_s:.2f} s")
        break

      logged["timestep"].append((step + 1) * raw_env.step_dt)
      logged["commands"].append(command.copy())
      logged["root_com_pos_w"].append(
        robot.data.root_com_pos_w[0].detach().cpu().numpy().copy()
      )
      logged["root_com_lin_vel_w"].append(
        robot.data.root_com_lin_vel_w[0].detach().cpu().numpy().copy()
      )
      logged["root_com_lin_vel_b"].append(
        robot.data.root_com_lin_vel_b[0].detach().cpu().numpy().copy()
      )
      logged["root_com_ang_vel_b"].append(
        robot.data.root_com_ang_vel_b[0].detach().cpu().numpy().copy()
      )
      logged["joint_pos"].append(robot.data.joint_pos[0].detach().cpu().numpy().copy())
      logged["joint_vel"].append(robot.data.joint_vel[0].detach().cpu().numpy().copy())
      logged["actuator_force"].append(
        robot.data.actuator_force[0].detach().cpu().numpy().copy()
      )
      logged["actions"].append(actions[0].detach().cpu().numpy().copy())
      if native_viewer is not None:
          sim_data = raw_env.sim.data
          viewer_data = raw_env.sim.mj_data

          viewer_data.qpos[:] = sim_data.qpos[0].detach().cpu().numpy()
          viewer_data.qvel[:] = sim_data.qvel[0].detach().cpu().numpy()
          viewer_data.xfrc_applied[:] = sim_data.xfrc_applied[0].detach().cpu().numpy()

          mujoco.mj_forward(raw_env.sim.mj_model, viewer_data)
          native_viewer.sync()

          time.sleep(
          max(0.0, raw_env.step_dt - (time.perf_counter() - step_start))
          )
  finally:
      if native_viewer is not None:
        native_viewer.close()
      env.close()

  if not logged["timestep"]:
    raise RuntimeError("No rollout samples were recorded.")

  data = {
    name: np.asarray(values, dtype=np.float32)
    for name, values in logged.items()
  }

  output_dir = (
    Path(cfg.output_dir).expanduser().resolve()
    if cfg.output_dir
    else checkpoint.parent
    / "evaluations"
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  )

  metadata = {
    "task_id": task_id,
    "checkpoint_file": str(checkpoint),
    "seed": cfg.seed,
    "step_dt": raw_env.step_dt,
    "duration_s": float(data["timestep"][-1]),
    "profile": "command_segments",
    "command_mode": "direct_yaw_rate",
    "command_segments": COMMAND_SEGMENTS,
    "joint_names": joint_names,
    "joint_pos_limits": pos_limits.tolist(),
    "joint_vel_limits": None,
    "torque_signal": "MjLab robot.data.actuator_force",
    "terminated_early": terminated_early,
    "termination_time_s": termination_time_s,
  }
  save_outputs(output_dir, data, joint_names, pos_limits, vel_limits, metadata)


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  task_id, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(
    EvaluateConfig,
    args=remaining_args,
    default=EvaluateConfig(checkpoint_file=""),
    prog=sys.argv[0] + f" {task_id}",
    config=mjlab.TYRO_FLAGS,
  )
  run_evaluation(task_id, cfg)


if __name__ == "__main__":
  main()