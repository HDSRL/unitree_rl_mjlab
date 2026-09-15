"""Load and validate a G1 deployment CSV log."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


JOINT_COUNT = 29

# Must match the deployment policy joint_ids_map: [0, ..., 28].
# This is the same canonical order used by evaluate_policy.py.
G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

JOINT_GROUPS = {
    "Lower body": list(range(0, 12)),
    "Waist": list(range(12, 15)),
    "Arms": list(range(15, 29)),
}

REQUIRED_BASE_COLUMNS = [
    "timestep",
    "sequence",
    "command_vx",
    "command_vy",
    "command_yaw_rate",
    "root_com_ang_vel_b_x",
    "root_com_ang_vel_b_y",
    "root_com_ang_vel_b_z",
    "imu_quat_w",
    "imu_quat_x",
    "imu_quat_y",
    "imu_quat_z",
]

def joint_columns(prefix: str) -> list[str]:
    """Return the expected 29 joint columns in policy-joint order."""
    return [f"{prefix}_{index}" for index in range(JOINT_COUNT)]


def load_deployment_log(csv_file: Path) -> pd.DataFrame:
    """Load one deployment CSV and reject incomplete or invalid logs."""
    if not csv_file.is_file():
        raise FileNotFoundError(f"Deployment CSV not found: {csv_file}")

    data = pd.read_csv(csv_file)

    required_columns = [
        *REQUIRED_BASE_COLUMNS,
        *joint_columns("joint_pos"),
        *joint_columns("joint_vel"),
        *joint_columns("joint_tau_est"),
        *joint_columns("target_joint_pos"),
    ]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"CSV is missing {len(missing_columns)} required columns: "
            f"{missing_columns}"
        )

    if data.empty:
        raise ValueError("CSV contains no deployment samples.")

    if not data["timestep"].is_monotonic_increasing:
        raise ValueError("CSV timestep must be monotonically increasing.")

    return data

def sample_period_summary(data: pd.DataFrame) -> dict[str, float]:
    """Summarize the observed interval between logged policy samples."""
    intervals_ms = data["timestep"].diff().iloc[1:] * 1000.0

    return {
        "mean_ms": float(intervals_ms.mean()),
        "min_ms": float(intervals_ms.min()),
        "max_ms": float(intervals_ms.max()),
    }

def command_and_yaw_figure(data: pd.DataFrame) -> go.Figure:
    """Plot velocity commands and the measurable yaw-rate response."""
    time_s = data["timestep"]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Forward command vx",
            "Lateral command vy",
            "Yaw-rate tracking",
        ],
    )

    fig.add_trace(
        go.Scatter(x=time_s, y=data["command_vx"], name="Command vx"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_s, y=data["command_vy"], name="Command vy"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_s,
            y=data["root_com_ang_vel_b_z"],
            name="Measured gyro z",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_s,
            y=data["command_yaw_rate"],
            name="Command yaw rate",
            line={"dash": "dash"},
        ),
        row=3,
        col=1,
    )

    fig.update_yaxes(title_text="m/s", row=1, col=1)
    fig.update_yaxes(title_text="m/s", row=2, col=1)
    fig.update_yaxes(title_text="rad/s", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_layout(
        title="Commands and Yaw-Rate Tracking",
        height=750,
        hovermode="x unified",
    )
    return fig

def imu_orientation_rpy(data: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Convert logged IMU quaternion w,x,y,z into roll, pitch, yaw in radians."""
    w = data["imu_quat_w"].to_numpy()
    x = data["imu_quat_x"].to_numpy()
    y = data["imu_quat_y"].to_numpy()
    z = data["imu_quat_z"].to_numpy()

    roll = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return roll, pitch, yaw


def imu_figure(data: pd.DataFrame) -> go.Figure:
    """Plot IMU orientation and body angular velocity."""
    time_s = data["timestep"]
    roll, pitch, yaw = imu_orientation_rpy(data)

    fig = make_subplots(
        rows=2,
        cols=3,
        shared_xaxes=True,
        subplot_titles=[
            "Roll",
            "Pitch",
            "Yaw",
            "Gyro x",
            "Gyro y",
            "Gyro z",
        ],
    )

    for column, values in enumerate((roll, pitch, yaw), start=1):
        fig.add_trace(
            go.Scatter(x=time_s, y=values, showlegend=False),
            row=1,
            col=column,
        )
        fig.update_yaxes(title_text="rad", row=1, col=column)

    gyro_columns = [
        "root_com_ang_vel_b_x",
        "root_com_ang_vel_b_y",
        "root_com_ang_vel_b_z",
    ]
    for column, gyro_column in enumerate(gyro_columns, start=1):
        fig.add_trace(
            go.Scatter(x=time_s, y=data[gyro_column], showlegend=False),
            row=2,
            col=column,
        )
        fig.update_yaxes(title_text="rad/s", row=2, col=column)
        fig.update_xaxes(title_text="Time (s)", row=2, col=column)

    fig.update_layout(
        title="IMU Orientation and Angular Velocity",
        height=600,
        hovermode="x unified",
    )
    return fig


def joint_figure(
    group_name: str,
    joint_indices: list[int],
    data: pd.DataFrame,
) -> go.Figure:
    """Plot actual/target position, velocity, and torque estimate per joint."""
    fig = make_subplots(
        rows=len(joint_indices),
        cols=3,
        shared_xaxes=True,
        vertical_spacing=0.02,
        horizontal_spacing=0.08,
        column_titles=[
            "Position (rad)",
            "Velocity (rad/s)",
            "Torque estimate (Nm)",
        ],
    )
    time_s = data["timestep"]

    for row, joint_index in enumerate(joint_indices, start=1):
        joint_name = G1_JOINT_NAMES[joint_index]

        fig.add_trace(
            go.Scatter(
                x=time_s,
                y=data[f"joint_pos_{joint_index}"],
                name="Actual position",
                line={"color": "#1f77b4"},
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=time_s,
                y=data[f"target_joint_pos_{joint_index}"],
                name="Target position",
                line={"color": "#ff7f0e", "dash": "dash"},
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=time_s,
                y=data[f"joint_vel_{joint_index}"],
                name=joint_name,
                showlegend=False,
            ),
            row=row,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=time_s,
                y=data[f"joint_tau_est_{joint_index}"],
                name=joint_name,
                showlegend=False,
            ),
            row=row,
            col=3,
        )

        fig.update_yaxes(title_text=joint_name, row=row, col=1)

    for column in range(1, 4):
        fig.update_xaxes(title_text="Time (s)", row=len(joint_indices), col=column)

    fig.update_layout(
        title=f"{group_name}: Joint Diagnostics",
        height=max(500, 190 * len(joint_indices)),
        hovermode="x unified",
    )
    return fig

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and validate a G1 deployment CSV log."
    )
    parser.add_argument("csv_file", type=Path)
    parser.add_argument(
        "--source",
        default="deployment",
        help="Run source label, for example unitree_mujoco or hardware.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/deployment_reports"),
        help="Directory for generated HTML reports.",
    )
        
    args = parser.parse_args()

    data = load_deployment_log(args.csv_file)
    duration_s = data["timestep"].iloc[-1] - data["timestep"].iloc[0]
    timing = sample_period_summary(data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.csv_file.stem}_report.html"

    figures = [
        command_and_yaw_figure(data),
        imu_figure(data),
        *[
            joint_figure(group_name, indices, data)
            for group_name, indices in JOINT_GROUPS.items()
        ],
    ]
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>G1 Deployment Report</title></head><body>",
        "<h1>G1 Deployment Report</h1>",
        f"<p><b>Source:</b> {args.source}</p>",
        f"<p><b>CSV:</b> {args.csv_file.resolve()}</p>",
        (
            f"<p><b>Samples:</b> {len(data)} | "
            f"<b>Duration:</b> {duration_s:.3f} s | "
            f"<b>Sample period:</b> mean {timing['mean_ms']:.3f} ms, "
            f"min {timing['min_ms']:.3f} ms, "
            f"max {timing['max_ms']:.3f} ms</p>"
        ),
        (
            "<p><b>Note:</b> Standard LowState provides no measured base "
            "linear velocity, so vx and vy show commands only. "
            "Torque estimates are not directly equivalent to MuJoCo "
            "actuator forces.</p>"
        ),
    ]

    for index, figure in enumerate(figures):
        html.append(
            figure.to_html(
                full_html=False,
                include_plotlyjs=index == 0,
            )
        )

    html.append("</body></html>")
    output_path.write_text("\n".join(html), encoding="utf-8")

    print(f"[INFO] Source: {args.source}")
    print(f"[INFO] CSV: {args.csv_file.resolve()}")
    print(f"[INFO] Saved report: {output_path.resolve()}")

if __name__ == "__main__":
    main()