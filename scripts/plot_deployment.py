"""Load and validate a G1 deployment CSV log."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

# 27-DoF policy: waist roll/pitch are mechanically locked and not policy joints.
G1_27DOF_JOINT_NAMES = [
    name for name in G1_JOINT_NAMES
    if name not in ("waist_roll_joint", "waist_pitch_joint")
]
JOINT_NAMES_BY_COUNT = {29: G1_JOINT_NAMES, 27: G1_27DOF_JOINT_NAMES}


def joint_groups(joint_names: list[str]) -> dict[str, list[int]]:
    """Group policy-joint indices by body part, using joint names."""
    groups = {"Lower body": [], "Waist": [], "Arms": []}
    for index, name in enumerate(joint_names):
        if name.startswith("waist"):
            groups["Waist"].append(index)
        elif any(part in name for part in ("shoulder", "elbow", "wrist")):
            groups["Arms"].append(index)
        else:
            groups["Lower body"].append(index)
    return groups



def joint_columns(prefix: str, joint_count: int) -> list[str]:
    """Return the expected joint columns in policy-joint order."""
    return [f"{prefix}_{index}" for index in range(joint_count)]


def load_deployment_log(csv_file: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load one deployment CSV and reject incomplete or invalid logs."""
    if not csv_file.is_file():
        raise FileNotFoundError(f"Deployment CSV not found: {csv_file}")

    data = pd.read_csv(csv_file)
    
    joint_count = sum(column.startswith("joint_pos_") for column in data.columns)
    if joint_count not in JOINT_NAMES_BY_COUNT:
        raise ValueError(f"Unsupported joint count {joint_count}; expected 29 or 27.")

    required_columns = [
        *REQUIRED_BASE_COLUMNS,
        *joint_columns("joint_pos", joint_count),
        *joint_columns("joint_vel", joint_count),
        *joint_columns("joint_tau_est", joint_count),
        *joint_columns("target_joint_pos", joint_count),
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

    return data,  JOINT_NAMES_BY_COUNT[joint_count]

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
    joint_names: list[str]
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
        joint_name = joint_names[joint_index]

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

# Adding hand plotting
HAND_AXES = ("Lx", "Ly", "Lz", "Rx", "Ry", "Rz")
G1_27DOF_XML = Path("src/assets/robots/unitree_g1/xmls/g1_27dof.xml")


def has_hand_data(data: pd.DataFrame) -> bool:
    """True if this log came from a hands policy.

    Every log has the hand_* columns, but non-hands policies write NaN there,
    so check for real values, not just for the column names.
    """
    columns = [f"hand_target_{axis}" for axis in HAND_AXES]
    return all(c in data.columns for c in columns) and bool(data[columns].notna().all(axis=1).any())


def actual_palm_positions(
    data: pd.DataFrame, joint_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Palm positions in the heading frame from logged joints + IMU, and pelvis tilt.

    The heading frame removes only the pelvis yaw (same frame as the policy's
    hand command), so the IMU roll/pitch is applied in the forward kinematics.
    Pelvis height is not logged: it is estimated by assuming the lower foot
    stands on flat ground.
    """
    import mujoco  # Imported here so reports for velocity-only logs don't need MuJoCo.

    model = mujoco.MjModel.from_xml_path(str(G1_27DOF_XML))
    mj_data = mujoco.MjData(model)

    def site_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)

    palms = [site_id("left_palm"), site_id("right_palm")]
    feet = [site_id("left_foot"), site_id("right_foot")]

    # Height of the foot site above the ground in the default stance (pelvis at 0.79 m).
    mj_data.qpos[:] = model.qpos0
    mj_data.qpos[2] = 0.79
    mujoco.mj_kinematics(model, mj_data)
    foot_height = mj_data.site_xpos[feet, 2].min()

    joints = data[joint_columns("joint_pos", joint_count)].to_numpy()
    quats = data[["imu_quat_w", "imu_quat_x", "imu_quat_y", "imu_quat_z"]].to_numpy()
    palm_pos = np.zeros((len(data), 6))
    palm_rot = np.zeros((len(data), 2, 3, 3))  # palm orientation in the heading frame
    tilt_deg = np.zeros(len(data))
    rot = np.zeros(9)
    quat_no_yaw = np.zeros(4)
    for k in range(len(data)):
        mujoco.mju_quat2Mat(rot, quats[k])
        r = rot.reshape(3, 3)
        # Remove yaw: rotate back by the pelvis heading, keep roll and pitch.
        yaw = np.arctan2(r[1, 0], r[0, 0])
        c, s = np.cos(-yaw), np.sin(-yaw)
        r_no_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ r
        mujoco.mju_mat2Quat(quat_no_yaw, r_no_yaw.reshape(-1))

        mj_data.qpos[:3] = 0.0  # pelvis at the origin; height added below
        mj_data.qpos[3:7] = quat_no_yaw
        mj_data.qpos[7:7 + joint_count] = joints[k]
        mujoco.mj_kinematics(model, mj_data)
        pelvis_height = foot_height - mj_data.site_xpos[feet, 2].min()
        palm_pos[k] = (mj_data.site_xpos[palms] + [0.0, 0.0, pelvis_height]).reshape(-1)
        # site_xmat is each site's 3x3 rotation, stored row by row (9 numbers).
        palm_rot[k] = mj_data.site_xmat[palms].reshape(2, 3, 3)
        # Tilt = angle between the pelvis z-axis and vertical.
        tilt_deg[k] = np.degrees(np.arccos(np.clip(r_no_yaw[2, 2], -1.0, 1.0)))
    return palm_pos, palm_rot, tilt_deg


HAND_QUAT_AXES = ("Lw", "Lx", "Ly", "Lz", "Rw", "Rx", "Ry", "Rz")


def orientation_error_deg(data: pd.DataFrame, palm_rot: np.ndarray) -> np.ndarray:
    """Angle [deg] between each palm and its logged orientation target, (N, 2).

    All NaN for logs without the hand_target_quat_* columns (older logs and
    policies without an orientation command).
    """
    import mujoco

    error = np.full((len(data), 2), np.nan)
    columns = [f"hand_target_quat_{axis}" for axis in HAND_QUAT_AXES]
    if not all(c in data.columns for c in columns):
        return error
    quats = data[columns].to_numpy()
    target_rot = np.zeros(9)
    for k in range(len(data)):
        for hand in range(2):
            q = quats[k, 4 * hand:4 * hand + 4]
            if np.isnan(q).any():
                continue
            mujoco.mju_quat2Mat(target_rot, q / np.linalg.norm(q))
            # Rotation from the actual palm to the target; its angle is the error.
            r_err = palm_rot[k, hand].T @ target_rot.reshape(3, 3)
            error[k, hand] = np.degrees(np.arccos(np.clip((np.trace(r_err) - 1.0) / 2.0, -1.0, 1.0)))
    return error


def rpy_deg_from_matrices(rot: np.ndarray) -> np.ndarray:
    """Roll, pitch, yaw [deg] (Z-Y-X order) of rotation matrices (N, 3, 3), unwrapped over time.

    Unwrapping removes the jump from +180 to -180 deg so the curves stay continuous.
    Only singular at pitch = +-90 deg (fingers pointing straight up or down).
    """
    roll = np.arctan2(rot[:, 2, 1], rot[:, 2, 2])
    pitch = np.arcsin(np.clip(-rot[:, 2, 0], -1.0, 1.0))
    yaw = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])
    return np.degrees(np.unwrap(np.stack([roll, pitch, yaw], axis=1), axis=0))


def quat_to_matrices(quats: np.ndarray) -> np.ndarray:
    """(N, 4) quaternions w x y z -> (N, 3, 3) rotation matrices (NaN rows stay NaN)."""
    w, x, y, z = (quats / np.linalg.norm(quats, axis=1, keepdims=True)).T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
    ], axis=1)


def has_hand_quat_data(data: pd.DataFrame) -> bool:
    """True if the log has the orientation command (logged since the hand_target_quat_* columns)."""
    columns = [f"hand_target_quat_{axis}" for axis in HAND_QUAT_AXES]
    return all(c in data.columns for c in columns) and bool(data[columns].notna().all(axis=1).any())


def hand_orientation_figure(data: pd.DataFrame, joint_count: int) -> go.Figure:
    """Commanded vs actual palm orientation per hand as roll / pitch / yaw (heading frame)."""
    time_s = data["timestep"]
    _, palm_rot, _ = actual_palm_positions(data, joint_count)
    series = []  # (label, rpy (N, 2, 3), line style)
    for kind, label, line in (
        ("hand_goal_quat", "button goal", dict(color="#52514e", width=1, dash="dash")),
        ("hand_target_quat", "commanded target (policy input)", dict(color="#2a78d6", width=2)),
    ):
        columns = [f"{kind}_{axis}" for axis in HAND_QUAT_AXES]
        if all(c in data.columns for c in columns):
            quats = data[columns].to_numpy().reshape(-1, 2, 4)
            series.append((label, np.stack([rpy_deg_from_matrices(quat_to_matrices(quats[:, h])) for h in (0, 1)], axis=1), line))
    actual = np.stack([rpy_deg_from_matrices(palm_rot[:, h]) for h in (0, 1)], axis=1)
    series.append(("actual palm (FK + IMU)", actual, dict(color="#eb6834", width=2)))

    fig = make_subplots(
        rows=3, cols=2, shared_xaxes=True, vertical_spacing=0.06, horizontal_spacing=0.08,
        subplot_titles=["Left palm roll", "Right palm roll", "Left palm pitch", "Right palm pitch",
                        "Left palm yaw", "Right palm yaw"],
    )
    for hand in range(2):
        for axis, name in enumerate(("roll", "pitch", "yaw")):
            for label, values, line in series:
                fig.add_trace(
                    go.Scatter(x=time_s, y=values[:, hand, axis], name=label, line=line, legendgroup=label,
                               showlegend=(hand == 0 and axis == 0)),
                    row=axis + 1, col=hand + 1,
                )
            fig.update_yaxes(title_text=f"{name} [deg]", row=axis + 1, col=hand + 1)
    fig.update_xaxes(title_text="time [s]", row=3, col=1)
    fig.update_xaxes(title_text="time [s]", row=3, col=2)
    fig.update_layout(
        title="Hand orientation (heading frame, roll-pitch-yaw; actual = FK from logged joints + IMU)",
        height=950, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def hand_tracking_figure(data: pd.DataFrame, joint_count: int) -> go.Figure:
    """Commanded vs actual palm position per hand and axis, error, and pelvis tilt."""
    time_s = data["timestep"]
    goal = data[[f"hand_goal_{axis}" for axis in HAND_AXES]].to_numpy()
    target = data[[f"hand_target_{axis}" for axis in HAND_AXES]].to_numpy()
    actual, palm_rot, tilt_deg = actual_palm_positions(data, joint_count)
    rot_error_deg = orientation_error_deg(data, palm_rot)
    error_cm = 100.0 * np.stack(
        [np.linalg.norm(actual[:, 3 * h:3 * h + 3] - target[:, 3 * h:3 * h + 3], axis=1)
         for h in (0, 1)],
        axis=1,
    )

    fig = make_subplots(
        rows=6,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.08,
        # Last row: one wide tilt plot across both columns.
        specs=[[{}, {}]] * 5 + [[{"colspan": 2}, None]],
        subplot_titles=[
            "Left palm x", "Right palm x",
            "Left palm y", "Right palm y",
            "Left palm z", "Right palm z",
            "Left tracking error", "Right tracking error",
            "Left orientation error", "Right orientation error",
            "Pelvis tilt",
        ],
    )
    series = [  # (label, data, line style); colors match across all panels
        ("button goal", goal, dict(color="#52514e", width=1, dash="dash")),
        ("commanded target (policy input)", target, dict(color="#2a78d6", width=2)),
        ("actual palm (FK + IMU)", actual, dict(color="#eb6834", width=2)),
    ]
    for hand in range(2):
        for axis in range(3):
            column = 3 * hand + axis
            for label, values, line in series:
                fig.add_trace(
                    go.Scatter(
                        x=time_s, y=values[:, column], name=label, line=line,
                        legendgroup=label,
                        showlegend=(hand == 0 and axis == 0),  # one legend entry per series
                    ),
                    row=axis + 1, col=hand + 1,
                )
            fig.update_yaxes(title_text=f"{'xyz'[axis]} [m]", row=axis + 1, col=hand + 1)
        fig.add_trace(
            go.Scatter(x=time_s, y=error_cm[:, hand], name="tracking error",
                       showlegend=False, line=dict(color="#0b0b0b", width=1.5)),
            row=4, col=hand + 1,
        )
        fig.update_yaxes(title_text="error [cm]", row=4, col=hand + 1)
        fig.add_trace(
            go.Scatter(x=time_s, y=rot_error_deg[:, hand], name="orientation error",
                       showlegend=False, line=dict(color="#0b0b0b", width=1.5)),
            row=5, col=hand + 1,
        )
        fig.update_yaxes(title_text="error [deg]", row=5, col=hand + 1)

    fig.add_trace(
        go.Scatter(x=time_s, y=tilt_deg, name="pelvis tilt", showlegend=False,
                   line=dict(color="#0b0b0b", width=1.5)),
        row=6, col=1,
    )
    fig.add_hline(y=15.0, line=dict(color="#52514e", width=1, dash="dash"),
                  annotation_text="safety trip 15°", row=6, col=1)
    fig.update_yaxes(title_text="tilt [deg]", row=6, col=1)
    fig.update_xaxes(title_text="time [s]", row=6, col=1)
    fig.update_layout(
        title="Hand tracking (heading frame; actual = FK from logged joints + IMU)",
        height=1550,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
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

    data, joint_names = load_deployment_log(args.csv_file)
    duration_s = data["timestep"].iloc[-1] - data["timestep"].iloc[0]
    timing = sample_period_summary(data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.csv_file.stem}_report.html"

    figures = [
        command_and_yaw_figure(data),
        imu_figure(data),
        *[
            joint_figure(group_name, indices, data, joint_names)
            for group_name, indices in joint_groups(joint_names).items()
        ],
    ]
    # Hands policies log their hand command; add the tracking section only then.
    if len(joint_names) == 27 and has_hand_data(data):
        figures.append(hand_tracking_figure(data, len(joint_names)))
        # Orientation command is logged since the hand_*_quat columns were added.
        if has_hand_quat_data(data):
            figures.append(hand_orientation_figure(data, len(joint_names)))

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