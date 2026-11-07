"""Compare commands with mocap torso and feet from a G1 deployment CSV log."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plot_deployment import load_deployment_log


BODIES = {"Torso": 0, "Left foot": 1, "Right foot": 2}
COMMAND_COLUMNS = ["command_vx", "command_vy", "command_yaw_rate"]
WAIST_YAW_COLUMN = "joint_pos_12"  # 27-DoF policy joint order.
GYRO_Z_COLUMN = "root_com_ang_vel_b_z"

# (label, command column, measured column, off-axis column)
MOTION_AXES = (
    ("Forward (m/s)", "command_vx", "forward", "lateral"),
    ("Sideways (m/s)", "command_vy", "lateral", "forward"),
    ("Yaw (rad/s)", "command_yaw_rate", "yaw_rate", None),
)

LOG_PERIOD_S = 0.02
STRIDE_S = 0.6  # Averaging over one stride cancels walking sway.
SETTLE_S = 2.0  # A command must hold this long before it counts as steady.
MAX_COMMAND_STD = 0.03
MOVING_COMMAND = 0.1  # The policy clock stops below this command norm.
PURE_COMMAND = 0.05  # Other command axes must stay below this.

# Limits for rejecting poses that cannot be real.
MAX_TORSO_HEIGHT_ERROR_M = 0.06
MAX_TORSO_TILT_ERROR_DEG = 12.0
MAX_TORSO_TRACK_ERROR_M = 0.12
MAX_FOOT_HEIGHT_M = 0.25
MAX_FOOT_REACH_M = 0.7

# Foot height above stance that marks a swing and a landing. Slow walking barely
# lifts the feet, so these miss some steps, but they rarely invent one.
SWING_CLEARANCE_M = 0.012
LANDED_CLEARANCE_M = 0.006

OK, STALE, WRONG = 0, 1, 2

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
COLORS = {
    "Commanded": SECONDARY_INK,
    "Mocap": "#2a78d6",
    "Left foot": "#eb6834",
    "Right foot": "#1baf7a",
}
STALE_COLOR = "#fab219"
WRONG_COLOR = "#d03b3b"

PAGE_STYLE = f"""
<style>
  body {{
    background: {SURFACE}; color: {INK}; margin: 24px;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; }}
  th, td {{ padding: 4px 16px 4px 0; text-align: right; border-bottom: 1px solid {GRID}; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: {SECONDARY_INK}; font-weight: 600; }}
  p {{ max-width: 80ch; }}
</style>
"""


@dataclass
class Body:
    pos: np.ndarray
    quat: np.ndarray
    status: np.ndarray


def load_body(data: pd.DataFrame, index: int) -> Body:
    """Load one mocap body and mark rows where its sequence did not advance."""
    prefix = f"mocap_packet_{index}_"
    if f"{prefix}seq" not in data.columns:
        raise ValueError(f"CSV has no mocap columns for body {index}.")

    seq = data[f"{prefix}seq"].to_numpy()
    fresh = np.r_[True, np.diff(seq) > 0] & (seq > 0)
    return Body(
        pos=data[[f"{prefix}pos_{axis}" for axis in "xyz"]].to_numpy(),
        quat=data[[f"{prefix}quat_{axis}" for axis in "wxyz"]].to_numpy(),
        status=np.where(fresh, OK, STALE),
    )


def yaw_of(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat.T
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def tilt_deg_of(quat: np.ndarray) -> np.ndarray:
    _, x, y, _ = quat.T
    return np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return start and end rows of each contiguous True stretch."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return edges[::2], edges[1::2]


def mark_wrong_torso(torso: Body) -> None:
    """Flag poses that cannot be the torso, such as a solve onto foot markers."""
    height = torso.pos[:, 2]
    tilt = tilt_deg_of(torso.quat)
    plausible = (np.abs(height - np.median(height)) < MAX_TORSO_HEIGHT_ERROR_M) & (
        np.abs(tilt - np.median(tilt)) < MAX_TORSO_TILT_ERROR_DEG
    )
    track = (
        pd.DataFrame(np.where(plausible[:, None], torso.pos[:, :2], np.nan))
        .rolling(51, center=True, min_periods=5)
        .median()
        .to_numpy()
    )
    on_track = np.linalg.norm(torso.pos[:, :2] - track, axis=1) < MAX_TORSO_TRACK_ERROR_M

    # A bad pose stays wrong while it is held, so WRONG overrides STALE.
    torso.status[~(plausible & on_track)] = WRONG


def mark_wrong_foot(foot: Body, torso: Body) -> None:
    """Flag poses that are too high or out of reach of the torso."""
    reach = np.linalg.norm(foot.pos[:, :2] - torso.pos[:, :2], axis=1)
    out_of_reach = (reach > MAX_FOOT_REACH_M) & (torso.status == OK)
    foot.status[(foot.pos[:, 2] > MAX_FOOT_HEIGHT_M) | out_of_reach] = WRONG


def pelvis_heading(torso: Body, waist_yaw: np.ndarray) -> np.ndarray:
    """Return unwrapped pelvis yaw, because commands are in the pelvis frame."""
    rows = np.arange(len(waist_yaw))
    ok = torso.status == OK
    return np.interp(rows, rows[ok], np.unwrap(yaw_of(torso.quat[ok]))) - waist_yaw


def standing_yaw_offset(
    time: np.ndarray, commands: pd.DataFrame, bodies: dict[str, Body], heading: np.ndarray
) -> float | None:
    """Compare heading with the line between the feet before the first command."""
    moving = np.linalg.norm(commands.to_numpy(), axis=1) >= MOVING_COMMAND
    first_move = time[np.argmax(moving)] if moving.any() else time[-1]
    left, right = bodies["Left foot"], bodies["Right foot"]
    rows = np.flatnonzero(
        (time > 3.0)
        & (time < first_move - 0.5)
        & (bodies["Torso"].status == OK)
        & (left.status == OK)
        & (right.status == OK)
    )
    if len(rows) < 100:
        return None

    across = left.pos[rows, :2] - right.pos[rows, :2]
    feet_heading = np.arctan2(across[:, 1], across[:, 0]) - np.pi / 2.0
    return float(np.angle(np.mean(np.exp(1j * (heading[rows] - feet_heading)))))


def stride_motion(time: np.ndarray, torso: Body, heading: np.ndarray) -> pd.DataFrame:
    """Return pelvis-frame velocity over one stride, NaN where tracking is poor."""
    half = round(STRIDE_S / 2.0 / np.median(np.diff(time)))
    clean = torso.status == OK
    mostly_clean = np.convolve(clean, np.ones(2 * half + 1) / (2 * half + 1), "same") > 0.9
    rows = np.arange(half, len(time) - half)
    rows = rows[clean[rows - half] & clean[rows + half] & mostly_clean[rows]]

    elapsed = time[rows + half] - time[rows - half]
    world = (torso.pos[rows + half, :2] - torso.pos[rows - half, :2]) / elapsed[:, None]
    cos_h, sin_h = np.cos(heading[rows]), np.sin(heading[rows])

    motion = pd.DataFrame(
        np.nan, index=range(len(time)), columns=["forward", "lateral", "yaw_rate"]
    )
    motion.loc[rows, "forward"] = cos_h * world[:, 0] + sin_h * world[:, 1]
    motion.loc[rows, "lateral"] = -sin_h * world[:, 0] + cos_h * world[:, 1]
    motion.loc[rows, "yaw_rate"] = (heading[rows + half] - heading[rows - half]) / elapsed
    return motion


def steady_rows(time: np.ndarray, commands: pd.DataFrame) -> pd.Series:
    """Mark rows whose stride window sits inside a held command."""
    period = np.median(np.diff(time))
    window = round((SETTLE_S + STRIDE_S) / period)
    half = round(STRIDE_S / 2.0 / period)
    held = commands.rolling(window).std().max(axis=1) < MAX_COMMAND_STD
    return held.shift(-half, fill_value=False)


def steady_response(
    commands: pd.DataFrame, motion: pd.DataFrame, steady: pd.Series
) -> pd.DataFrame:
    """Average the achieved motion for each held command, one axis at a time."""
    rows = []
    for label, column, measured, off_axis in MOTION_AXES:
        pure = commands.drop(columns=column).abs().max(axis=1) < PURE_COMMAND
        selected = (
            steady
            & pure
            & (commands[column].abs() >= MOVING_COMMAND)
            & motion[measured].notna()
        )
        bins = (commands.loc[selected, column] / 0.1).round() * 0.1
        for _, group in motion[selected].groupby(bins):
            if len(group) < 50:
                continue
            commanded = commands.loc[group.index, column].mean()
            achieved = group[measured].mean()
            rows.append(
                {
                    "Motion": label,
                    "Commanded": commanded,
                    "Achieved": achieved,
                    "Ratio": achieved / commanded,
                    "Off-axis": group[off_axis].mean() if off_axis else np.nan,
                    "Data (s)": len(group) * LOG_PERIOD_S,
                }
            )
    return pd.DataFrame(rows)


def find_touchdowns(foot: Body, moving: np.ndarray) -> np.ndarray:
    """Return rows where a swinging foot comes back down to stance height."""
    ok = foot.status == OK
    height = foot.pos[:, 2]
    stance = np.percentile(height[ok & moving], 25)
    touchdowns, in_swing = [], False
    for row in np.flatnonzero(ok):
        if height[row] > stance + SWING_CLEARANCE_M:
            in_swing = True
        elif in_swing and height[row] < stance + LANDED_CLEARANCE_M:
            touchdowns.append(row)
            in_swing = False
    return np.array(touchdowns, dtype=int)


def foot_offsets(foot: Body, torso: Body, heading: np.ndarray) -> np.ndarray:
    """Return foot forward and lateral offset from the torso in the pelvis frame."""
    delta = foot.pos[:, :2] - torso.pos[:, :2]
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    offsets = np.column_stack(
        (cos_h * delta[:, 0] + sin_h * delta[:, 1], -sin_h * delta[:, 0] + cos_h * delta[:, 1])
    )
    offsets[(foot.status != OK) | (torso.status != OK)] = np.nan
    return offsets


def yaw_rate_check(
    time: np.ndarray, heading: np.ndarray, gyro_z: np.ndarray, torso: Body, moving: np.ndarray
) -> float:
    """Correlate mocap pelvis yaw rate with the IMU gyro while walking."""
    half = 3
    width = 2 * half + 1
    rate = (heading[2 * half:] - heading[: -2 * half]) / (time[2 * half:] - time[: -2 * half])
    gyro = np.convolve(gyro_z, np.ones(width) / width, "valid")
    clean = np.convolve(torso.status == OK, np.ones(width), "valid") == width
    selected = clean & moving[half:-half]
    return float(np.corrcoef(rate[selected], gyro[selected])[0, 1])


def quality_summary(
    time: np.ndarray, bodies: dict[str, Body], moving: np.ndarray
) -> pd.DataFrame:
    """Summarize tracking per body."""
    period = np.median(np.diff(time))
    rows = []
    for name, body in bodies.items():
        starts, ends = runs(body.status == STALE)
        rows.append(
            {
                "Body": name,
                "OK (%)": 100.0 * np.mean(body.status == OK),
                "OK walking (%)": 100.0 * np.mean(body.status[moving] == OK),
                "Stale (%)": 100.0 * np.mean(body.status == STALE),
                "Wrong (%)": 100.0 * np.mean(body.status == WRONG),
                "Longest stale (s)": (ends - starts).max(initial=0) * period,
            }
        )
    return pd.DataFrame(rows)


def foot_summary(
    time: np.ndarray,
    bodies: dict[str, Body],
    touchdowns: dict[str, np.ndarray],
    placements: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize gait timing and foot placement per foot."""
    rows = []
    for name, rows_down in touchdowns.items():
        foot = bodies[name]
        height = foot.pos[foot.status == OK, 2]
        intervals = np.diff(time[rows_down])
        placement = placements[name]
        slope, intercept = (
            np.polyfit(placement["command"], placement["forward"], 1)
            if len(placement) >= 10
            else (np.nan, np.nan)
        )
        rows.append(
            {
                "Foot": name,
                "Touchdowns": len(rows_down),
                "Median stride (s)": np.median(intervals[intervals < 1.5]),
                "Peak lift (cm)": 100.0 * (np.percentile(height, 99) - np.percentile(height, 25)),
                "Placement at 0 (m)": intercept,
                "Placement slope (s)": slope,
            }
        )
    return pd.DataFrame(rows)


def touchdown_placements(
    rows_down: np.ndarray, offsets: np.ndarray, commands: pd.DataFrame, steady: pd.Series
) -> pd.DataFrame:
    """Return forward foot placement at touchdown during steady straight walking."""
    straight = commands[["command_vy", "command_yaw_rate"]].abs().max(axis=1) < PURE_COMMAND
    selected = rows_down[(steady & straight).to_numpy()[rows_down]]
    selected = selected[~np.isnan(offsets[selected, 0])]
    return pd.DataFrame(
        {"command": commands["command_vx"].to_numpy()[selected], "forward": offsets[selected, 0]}
    )


def masked(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Replace rejected rows with NaN so plots show a gap, not a held value."""
    return np.where(keep, values, np.nan)


def line(time: np.ndarray, values: np.ndarray, name: str, show_legend: bool, width: float = 2.0):
    return go.Scattergl(
        x=time,
        y=values,
        mode="lines",
        name=name,
        legendgroup=name,
        showlegend=show_legend,
        line=dict(color=COLORS[name], width=width),
    )


def style(figure: go.Figure, title: str, height: int) -> go.Figure:
    """Apply the shared report look."""
    figure.update_layout(
        title=dict(text=title, x=0.0, xanchor="left"),
        height=height,
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(
            family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK, size=13
        ),
        legend=dict(orientation="h", x=1.0, xanchor="right", y=1.0, yanchor="bottom"),
        margin=dict(l=64, r=24, t=72, b=48),
    )
    figure.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS)
    figure.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS)
    return figure


def status_figure(time: np.ndarray, bodies: dict[str, Body]) -> go.Figure:
    """Show when each body was tracked, held, or wrong."""
    bins = np.arange(0, len(time), 10)
    names = list(bodies)
    status = np.array([np.maximum.reduceat(bodies[name].status, bins) for name in names])
    labels = np.array(["OK", "Stale", "Wrong"])[status]
    figure = go.Figure(
        go.Heatmap(
            x=time[bins],
            y=names,
            z=status,
            zmin=0,
            zmax=2,
            text=labels,
            hovertemplate="%{y}, t=%{x:.1f} s: %{text}<extra></extra>",
            colorscale=[
                [0.0, GRID],
                [1 / 3, GRID],
                [1 / 3, STALE_COLOR],
                [2 / 3, STALE_COLOR],
                [2 / 3, WRONG_COLOR],
                [1.0, WRONG_COLOR],
            ],
            colorbar=dict(
                tickvals=[1 / 3, 1, 5 / 3], ticktext=["OK", "Stale", "Wrong"], len=0.9, thickness=12
            ),
        )
    )
    figure.update_xaxes(title_text="Time (s)")
    return style(figure, "Tracking status", 220)


def velocity_figure(
    time: np.ndarray, commands: pd.DataFrame, motion: pd.DataFrame
) -> go.Figure:
    """Plot commanded velocity against stride-averaged mocap velocity."""
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[axis[0] for axis in MOTION_AXES],
    )
    for row, (_, column, measured, _) in enumerate(MOTION_AXES, start=1):
        show = row == 1
        figure.add_trace(line(time, commands[column], "Commanded", show), row=row, col=1)
        figure.add_trace(line(time, motion[measured], "Mocap", show), row=row, col=1)
    figure.update_xaxes(title_text="Time (s)", row=3, col=1)
    figure.update_layout(hovermode="x unified")
    return style(figure, "Commanded vs mocap velocity (pelvis frame, one-stride average)", 720)


def response_figure(
    commands: pd.DataFrame, motion: pd.DataFrame, steady: pd.Series, response: pd.DataFrame
) -> go.Figure:
    """Plot achieved motion against each held command, with a perfect-tracking line."""
    figure = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.07,
        subplot_titles=[axis[0] for axis in MOTION_AXES],
    )
    for col, (label, column, measured, _) in enumerate(MOTION_AXES, start=1):
        show = col == 1
        pure = commands.drop(columns=column).abs().max(axis=1) < PURE_COMMAND
        selected = steady & pure & (commands[column].abs() >= MOVING_COMMAND)
        selected = selected.to_numpy(copy=True)
        selected[::2] = False  # Stride windows overlap heavily, so thin them.
        x, y = commands[column].to_numpy()[selected], motion[measured].to_numpy()[selected]
        figure.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                name="Stride windows",
                legendgroup="windows",
                showlegend=show,
                marker=dict(color=COLORS["Mocap"], size=5, opacity=0.25),
            ),
            row=1,
            col=col,
        )
        means = response[response["Motion"] == label]
        figure.add_trace(
            go.Scatter(
                x=means["Commanded"],
                y=means["Achieved"],
                mode="markers",
                name="Mean per command",
                legendgroup="means",
                showlegend=show,
                marker=dict(color=INK, size=9, line=dict(color=SURFACE, width=2)),
            ),
            row=1,
            col=col,
        )
        ends = [min(x.min(), 0.0), max(x.max(), 0.0)] if len(x) else [-1.0, 1.0]
        figure.add_trace(
            go.Scatter(
                x=ends,
                y=ends,
                mode="lines",
                name="Perfect tracking",
                legendgroup="perfect",
                showlegend=show,
                line=dict(color=AXIS, width=2),
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )
        figure.update_xaxes(title_text="Commanded", row=1, col=col)
    figure.update_yaxes(title_text="Achieved", row=1, col=1)
    style(figure, "Steady response", 460)
    # Subplot titles sit where the shared legend normally goes.
    figure.update_layout(
        legend=dict(x=0.5, xanchor="center", y=-0.2, yanchor="top"),
        margin=dict(b=96),
    )
    return figure


def floor_figure(time: np.ndarray, torso: Body) -> go.Figure:
    """Show the torso path and where tracking was lost."""
    ok = torso.status == OK
    last_ok = np.maximum.accumulate(np.where(ok, np.arange(len(ok)), 0))
    figure = go.Figure(
        go.Scattergl(
            x=masked(torso.pos[:, 0], ok),
            y=masked(torso.pos[:, 1], ok),
            mode="lines",
            name="Torso path",
            line=dict(color=COLORS["Mocap"], width=1.5),
        )
    )
    for status, name, color, symbol in (
        (STALE, "Lost tracking", STALE_COLOR, "circle"),
        (WRONG, "Wrong pose", WRONG_COLOR, "x"),
    ):
        starts, ends = runs(torso.status == status)
        long_enough = (time[ends - 1] - time[starts]) >= 0.2
        spots = last_ok[starts[long_enough]]
        figure.add_trace(
            go.Scatter(
                x=torso.pos[spots, 0],
                y=torso.pos[spots, 1],
                mode="markers",
                name=name,
                text=[f"t={time[row]:.1f} s" for row in spots],
                hovertemplate="%{text}<extra>" + name + "</extra>",
                marker=dict(color=color, size=9, symbol=symbol, line=dict(color=INK, width=1)),
            )
        )
    figure.update_xaxes(title_text="Room x (m)")
    figure.update_yaxes(title_text="Room y (m)", scaleanchor="x", scaleratio=1.0)
    return style(figure, "Where the robot walked, and where tracking failed", 560)


def zoom_window(time: np.ndarray, usable: np.ndarray, length_s: float = 8.0) -> list[float]:
    """Pick a short window in the longest clean walking stretch."""
    starts, ends = runs(usable)
    if len(starts) == 0:
        return [float(time[0]), float(time[0] + length_s)]
    longest = np.argmax(ends - starts)
    start = float(time[starts[longest]])
    return [start, start + length_s]


def foot_height_figure(
    time: np.ndarray, bodies: dict[str, Body], window: list[float]
) -> go.Figure:
    figure = go.Figure()
    for name in ("Left foot", "Right foot"):
        foot = bodies[name]
        figure.add_trace(
            line(time, masked(foot.pos[:, 2], foot.status == OK), name, True, width=1.5)
        )
    figure.update_xaxes(title_text="Time (s)", range=window)
    figure.update_yaxes(title_text="Height (m)")
    figure.update_layout(hovermode="x unified")
    return style(figure, "Foot height (double-click to see the whole run)", 320)


def foot_offset_figure(
    time: np.ndarray, offsets: dict[str, np.ndarray], window: list[float]
) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Forward of torso (m)", "Left of torso (m)"),
    )
    for name, values in offsets.items():
        figure.add_trace(line(time, values[:, 0], name, True, width=1.5), row=1, col=1)
        figure.add_trace(line(time, values[:, 1], name, False, width=1.5), row=2, col=1)
    figure.update_xaxes(title_text="Time (s)", range=window, row=2, col=1)
    figure.update_layout(hovermode="x unified")
    return style(figure, "Foot position relative to the torso (double-click for the whole run)", 480)


def placement_figure(placements: dict[str, pd.DataFrame]) -> go.Figure:
    figure = go.Figure()
    for name, placement in placements.items():
        figure.add_trace(
            go.Scatter(
                x=placement["command"],
                y=placement["forward"],
                mode="markers",
                name=name,
                marker=dict(color=COLORS[name], size=8, opacity=0.6),
            )
        )
    figure.update_xaxes(title_text="Commanded forward speed (m/s)")
    figure.update_yaxes(title_text="Foot forward of torso at touchdown (m)")
    return style(figure, "Foot placement vs command (steady straight walking)", 420)


def table_html(table: pd.DataFrame) -> str:
    return table.to_html(index=False, border=0, na_rep="", float_format="{:.3f}".format)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare commands with mocap torso and feet from a deployment CSV log."
    )
    parser.add_argument("csv_file", type=Path)
    parser.add_argument(
        "--source",
        default="deployment",
        help="Run source label, for example unitree_mujoco or hardware.",
    )
    parser.add_argument(
        "--yaw-offset-deg",
        type=float,
        default=None,
        help="Torso frame yaw offset. Default: estimated from the feet before the first command.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/deployment_reports"),
        help="Directory for generated HTML reports.",
    )
    args = parser.parse_args()

    data, _ = load_deployment_log(args.csv_file)
    time = data["timestep"].to_numpy()
    commands = data[COMMAND_COLUMNS]
    moving = np.linalg.norm(commands.to_numpy(), axis=1) >= MOVING_COMMAND

    bodies = {name: load_body(data, index) for name, index in BODIES.items()}
    torso = bodies["Torso"]
    mark_wrong_torso(torso)
    if not np.any(torso.status == OK):
        raise ValueError("CSV has no usable torso mocap samples.")
    mark_wrong_foot(bodies["Left foot"], torso)
    mark_wrong_foot(bodies["Right foot"], torso)

    heading = pelvis_heading(torso, data[WAIST_YAW_COLUMN].to_numpy())
    estimated = standing_yaw_offset(time, commands, bodies, heading)
    if args.yaw_offset_deg is not None:
        yaw_offset = np.radians(args.yaw_offset_deg)
    else:
        yaw_offset = estimated if estimated is not None else 0.0
    heading = heading - yaw_offset

    motion = stride_motion(time, torso, heading)
    steady = steady_rows(time, commands)
    response = steady_response(commands, motion, steady)
    feet = ("Left foot", "Right foot")
    offsets = {name: foot_offsets(bodies[name], torso, heading) for name in feet}
    touchdowns = {name: find_touchdowns(bodies[name], moving) for name in feet}
    placements = {
        name: touchdown_placements(touchdowns[name], offsets[name], commands, steady)
        for name in feet
    }
    quality = quality_summary(time, bodies, moving)
    gait = foot_summary(time, bodies, touchdowns, placements)
    yaw_correlation = yaw_rate_check(
        time, heading, data[GYRO_Z_COLUMN].to_numpy(), torso, moving
    )

    all_ok = np.all([body.status == OK for body in bodies.values()], axis=0)
    window = zoom_window(time, all_ok & moving)
    figures = [
        status_figure(time, bodies),
        velocity_figure(time, commands, motion),
        response_figure(commands, motion, steady, response),
        floor_figure(time, torso),
        foot_height_figure(time, bodies, window),
        foot_offset_figure(time, offsets, window),
        placement_figure(placements),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.csv_file.stem}_mocap_report.html"
    offset_note = (
        f"{np.degrees(yaw_offset):.2f} deg"
        if args.yaw_offset_deg is not None or estimated is not None
        else "not estimated, assumed 0 deg"
    )
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>G1 Mocap Report</title>{PAGE_STYLE}</head><body>",
        "<h1>G1 Mocap Report</h1>",
        f"<p><b>Source:</b> {args.source}</p>",
        f"<p><b>CSV:</b> {args.csv_file.resolve()}</p>",
        (
            f"<p><b>Samples:</b> {len(data)} | "
            f"<b>Duration:</b> {time[-1] - time[0]:.1f} s | "
            f"<b>Walking:</b> {moving.sum() * LOG_PERIOD_S:.1f} s | "
            f"<b>Yaw offset:</b> {offset_note} | "
            f"<b>Mocap yaw rate vs IMU gyro:</b> r = {yaw_correlation:.2f}</p>"
        ),
        (
            "<p><b>Stale</b> means the sequence number did not advance, so the "
            "row repeats an old pose. <b>Wrong</b> means a new or held pose "
            "that cannot be real, such as the torso solved onto foot markers. "
            "Both are removed from every plot and table. Rigid-body frames are "
            "not calibrated against the robot yet, so foot positions are "
            "relative to the torso marker reference points, and the yaw offset "
            "is a rough estimate from the line between the feet.</p>"
        ),
        "<h2>Tracking</h2>",
        table_html(quality),
        "<h2>Steady response</h2>",
        table_html(response),
        "<h2>Feet</h2>",
        table_html(gait),
    ]
    for index, figure in enumerate(figures):
        html.append(figure.to_html(full_html=False, include_plotlyjs=index == 0))
    html.append("</body></html>")
    output_path.write_text("\n".join(html), encoding="utf-8")

    print(f"[INFO] Source: {args.source}")
    print(f"[INFO] Yaw offset: {offset_note}")
    print(f"[INFO] Mocap yaw rate vs IMU gyro: r = {yaw_correlation:.2f}")
    print(quality.round(3).to_string(index=False))
    print(response.round(3).to_string(index=False))
    print(gait.round(3).to_string(index=False))
    print(f"[INFO] Saved report: {output_path.resolve()}")


if __name__ == "__main__":
    main()
