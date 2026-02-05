#!/usr/bin/env python3
"""
Coordinate Frame Visualization for Grasp System
================================================
Visualizes:
1. Robot base (world coordinate system)
2. Camera coordinate system (using extrinsics + EE pose)
3. End effector pose (from Franky)
4. Allegro Hand palm center (from URDF)

Usage:
  python visualize_coordinate_frames.py [--eye-in-hand] [--robot-ip IP]
"""

import numpy as np
import argparse
import os
import sys
from typing import Optional, List, Tuple

# Visualization imports
import plotly.graph_objects as go

# Franky imports
from franky import Robot

# ============================================================================
# Configuration
# ============================================================================
DEFAULT_ROBOT_IP = "172.16.1.22"
CALIBRATION_FILE_EYE_TO_HAND = "./calibration_results/eye_to_hand_calibration.npz"
CALIBRATION_FILE_EYE_ON_HAND = "./calibration_results/eye_on_hand_calibration.npz"

# Flange to Palm Transformation (same as grasp_client.py)
FLANGE_TO_PALM_TRANSLATION = [0.026, -0.026, 0.07]
FLANGE_TO_PALM_ROTATION_DEG = [0, -90, 135]

# URDF path
ALLEGRO_HAND_URDF = "./dataset/urdf/allegro_hand_description/allegro_hand_description_right.urdf"

# ============================================================================
# Math Helper Functions
# ============================================================================
def euler_to_matrix(r_deg, p_deg, y_deg):
    """Convert Euler angles (degrees, XYZ sequence) to 3x3 Rotation Matrix."""
    r = np.radians(r_deg)
    p = np.radians(p_deg)
    y = np.radians(y_deg)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r), np.cos(r)]])

    Ry = np.array([[np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])

    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y), np.cos(y), 0],
                   [0, 0, 1]])

    return Rz @ Ry @ Rx


def get_flange_to_palm_transform():
    """Constructs the static transform T_flange_palm from configuration."""
    T = np.eye(4)
    R = euler_to_matrix(*FLANGE_TO_PALM_ROTATION_DEG)
    T[:3, :3] = R
    T[:3, 3] = FLANGE_TO_PALM_TRANSLATION
    return T


# ============================================================================
# Visualization Functions
# ============================================================================
def create_axis_arrow(origin, direction, color, name, size=0.1):
    """
    Create a 3D arrow representing a coordinate axis.

    Args:
        origin: Starting point [x, y, z]
        direction: Direction vector [x, y, z]
        color: RGB color string (e.g., 'red', 'green', 'blue')
        name: Label for the axis
        size: Length of the axis

    Returns:
        go.Scatter3d: Plotly trace for the axis arrow
    """
    end = origin + direction * size

    # Create line from origin to end
    x = [origin[0], end[0]]
    y = [origin[1], end[1]]
    z = [origin[2], end[2]]

    trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+text',
        line=dict(color=color, width=6),
        name=name,
        text=['', name[-1]],  # Label at the end (X, Y, or Z)
        textposition='top center',
        showlegend=True
    )

    return trace


def create_coordinate_frame_traces(transform, label, size=0.1):
    """
    Create traces for a coordinate frame (X, Y, Z axes) at a given transformation.

    Args:
        transform: 4x4 transformation matrix
        label: String label for the frame
        size: Size of the coordinate frame axes

    Returns:
        List[go.Scatter3d]: List of Plotly traces for the frame
    """
    origin = transform[:3, 3]
    R = transform[:3, :3]

    # Extract the transformed axis directions
    x_axis = R[:, 0]  # First column = X direction
    y_axis = R[:, 1]  # Second column = Y direction
    z_axis = R[:, 2]  # Third column = Z direction

    traces = []

    # X axis - Red
    traces.append(create_axis_arrow(
        origin, x_axis, 'red', f'{label}-X', size
    ))

    # Y axis - Green
    traces.append(create_axis_arrow(
        origin, y_axis, 'green', f'{label}-Y', size
    ))

    # Z axis - Blue
    traces.append(create_axis_arrow(
        origin, z_axis, 'blue', f'{label}-Z', size
    ))

    # Add a marker at the origin with label
    origin_marker = go.Scatter3d(
        x=[origin[0]],
        y=[origin[1]],
        z=[origin[2]],
        mode='markers+text',
        marker=dict(size=8, color='black'),
        text=[label],
        textposition='top center',
        name=label,
        showlegend=True
    )
    traces.append(origin_marker)

    return traces


def create_line_trace(p1, p2, color='gray', name='Connection', width=3):
    """
    Create a line trace between two points.

    Args:
        p1: Start point [x, y, z]
        p2: End point [x, y, z]
        color: Line color
        name: Trace name
        width: Line width

    Returns:
        go.Scatter3d: Plotly trace for the line
    """
    trace = go.Scatter3d(
        x=[p1[0], p2[0]],
        y=[p1[1], p2[1]],
        z=[p1[2], p2[2]],
        mode='lines',
        line=dict(color=color, width=width, dash='dash'),
        name=name,
        showlegend=True
    )
    return trace


def create_ground_plane(size=1.0, grid_step=0.1):
    """
    Create a ground plane grid.

    Args:
        size: Size of the plane
        grid_step: Step size for grid lines

    Returns:
        List[go.Scatter3d]: List of traces for grid lines
    """
    traces = []

    # Create grid lines parallel to X axis
    for y in np.arange(-size/2, size/2 + grid_step, grid_step):
        trace = go.Scatter3d(
            x=[-size/2, size/2],
            y=[y, y],
            z=[0, 0],
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            hoverinfo='skip'
        )
        traces.append(trace)

    # Create grid lines parallel to Y axis
    for x in np.arange(-size/2, size/2 + grid_step, grid_step):
        trace = go.Scatter3d(
            x=[x, x],
            y=[-size/2, size/2],
            z=[0, 0],
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            hoverinfo='skip'
        )
        traces.append(trace)

    return traces


def load_calibration(calibration_file, eye_in_hand):
    """
    Load calibration transform.

    Args:
        calibration_file: Path to calibration .npz file
        eye_in_hand: Whether using eye-in-hand configuration

    Returns:
        np.ndarray or None: Calibration transform matrix
    """
    if not os.path.exists(calibration_file):
        print(f"Warning: Calibration file {calibration_file} not found.")
        return None

    try:
        data = np.load(calibration_file)
        print(f"Loading calibration from: {calibration_file}")
        print(f"Keys found: {list(data.keys())}")

        if eye_in_hand:
            if 'T_flange_cam' in data:
                T_ref_cam = data['T_flange_cam']
                print("Loaded 'T_flange_cam' (Eye-in-Hand: flange to camera).")
                return T_ref_cam
            elif 'T_ee_cam' in data:
                T_ref_cam = data['T_ee_cam']
                print("Loaded 'T_ee_cam' (Fallback for Eye-in-Hand).")
                return T_ref_cam
            else:
                print("Error: Eye-in-Hand mode selected but 'T_flange_cam' or 'T_ee_cam' key not found.")
                return None
        else:
            if 'T_base_cam' in data:
                T_ref_cam = data['T_base_cam']
                print("Loaded 'T_base_cam' (Eye-to-Hand: base to camera).")
                return T_ref_cam
            else:
                print("Error: Eye-to-Hand mode selected but 'T_base_cam' key not found.")
                return None

    except Exception as e:
        print(f"Warning: Failed to load calibration: {e}")
        return None


def get_robot_pose(robot_ip):
    """
    Connect to robot and get current end-effector pose.

    Args:
        robot_ip: IP address of the robot

    Returns:
        np.ndarray or None: 4x4 transformation matrix of EE pose
    """
    try:
        print(f"Connecting to Franka robot at {robot_ip}...")
        robot = Robot(robot_ip)
        robot.recover_from_errors()
        print("Connected to Franka robot.")

        current_affine = robot.current_cartesian_state.pose.end_effector_pose
        T_base_flange = np.array(current_affine.matrix).reshape(4, 4, order='F')

        print("Current EE pose retrieved.")
        return T_base_flange

    except Exception as e:
        print(f"Error connecting to robot: {e}")
        return None


def print_transform_info(name, T):
    """Print transformation matrix information."""
    print(f"\n{'='*60}")
    print(f"{name}:")
    print(f"{'='*60}")
    print(f"Translation: {T[:3, 3]}")
    print(f"Rotation:\n{T[:3, :3]}")


# ============================================================================
# Main Visualization
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Visualize Coordinate Frames')
    parser.add_argument('--robot-ip', type=str, default=DEFAULT_ROBOT_IP,
                        help='Franka robot IP address')
    parser.add_argument('--eye-in-hand', action='store_true',
                        help='Enable eye-in-hand mode (camera on robot end-effector)')
    parser.add_argument('--frame-size', type=float, default=0.15,
                        help='Size of coordinate frame axes (meters)')
    args = parser.parse_args()

    # Select calibration file
    calib_file = CALIBRATION_FILE_EYE_ON_HAND if args.eye_in_hand else CALIBRATION_FILE_EYE_TO_HAND

    print("\n" + "="*60)
    print("Coordinate Frame Visualization")
    print("="*60)
    print(f"Mode: {'Eye-in-Hand' if args.eye_in_hand else 'Eye-to-Hand'}")
    print(f"Calibration file: {calib_file}")
    print(f"Robot IP: {args.robot_ip}")
    print("="*60 + "\n")

    # ========================================================================
    # 1. Load Calibration
    # ========================================================================
    T_ref_cam = load_calibration(calib_file, args.eye_in_hand)

    # ========================================================================
    # 2. Get Robot Pose (T_base_flange)
    # ========================================================================
    T_base_flange = get_robot_pose(args.robot_ip)
    if T_base_flange is None:
        print("Could not retrieve robot pose. Using identity matrix.")
        T_base_flange = np.eye(4)

    # ========================================================================
    # 3. Calculate Camera Pose (T_base_cam)
    # ========================================================================
    T_base_cam = None
    if T_ref_cam is not None:
        if args.eye_in_hand:
            # T_base_cam = T_base_flange @ T_flange_cam
            T_base_cam = np.dot(T_base_flange, T_ref_cam)
            print("\nCamera pose calculated using eye-in-hand configuration.")
        else:
            # T_base_cam is static
            T_base_cam = T_ref_cam
            print("\nCamera pose loaded from static eye-to-hand calibration.")
    else:
        print("\nWarning: No calibration available. Camera frame will not be displayed.")

    # ========================================================================
    # 4. Calculate Palm Pose (T_base_palm)
    # ========================================================================
    T_flange_palm = get_flange_to_palm_transform()
    T_base_palm = np.dot(T_base_flange, T_flange_palm)

    # ========================================================================
    # Print all transforms
    # ========================================================================
    print_transform_info("T_base_flange (End-Effector)", T_base_flange)
    if T_base_cam is not None:
        print_transform_info("T_base_cam (Camera)", T_base_cam)
    print_transform_info("T_flange_palm (Flange to Palm)", T_flange_palm)
    print_transform_info("T_base_palm (Palm in Base Frame)", T_base_palm)

    # ========================================================================
    # 5. Create Visualization
    # ========================================================================
    print("\n" + "="*60)
    print("Creating 3D Visualization...")
    print("="*60)

    traces = []

    # Add ground plane grid
    traces.extend(create_ground_plane(size=1.0, grid_step=0.1))

    # Base frame (World/Robot Base) - Larger frame
    T_base = np.eye(4)
    base_traces = create_coordinate_frame_traces(T_base, "Base", size=args.frame_size * 2)
    traces.extend(base_traces)
    print("✓ Robot Base Frame (World)")

    # End-Effector frame (Flange)
    ee_traces = create_coordinate_frame_traces(T_base_flange, "EE", size=args.frame_size)
    traces.extend(ee_traces)
    pos_ee = T_base_flange[:3, 3]
    print(f"✓ End-Effector Frame at {pos_ee}")

    # Camera frame
    if T_base_cam is not None:
        cam_traces = create_coordinate_frame_traces(T_base_cam, "Camera", size=args.frame_size)
        traces.extend(cam_traces)
        pos_cam = T_base_cam[:3, 3]
        print(f"✓ Camera Frame at {pos_cam}")

    # Palm frame
    palm_traces = create_coordinate_frame_traces(T_base_palm, "Palm", size=args.frame_size * 0.8)
    traces.extend(palm_traces)
    pos_palm = T_base_palm[:3, 3]
    print(f"✓ Palm Frame at {pos_palm}")

    # Add connection lines between frames
    # Line from base to EE
    line_base_ee = create_line_trace(
        [0, 0, 0], T_base_flange[:3, 3],
        color='rgba(255, 0, 0, 0.3)', name='Base→EE', width=4
    )
    traces.append(line_base_ee)

    # Line from EE to Palm
    line_ee_palm = create_line_trace(
        T_base_flange[:3, 3], T_base_palm[:3, 3],
        color='rgba(0, 255, 0, 0.3)', name='EE→Palm', width=4
    )
    traces.append(line_ee_palm)

    # Line from EE to Camera (if available)
    if T_base_cam is not None:
        line_ee_cam = create_line_trace(
            T_base_flange[:3, 3], T_base_cam[:3, 3],
            color='rgba(0, 0, 255, 0.3)', name='EE→Camera', width=4
        )
        traces.append(line_ee_cam)

    # ========================================================================
    # 6. Create and Display Figure
    # ========================================================================
    print("\n" + "="*60)
    print("Launching Visualization...")
    print("="*60)
    print("\nCoordinate Frame Color Convention:")
    print("  X-axis: RED")
    print("  Y-axis: GREEN")
    print("  Z-axis: BLUE")
    print("\nFrames shown:")
    print("  1. Robot Base (World) - Large frame at origin")
    print("  2. End-Effector (Flange) - Connected to base")
    print("  3. Palm Center - Connected to EE")
    if T_base_cam is not None:
        print("  4. Camera - Connected to EE")
    print("\nInteractive controls:")
    print("  - Drag to rotate")
    print("  - Scroll to zoom")
    print("  - Right-click drag to pan")
    print("="*60 + "\n")

    # Create figure
    fig = go.Figure(data=traces)

    # Update layout
    fig.update_layout(
        title=dict(
            text="Coordinate Frame Visualization",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(
                title='X (m)',
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white",
                showbackground=True,
            ),
            yaxis=dict(
                title='Y (m)',
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white",
                showbackground=True,
            ),
            zaxis=dict(
                title='Z (m)',
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white",
                showbackground=True,
            ),
            aspectmode='data',  # Equal aspect ratio
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0)
            )
        ),
        width=1280,
        height=720,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )

    # Show the plot
    fig.show()

    print("\nVisualization opened in browser.")


if __name__ == '__main__':
    main()
