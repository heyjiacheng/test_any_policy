# compute grasp pose in the camera frame

import argparse
from pathlib import Path
import os
import glob
import numpy as np
import open3d as o3d
import trimesh.transformations as tra
import json

import numpy as np
from typing import Dict, Tuple, List, Iterable


def compute_palm_pose(
    joints: np.ndarray,
    indices: Dict[str, int] = None,
    use_thumb: bool = False
):
    if indices is None:
        indices = dict(
            wrist=0,
            index_mcp=7,
            middle_mcp=11,
            ring_mcp=15,
            pinky_mcp=19,
            thumb_base=3,
        )
    eps = 1e-12

    # keypoints
    w  = joints[indices["wrist"]]
    iM = joints[indices["index_mcp"]]
    mM = joints[indices["middle_mcp"]]
    rM = joints[indices["ring_mcp"]]
    pM = joints[indices["pinky_mcp"]]
    has_thumb = ("thumb_base" in indices) and (indices["thumb_base"] is not None)
    tB = joints[indices["thumb_base"]] if has_thumb else None

    # 掌心：四个“mcp”点的平均（保持与你代码一致）
    # center = (iM + mM + rM + pM) / 4.0
    # TODO:hardcode for now
    center = (joints[3] + joints[15]) / 2.0

    # 拟合掌面用于稳定
    fit_ids = [indices["wrist"], indices["index_mcp"], indices["middle_mcp"],
               indices["ring_mcp"], indices["pinky_mcp"]]
    if use_thumb and has_thumb:
        fit_ids.append(indices["thumb_base"])
    pts = joints[np.asarray(fit_ids)]
    pts_c = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts_c, full_matrices=False)
    plane_n = vh[-1] / (np.linalg.norm(vh[-1]) + eps)

    # +X: wrist -> center
    x_axis = center - w
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = ((iM + mM + rM + pM) / 4.0) - w
        if np.linalg.norm(x_axis) < 1e-8:
            x_axis = vh[-2]
    x_axis = x_axis / (np.linalg.norm(x_axis) + eps)

    # +Y: index -> thumb (无thumb时用 index - pinky 近似)
    if has_thumb:
        y_raw = tB - iM
    else:
        y_raw = iM - pM
    y_axis = y_raw - x_axis * np.dot(x_axis, y_raw)
    if np.linalg.norm(y_axis) < 1e-8:
        width_vec = iM - pM
        y_axis = width_vec - x_axis * np.dot(x_axis, width_vec)
        if np.linalg.norm(y_axis) < 1e-8:
            y_axis = vh[-2] - x_axis * np.dot(x_axis, vh[-2])
    y_axis = y_axis / (np.linalg.norm(y_axis) + eps)

    # +Z: X × Y，并与掌面法线同向以减少翻转
    z_axis = np.cross(x_axis, y_axis)
    if np.linalg.norm(z_axis) < 1e-8:
        z_axis = np.cross(x_axis, plane_n)
        if np.linalg.norm(z_axis) < 1e-8:
            z_axis = vh[-1]
    z_axis = z_axis / (np.linalg.norm(z_axis) + eps)
    if np.dot(z_axis, plane_n) < 0:
        y_axis = -y_axis
        z_axis = -z_axis

    # 重新正交化保证数值稳定
    z_axis = np.cross(x_axis, y_axis); z_axis /= (np.linalg.norm(z_axis)+eps)
    y_axis = np.cross(z_axis, x_axis); y_axis /= (np.linalg.norm(y_axis)+eps)

    R = np.column_stack([x_axis, y_axis, z_axis])
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = center
    return dict(center=center, axes=R, T=T)

def _closest_rotation(R: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix to the nearest proper rotation (polar decomposition)."""
    U, _, Vt = np.linalg.svd(R)
    Rproj = U @ Vt
    if np.linalg.det(Rproj) < 0:  # fix improper rotation
        U[:, -1] *= -1
        Rproj = U @ Vt
    return Rproj

def rotation_angle(R: np.ndarray) -> float:
    """
    Geodesic angle on SO(3): angle in [0, pi] between R and I.
    R should be a rotation; we project it to SO(3) for numerical stability.
    """
    R = _closest_rotation(R)
    c = (np.trace(R) - 1.0) * 0.5
    c = np.clip(c, -1.0, 1.0)
    return float(np.arccos(c))

def default_parallel_jaw_symmetries(hand_axis: str = "z"):
    def Rot(axis, angle):
        c,s = np.cos(angle), np.sin(angle)
        if axis == "x":
            R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        elif axis == "y":
            R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        elif axis == "z":
            R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        T = np.eye(4); T[:3,:3] = R; return T
    return [np.eye(4), Rot(hand_axis, np.pi)]

def se3_distance_parallel_jaw(
    T_target: np.ndarray,
    T_candidate: np.ndarray,
    r: float = 0.05,
    symmetries: Iterable[np.ndarray] = None
) -> Tuple[float, Dict[str, float], int]:
    """
    Option 1: SE(3) distance = sqrt( ||Δt||^2 + (r * θ)^2 ), minimized over gripper symmetries.

    Args
    ----
    T_target:    (4,4) target pose.
    T_candidate: (4,4) candidate pose (to be compared against target).
    r:           Characteristic length (meters) that trades rotation vs translation.
                 Choose ~ gripper size (e.g., 0.03~0.10 m).
    symmetries:  Iterable of (4,4) right-multipliers S in the *candidate's local frame*.
                 If None, uses identity and 180° about local Z (parallel-jaw yaw symmetry).

    Returns
    -------
    d:           Combined distance (meters).
    parts:       Dict with components:
                   - trans_err (m)
                   - rot_err_rad
                   - rot_err_deg
    best_idx:    Index of the symmetry in 'symmetries' that achieved the minimum.
    """
    if symmetries is None:
        symmetries = default_parallel_jaw_symmetries()
    symmetries = list(symmetries)

    T_inv = np.linalg.inv(T_target)
    best = (np.inf, None, None)  # (d, trans_err, rot_err)
    best_idx = 0

    for i, S in enumerate(symmetries):
        # Right-multiply symmetry in candidate's local coordinates.
        T_rel = T_inv @ (T_candidate @ S)

        dt = T_rel[:3, 3]
        R_rel = T_rel[:3, :3]
        theta = rotation_angle(R_rel)

        trans_err = float(np.linalg.norm(dt))
        d = float(np.sqrt(trans_err**2 + (r * theta)**2))

        if d < best[0]:
            best = (d, trans_err, theta)
            best_idx = i

    d, trans_err, theta = best
    parts = dict(trans_err=trans_err, rot_err_rad=theta, rot_err_deg=np.degrees(theta))
    return d, parts, best_idx

def rank_grasps_parallel_jaw(
    T_target: np.ndarray,
    candidates: List[np.ndarray],
    r: float = 0.05,
    symmetries: Iterable[np.ndarray] = None
) -> List[Tuple[int, float, Dict[str, float], int]]:
    """
    Rank candidates by the above distance (lower is better).

    Returns a list of tuples:
      (index_in_input_list, distance, parts_dict, best_symmetry_idx)
    sorted by distance ascending.
    """
    results = []
    for i, Tc in enumerate(candidates):
        d, parts, sidx = se3_distance_parallel_jaw(T_target, Tc, r=r, symmetries=symmetries)
        results.append((i, d, parts, sidx))
    results.sort(key=lambda x: x[1])
    return results

def pcd_from_rgbd_cpu(color_img_o3d: o3d.geometry.Image,
                      depth_np: np.ndarray,
                      K: np.ndarray,
                      depth_unit: str = "m",
                      flip_for_view: bool = True,
                      add_axis: bool = True,
                      axis_size: float = 0.1) -> list[o3d.geometry.Geometry]:
    """
    Build a point cloud (CPU) by manual back-projection, optionally add a coordinate axis.

    Args:
        color_img_o3d: Open3D Image (HxWx3 uint8) for color.
        depth_np:      HxW depth array; meters if depth_unit='m', millimeters if 'mm'.
        K:             3x3 camera intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]].
        depth_unit:    'm' (meters) or 'mm' (millimeters).
        flip_for_view: If True, flip Y/Z for nicer Open3D view.
        add_axis:      If True, include a coordinate frame mesh at the origin.
        axis_size:     Length scaling for the axis (in meters).

    Returns:
        List of Open3D geometries (point cloud first, then axis if requested).
    """
    if K.shape != (3, 3):
        raise ValueError(f"Expected K as 3x3, got {K.shape}")
    if depth_np.ndim != 2:
        raise ValueError(f"Expected depth as HxW, got {depth_np.shape}")

    # Depth → meters
    if depth_unit == "mm":
        z = depth_np.astype(np.float32) / 1000.0
    elif depth_unit == "m":
        z = depth_np.astype(np.float32)
    else:
        raise ValueError("depth_unit must be 'm' or 'mm'")

    color_np = np.asarray(color_img_o3d)  # HxWx3 uint8
    if color_np.ndim != 3 or color_np.shape[2] < 3:
        raise ValueError(f"Color image must be HxWx3, got {color_np.shape}")

    H, W = z.shape
    if color_np.shape[0] != H or color_np.shape[1] != W:
        raise ValueError(f"RGB/Depth size mismatch: color {color_np.shape[:2]} vs depth {(H, W)}")

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    # Pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = (z > 0) & (z < 1)  # tweak as needed (e.g., (z > 0) & (z < max_range))

    # Back-project
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z

    pts  = np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float64)      # Nx3
    cols = (color_np[valid, :3].astype(np.float32) / 255.0).astype(np.float64)      # Nx3

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    if flip_for_view:
        pcd.transform([[1, 0, 0, 0],
                       [0,-1, 0, 0],
                       [0, 0,-1, 0],
                       [0, 0, 0, 1]])

    geoms: list[o3d.geometry.Geometry] = [pcd]
    if add_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
        geoms.append(axis)
        
    return geoms

def get_gripper_control_points() -> np.ndarray:
    """Control points (homogeneous) for grasp visualization (7x4)."""
    return np.array([
        [-0.10, 0.00, 0.00, 1.0],
        [-0.03, 0.00, 0.00, 1.0],
        [-0.03, 0.07, 0.00, 1.0],
        [ 0.03, 0.07, 0.00, 1.0],
        [-0.03, 0.07, 0.00, 1.0],
        [-0.03,-0.07, 0.00, 1.0],
        [ 0.03,-0.07, 0.00, 1.0],
    ], dtype=np.float32)

def get_gripper_mesh_o3d(grasp: np.ndarray,
                         color=(0.2, 0.8, 0.0),
                         show_sweep_volume: bool = False) -> list[o3d.geometry.TriangleMesh]:
    """Create Open3D mesh representation of a parallel-jaw gripper at pose `grasp` (4x4)."""
    meshes = []
    align = tra.euler_matrix(0, 0, 0)  # rotate to O3D's frame for cylinders

    # Main body (cylinder: length along local +Z before alignment)
    cyl_body = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=0.139)
    T = np.eye(4)
    T[0, 3] = -0.03
    T = grasp @ (align @ T)
    cyl_body.paint_uniform_color(color)
    cyl_body.transform(T)

    # Handle
    cyl_handle = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=0.07)
    T = tra.euler_matrix(0, np.pi / 2, 0)
    T[0, 3] = -0.065
    T = grasp @ (align @ T)
    cyl_handle.paint_uniform_color(color)
    cyl_handle.transform(T)

    # Top finger
    cyl_top = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=0.06)
    T = tra.euler_matrix(0, np.pi / 2, 0)
    T[2, 3] = 0.065
    T = grasp @ (align @ T)
    cyl_top.paint_uniform_color(color)
    cyl_top.transform(T)

    # Bottom finger
    cyl_bottom = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=0.06)
    T = tra.euler_matrix(0, np.pi / 2, 0)
    T[2, 3] = -0.065
    T = grasp @ (align @ T)
    cyl_bottom.paint_uniform_color(color)
    cyl_bottom.transform(T)

    for m in (cyl_body, cyl_handle, cyl_top, cyl_bottom):
        m.compute_vertex_normals()
        meshes.append(m)

    # Optional sweep volume
    if show_sweep_volume:
        sweep = o3d.geometry.TriangleMesh.create_box(width=0.06, height=0.02, depth=0.14)
        T = np.eye(4)
        T[0, 3] = -0.06 / 2
        T[1, 3] = -0.02 / 2
        T[2, 3] = -0.14 / 2
        T = grasp @ (align @ T)
        sweep.paint_uniform_color([0.0, 0.2, 0.8])
        sweep.transform(T)
        sweep.compute_vertex_normals()
        meshes.append(sweep)

    return meshes

def process_single_frame(base_dir, frame_id, five_digits_mode=False):
     # if frame_id is not provided, use the latest frame

    if five_digits_mode:
        frame_id_5digits = frame_id
    else:
        if frame_id is None:
            frame_folders = [f for f in os.listdir(os.path.join(base_dir, 'depth_align_out')) if f.startswith('frame_')]
            frame_folders.sort()
            frame_id_5digits = frame_folders[-1]
        else:
            frame_id_5digits = f"{frame_id:05d}"

    # load predicted hand joints
    hand_joints_path = os.path.join(base_dir, 'depth_align_out', frame_id_5digits, 'outputs_grasp', f"{frame_id_5digits}_3d_hand_joints_aligned.npy")

    # construct palm pose from predicted hand joints
    hand_joints = np.load(hand_joints_path)
    palm_pose_dict = compute_palm_pose(hand_joints, use_thumb=True)
    T_palm_pose = palm_pose_dict['T']
    T_palm_pose_adj = T_palm_pose

    return T_palm_pose_adj


def main():
    """
    Convert the predicted hand trajectory to gripper trajectory
    """
    parser = argparse.ArgumentParser()
    # add base_dir
    parser.add_argument("--base_dir", type=str, default="/home/supertc/repo/hamer/jc_test_folder/data/000_mug")
    parser.add_argument("--frame_id", type=int, default=None)
    parser.add_argument("--viz", action="store_true")
    args = parser.parse_args()

    # --- inputs ---
    video_path = glob.glob(os.path.join(args.base_dir, "*.mp4"))[0]
    view_id = (video_path.split("/")[-1].split(".")[0]).split('_')[0]

    rgb_path   = os.path.join(args.base_dir, f"{view_id}_color.png")
    depth_path = os.path.join(args.base_dir, f"{view_id}_depth_aligned_rgb.npy")
    cam_path   = os.path.join(args.base_dir, f"{view_id}_camerainfo.npy")

    frame_folders = [f for f in os.listdir(os.path.join(args.base_dir, 'depth_align_out')) if f.startswith('frame_')]
    frame_folders.sort()

    T_palm_pose_adj_list = []
    if args.frame_id == -1:
        print("Processing all frames...")
        for frame_id in frame_folders:
            T_palm_pose_adj = process_single_frame(args.base_dir, frame_id, five_digits_mode=True)
            print(f"Processing frame {frame_id}...")

            # Ensure trajectory continuity by checking symmetries
            if len(T_palm_pose_adj_list) > 0:
                T_prev = T_palm_pose_adj_list[-1]

                # Try all three axes for 180° rotation symmetry
                best_overall = (np.inf, None, None, None, None)  # (distance, parts, sym_idx, axis_name, symmetries)

                for axis_name in ["x", "y", "z"]:
                    symmetries = default_parallel_jaw_symmetries(hand_axis=axis_name)
                    d, parts, best_idx = se3_distance_parallel_jaw(
                        T_target=T_prev,
                        T_candidate=T_palm_pose_adj,
                        r=0.05,
                        symmetries=symmetries
                    )

                    if d < best_overall[0]:
                        best_overall = (d, parts, best_idx, axis_name, symmetries)

                d, parts, best_idx, best_axis, best_symmetries = best_overall

                # Apply the best symmetry transformation
                if best_idx != 0:  # If not identity (i.e., 180° rotation is better)
                    T_palm_pose_adj = T_palm_pose_adj @ best_symmetries[best_idx]
                    print(f"  -> Applied 180° rotation around {best_axis.upper()}-axis (trans_err={parts['trans_err']:.4f}m, rot_err={parts['rot_err_deg']:.2f}°)")
                else:
                    print(f"  -> Pose continuous (trans_err={parts['trans_err']:.4f}m, rot_err={parts['rot_err_deg']:.2f}°)")

            T_palm_pose_adj_list.append(T_palm_pose_adj)
            print("T_palm_pose_adj:", T_palm_pose_adj)
    else:
        print(f"Processing frame {args.frame_id}...")
        T_palm_pose_adj = process_single_frame(args.base_dir, args.frame_id)
        T_palm_pose_adj_list.append(T_palm_pose_adj)

    # --- load images & intrinsics ---
    color_o3d = o3d.io.read_image(rgb_path)          # HxWx3 uint8
    depth_np  = np.load(depth_path)                  # HxW depth (meters or mm)
    K         = np.load(cam_path)                    # 3x3 intrinsics

    # --- scene point cloud visualization ---
    geoms = pcd_from_rgbd_cpu(
        color_img_o3d=color_o3d,
        depth_np=depth_np,
        K=K,
        depth_unit="m",        # or "mm" if your depth is millimeters
        flip_for_view=False,   # keep camera-frame alignment so gripper poses match
        add_axis=True,
        axis_size=0.1
    )

    if args.viz:    
        for T_palm_pose_adj in T_palm_pose_adj_list:
            T_palm_pose_adj = T_palm_pose_adj @ np.array([[ 1, 0, 0, 0.02],
                                                           [0, 1, 0, 0],
                                                           [0, 0, 1, 0],
                                                           [0, 0, 0, 1]])
            meshes = get_gripper_mesh_o3d(
                grasp=T_palm_pose_adj,
                # grasp=np.eye(4),  # visualize the gripper at the origin
                color=(0.2, 0.2, 0.8),  # blue
                show_sweep_volume=False
            )
            geoms.extend(meshes)

        # --- visualize (GUI) ---
        o3d.visualization.draw_geometries(
            geoms,
            window_name="Scene Point Cloud + Gripper(s) in Camera Frame"
        )

    # --- Convert trajectory to JSON format ---
    trajectory = []
    for step, T_palm_pose_adj in enumerate(T_palm_pose_adj_list):
        T_palm_pose_adj = T_palm_pose_adj @ np.array([[ 1, 0, 0, 0.02],
                                                           [0, 1, 0, 0],
                                                           [0, 0, 1, 0],
                                                           [0, 0, 0, 1]])
        trajectory.append({
            "step": step,
            "tcp_matrix": T_palm_pose_adj.tolist()
        })

    # --- Save trajectory to JSON file ---
    output_dir = Path("/home/supertc/jc_workspace/ManiSkill_VLA/dataset/trajectory/jc_test_folder/data/000_mug/trajectory")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trajectory.json"

    with open(output_path, 'w') as f:
        json.dump(trajectory, f, indent=2)

    print(f"Trajectory saved to: {output_path}")
    print(f"Total frames: {len(trajectory)}")


if __name__ == "__main__":
    main()
