# compute grasp pose in the camera frame

import argparse
from pathlib import Path
import os
import glob
import numpy as np
import open3d as o3d
import trimesh.transformations as tra
import json
from PIL import Image

import numpy as np
from typing import Dict, Tuple, List, Iterable, Optional
from transformers import (
    OwlViTProcessor, OwlViTForObjectDetection,
    SamModel, SamProcessor
)
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Global cache for models to avoid reloading
_MODEL_CACHE = {}

def get_models(
    owl_model_name: str = "google/owlvit-base-patch32",
    sam_model_name: str = "facebook/sam-vit-huge",
    device: str = "cuda"
):
    """
    Get or create OwlViT and SAM models with caching.

    Args:
        owl_model_name: OwlViT model name from HuggingFace
        sam_model_name: SAM model name from HuggingFace
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Tuple of (owl_processor, owl_model, sam_processor, sam_model)
    """
    cache_key = (owl_model_name, sam_model_name, device)

    if cache_key not in _MODEL_CACHE:
        print(f"Loading models from HuggingFace...")
        print(f"  - OwlViT: {owl_model_name}")
        print(f"  - SAM: {sam_model_name}")

        # Load OwlViT for object detection
        owl_processor = OwlViTProcessor.from_pretrained(owl_model_name)
        owl_model = OwlViTForObjectDetection.from_pretrained(owl_model_name)
        owl_model.to(device)
        owl_model.eval()

        # Load SAM for segmentation
        sam_processor = SamProcessor.from_pretrained(sam_model_name)
        sam_model = SamModel.from_pretrained(sam_model_name)
        sam_model.to(device)
        sam_model.eval()

        _MODEL_CACHE[cache_key] = (owl_processor, owl_model, sam_processor, sam_model)
        print("Models loaded and cached")

    return _MODEL_CACHE[cache_key]


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

    # 掌心：四个"mcp"点的平均（保持与你代码一致）
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

def save_segmentation_visualization(rgb: np.ndarray, segmap: np.ndarray, output_path: str):
    """
    Save segmentation visualization as a colored image.

    Args:
        rgb: HxWx3 RGB image (uint8)
        segmap: HxW segmentation map with integer labels
        output_path: Path to save the visualization image
    """
    segment_ids = np.unique(segmap)
    segment_ids = segment_ids[segment_ids > 0]  # 排除背景
    num_segments = len(segment_ids)

    # 创建彩色分割图
    segmap_colored = np.zeros_like(rgb)
    if num_segments > 0:
        colors = plt.cm.tab20(np.linspace(0, 1, max(num_segments, 20)))

        for idx, seg_id in enumerate(segment_ids):
            mask = (segmap == seg_id)
            segmap_colored[mask] = (colors[idx % 20, :3] * 255).astype(np.uint8)

    # 创建叠加图
    overlay = rgb.astype(np.float32) / 255.0
    seg_overlay = segmap_colored.astype(np.float32) / 255.0
    combined = (overlay * 0.5 + seg_overlay * 0.5 * 255).astype(np.uint8)

    # 保存图片
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title('Original RGB')
    axes[0].axis('off')

    axes[1].imshow(segmap_colored)
    axes[1].set_title(f'Segmentation ({num_segments} objects)')
    axes[1].axis('off')

    axes[2].imshow(combined)
    axes[2].set_title('RGB + Segmentation Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Segmentation visualization saved to: {output_path}")
    print(f"  - Total segments: {num_segments}")


def generate_segmentation_map(
    rgb: np.ndarray,
    depth: np.ndarray,
    owl_model_name: str = "google/owlvit-base-patch32",
    sam_model_name: str = "facebook/sam-vit-huge",
    device: str = "cuda",
    text_queries: List[str] = None,
    box_threshold: float = 0.1,
    nms_threshold: float = 0.3
) -> np.ndarray:
    """
    Generate segmentation map from RGB and depth images using OwlViT + SAM.

    Args:
        rgb: HxWx3 RGB image (uint8)
        depth: HxW depth map (float, in meters)
        owl_model_name: OwlViT model name from HuggingFace
        sam_model_name: SAM model name from HuggingFace
        device: Device to run the model on ('cuda' or 'cpu')
        text_queries: List of text queries for object detection (e.g., ["object", "thing", "item"])
        box_threshold: Confidence threshold for bounding boxes
        nms_threshold: NMS threshold for overlapping boxes

    Returns:
        segmap: HxW segmentation map with integer labels (0=background, 1,2,3...=objects)
    """
    H, W = depth.shape

    # Default text queries for generic object detection
    if text_queries is None:
        text_queries = ["mug"]

    try:
        # Get cached models
        owl_processor, owl_model, sam_processor, sam_model = get_models(
            owl_model_name, sam_model_name, device
        )

        # Convert to PIL Image for OwlViT
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(rgb)

        # Step 1: Detect objects using OwlViT
        print(f"Detecting objects with text queries: {text_queries}")
        inputs = owl_processor(text=text_queries, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = owl_model(**inputs)

        # Process detection results
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
        results = owl_processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=box_threshold
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        print(f"OwlViT detected {len(boxes)} objects")

        # Apply NMS to remove overlapping boxes
        if len(boxes) > 0:
            keep_indices = []
            for i in range(len(boxes)):
                keep = True
                for j in range(i):
                    if scores[j] > scores[i]:
                        iou = compute_iou(boxes[i], boxes[j])
                        if iou > nms_threshold:
                            keep = False
                            break
                if keep:
                    keep_indices.append(i)

            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
            print(f"After NMS: {len(boxes)} objects")

        # Step 2: Use SAM to segment each detected box
        segmap = np.zeros((H, W), dtype=np.uint8)

        if len(boxes) == 0:
            print("No objects detected, using depth-based segmentation")
            valid_mask = (depth > 0) & (depth < 2.0)
            segmap[valid_mask] = 1
            return segmap

        # Process each detected box with SAM
        for idx, (box, score) in enumerate(zip(boxes, scores), start=1):
            # Convert box to input_boxes format [x_min, y_min, x_max, y_max]
            input_boxes = [[box.tolist()]]

            # Prepare SAM inputs
            sam_inputs = sam_processor(
                pil_image,
                input_boxes=input_boxes,
                return_tensors="pt"
            )
            sam_inputs = {k: v.to(device) for k, v in sam_inputs.items()}

            with torch.no_grad():
                sam_outputs = sam_model(**sam_inputs)

            # Get the best mask
            masks = sam_processor.post_process_masks(
                sam_outputs.pred_masks,
                sam_inputs["original_sizes"],
                sam_inputs["reshaped_input_sizes"]
            )[0]

            # Use the mask with highest score (first one)
            mask = masks[0, 0].cpu().numpy() > 0

            # Filter by depth validity
            valid_depth = (depth > 0) & (depth < 2.0)
            mask = mask & valid_depth

            # Only assign if mask overlaps with valid depth region
            if np.any(mask):
                # Avoid overwriting existing segments (keep highest confidence)
                segmap[mask & (segmap == 0)] = idx

        print(f"SAM segmentation: Generated {idx} segments")

    except Exception as e:
        print(f"Warning: Segmentation failed ({e}), falling back to depth-based segmentation")
        import traceback
        traceback.print_exc()

        # Fallback: Simple depth-based segmentation
        segmap = np.zeros((H, W), dtype=np.uint8)
        valid_mask = (depth > 0) & (depth < 2.0)
        segmap[valid_mask] = 1

    return segmap


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x_min, y_min, x_max, y_max]"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


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
    parser.add_argument("--base_dir", type=str, default="dataset/trajectory/jc_test_folder/data/000_mug",
                        help="Base directory containing the hand tracking data")
    parser.add_argument("--frame_id", type=int, default=-1,
                        help="Specific frame ID to process, or -1 for all frames")
    parser.add_argument("--viz", action="store_true",
                        help="Visualize the results with Open3D")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for trajectory (default: base_dir/trajectory)")
    parser.add_argument("--owl_model", type=str, default="google/owlvit-base-patch32",
                        help="OwlViT model name from HuggingFace")
    parser.add_argument("--sam_model", type=str, default="facebook/sam-vit-huge",
                        help="SAM model name from HuggingFace")
    parser.add_argument("--text_queries", type=str, nargs="+",
                        default=["mug"],
                        help="Text queries for object detection")
    parser.add_argument("--box_threshold", type=float, default=0.1,
                        help="Confidence threshold for bounding boxes")
    parser.add_argument("--nms_threshold", type=float, default=0.3,
                        help="NMS threshold for overlapping boxes")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run models on (cuda or cpu)")
    args = parser.parse_args()

    # --- inputs ---
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        return

    video_files = list(base_dir.glob("*.mp4"))
    if not video_files:
        print(f"Error: No video files found in {base_dir}")
        return

    video_path = video_files[0]
    view_id = (video_path.stem).split('_')[0]

    rgb_path   = base_dir / f"{view_id}_color.png"
    depth_path = base_dir / f"{view_id}_depth_aligned_rgb.npy"
    cam_path   = base_dir / f"{view_id}_camerainfo.npy"

    # Check if required files exist
    if not rgb_path.exists():
        print(f"Error: RGB image not found: {rgb_path}")
        return
    if not depth_path.exists():
        print(f"Error: Depth file not found: {depth_path}")
        return
    if not cam_path.exists():
        print(f"Error: Camera info not found: {cam_path}")
        return

    depth_align_out = base_dir / 'depth_align_out'
    if not depth_align_out.exists():
        print(f"Error: depth_align_out directory not found: {depth_align_out}")
        return

    frame_folders = sorted([f for f in os.listdir(depth_align_out) if f.startswith('frame_')])
    if not frame_folders:
        print(f"Error: No frame folders found in {depth_align_out}")
        return

    T_palm_pose_adj_list = []
    if args.frame_id == -1:
        print("Processing all frames...")
        for frame_id in frame_folders:
            T_palm_pose_adj = process_single_frame(str(base_dir), frame_id, five_digits_mode=True)
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
        T_palm_pose_adj = process_single_frame(str(base_dir), args.frame_id)
        T_palm_pose_adj_list.append(T_palm_pose_adj)

    # --- load images & intrinsics ---
    color_o3d = o3d.io.read_image(str(rgb_path))          # HxWx3 uint8
    depth_np  = np.load(str(depth_path))                  # HxW depth (meters or mm)
    K         = np.load(str(cam_path))                    # 3x3 intrinsics

    # Load RGB as numpy array for segmentation
    rgb_np = np.array(Image.open(str(rgb_path)))          # HxWx3 uint8

    # --- Generate segmentation map using OwlViT + SAM ---
    segmap = generate_segmentation_map(
        rgb=rgb_np,
        depth=depth_np,
        owl_model_name=args.owl_model,
        sam_model_name=args.sam_model,
        device=args.device,
        text_queries=args.text_queries,
        box_threshold=args.box_threshold,
        nms_threshold=args.nms_threshold
    )

    # --- Prepare output directory ---
    output_dir = Path("scripts/grasp/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Save data in contact_graspnet_pytorch format ---
    # Format: .npy file with dict containing 'depth', 'K', 'rgb', 'segmap'
    output_data = {
        'depth': depth_np,      # HxW depth map
        'K': K,                 # 3x3 camera intrinsics
        'rgb': rgb_np,          # HxWx3 RGB image
        'segmap': segmap        # HxW segmentation map
    }

    output_path = output_dir / f"{view_id}_scene.npy"
    np.save(str(output_path), output_data)

    print(f"Scene data saved to: {output_path}")
    print(f"  - RGB shape: {rgb_np.shape}")
    print(f"  - Depth shape: {depth_np.shape}")
    print(f"  - K shape: {K.shape}")
    print(f"  - Segmap shape: {segmap.shape}")

    # --- Save segmentation visualization ---
    seg_viz_path = output_dir / f"{view_id}_segmentation.png"
    save_segmentation_visualization(rgb_np, segmap, str(seg_viz_path))

    if args.viz:
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

        for T_palm_pose_adj in T_palm_pose_adj_list:
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
    if args.output_dir:
        output_dir = Path(args.output_dir)
    # else:
    #     output_dir = base_dir / "trajectory"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trajectory.json"

    with open(output_path, 'w') as f:
        json.dump(trajectory, f, indent=2)

    print(f"Trajectory saved to: {output_path}")
    print(f"Total frames: {len(trajectory)}")


if __name__ == "__main__":
    main()
