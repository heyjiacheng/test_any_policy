"""相机设置和参数保存工具"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from mani_skill.sensors.camera import Camera, CameraConfig
from mani_skill.utils import sapien_utils

from .image_utils import to_numpy_uint8


def setup_cameras(
    scene,
    camera_views: Dict[str, tuple],
    shader: str,
    width: int = 640,
    height: int = 480
) -> Dict[str, Camera]:
    """设置多视角相机

    Args:
        scene: SAPIEN场景
        camera_views: 相机视角配置字典 {uid: (eye, target)}
        shader: 着色器类型
        width: 图像宽度
        height: 图像高度

    Returns:
        相机字典 {uid: Camera}
    """
    cameras = {}
    for uid, (eye, target) in camera_views.items():
        config = CameraConfig(
            uid=uid,
            pose=sapien_utils.look_at(eye=eye, target=target),
            width=width,
            height=height,
            fov=np.deg2rad(55.0),
            near=0.01,
            far=3.0,
            shader_pack=shader,
        )
        camera = Camera(config, scene)
        cameras[uid] = camera
        scene.human_render_cameras[uid] = camera
    return cameras


def capture_images(
    cameras: Dict[str, Camera],
    scene,
    output_dir: Path,
    step: int,
):
    """捕获所有相机的图像（RGB + 深度）

    Args:
        cameras: 相机字典
        scene: SAPIEN场景
        output_dir: 输出目录
        step: 当前步数
    """
    scene.update_render(update_sensors=False, update_human_render_cameras=True)
    step_dir = output_dir / "images" / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    for uid, camera in cameras.items():
        camera.capture()
        obs = camera.get_obs(rgb=True, depth=True, position=False, segmentation=False)

        # 保存 RGB 图像
        rgb = to_numpy_uint8(obs["rgb"])
        image = Image.fromarray(np.ascontiguousarray(rgb))
        image.save(step_dir / f"{uid}.png")

        # 保存深度图（原始浮点值）
        depth = obs["depth"]
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        if depth.ndim == 4:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        np.save(step_dir / f"{uid}_depth.npy", depth)

        # 保存深度图可视化（用于预览）
        depth_vis = depth.copy()
        depth_vis = np.nan_to_num(depth_vis, nan=0.0, posinf=0.0, neginf=0.0)
        if depth_vis.max() > 0:
            depth_vis = (depth_vis / depth_vis.max() * 255).astype(np.uint8)
        else:
            depth_vis = np.zeros_like(depth_vis, dtype=np.uint8)
        Image.fromarray(depth_vis).save(step_dir / f"{uid}_depth.png")


def save_camera_params(cameras: Dict[str, Camera], output_dir: Path) -> dict:
    """保存相机的内参和外参

    Args:
        cameras: 相机字典
        output_dir: 输出目录

    Returns:
        相机参数字典
    """
    camera_dir = output_dir / "camera_params"
    camera_dir.mkdir(parents=True, exist_ok=True)

    camera_params = {}

    for uid, camera in cameras.items():
        # 获取相机内参矩阵
        intrinsic_matrix_raw = camera.camera.get_intrinsic_matrix()
        if isinstance(intrinsic_matrix_raw, torch.Tensor):
            intrinsic_matrix_np = intrinsic_matrix_raw.detach().cpu().numpy()
        else:
            intrinsic_matrix_np = intrinsic_matrix_raw
        if intrinsic_matrix_np.ndim == 3:
            intrinsic_matrix_np = intrinsic_matrix_np[0]

        # 获取相机的模型矩阵（cam2world，GL坐标系）
        model_matrix_raw = camera.camera.get_model_matrix()
        if isinstance(model_matrix_raw, torch.Tensor):
            model_matrix = model_matrix_raw.detach().cpu().numpy()
        else:
            model_matrix = np.array(model_matrix_raw)
        if model_matrix.ndim == 3:
            model_matrix = model_matrix[0]

        # 获取OpenCV风格的外参矩阵（world->cam）
        extrinsic_cv_raw = camera.camera.get_extrinsic_matrix()
        if isinstance(extrinsic_cv_raw, torch.Tensor):
            extrinsic_cv = extrinsic_cv_raw.detach().cpu().numpy()
        else:
            extrinsic_cv = np.array(extrinsic_cv_raw)
        if extrinsic_cv.ndim == 3:
            extrinsic_cv = extrinsic_cv[0]
        if extrinsic_cv.shape == (3, 4):
            extrinsic_cv_h = np.eye(4)
            extrinsic_cv_h[:3, :4] = extrinsic_cv
        elif extrinsic_cv.shape == (4, 4):
            extrinsic_cv_h = extrinsic_cv
        else:
            raise ValueError(f"Unexpected extrinsic matrix shape: {extrinsic_cv.shape}")
        # 转为cam->world（OpenCV坐标系）
        model_matrix_cv = np.linalg.inv(extrinsic_cv_h)

        # 从配置中获取位姿信息
        pose = camera.config.pose
        if hasattr(pose, 'p') and hasattr(pose, 'q'):
            position = pose.p
            quaternion_wxyz = pose.q  # SAPIEN格式: [w, x, y, z]
            if isinstance(position, torch.Tensor):
                position = position.detach().cpu().numpy()
            if isinstance(quaternion_wxyz, torch.Tensor):
                quaternion_wxyz = quaternion_wxyz.detach().cpu().numpy()
            if position.ndim > 1:
                position = position[0]
            if quaternion_wxyz.ndim > 1:
                quaternion_wxyz = quaternion_wxyz[0]
            # 转换为 [x, y, z, w] 格式（ROS/机器人标准）
            quaternion = np.array([quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]])
        else:
            # 从模型矩阵提取
            position = model_matrix[:3, 3]
            quaternion = Rotation.from_matrix(model_matrix[:3, :3]).as_quat()

        # 从内参矩阵提取参数
        fx = float(intrinsic_matrix_np[0, 0])
        fy = float(intrinsic_matrix_np[1, 1])
        cx = float(intrinsic_matrix_np[0, 2])
        cy = float(intrinsic_matrix_np[1, 2])

        # 相机参数
        width = camera.camera.width
        height = camera.camera.height
        fov = camera.camera.fovy if hasattr(camera.camera, 'fovy') else None
        near = camera.camera.near
        far = camera.camera.far

        # 转换内参矩阵为列表
        intrinsic_matrix = intrinsic_matrix_np.tolist()

        camera_params[uid] = {
            "intrinsics": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": int(width),
                "height": int(height),
                "fov": float(fov) if fov is not None else None,
                "near": float(near),
                "far": float(far),
                "matrix": intrinsic_matrix
            },
            "extrinsics": {
                "position": [float(x) for x in position],
                "quaternion": [float(x) for x in quaternion],  # [x, y, z, w]
                "model_matrix": model_matrix.tolist(),
                "model_matrix_cv": model_matrix_cv.tolist(),
                "extrinsic_matrix_cv": extrinsic_cv_h.tolist(),
            }
        }

    # 保存为 JSON
    with (camera_dir / "camera_params.json").open("w") as f:
        json.dump(camera_params, f, indent=2)

    # 保存为 NPZ
    npz_data = {}
    for uid, params in camera_params.items():
        npz_data[f"{uid}_intrinsic_matrix"] = np.array(params["intrinsics"]["matrix"])
        npz_data[f"{uid}_model_matrix"] = np.array(params["extrinsics"]["model_matrix"])
        npz_data[f"{uid}_model_matrix_cv"] = np.array(params["extrinsics"]["model_matrix_cv"])
        npz_data[f"{uid}_extrinsic_matrix_cv"] = np.array(params["extrinsics"]["extrinsic_matrix_cv"])
        npz_data[f"{uid}_position"] = np.array(params["extrinsics"]["position"])
        npz_data[f"{uid}_quaternion"] = np.array(params["extrinsics"]["quaternion"])

    np.savez(camera_dir / "camera_params.npz", **npz_data)

    return camera_params
