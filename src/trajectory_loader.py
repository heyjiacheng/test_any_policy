"""轨迹加载工具"""

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def load_trajectory(
    trajectory_path: str,
    reference_camera,
    add_grasp_and_lift: bool = False,
    lift_height: float = 0.2,
    num_lift_points: int = 5,
    num_grasp_wait_points: int = 10
) -> tuple[List[dict], int]:
    """
    加载轨迹文件（新格式：相机坐标系下的4x4矩阵）

    注意：add_grasp_and_lift 参数已废弃，推荐使用 trajectory_executor.grasp_and_lift() 单独处理抓取和抬升。

    Args:
        trajectory_path: 轨迹文件路径
        reference_camera: 参考相机对象（ManiSkill Camera）
        add_grasp_and_lift: [已废弃] 是否在轨迹末尾添加抓取和抬升动作
        lift_height: [已废弃] 抬升高度（米）
        num_lift_points: [已废弃] 抬升过程的插值点数量
        num_grasp_wait_points: [已废弃] 闭合夹爪后的等待点数量

    Returns:
        (轨迹列表, 0)
        - 轨迹列表: 每个元素包含世界坐标系下的tcp_position和tcp_quaternion
        - 第二个返回值始终为0（保留用于兼容性）
    """
    path = Path(trajectory_path)
    if not path.exists():
        raise FileNotFoundError(f"轨迹文件不存在: {trajectory_path}")

    with path.open("r") as f:
        trajectory_data = json.load(f)

    # 从sapien相机直接读取cam2world变换矩阵 (OpenGL坐标系)
    model_matrix_raw = reference_camera.camera.get_model_matrix()
    if isinstance(model_matrix_raw, torch.Tensor):
        T_cam_to_world = model_matrix_raw.detach().cpu().numpy()
    else:
        T_cam_to_world = np.array(model_matrix_raw, dtype=np.float32)
    if T_cam_to_world.ndim == 3:
        T_cam_to_world = T_cam_to_world[0]

    print(f"✓ 参考相机: {reference_camera.config.uid}")

    # 转换每个轨迹点
    trajectory = []
    for step_data in trajectory_data:
        # 读取相机坐标系下的TCP矩阵
        tcp_matrix_cam = np.array(step_data["tcp_matrix"], dtype=np.float32)

        # OpenCV 到 OpenGL 坐标系转换
        opencv_to_opengl = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=np.float32)

        tcp_matrix_opengl = opencv_to_opengl @ tcp_matrix_cam

        # 转换到世界坐标系
        tcp_matrix_world = T_cam_to_world @ tcp_matrix_opengl

        # o3d夹爪到SAPIEN夹爪的坐标系转换
        # o3d: 主体在-X, 夹指在±Z, Y垂直 → SAPIEN: 主体在-Z, 夹指在±Y, X垂直
        # 转换关系: o3d_X→SAPIEN_Y, o3d_Y→SAPIEN_Z, o3d_Z→SAPIEN_X
        o3d_to_sapien_rotation = np.array([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        tcp_matrix_world = tcp_matrix_world @ o3d_to_sapien_rotation

        # 提取位置
        position = tcp_matrix_world[:3, 3]

        # 提取旋转矩阵并转换为四元数 [x, y, z, w]
        rotation_matrix = tcp_matrix_world[:3, :3]
        quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()

        trajectory.append({
            "step": step_data["step"],
            "tcp_position": position,
            "tcp_quaternion": quat_xyzw,
        })

    return trajectory, 0
