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

    Args:
        trajectory_path: 轨迹文件路径
        reference_camera: 参考相机对象（ManiSkill Camera）
        add_grasp_and_lift: 是否在轨迹末尾添加抓取和抬升动作
        lift_height: 抬升高度（米），默认0.2m (20cm)
        num_lift_points: 抬升过程的插值点数量
        num_grasp_wait_points: 闭合夹爪后的等待点数量（确保抓稳后再lift）

    Returns:
        (轨迹列表, 等待点数量)
        - 轨迹列表: 每个元素包含世界坐标系下的tcp_position和tcp_quaternion
        - 等待点数量: 用于闭合夹爪的等待点数量（如果add_grasp_and_lift=False则为0）
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

    # 如果需要，添加抓取和抬升轨迹点
    num_grasp_wait_points_actual = 0  # 实际添加的等待点数量

    if add_grasp_and_lift and len(trajectory) > 0:
        # 获取最后一个轨迹点的位置和姿态
        last_point = trajectory[-1]
        last_position = last_point["tcp_position"].copy()
        last_quaternion = last_point["tcp_quaternion"].copy()

        # 步骤1: 添加闭合夹爪的等待点（保持位置和姿态不变，让夹爪有时间完全闭合）
        # 闭合夹爪需要GRASP_STEPS步，每个轨迹点会被细化为refine_steps步
        # 为了确保有足够时间闭合，需要：num_grasp_wait_points * refine_steps >= GRASP_STEPS
        # 假设 GRASP_STEPS=20, refine_steps=2，则需要至少 20/2=10 个点
        # 使用传入的 num_grasp_wait_points 参数来确保足够时间闭合并稳定
        num_grasp_wait_points_actual = num_grasp_wait_points
        for i in range(num_grasp_wait_points_actual):
            trajectory.append({
                "step": len(trajectory),
                "tcp_position": last_position.copy(),
                "tcp_quaternion": last_quaternion.copy(),
            })

        # 步骤2: 在世界坐标系的Z轴方向上添加抬升轨迹点
        for i in range(1, num_lift_points + 1):
            alpha = i / num_lift_points
            lift_position = last_position.copy()
            lift_position[2] += lift_height * alpha  # Z轴是向上的

            trajectory.append({
                "step": len(trajectory),
                "tcp_position": lift_position,
                "tcp_quaternion": last_quaternion.copy(),  # 保持姿态不变
            })

        print(f"✓ 添加了 {num_grasp_wait_points_actual} 个闭合夹爪点 + {num_lift_points} 个抬升点 (高度: {lift_height*100:.1f}cm)")

    # print(f"✓ 加载轨迹: {len(trajectory)} 个关键点")
    return trajectory, num_grasp_wait_points_actual
