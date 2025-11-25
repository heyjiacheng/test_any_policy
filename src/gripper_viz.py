"""Gripper 可视化工具"""

from typing import List

import numpy as np
import sapien
from transforms3d import quaternions


def create_gripper_actor(
    scene: sapien.Scene,
    position: np.ndarray,
    quaternion: np.ndarray,
    color: tuple = (0.2, 0.8, 0.0),
    name: str = "gripper",
    alpha: float = 0.7
) -> sapien.Entity:
    """
    在场景中创建gripper的可视化actor
    参考ManiSkill源代码: mani_skill/examples/motionplanning/two_finger_gripper/motionplanner.py

    Args:
        scene: SAPIEN场景
        position: [x, y, z] 位置向量（世界坐标系）
        quaternion: [x, y, z, w] 四元数（ROS/机器人标准格式，世界坐标系）
        color: RGB颜色(0-1范围)
        name: actor名称
        alpha: 透明度

    Returns:
        创建的gripper actor
    """
    builder = scene.create_actor_builder()

    # 参考ManiSkill的两指夹爪可视化
    grasp_pose_visual_width = 0.01
    grasp_width = 0.05

    # TCP中心点 - 使用球体标记
    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, alpha])
    )

    # 主体 - 从TCP到夹爪基座的连杆
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.08]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, alpha]),
    )

    # 夹爪开口部分 - 水平横杆
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, alpha]),
    )

    # 蓝色夹爪 - 表示一侧夹爪
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[0.03 - grasp_pose_visual_width * 3, grasp_width + grasp_pose_visual_width, 0.03 - 0.05],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, alpha]),
    )

    # 红色夹爪 - 表示另一侧夹爪
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[0.03 - grasp_pose_visual_width * 3, -grasp_width - grasp_pose_visual_width, 0.03 - 0.05],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, alpha]),
    )

    # 直接转换四元数格式: [x, y, z, w] (ROS/机器人标准) -> [w, x, y, z] (SAPIEN格式)
    quat_wxyz = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]

    builder.initial_pose = sapien.Pose(p=position.tolist(), q=quat_wxyz)

    # 创建kinematic actor (不参与物理模拟,仅用于可视化)
    actor = builder.build_kinematic(name=name)

    return actor


def create_trajectory_grippers(
    scene: sapien.Scene,
    positions: list,
    quaternions: list,
    max_grippers: int = 20,
    alpha: float = 0.5
) -> List[sapien.Entity]:
    """
    在场景中创建一系列gripper actors来可视化轨迹

    Args:
        scene: SAPIEN场景
        positions: 位置列表 [N, 3]
        quaternions: 四元数列表 [N, 4] (格式: [x, y, z, w])
        max_grippers: 最多显示的gripper数量（均匀采样）
        alpha: gripper透明度

    Returns:
        创建的gripper actors列表
    """
    if not positions or len(positions) == 0:
        return []

    actors = []
    num_points = len(positions)

    # 如果轨迹点太多，均匀采样
    if num_points > max_grippers:
        indices = np.linspace(0, num_points - 1, max_grippers, dtype=int)
    else:
        indices = range(num_points)

    for idx, i in enumerate(indices):
        actor = create_gripper_actor(
            scene=scene,
            position=positions[i],
            quaternion=quaternions[i],
            name=f"trajectory_gripper_{idx}",
            alpha=alpha
        )
        actors.append(actor)

    return actors


def remove_actors(actors: List[sapien.Entity]):
    """移除场景中的actors

    Args:
        actors: 要移除的actor列表
    """
    for actor in actors:
        if actor:
            actor.remove_from_scene()
