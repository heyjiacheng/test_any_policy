"""
环境工具模块

提供 ManiSkill 环境的初始化和配置功能
"""

from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch

from mani_skill.envs.sapien_env import BaseEnv

from .object_loader import ObjectConfig, ObjectLoader


def create_maniskill_env(
    env_id: str,
    control_mode: Optional[str] = "pd_joint_pos",
    num_envs: int = 1,
    sim_backend: str = "auto",
    render_backend: str = "gpu",
    shader: str = "rt",
    enable_shadow: bool = False,
    seed: Optional[int] = None,
) -> BaseEnv:
    """
    创建 ManiSkill 环境

    Args:
        env_id: 环境ID (如 "PickCube-v1")
        control_mode: 控制模式 (如 "pd_joint_pos")，None 表示仅用于场景渲染
        num_envs: 并行环境数量
        sim_backend: 仿真后端 ("auto", "cpu", "gpu")
        render_backend: 渲染后端 ("gpu", "cpu")
        shader: 着色器类型 ("rt", "rt-fast", "default")
        enable_shadow: 是否启用阴影
        seed: 随机种子

    Returns:
        初始化好的环境实例
    """
    print("=" * 60)
    print("初始化环境...")
    print("=" * 60)
    print(f"  环境: {env_id}")
    print(f"  控制模式: {control_mode or 'None (仅渲染)'}")
    print(f"  并行数: {num_envs}")
    print(f"  仿真后端: {sim_backend}")
    print(f"  渲染后端: {render_backend}")
    print(f"  着色器: {shader}")

    env: BaseEnv = gym.make(
        env_id,
        obs_mode="none",
        control_mode=control_mode,
        render_mode=None,
        num_envs=num_envs,
        sim_backend=sim_backend,
        render_backend=render_backend,
        enable_shadow=enable_shadow,
        sensor_configs=dict(shader_pack=shader),
        human_render_camera_configs=dict(shader_pack=shader),
    )

    if seed is not None:
        env.reset(seed=seed, options=dict(reconfigure=True))
    else:
        env.reset(options=dict(reconfigure=True))
    return env


def remove_default_objects(env: BaseEnv) -> None:
    """
    移除环境中的默认物体（如 cube）

    Args:
        env: ManiSkill 环境
    """
    removed_count = 0

    # 移除 cube
    if hasattr(env.unwrapped, 'cube') and env.unwrapped.cube:
        env.unwrapped.cube.remove_from_scene()
        removed_count += 1

    # if removed_count > 0:
        # print(f"✓ 已移除 {removed_count} 个默认物体\n")


def load_objects_to_env(
    env: BaseEnv,
    object_configs: List[ObjectConfig],
    set_as_main_object: bool = True
) -> List:
    """
    将物体加载到环境中

    Args:
        env: ManiSkill 环境
        object_configs: 物体配置列表
        set_as_main_object: 是否将第一个物体设置为 env.unwrapped.cube

    Returns:
        加载的物体列表
    """
    if not object_configs:
        print("警告: 没有配置任何物体！")
        return []

    print("=" * 60)
    print("加载自定义物体...")
    print("=" * 60)

    loader = ObjectLoader(env.unwrapped.scene)
    loaded_objects = loader.load_multiple_objects(object_configs)

    if not loaded_objects:
        print("错误: 没有成功加载任何物体！")
        return []

    # 设置主要物体
    if set_as_main_object and loaded_objects:
        env.unwrapped.cube = loaded_objects[0]
        # print(f"✓ 主要物体: {object_configs[0].name}\n")

    return loaded_objects


def set_robot_base_pose(
    env: BaseEnv,
    position: tuple = (0, 0, 0),
    rotation_deg: tuple = (0, 0, 0)
) -> None:
    """
    设置机械臂基座的位姿

    Args:
        env: ManiSkill 环境
        position: 基座位置 [x, y, z] (米)
        rotation_deg: 基座旋转欧拉角 [rx, ry, rz] (度)
    """
    import sapien
    from scipy.spatial.transform import Rotation

    if not hasattr(env.unwrapped, 'agent') or not hasattr(env.unwrapped.agent, 'robot'):
        print("警告: 环境中没有找到机械臂 agent")
        return

    # 将角度转换为四元数
    rotation_rad = np.deg2rad(rotation_deg)
    quat = Rotation.from_euler('xyz', rotation_rad).as_quat()  # [x, y, z, w]
    quaternion = [quat[3], quat[0], quat[1], quat[2]]  # 转换为 [w, x, y, z] 格式

    # 设置机械臂基座位姿
    new_pose = sapien.Pose(p=position, q=quaternion)
    env.unwrapped.agent.robot.set_pose(new_pose)



def hide_goal_markers(env: BaseEnv) -> None:
    """
    隐藏环境中的目标标记

    Args:
        env: ManiSkill 环境
    """
    if hasattr(env.unwrapped, '_hidden_objects'):
        for obj in env.unwrapped._hidden_objects:
            obj.hide_visual()
        # print("✓ 已隐藏目标标记")


def step_environment_static(
    env: BaseEnv,
    num_steps: int = 1
) -> None:
    """
    执行静态环境步进（零动作）

    Args:
        env: ManiSkill 环境
        num_steps: 步进次数
    """
    for _ in range(num_steps):
        env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))


def setup_environment(
    env_id: str,
    object_configs: List[ObjectConfig],
    control_mode: Optional[str] = "pd_joint_pos",
    shader: str = "rt",
    hide_goals: bool = True,
    seed: Optional[int] = None,
    **env_kwargs
) -> BaseEnv:
    """
    一键配置完整环境（创建、移除默认物体、加载自定义物体）

    Args:
        env_id: 环境ID
        object_configs: 物体配置列表
        control_mode: 控制模式
        shader: 着色器类型
        hide_goals: 是否隐藏目标标记
        seed: 随机种子
        **env_kwargs: 其他环境参数

    Returns:
        配置好的环境实例
    """
    # 创建环境
    env = create_maniskill_env(
        env_id=env_id,
        control_mode=control_mode,
        shader=shader,
        seed=seed,
        **env_kwargs
    )

    # 移除默认物体
    remove_default_objects(env)

    # 加载自定义物体
    load_objects_to_env(env, object_configs)

    # 隐藏目标标记
    if hide_goals:
        hide_goal_markers(env)

    return env
