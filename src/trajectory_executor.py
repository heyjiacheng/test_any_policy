"""
轨迹执行模块

该模块提供机械臂轨迹执行功能，包括IK求解、关节插值和图像捕获。
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import sapien
import torch

from mani_skill.agents.controllers.utils.kinematics import Kinematics
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import Pose

from .camera_utils import capture_images


def execute_trajectory_with_arm(
    env: BaseEnv,
    positions: List[np.ndarray],
    quaternions: List[np.ndarray],
    kinematics: Kinematics,
    cameras: Optional[dict] = None,
    output_dir: Optional[Path] = None,
    refine_steps: int = 5,
    gripper_open: bool = True,
) -> int:
    """
    让机械臂执行轨迹，并可选地捕获图像

    Args:
        env: ManiSkill环境
        positions: TCP位置列表 (世界坐标系)
        quaternions: TCP四元数列表 [x,y,z,w] (世界坐标系)
        kinematics: IK求解器
        cameras: 相机字典（如果需要捕获图像）
        output_dir: 输出目录（如果需要捕获图像）
        refine_steps: 每个轨迹点的细化步数
        gripper_open: 夹爪状态（True=张开1.0, False=闭合-1.0）

    Returns:
        执行的总步数
    """
    num_points = len(positions)
    print(f"\n开始执行轨迹 ({num_points} 个关键点)...")

    capture_images_flag = cameras is not None and output_dir is not None
    total_steps = 0
    gripper_state = 1.0 if gripper_open else -1.0

    for idx, (position, quaternion) in enumerate(zip(positions, quaternions)):
        try:
            # 构建目标TCP姿态 (世界坐标系)
            # 注意：quaternion是 [x,y,z,w] 格式，需要转换为 [w,x,y,z] (SAPIEN格式)
            quat_wxyz = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
            target_tcp_pose = sapien.Pose(p=position.tolist(), q=quat_wxyz)

            # 将世界坐标系的姿态转换到机械臂基座坐标系
            base_pose = env.unwrapped.agent.robot.pose
            print(f"IK前基座位姿: 位置 {base_pose.p}, 四元数 {base_pose.q}")
            
            # import pdb; pdb.set_trace()
            # target_tcp_pose_at_base = sapien.Pose(p=[-0.2,0,0], q=[-0.4480736,0.8939967,0, 0])
            target_tcp_pose_at_base = base_pose.inv() * target_tcp_pose

            # 转换为 Pose 对象（用于IK）
            device = env.unwrapped.device
            if isinstance(target_tcp_pose_at_base.p, torch.Tensor):
                p_tensor = target_tcp_pose_at_base.p.reshape(1, 3).to(device)
                q_tensor = target_tcp_pose_at_base.q.reshape(1, 4).to(device)
            else:
                p_tensor = torch.tensor([target_tcp_pose_at_base.p], dtype=torch.float32, device=device)
                q_tensor = torch.tensor([target_tcp_pose_at_base.q], dtype=torch.float32, device=device)

            target_pose = Pose.create_from_pq(p=p_tensor, q=q_tensor)

            # 使用IK求解关节角度
            current_qpos = env.unwrapped.agent.robot.get_qpos()
            target_qpos = kinematics.compute_ik(
                target_pose,
                q0=current_qpos,
                use_delta_ik_solver=False,
            )

            if target_qpos is None:
                print(f"警告: 第 {idx} 个轨迹点IK求解失败，跳过")
                continue

        except Exception as e:
            print(f"错误: 第 {idx} 个轨迹点处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

        # 执行运动（插值到目标关节位置）
        start_qpos = current_qpos[0, :len(target_qpos[0])].cpu().numpy()
        target_qpos_np = target_qpos[0].cpu().numpy()

        for step in range(refine_steps):
            # 线性插值
            alpha = (step + 1) / refine_steps
            interp_qpos = start_qpos * (1 - alpha) + target_qpos_np * alpha

            # 构建动作：关节位置 + gripper状态
            action = np.hstack([interp_qpos, gripper_state])
            action_tensor = torch.from_numpy(action).unsqueeze(0).float().to(env.unwrapped.device)

            # 执行一步
            env.step(action_tensor)

            # 捕获图像（如果启用）
            if capture_images_flag:
                capture_images(
                    cameras,
                    env.unwrapped.scene,
                    output_dir,
                    total_steps,
                )

            total_steps += 1

        if (idx + 1) % 10 == 0 or idx == num_points - 1:
            print(f"  进度: {idx + 1}/{num_points}")

    print(f"✓ 轨迹执行完成 (总步数: {total_steps})\n")
    return total_steps


def grasp_and_lift(
    env: BaseEnv,
    kinematics: Kinematics,
    lift_height: float = 0.2,
    grasp_steps: int = 20,
    wait_steps: int = 10,
    lift_steps: int = 30,
    cameras: Optional[dict] = None,
    output_dir: Optional[Path] = None,
    start_step: int = 0,
) -> int:
    """
    闭合夹爪、等待稳定、然后抬升

    这个函数按以下步骤执行：
    1. 逐渐闭合夹爪（grasp_steps步）
    2. 保持当前姿态等待，让物理引擎稳定（wait_steps步）
    3. 垂直抬升指定高度（lift_steps步）

    Args:
        env: ManiSkill环境
        kinematics: IK求解器
        lift_height: 抬升高度（米），默认0.2m (20cm)
        grasp_steps: 闭合夹爪的步数，默认20
        wait_steps: 等待稳定的步数，默认10
        lift_steps: 抬升的步数，默认30
        cameras: 相机字典（如果需要捕获图像）
        output_dir: 输出目录（如果需要捕获图像）
        start_step: 起始步数（用于图像编号连续）

    Returns:
        执行的总步数
    """
    print(f"\n开始抓取和抬升 (高度: {lift_height*100:.1f}cm)...")
    print(f"  阶段: 闭合夹爪({grasp_steps}步) -> 等待稳定({wait_steps}步) -> 抬升({lift_steps}步)")

    capture_images_flag = cameras is not None and output_dir is not None
    total_steps = 0
    current_step = start_step

    # 步骤1: 闭合夹爪
    print(f"\n  1. 闭合夹爪 ({grasp_steps} 步)...")

    try:
        current_qpos = env.unwrapped.agent.robot.get_qpos()
        num_arm_joints = len(env.unwrapped.agent.arm_joint_names)
        arm_qpos = current_qpos[0, :num_arm_joints].cpu().numpy()

        for step in range(grasp_steps):
            # gripper从1.0(开)逐渐到-1.0(闭)
            alpha = (step + 1) / grasp_steps
            gripper_state = 1.0 * (1 - alpha) + (-1.0) * alpha

            action = np.hstack([arm_qpos, gripper_state])
            action_tensor = torch.from_numpy(action).unsqueeze(0).float().to(env.unwrapped.device)
            env.step(action_tensor)

            if capture_images_flag:
                capture_images(cameras, env.unwrapped.scene, output_dir, current_step)

            current_step += 1
            total_steps += 1

        print(f"     ✓ 闭合夹爪完成")

    except Exception as e:
        print(f"     错误: 闭合夹爪失败: {e}")
        import traceback
        traceback.print_exc()
        return total_steps

    # 步骤2: 等待稳定
    print(f"\n  2. 等待稳定 ({wait_steps} 步)...")

    try:
        current_qpos = env.unwrapped.agent.robot.get_qpos()
        num_arm_joints = len(env.unwrapped.agent.arm_joint_names)
        arm_qpos = current_qpos[0, :num_arm_joints].cpu().numpy()

        for step in range(wait_steps):
            # 保持当前姿态和闭合状态不变，让物理引擎稳定
            action = np.hstack([arm_qpos, -1.0])  # gripper保持闭合
            action_tensor = torch.from_numpy(action).unsqueeze(0).float().to(env.unwrapped.device)
            env.step(action_tensor)

            if capture_images_flag:
                capture_images(cameras, env.unwrapped.scene, output_dir, current_step)

            current_step += 1
            total_steps += 1

        print(f"     ✓ 等待稳定完成")

    except Exception as e:
        print(f"     错误: 等待稳定失败: {e}")
        import traceback
        traceback.print_exc()
        return total_steps

    # 步骤3: 抬升
    print(f"\n  3. 垂直抬升 {lift_height*100:.1f}cm ({lift_steps} 步)...")

    try:
        device = env.unwrapped.device
        agent = env.unwrapped.agent
        current_qpos = agent.robot.get_qpos()

        # 获取当前TCP在世界坐标系的姿态
        current_tcp_pose_world = agent.tcp_pose

        # 在世界坐标系中垂直向上抬升（world Z轴）
        if isinstance(current_tcp_pose_world.p, torch.Tensor):
            # 处理批量数据
            target_position_world = current_tcp_pose_world.p.clone()
            if target_position_world.ndim == 1:
                target_position_world = target_position_world.unsqueeze(0)
            target_position_world[:, 2] += lift_height  # 世界坐标系Z轴向上
            target_quaternion_world = current_tcp_pose_world.q
            if target_quaternion_world.ndim == 1:
                target_quaternion_world = target_quaternion_world.unsqueeze(0)
        else:
            target_position_world = np.array(current_tcp_pose_world.p).reshape(-1, 3)
            target_position_world[:, 2] += lift_height
            target_quaternion_world = np.array(current_tcp_pose_world.q).reshape(-1, 4)

        # 构建目标姿态（世界坐标系）
        target_tcp_pose_world = sapien.Pose(
            p=(target_position_world[0] if target_position_world.shape[0] == 1 else target_position_world).tolist() if isinstance(target_position_world, np.ndarray) else target_position_world[0].cpu().numpy().tolist(),
            q=(target_quaternion_world[0] if target_quaternion_world.shape[0] == 1 else target_quaternion_world).tolist() if isinstance(target_quaternion_world, np.ndarray) else target_quaternion_world[0].cpu().numpy().tolist()
        )

        # 转换到基座坐标系
        base_pose = agent.robot.pose
        target_tcp_pose_at_base = base_pose.inv() * target_tcp_pose_world

        # 转换为 Pose 对象（用于IK）
        if isinstance(target_tcp_pose_at_base.p, torch.Tensor):
            p_tensor = target_tcp_pose_at_base.p.reshape(1, 3).to(device)
            q_tensor = target_tcp_pose_at_base.q.reshape(1, 4).to(device)
        else:
            p_tensor = torch.tensor([target_tcp_pose_at_base.p], dtype=torch.float32, device=device)
            q_tensor = torch.tensor([target_tcp_pose_at_base.q], dtype=torch.float32, device=device)

        target_pose = Pose.create_from_pq(p=p_tensor, q=q_tensor)

        # 使用IK求解目标位置
        target_qpos = kinematics.compute_ik(
            target_pose,
            q0=current_qpos,
            use_delta_ik_solver=False,  # 使用绝对位置模式
        )

        if target_qpos is None:
            print(f"     警告: 抬升位置IK求解失败，跳过抬升")
            return total_steps

        # 执行抬升运动（插值）
        start_qpos = current_qpos[0, :len(target_qpos[0])].cpu().numpy()
        target_qpos_np = target_qpos[0].cpu().numpy()

        for step in range(lift_steps):
            alpha = (step + 1) / lift_steps
            interp_qpos = start_qpos * (1 - alpha) + target_qpos_np * alpha

            # 保持夹爪闭合状态
            action = np.hstack([interp_qpos, -1.0])  # -1.0 = gripper closed
            action_tensor = torch.from_numpy(action).unsqueeze(0).float().to(env.unwrapped.device)
            env.step(action_tensor)

            if capture_images_flag:
                capture_images(cameras, env.unwrapped.scene, output_dir, current_step)

            current_step += 1
            total_steps += 1

    except Exception as e:
        print(f"  错误: 抬升动作失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"✓ 抓取和抬升完成 (总步数: {total_steps})\n")
    return total_steps


def initialize_ik_solver(env: BaseEnv) -> Kinematics:
    """
    初始化IK求解器

    Args:
        env: ManiSkill环境

    Returns:
        Kinematics: 初始化好的IK求解器
    """
    print("\n" + "=" * 60)
    print("初始化IK求解器...")
    print("=" * 60)

    # 获取机械臂的URDF路径和末端执行器链接名
    agent = env.unwrapped.agent
    urdf_path = agent.urdf_path
    ee_link_name = agent.ee_link_name
    arm_joint_names = agent.arm_joint_names

    # 获取活动关节索引
    active_joint_indices = []
    for joint_name in arm_joint_names:
        for idx, joint in enumerate(agent.robot.get_active_joints()):
            if joint.name == joint_name:
                active_joint_indices.append(idx)
                break

    # 转换为 tensor
    active_joint_indices_tensor = torch.tensor(
        active_joint_indices, dtype=torch.int32, device=env.unwrapped.device
    )

    # 创建IK求解器
    kinematics = Kinematics(
        urdf_path=urdf_path,
        end_link_name=ee_link_name,
        articulation=agent.robot,
        active_joint_indices=active_joint_indices_tensor
    )
    print(f"  - URDF: {urdf_path}")
    print(f"  - 末端执行器: {ee_link_name}")
    print(f"  - 活动关节数: {len(active_joint_indices)}\n")

    return kinematics
