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
    num_original_points: Optional[int] = None,
    num_grasp_wait_points: int = 0
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
        num_original_points: 原始轨迹点数量（如果提供，将在等待点闭合夹爪，在抬升阶段保持闭合）
        num_grasp_wait_points: 等待点数量（用于闭合夹爪）

    Returns:
        执行的总步数
    """
    num_points = len(positions)
    print(f"\n开始执行轨迹 ({num_points} 个关键点)...")

    capture_images_flag = cameras is not None and output_dir is not None
    total_steps = 0

    # 闭合夹爪所需的步数
    GRASP_STEPS = 5

    for idx, (position, quaternion) in enumerate(zip(positions, quaternions)):
        try:
            # 构建目标TCP姿态 (世界坐标系)
            # 注意：quaternion是 [x,y,z,w] 格式，需要转换为 [w,x,y,z] (SAPIEN格式)
            quat_wxyz = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
            target_tcp_pose = sapien.Pose(p=position.tolist(), q=quat_wxyz)

            # 将世界坐标系的姿态转换到机械臂基座坐标系
            base_pose = env.unwrapped.agent.robot.pose
            target_tcp_pose_at_base = base_pose.inv() * target_tcp_pose

            # if idx == 0:
            #     print(f"  第一个轨迹点调试信息:")
            #     print(f"    世界坐标: p={position}, q={quat_wxyz}")
            #     print(f"    基座坐标: p={target_tcp_pose_at_base.p}, q={target_tcp_pose_at_base.q}")

            # 转换为 Pose 对象（用于IK）
            # target_tcp_pose_at_base.p 和 .q 已经是 tensor，直接使用
            # 确保形状为 [1, 3] 和 [1, 4]
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
                pose=target_pose,
                q0=current_qpos,
                is_delta_pose=False,
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

        # 判断当前处于哪个阶段：
        # 1. 原始轨迹阶段: idx < num_original_points
        # 2. 闭合夹爪等待阶段: num_original_points <= idx < num_original_points + num_grasp_wait_points
        # 3. 抬升阶段: idx >= num_original_points + num_grasp_wait_points
        is_grasping = (num_original_points is not None and
                       num_original_points <= idx < num_original_points + num_grasp_wait_points)
        is_lifting = (num_original_points is not None and
                      idx >= num_original_points + num_grasp_wait_points)

        for step in range(refine_steps):
            # 线性插值
            alpha = (step + 1) / refine_steps
            interp_qpos = start_qpos * (1 - alpha) + target_qpos_np * alpha

            # 计算夹爪状态
            if is_grasping:
                # 在等待点阶段，逐渐闭合夹爪
                # 计算从闭合开始到现在的总步数
                steps_since_grasp_start = (idx - num_original_points) * refine_steps + step
                grasp_alpha = min(1.0, steps_since_grasp_start / GRASP_STEPS)
                gripper_state = 1.0 * (1 - grasp_alpha) + (-1.0) * grasp_alpha  # 从1.0(开)到-1.0(闭)
            elif is_lifting:
                # 在抬升阶段，保持夹爪完全闭合
                gripper_state = -1.0
            else:
                # 在原始轨迹阶段，保持夹爪张开
                gripper_state = 1.0

            # 构建动作：关节位置 + gripper状态
            action = np.hstack([interp_qpos, gripper_state])
            # 将numpy数组转换为tensor，避免从列表创建（性能优化）
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
    lift_steps: int = 30,
    cameras: Optional[dict] = None,
    output_dir: Optional[Path] = None,
    start_step: int = 0,
) -> int:
    """
    闭合夹爪并抬升

    Args:
        env: ManiSkill环境
        kinematics: IK求解器
        lift_height: 抬升高度（米），默认0.2m (20cm)
        grasp_steps: 闭合夹爪的步数
        lift_steps: 抬升的步数
        cameras: 相机字典（如果需要捕获图像）
        output_dir: 输出目录（如果需要捕获图像）
        start_step: 起始步数（用于图像编号连续）

    Returns:
        执行的总步数
    """
    print(f"\n开始抓取和抬升 (高度: {lift_height*100:.1f}cm)...")

    capture_images_flag = cameras is not None and output_dir is not None
    total_steps = 0
    current_step = start_step

    # 步骤1: 闭合夹爪
    print(f"  1. 闭合夹爪 ({grasp_steps} 步)...")

    try:
        current_qpos = env.unwrapped.agent.robot.get_qpos()
        print(f"  调试: current_qpos shape = {current_qpos.shape}")

        # 获取arm关节数量
        num_arm_joints = len(env.unwrapped.agent.arm_joint_names)
        print(f"  调试: arm joints = {num_arm_joints}")

        arm_qpos = current_qpos[0, :num_arm_joints].cpu().numpy()
        print(f"  调试: arm_qpos shape = {arm_qpos.shape}")

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

        print(f"  ✓ 闭合夹爪完成")

    except Exception as e:
        print(f"  错误: 闭合夹爪失败: {e}")
        import traceback
        traceback.print_exc()
        return total_steps

    # 步骤2: 抬升
    print(f"  2. 抬升 {lift_height*100:.1f}cm ({lift_steps} 步)...")

    try:
        device = env.unwrapped.device
        current_qpos = env.unwrapped.agent.robot.get_qpos()

        # 使用相对位移：只在Z轴上抬升，姿态不变
        # 创建一个相对位移的 Pose（基座坐标系）
        delta_p = torch.tensor([[0.0, 0.0, lift_height]], dtype=torch.float32, device=device)
        delta_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)  # 无旋转
        delta_pose = Pose.create_from_pq(p=delta_p, q=delta_q)

        # 使用IK求解相对位移
        target_qpos = kinematics.compute_ik(
            pose=delta_pose,
            q0=current_qpos,
            is_delta_pose=True,  # 使用相对位移模式
        )

        if target_qpos is None:
            print(f"  警告: 抬升位置IK求解失败，跳过抬升")
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
