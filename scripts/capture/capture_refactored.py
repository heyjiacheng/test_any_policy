import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tyro

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.building import articulations
from src.object_loader import ObjectConfig, get_dataset_objects, load_custom_objects
from src.camera_utils import capture_images, save_camera_params, setup_cameras
from src.camera_config import CAMERA_VIEWS
from src.gripper_viz import create_trajectory_grippers, remove_actors
from src.trajectory_loader import load_trajectory
from src.trajectory_executor import execute_trajectory_with_arm, initialize_ik_solver
from src.video_utils import generate_videos
from src.env_utils import create_maniskill_env, remove_default_objects, load_objects_to_env, hide_goal_markers


# ============================================================================
# 配置
# ============================================================================
#
# 使用方法:
#
# 1. 加载自定义 .obj 物体（默认方式）:
#    python scripts/capture/capture_refactored.py
#
# 2. 加载 PartNet-Mobility 数据集的 articulation 物体:
#    python scripts/capture/capture_refactored.py --use-articulation --articulation-id 12536
#
# 3. 从其他数据集加载:
#    python scripts/capture/capture_refactored.py --use-articulation --articulation-id <ID> --articulation-dataset <dataset-name>
#
# ============================================================================

def get_my_objects() -> List[ObjectConfig]:
    """配置要加载的物体"""
    # 方式1: 从 dataset 加载
    # return get_dataset_objects(max_objects=1, category_filter="bottle", auto_scale=True)

    # 方式2: 从自定义路径加载（推荐）
    return load_custom_objects("dataset/customize/mug_obj/base.obj", auto_scale=True, target_size=0.1)

    # 方式3: 加载多个物体
    # return load_custom_objects(
    #     ["dataset/customize/mug_obj/base.obj", "dataset/customize/bottle_obj/model.obj"],
    #     names=["my_mug", "my_bottle"],
    #     auto_scale=True
    # )


def load_articulation(env: BaseEnv, model_id: str, dataset: str = "partnet-mobility", position: tuple = (-0.15, 0, 0)):
    """
    从数据集加载 articulation 物体

    Args:
        env: ManiSkill 环境
        model_id: 模型 ID（例如 "12536"）
        dataset: 数据集名称（例如 "partnet-mobility"）
        position: 初始位置 [x, y, z]

    Returns:
        加载的 articulation 对象
    """
    import sapien

    print(f"\n{'='*60}")
    print(f"加载 Articulation 物体...")
    print(f"{'='*60}")
    print(f"  数据集: {dataset}")
    print(f"  模型ID: {model_id}")
    print(f"  位置: {position}")

    # 使用 articulation builder 加载物体
    builder = articulations.get_articulation_builder(
        env.unwrapped.scene, f"{dataset}:{model_id}"
    )

    # 设置初始位姿，避免与其他物体碰撞
    builder.initial_pose = sapien.Pose(p=position)

    builder.fix_root_link = False

    # 构建 articulation
    articulation = builder.build(name="object")

    print(f"✓ 加载成功: {articulation.name}")
    print(f"  关节数量: {len(articulation.joints)}")
    print(f"  连杆数量: {len(articulation.links)}")
    print(f"{'='*60}\n")

    return articulation


@dataclass
class Args:
    """命令行参数"""
    env_id: str = "PickCube-v1"
    trajectory_path: str = "dataset/trajectory/jc_test_folder/data/000_mug/trajectory/trajectory.json"
    reference_camera: str = "front"  # 轨迹文件所在的参考相机坐标系
    output_root: str = "outputs"
    image_width: int = 640
    image_height: int = 480
    shader: str = "rt"  # 可选: "rt", "rt-fast", "default"
    hide_goal: bool = True
    save_video: bool = True
    video_fps: int = 10
    sim_backend: str = "auto"
    render_backend: str = "gpu"
    seed: Optional[int] = None
    max_trajectory_grippers: int = 30  # 最多显示的gripper数量
    gripper_alpha: float = 0.7  # gripper透明度
    num_capture_steps: int = 100  # 捕获的图像步数
    execute_trajectory: bool = True  # 是否让机械臂执行轨迹
    show_trajectory_viz: bool = False  # 是否显示轨迹可视化（半透明grippers）
    ik_refine_steps: int = 2  # 每个轨迹点的IK细化步数
    do_grasp_and_lift: bool = True  # 轨迹执行完后是否闭合夹爪并抬升20cm
    num_grasp_wait_points: int = 10  # 闭合夹爪后的等待点数量（确保抓稳后再lift）
    # Articulation 加载选项
    use_articulation: bool = False  # 是否使用 articulation 物体（从 PartNet-Mobility 等数据集）
    articulation_id: Optional[str] = None  # articulation 模型 ID（例如 "12536"）
    articulation_dataset: str = "partnet-mobility"  # 数据集名称


# ============================================================================
# 主函数
# ============================================================================

def main(args: Args):
    """主函数"""
    # 创建环境
    env: BaseEnv = create_maniskill_env(
        env_id=args.env_id,
        control_mode="pd_joint_pos" if args.execute_trajectory else None,
        num_envs=1,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        shader=args.shader,
        enable_shadow=False,
        seed=args.seed,
    )

    # 移除默认物体
    remove_default_objects(env)

    # 加载物体（articulation 或普通物体）
    if args.use_articulation:
        # 加载 articulation 物体
        if not args.articulation_id:
            print("错误: 使用 articulation 模式时必须指定 --articulation-id 参数！")
            print("例如: --use-articulation --articulation-id 12536")
            return

        articulation_obj = load_articulation(
            env=env,
            model_id=args.articulation_id,
            dataset=args.articulation_dataset,
            position=(-0.15, 0, 0.15)
        )
    else:
        # 加载自定义物体
        object_configs = get_my_objects()
        if not object_configs:
            print("错误: 没有配置任何物体！")
            return

        loaded_objects = load_objects_to_env(env, object_configs, set_as_main_object=True)
        if not loaded_objects:
            return

    # 设置相机
    cameras = setup_cameras(env.unwrapped.scene, CAMERA_VIEWS, args.shader, args.image_width, args.image_height)

    # 加载轨迹
    print("\n" + "=" * 60)
    print("加载轨迹数据...")
    print("=" * 60)

    # 抓取和抬升配置
    LIFT_HEIGHT = 0.2  # 抬升高度（米）
    NUM_LIFT_POINTS = 10  # 抬升轨迹点数量

    reference_camera = cameras[args.reference_camera]
    trajectory_data, num_grasp_wait_points = load_trajectory(
        args.trajectory_path,
        reference_camera,
        add_grasp_and_lift=args.do_grasp_and_lift,
        lift_height=LIFT_HEIGHT,
        num_lift_points=NUM_LIFT_POINTS,
        num_grasp_wait_points=args.num_grasp_wait_points
    )

    positions = [np.array(step["tcp_position"], dtype=np.float32) for step in trajectory_data]
    quaternions = [np.array(step["tcp_quaternion"], dtype=np.float32) for step in trajectory_data]

    # 计算原始轨迹点数量（用于判断何时开始闭合夹爪）
    if args.do_grasp_and_lift:
        num_added_points = num_grasp_wait_points + NUM_LIFT_POINTS
        num_original_points = len(positions) - num_added_points
    else:
        num_original_points = len(positions)
        num_grasp_wait_points = 0

    # 隐藏目标标记
    if args.hide_goal:
        hide_goal_markers(env)

    # 创建轨迹可视化grippers（可选）
    trajectory_gripper_actors = []
    if args.show_trajectory_viz:
        trajectory_gripper_actors = create_trajectory_grippers(
            scene=env.unwrapped.scene,
            positions=positions,
            quaternions=quaternions,
            max_grippers=args.max_trajectory_grippers,
            alpha=args.gripper_alpha
        )
        print(f"✓ 已创建 {len(trajectory_gripper_actors)} 个轨迹gripper\n")

    # 初始化IK求解器（如果需要执行轨迹）
    kinematics = None
    if args.execute_trajectory:
        kinematics = initialize_ik_solver(env)

    # 准备输出
    output_dir = Path(__file__).parent / args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_captured_steps = 0

    try:
        # 执行轨迹并捕获图像
        if args.execute_trajectory:
            print("\n" + "=" * 60)
            print("执行轨迹并同时捕获图像...")
            print(f"分辨率: {args.image_width}x{args.image_height}, 着色器: {args.shader}")
            print(f"输出: {output_dir}")
            print("=" * 60)

            total_captured_steps = execute_trajectory_with_arm(
                env=env,
                positions=positions,
                quaternions=quaternions,
                kinematics=kinematics,
                cameras=cameras,
                output_dir=output_dir,
                refine_steps=args.ik_refine_steps,
                num_original_points=num_original_points if args.do_grasp_and_lift else None,
                num_grasp_wait_points=num_grasp_wait_points if args.do_grasp_and_lift else 0
            )
        else:
            # 静态场景捕获
            print("\n" + "=" * 60)
            print(f"捕获静态场景图像 ({args.num_capture_steps} 步)...")
            print(f"分辨率: {args.image_width}x{args.image_height}, 着色器: {args.shader}")
            print(f"输出: {output_dir}")
            print("=" * 60 + "\n")

            for step in range(args.num_capture_steps):
                # step来保持物理场景稳定
                env.step(torch.zeros(*env.action_space.shape, device=env.unwrapped.device))

                # 捕获图像
                capture_images(
                    cameras,
                    env.unwrapped.scene,
                    output_dir,
                    step,
                )
                total_captured_steps += 1

                if (step + 1) % 20 == 0:
                    print(f"进度: {step + 1}/{args.num_capture_steps}")

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        # 清理轨迹可视化actors
        if trajectory_gripper_actors:
            remove_actors(trajectory_gripper_actors)
            trajectory_gripper_actors.clear()

        print("\n" + "=" * 60)
        print("保存结果...")
        print("=" * 60)

        print(f"✓ 图像: {output_dir / 'images'} ({total_captured_steps} 步)")

        # 保存相机参数
        camera_params = save_camera_params(cameras, output_dir)
        print(f"✓ 相机参数: {output_dir / 'camera_params'}")
        print(f"  - {len(camera_params)} 个相机视角")

        if args.save_video:
            generate_videos(output_dir, list(CAMERA_VIEWS.keys()), args.video_fps)

        env.close()
        print("\n完成！")

        # 强制退出，避免 Python 清理时的 segfault
        os._exit(0)


if __name__ == "__main__":
    main(tyro.cli(Args))
