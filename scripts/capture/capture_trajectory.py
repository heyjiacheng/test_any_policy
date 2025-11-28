import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import sapien_utils
from src.object_loader import ObjectConfig, ObjectLoader, load_custom_objects
from src.camera_utils import capture_images, save_camera_params, setup_cameras
from src.camera_config import CAMERA_VIEWS
from src.gripper_viz import create_trajectory_grippers, remove_actors
from src.trajectory_loader import load_trajectory
from src.trajectory_executor import execute_trajectory_with_arm, grasp_and_lift, initialize_ik_solver
from src.video_utils import generate_videos
from src.env_utils import create_maniskill_env, remove_default_objects, load_objects_to_env, hide_goal_markers, set_robot_base_pose


# ============================================================================
# 配置
# ============================================================================
#
# 使用方法:
#
# 1. 加载自定义 .obj 物体（默认方式）:
#    python scripts/capture/capture_trajectory.py
#    python scripts/capture/capture_trajectory.py --object-mesh-path dataset/customize/mug_obj/base.obj
#
# 2. 加载 PartNet-Mobility 数据集的 articulation 物体:
#    python scripts/capture/capture_trajectory.py --use-articulation --articulation-id 12536
#
# 3. 从其他数据集加载:
#    python scripts/capture/capture_trajectory.py --use-articulation --articulation-id <ID> --articulation-dataset <dataset-name>
#
# 4. 自定义物体位姿（适用于普通物体和 articulation 物体）:
#    python scripts/capture/capture_trajectory.py --object-position -0.15 0 0.2 --object-rotation 90 0 0
#
# 5. 自定义机械臂基座位置:
#    python scripts/capture/capture_trajectory.py --robot-position 0.5 0 0 --robot-rotation 0 0 45
#
# 6. 同时设置机械臂和物体位置:
#    python scripts/capture/capture_trajectory.py --robot-position 0.5 0 0 --object-position 0.3 0 0.2
#
# 注意: tyro 的 tuple 参数语法为空格分隔的值，不需要括号或引号
#
# ============================================================================

def load_single_object(mesh_path: str, position: tuple = (0, 0, 0), rotation_deg: tuple = (90, 0, 0)) -> ObjectConfig:
    """加载单个普通物体

    Args:
        mesh_path: 物体的 mesh 文件路径
        position: 物体初始位置 [x, y, z]
        rotation_deg: 物体旋转欧拉角（单位：度）

    Returns:
        ObjectConfig: 物体配置对象
    """
    configs = load_custom_objects(mesh_path, auto_scale=True, target_size=0.1)
    if not configs:
        raise ValueError(f"无法加载物体: {mesh_path}")

    # 设置位置和旋转
    config = configs[0]
    config.position = position
    config.rotation = tuple(np.deg2rad(rotation_deg))  # 转换为弧度
    return config


def load_articulation(env: BaseEnv, model_id: str, dataset: str = "partnet-mobility", position: tuple = (-0.15, 0, 0), rotation_deg: tuple = (0, 0, 0)):
    """从本地数据集加载 articulation 物体

    Args:
        env: ManiSkill 环境
        model_id: 模型 ID（例如 "12536"）
        dataset: 数据集名称（例如 "partnet-mobility"）
        position: 初始位置 [x, y, z]
        rotation_deg: 初始旋转欧拉角 (rx, ry, rz)，单位：度

    Returns:
        加载的 articulation 对象
    """
    import sapien
    from scipy.spatial.transform import Rotation

    print(f"  数据集: {dataset}")
    print(f"  模型ID: {model_id}")
    print(f"  位置: {position}")
    print(f"  旋转 (角度): {rotation_deg}")

    # 构建 URDF 文件路径
    # 将连字符转换为下划线以匹配实际目录名
    dataset_dir = dataset.replace("-", "_")
    model_dir = Path(f"dataset/{dataset_dir}/{model_id}")

    # 按优先级查找 URDF 文件
    urdf_names = ["mobility_cvx.urdf", "mobility_fixed.urdf", "mobility.urdf"]
    urdf_path = None
    for urdf_name in urdf_names:
        candidate_path = model_dir / urdf_name
        if candidate_path.exists():
            urdf_path = candidate_path
            break

    if urdf_path is None:
        raise FileNotFoundError(f"在 {model_dir} 中未找到任何 URDF 文件 ({', '.join(urdf_names)})")

    print(f"  URDF 文件: {urdf_path}")

    # 创建 URDF loader（参考 get_partnet_mobility_builder 的实现）
    scene = env.unwrapped.scene
    loader = scene.create_urdf_loader()
    loader.fix_root_link = False  # 允许物体移动
    loader.scale = 1.0  # 可以根据需要调整缩放
    loader.load_multiple_collisions_from_file = True

    # 应用 URDF 配置（设置材质属性）
    urdf_config = sapien_utils.parse_urdf_config(
        dict(
            material=dict(static_friction=1, dynamic_friction=1, restitution=0),
        )
    )
    sapien_utils.apply_urdf_config(loader, urdf_config)

    # 解析 URDF 文件
    articulation_builders = loader.parse(str(urdf_path))["articulation_builders"]
    builder = articulation_builders[0]

    # 将角度转换为四元数
    rotation_rad = np.deg2rad(rotation_deg)  # 角度转弧度
    quat = Rotation.from_euler('xyz', rotation_rad).as_quat()  # [x, y, z, w]
    quaternion = [quat[3], quat[0], quat[1], quat[2]]  # 转换为 [w, x, y, z] 格式

    # 设置初始位姿，避免与其他物体碰撞
    builder.initial_pose = sapien.Pose(p=position, q=quaternion)

    # 构建 articulation
    articulation = builder.build(name="articulation_object")

    print(f"✓ 加载成功: {articulation.name}")
    print(f"  - 关节数量: {len(articulation.joints)}")
    print(f"  - 连杆数量: {len(articulation.links)}\n")

    return articulation


@dataclass
class Args:
    """命令行参数"""
    env_id: str = "PickCube-v1"
    trajectory_path: str = "dataset/trajectory/jc_test_folder/data/000_mug/trajectory/trajectory.json"
    reference_camera: str = "behind"  # 轨迹文件所在的参考相机坐标系
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
    num_capture_steps: int = 10  # 捕获的图像步数
    execute_trajectory: bool = True  # 是否让机械臂执行轨迹
    show_trajectory_viz: bool = True  # 是否显示轨迹可视化（半透明grippers）
    ik_refine_steps: int = 2  # 每个轨迹点的IK细化步数
    do_grasp_and_lift: bool = False  # 轨迹执行完后是否闭合夹爪并抬升20cm
    num_grasp_wait_points: int = 10  # 闭合夹爪后的等待点数量（确保抓稳后再lift）
    # 普通物体加载选项
    object_mesh_path: str = "dataset/customize/mug_obj/base.obj"  # 物体的 mesh 文件路径
    # Articulation 加载选项
    use_articulation: bool = False  # 是否使用 articulation 物体（从 PartNet-Mobility 等数据集）
    articulation_id: Optional[str] = "12536"  # articulation 模型 ID（例如 "12536"）
    articulation_dataset: str = "partnet-mobility"  # 数据集名称
    # 物体位姿（适用于普通物体和 articulation 物体）
    object_position: tuple = (-0.05, 0.0, 0.15)  # 物体初始位置 [x, y, z]（米）
    object_rotation: tuple = (0, 0, 10)  # 物体旋转欧拉角 [rx, ry, rz]（度）
    # 机械臂基座位姿
    robot_position: Optional[tuple] = None  # 机械臂基座位置 [x, y, z]（米），None 表示使用默认位置
    robot_rotation: tuple = (0, 0, 0)  # 机械臂基座旋转欧拉角 [rx, ry, rz]（度）


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

    # 设置机械臂基座位姿（如果指定）
    if args.robot_position is not None:
        print("=" * 60)
        print("设置机械臂基座位姿...")
        print("=" * 60)
        set_robot_base_pose(env, args.robot_position, args.robot_rotation)
        print()

    # 加载物体（articulation 或普通物体）
    print("=" * 60)
    if args.use_articulation:
        # 加载 articulation 物体
        print("加载 Articulation 物体...")
        print("=" * 60)

        if not args.articulation_id:
            print("错误: 使用 articulation 模式时必须指定 --articulation-id 参数！")
            print("例如: --use-articulation --articulation-id 12536")
            return

        articulation_obj = load_articulation(
            env=env,
            model_id=args.articulation_id,
            dataset=args.articulation_dataset,
            position=args.object_position,
            rotation_deg=args.object_rotation
        )
    else:
        # 加载普通物体
        print("加载普通物体...")
        print("=" * 60)
        print(f"  Mesh 文件: {args.object_mesh_path}")
        print(f"  位置: {args.object_position}")
        print(f"  旋转 (角度): {args.object_rotation}")

        object_config = load_single_object(
            mesh_path=args.object_mesh_path,
            position=args.object_position,
            rotation_deg=args.object_rotation
        )

        loader = ObjectLoader(env.unwrapped.scene)
        loaded_object = loader.load_object(object_config)
        print(f"✓ 加载成功: {object_config.name}\n")

    # 设置相机
    cameras = setup_cameras(env.unwrapped.scene, CAMERA_VIEWS, args.shader, args.image_width, args.image_height)

    # 加载轨迹
    print("\n" + "=" * 60)
    print("加载轨迹数据...")
    print("=" * 60)

    reference_camera = cameras[args.reference_camera]
    trajectory_data, _ = load_trajectory(
        args.trajectory_path,
        reference_camera,
        add_grasp_and_lift=False,  
    )

    positions = [np.array(step["tcp_position"], dtype=np.float32) for step in trajectory_data]
    quaternions = [np.array(step["tcp_quaternion"], dtype=np.float32) for step in trajectory_data]

    print(f"✓ 加载了 {len(positions)} 个轨迹关键点")

    # 抓取和抬升配置
    LIFT_HEIGHT = 0.2  # 抬升高度（米）
    GRASP_STEPS = 20  # 闭合夹爪的步数
    WAIT_STEPS = args.num_grasp_wait_points  # 等待稳定的步数
    LIFT_STEPS = 30  # 抬升的步数

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

            # 阶段1: 执行原始轨迹（夹爪保持张开）
            total_captured_steps = execute_trajectory_with_arm(
                env=env,
                positions=positions,
                quaternions=quaternions,
                kinematics=kinematics,
                cameras=cameras,
                output_dir=output_dir,
                refine_steps=args.ik_refine_steps,
                gripper_open=True,  # 轨迹执行时夹爪张开
            )

            # 阶段2: 闭合夹爪、等待稳定、抬升（如果启用）
            if args.do_grasp_and_lift:
                total_captured_steps += grasp_and_lift(
                    env=env,
                    kinematics=kinematics,
                    lift_height=LIFT_HEIGHT,
                    grasp_steps=GRASP_STEPS,
                    wait_steps=WAIT_STEPS,
                    lift_steps=LIFT_STEPS,
                    cameras=cameras,
                    output_dir=output_dir,
                    start_step=total_captured_steps,
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
