"""
物体捕获脚本 - 加载和捕获单个物体（普通物体或 articulation 物体）

使用方法:

1. 加载默认的普通物体:
   python scripts/capture/capture_custom_objects.py

2. 加载指定的普通物体:
   python scripts/capture/capture_custom_objects.py --object-mesh-path dataset/customize/mug_obj/base.obj

3. 自定义普通物体的位置和旋转:
   python scripts/capture/capture_custom_objects.py --object-position 0 0 0.15 --object-rotation 90 0 0

4. 加载 articulation 物体:
   python scripts/capture/capture_custom_objects.py --use-articulation --articulation-id 12536

5. 自定义 articulation 物体的位置和旋转:
   python scripts/capture/capture_custom_objects.py --use-articulation --articulation-id 12536 --object-position -0.15 0 0.2 --object-rotation 0 0 90

注意：
- tuple 参数使用空格分隔，不需要括号和引号
- --object-position 和 --object-rotation 参数同时适用于普通物体和 articulation 物体
- 一次只能加载一个物体（要么普通物体，要么 articulation 物体）
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import gymnasium as gym
import imageio
import numpy as np
import sapien
import torch
import tyro
from PIL import Image

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import Camera, CameraConfig
from mani_skill.utils import sapien_utils
from object_loader import ObjectConfig, ObjectLoader, load_custom_objects


# ============================================================================
# 配置
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
    print(f"  数据集: {dataset}")
    print(f"  模型ID: {model_id}")
    print(f"  位置: {position}")
    print(f"  旋转 (角度): {rotation_deg}")

    # 构建 URDF 文件路径
    from pathlib import Path
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
    from scipy.spatial.transform import Rotation
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


CAMERA_VIEWS = (
    ("front", [1.3, 0.0, 1.2], [0.0, 0.0, 0.1]),
    ("behind", [-1.3, 0.0, 1.2], [0.0, 0.0, 0.1]),
    ("top", [0.0, 0.0, 1.8], [0.0, 0.0, 0.1]),
    ("left", [0.0, 1.3, 0.8], [0.0, 0.0, 0.1]),
    ("right", [0.0, -1.3, 0.8], [0.0, 0.0, 0.1]),
    ("diagonal", [1.0, 1.0, 1.2], [0.0, 0.0, 0.1]),
)


@dataclass
class Args:
    """命令行参数"""
    env_id: str = "PickCube-v1"
    max_steps: int = 1
    output_root: str = "outputs"
    image_width: int = 640
    image_height: int = 480
    shader: str = "default" # 可选: "rt", "rt-fast", "default"
    hide_robot: bool = True
    hide_goal: bool = True
    save_video: bool = True
    video_fps: int = 10
    sim_backend: str = "auto"
    render_backend: str = "gpu"
    seed: Optional[int] = None
    # 普通物体加载选项
    object_mesh_path: str = "dataset/customize/mug_obj/base.obj"  # 普通物体的 mesh 文件路径
    # Articulation 加载选项
    use_articulation: bool = False  # 是否使用 articulation 物体（从 PartNet-Mobility 等数据集）
    articulation_id: Optional[str] = "12536"  # articulation 模型 ID（例如 "12536"）
    articulation_dataset: str = "partnet-mobility"  # 数据集名称
    # 物体位姿（适用于普通物体和 articulation 物体）
    object_position: tuple = (-0.05, 0.0, 0.15)  # 物体的初始位置 [x, y, z]
    object_rotation: tuple = (0, 0, 10)  # 物体的初始旋转（欧拉角 [rx, ry, rz]，单位：度）


# ============================================================================
# 辅助函数
# ============================================================================

def _to_numpy_uint8(rgb):
    """将 tensor/array 转为 numpy uint8"""
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.dtype in (np.float32, np.float64):
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb


def setup_cameras(scene, shader: str, width: int = 640, height: int = 480):
    """设置多视角相机"""
    cameras = {}
    for uid, eye, target in CAMERA_VIEWS:
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


def capture_images(cameras, scene, output_dir: Path, step: int):
    """捕获所有相机的图像（RGB + 深度）"""
    scene.update_render(update_sensors=False, update_human_render_cameras=True)
    step_dir = output_dir / "images" / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    for uid, camera in cameras.items():
        camera.capture()
        obs = camera.get_obs(rgb=True, depth=True, position=False, segmentation=False)

        # 保存 RGB 图像
        rgb = _to_numpy_uint8(obs["rgb"])
        Image.fromarray(np.ascontiguousarray(rgb)).save(step_dir / f"{uid}.png")

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


def generate_videos(output_dir: Path, camera_uids: List[str], fps: int = 10):
    """从图像生成视频（RGB + 深度）"""
    images_dir = output_dir / "images"
    if not images_dir.exists():
        return

    step_dirs = sorted(images_dir.glob("step_*"))
    if not step_dirs:
        return

    video_dir = output_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    print(f"\n生成视频 ({len(step_dirs)} 帧)...")

    for camera_uid in camera_uids:
        # 生成 RGB 视频
        frames = [imageio.imread(step_dir / f"{camera_uid}.png")
                  for step_dir in step_dirs if (step_dir / f"{camera_uid}.png").exists()]
        if frames:
            imageio.mimwrite(video_dir / f"{camera_uid}.mp4", frames, fps=fps, codec="libx264", quality=8)
            print(f"  ✓ {camera_uid}.mp4 ({len(frames)} 帧)")

        # 生成深度视频
        depth_frames = [imageio.imread(step_dir / f"{camera_uid}_depth.png")
                        for step_dir in step_dirs if (step_dir / f"{camera_uid}_depth.png").exists()]
        if depth_frames:
            imageio.mimwrite(video_dir / f"{camera_uid}_depth.mp4", depth_frames, fps=fps, codec="libx264", quality=8)
            print(f"  ✓ {camera_uid}_depth.mp4 ({len(depth_frames)} 帧)")

    print(f"视频保存至: {video_dir}")


def save_camera_params(cameras: dict, output_dir: Path):
    """保存相机的内参和外参"""
    camera_dir = output_dir / "camera_params"
    camera_dir.mkdir(parents=True, exist_ok=True)

    camera_params = {}

    for uid, camera in cameras.items():
        # 获取相机内参矩阵（直接从 sapien camera 获取）
        intrinsic_matrix_raw = camera.camera.get_intrinsic_matrix()
        if isinstance(intrinsic_matrix_raw, torch.Tensor):
            intrinsic_matrix_np = intrinsic_matrix_raw.detach().cpu().numpy()
        else:
            intrinsic_matrix_np = intrinsic_matrix_raw
        # 处理批处理维度
        if intrinsic_matrix_np.ndim == 3:
            intrinsic_matrix_np = intrinsic_matrix_np[0]

        # 获取相机的模型矩阵（cam2world，GL坐标系）
        model_matrix_raw = camera.camera.get_model_matrix()
        if isinstance(model_matrix_raw, torch.Tensor):
            model_matrix = model_matrix_raw.detach().cpu().numpy()
        else:
            model_matrix = np.array(model_matrix_raw)
        # 处理批处理维度
        if model_matrix.ndim == 3:
            model_matrix = model_matrix[0]

        # 从配置中获取位姿信息
        pose = camera.config.pose
        if hasattr(pose, 'p') and hasattr(pose, 'q'):
            # Pose 对象
            position = pose.p
            quaternion = pose.q  # [w, x, y, z]
            if isinstance(position, torch.Tensor):
                position = position.detach().cpu().numpy()
            if isinstance(quaternion, torch.Tensor):
                quaternion = quaternion.detach().cpu().numpy()
            # 处理批处理维度
            if position.ndim > 1:
                position = position[0]
            if quaternion.ndim > 1:
                quaternion = quaternion[0]
        else:
            # 从模型矩阵提取
            position = model_matrix[:3, 3]
            from scipy.spatial.transform import Rotation
            quaternion = Rotation.from_matrix(model_matrix[:3, :3]).as_quat()  # [x, y, z, w]
            # 转换为 [w, x, y, z] 格式
            quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

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
                "quaternion": [float(x) for x in quaternion],  # [w, x, y, z]
                "model_matrix": model_matrix.tolist()  # cam2world (GL 坐标系)
            }
        }

    # 保存为 JSON
    with (camera_dir / "camera_params.json").open("w") as f:
        json.dump(camera_params, f, indent=2)

    # 保存为 NPZ（方便 numpy 加载）
    npz_data = {}
    for uid, params in camera_params.items():
        npz_data[f"{uid}_intrinsic_matrix"] = np.array(params["intrinsics"]["matrix"])
        npz_data[f"{uid}_model_matrix"] = np.array(params["extrinsics"]["model_matrix"])
        npz_data[f"{uid}_position"] = np.array(params["extrinsics"]["position"])
        npz_data[f"{uid}_quaternion"] = np.array(params["extrinsics"]["quaternion"])

    np.savez(camera_dir / "camera_params.npz", **npz_data)

    return camera_params


# ============================================================================
# 主函数
# ============================================================================

def _hide_robot(env):
    """隐藏机器人"""
    if env.unwrapped.agent:
        for link in env.unwrapped.agent.robot.links:
            for obj in link._objs:
                if rb := obj.entity.find_component_by_type(sapien.render.RenderBodyComponent):
                    rb.visibility = 0


def main(args: Args):
    """主函数"""
    print("=" * 60)
    print("初始化环境...")
    print("=" * 60)

    # 创建环境
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode="none",
        render_mode=None,
        num_envs=1,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        enable_shadow=False,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
    )

    env.reset(seed=args.seed, options=dict(reconfigure=True)) if args.seed else env.reset(options=dict(reconfigure=True))

    # 移除默认 cube
    if hasattr(env.unwrapped, 'cube') and env.unwrapped.cube:
        env.unwrapped.cube.remove_from_scene()
        print("✓ 已移除默认 cube\n")

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

        env.unwrapped.cube = load_articulation(
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
        env.unwrapped.cube = loader.load_object(object_config)
        print(f"✓ 加载成功: {object_config.name}\n")

    # 设置相机
    cameras = setup_cameras(env.unwrapped.scene, args.shader, args.image_width, args.image_height)

    # 隐藏选项
    if args.hide_robot:
        _hide_robot(env)
        print("✓ 已隐藏机器人")
    if args.hide_goal:
        for obj in env.unwrapped._hidden_objects:
            obj.hide_visual()
        print("✓ 已隐藏目标标记")

    # 准备输出
    output_dir = Path(__file__).parent / args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"开始捕获 {args.max_steps} 步...")
    print(f"分辨率: {args.image_width}x{args.image_height}, 着色器: {args.shader}")
    print(f"输出: {output_dir}")
    print("=" * 60 + "\n")

    try:
        for step in range(args.max_steps):
            # 对于 articulation 物体，直接更新场景而不调用 env.step()
            # 这样可以避免环境的 evaluate() 方法访问不兼容的对象属性
            if args.use_articulation:
                # 可选：随机设置关节角度来产生不同的姿态
                if hasattr(env.unwrapped.cube, 'set_qpos'):
                    # 获取关节限制
                    qpos = env.unwrapped.cube.get_qpos()
                    # 随机扰动关节角度（可选）
                    # qpos = qpos + torch.randn_like(qpos) * 0.1
                    env.unwrapped.cube.set_qpos(qpos)

                # 更新物理场景
                env.unwrapped.scene.px.step()
            else:
                # 普通物体使用正常的 step
                action = env.action_space.sample() if env.action_space else None
                _, _, terminated, truncated, _ = env.step(action)
                if (terminated | truncated).any():
                    env.reset()

            print(f"正在捕获第 {step + 1}/{args.max_steps} 帧...")
            capture_images(cameras, env.unwrapped.scene, output_dir, step)

            if (step + 1) % 50 == 0:
                print(f"进度: {step + 1}/{args.max_steps}")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "=" * 60)
        print("保存结果...")
        print("=" * 60)

        # 保存相机参数
        camera_params = save_camera_params(cameras, output_dir)
        print(f"✓ 相机参数: {output_dir / 'camera_params'}")
        print(f"  - {len(camera_params)} 个相机视角")
        print(f"  - 内参 (intrinsics): fx, fy, cx, cy, 内参矩阵")
        print(f"  - 外参 (extrinsics): 位置, 四元数, 变换矩阵")

        if args.save_video:
            generate_videos(output_dir, [uid for uid, _, _ in CAMERA_VIEWS], args.video_fps)

        env.close()
        print("\n完成！")

        # 强制退出，避免 Python 清理时的 segfault
        # 这会跳过正常的析构过程，但所有数据已经保存
        os._exit(0)


if __name__ == "__main__":
    main(tyro.cli(Args))
