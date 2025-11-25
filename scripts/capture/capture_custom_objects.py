"""
多物体捕获脚本 - 加载和捕获自定义物体
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
from object_loader import ObjectConfig, ObjectLoader, get_dataset_objects, load_custom_objects


# ============================================================================
# 配置
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


CAMERA_VIEWS = (
    ("front", [0.3, 0.0, 0.2], [0.0, 0.0, 0.1]),
    # ("top", [0.05, 0.0, 1.0], [0.0, 0.0, 0.1]),
    # ("left", [0.2, 0.6, 0.45], [0.0, 0.0, 0.2]),
    # ("right", [0.2, -0.6, 0.45], [0.0, 0.0, 0.2]),
    # ("diagonal", [0.55, 0.35, 0.6], [0.0, 0.0, 0.2]),
)


@dataclass
class Args:
    """命令行参数"""
    env_id: str = "PickCube-v1"
    max_steps: int = 100
    output_root: str = "outputs"
    image_width: int = 640
    image_height: int = 480
    shader: str = "rt" # 可选: "rt", "rt-fast", "default"
    hide_robot: bool = False
    hide_goal: bool = True
    save_video: bool = True
    video_fps: int = 10
    sim_backend: str = "auto"
    render_backend: str = "gpu"
    seed: Optional[int] = None


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


def save_trajectory(trajectory: List[dict], output_dir: Path):
    """保存轨迹数据"""
    traj_dir = output_dir / "trajectory"
    traj_dir.mkdir(exist_ok=True)

    with (traj_dir / "trajectory.json").open("w") as f:
        json.dump(trajectory, f, indent=2)

    if trajectory:
        np.savez(
            traj_dir / "trajectory.npz",
            steps=np.array([t["step"] for t in trajectory]),
            tcp_position=np.array([t["tcp_position"] for t in trajectory]),
            tcp_quaternion=np.array([t["tcp_quaternion"] for t in trajectory]),
        )


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


def _to_numpy(tensor_or_array):
    """转换 tensor 为 numpy"""
    if isinstance(tensor_or_array, torch.Tensor):
        arr = tensor_or_array.detach().cpu().numpy()
    else:
        arr = tensor_or_array
    return arr[0] if arr.ndim > 1 else arr


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

    # 加载自定义物体
    print("=" * 60)
    print("加载自定义物体...")
    print("=" * 60)

    object_configs = get_my_objects()
    if not object_configs:
        print("错误: 没有配置任何物体！")
        return

    loader = ObjectLoader(env.unwrapped.scene)
    loaded_objects = loader.load_multiple_objects(object_configs)
    if not loaded_objects:
        print("错误: 没有成功加载任何物体！")
        return

    env.unwrapped.cube = loaded_objects[0]
    print(f"✓ 主要物体: {object_configs[0].name}\n")

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

    trajectory = []

    try:
        for step in range(args.max_steps):
            action = env.action_space.sample() if env.action_space else None
            _, _, terminated, truncated, _ = env.step(action)

            capture_images(cameras, env.unwrapped.scene, output_dir, step)

            # 记录轨迹
            if env.unwrapped.agent:
                tcp_pose = env.unwrapped.agent.tcp_pose
                trajectory.append({
                    "step": step,
                    "tcp_position": _to_numpy(tcp_pose.p).tolist(),
                    "tcp_quaternion": _to_numpy(tcp_pose.q).tolist(),
                })

            if (terminated | truncated).any():
                env.reset()

            if (step + 1) % 50 == 0:
                print(f"进度: {step + 1}/{args.max_steps}")

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        print("\n" + "=" * 60)
        print("保存结果...")
        print("=" * 60)

        save_trajectory(trajectory, output_dir)
        print(f"✓ 轨迹: {output_dir / 'trajectory'}")
        print(f"✓ 图像: {output_dir / 'images'} ({len(trajectory)} 步)")

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
