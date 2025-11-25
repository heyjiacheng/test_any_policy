"""视频生成工具"""

from pathlib import Path
from typing import List

import imageio


def generate_videos(output_dir: Path, camera_uids: List[str], fps: int = 10):
    """从图像生成视频（RGB + 深度）

    Args:
        output_dir: 输出目录
        camera_uids: 相机ID列表
        fps: 视频帧率
    """
    images_dir = output_dir / "images"
    if not images_dir.exists():
        return

    step_dirs = sorted(images_dir.glob("step_*"))
    if not step_dirs:
        return

    video_dir = output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    for camera_uid in camera_uids:
        # 生成 RGB 视频
        frames = [imageio.imread(step_dir / f"{camera_uid}.png")
                  for step_dir in step_dirs if (step_dir / f"{camera_uid}.png").exists()]
        if frames:
            imageio.mimwrite(video_dir / f"{camera_uid}.mp4", frames, fps=fps, codec="libx264", quality=8)
            # print(f"  ✓ {camera_uid}.mp4 ({len(frames)} 帧)")

        # 生成深度视频
        depth_frames = [imageio.imread(step_dir / f"{camera_uid}_depth.png")
                        for step_dir in step_dirs if (step_dir / f"{camera_uid}_depth.png").exists()]
        if depth_frames:
            imageio.mimwrite(video_dir / f"{camera_uid}_depth.mp4", depth_frames, fps=fps, codec="libx264", quality=8)
            # print(f"  ✓ {camera_uid}_depth.mp4 ({len(depth_frames)} 帧)")

    print(f"视频保存至: {video_dir}")
