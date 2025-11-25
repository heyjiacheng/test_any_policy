"""图像处理工具"""

import numpy as np
import torch


def to_numpy_uint8(rgb):
    """将 tensor/array 转为 numpy uint8

    Args:
        rgb: 输入图像（torch.Tensor 或 numpy.ndarray）

    Returns:
        numpy.ndarray: uint8 格式的 RGB 图像
    """
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.dtype in (np.float32, np.float64):
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb
