"""
物体加载器 - 支持使用 trimesh 加载和管理多个 .obj 物体
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import sapien
import trimesh
from scipy.spatial.transform import Rotation as R


@dataclass
class ObjectConfig:
    """物体配置"""
    name: str
    mesh_path: str
    scale: float = 1.0
    density: float = 1000.0
    position: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = (0.0, 0.0, 0.0)  # (roll, pitch, yaw) 弧度
    use_decomposed: bool = True
    texture_diffuse: Optional[str] = None
    texture_metallic: Optional[str] = None
    texture_roughness: Optional[str] = None
    texture_normal: Optional[str] = None


def euler_to_quat_sapien(euler: tuple) -> list:
    """将欧拉角转换为 SAPIEN 四元数格式 [w,x,y,z]"""
    quat = R.from_euler('xyz', euler).as_quat()  # [x,y,z,w]
    return [quat[3], quat[0], quat[1], quat[2]]  # [w,x,y,z]


def _load_texture(texture_path: Optional[str], texture_name: str) -> Optional[object]:
    """加载单个纹理"""
    if texture_path and Path(texture_path).exists():
        texture = sapien.render.RenderTexture2D(filename=str(texture_path), mipmap_levels=4)
        print(f"  ✓ {texture_name}: {Path(texture_path).name}")
        return texture
    return None


def _create_material(config: ObjectConfig) -> Optional[object]:
    """创建 PBR 材质"""
    if not any([config.texture_diffuse, config.texture_metallic,
                config.texture_roughness, config.texture_normal]):
        return None

    try:
        material = sapien.render.RenderMaterial()
        material.set_base_color([1.0, 1.0, 1.0, 1.0])

        if tex := _load_texture(config.texture_diffuse, "漫反射"):
            material.base_color_texture = tex
        if tex := _load_texture(config.texture_metallic, "金属度"):
            material.metallic_texture = tex
        else:
            material.metallic = 0.0
        if tex := _load_texture(config.texture_roughness, "粗糙度"):
            material.roughness_texture = tex
        else:
            material.roughness = 0.5
        if tex := _load_texture(config.texture_normal, "法线"):
            material.normal_texture = tex

        return material
    except Exception as e:
        print(f"  警告: 加载纹理失败: {e}")
        return None


class ObjectLoader:
    """物体加载器"""

    def __init__(self, scene):
        self.scene = scene
        self.loaded_objects = {}

    def _get_mesh_info(self, mesh_path: str, scale: float) -> dict:
        """获取 mesh 信息"""
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            if scale != 1.0:
                mesh.apply_scale(scale)
            return {
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'extents': mesh.extents,
                'volume': mesh.volume,
            }
        except Exception:
            return {}

    def load_object(self, config: ObjectConfig) -> sapien.Entity:
        """加载单个物体"""
        mesh_file = Path(config.mesh_path)
        if not mesh_file.exists():
            raise FileNotFoundError(f"找不到 mesh 文件: {config.mesh_path}")

        # 打印 mesh 信息
        if mesh_info := self._get_mesh_info(str(mesh_file), config.scale):
            print(f"\n加载: {config.name}")
            print(f"  文件: {mesh_file.name}")
            print(f"  顶点/面: {mesh_info['vertices']}/{mesh_info['faces']}")
            print(f"  尺寸: {mesh_info['extents']}")

        # 创建 builder
        builder = self.scene.create_actor_builder()
        material = _create_material(config)

        # 加载 mesh
        builder.add_visual_from_file(
            filename=str(mesh_file),
            scale=[config.scale] * 3,
            material=material,
        )

        if config.use_decomposed:
            builder.add_multiple_convex_collisions_from_file(
                filename=str(mesh_file),
                scale=[config.scale] * 3,
                density=config.density,
            )
        else:
            builder.add_convex_collision_from_file(
                filename=str(mesh_file),
                scale=[config.scale] * 3,
                density=config.density,
            )

        # 设置初始位姿
        builder.initial_pose = sapien.Pose(
            p=config.position,
            q=euler_to_quat_sapien(config.rotation)
        )

        # 构建并存储
        actor = builder.build(name=config.name)
        self.loaded_objects[config.name] = actor

        print(f"  位置: {config.position}, 旋转: {config.rotation}")
        print(f"✓ 加载成功\n")
        return actor

    def load_multiple_objects(self, configs: List[ObjectConfig]) -> List[sapien.Entity]:
        """批量加载多个物体"""
        print(f"准备加载 {len(configs)} 个物体...")
        actors = []
        for config in configs:
            try:
                actors.append(self.load_object(config))
            except Exception as e:
                print(f"✗ 加载失败: {config.name} - {e}\n")
        print(f"成功加载 {len(actors)}/{len(configs)} 个物体")
        return actors

    def get_object(self, name: str) -> Optional[sapien.Entity]:
        """获取已加载的物体"""
        return self.loaded_objects.get(name)

    def remove_object(self, name: str) -> bool:
        """移除物体"""
        if name in self.loaded_objects:
            self.loaded_objects[name].remove_from_scene()
            del self.loaded_objects[name]
            return True
        return False

    def update_object_pose(self, name: str, position: tuple, rotation: tuple = None):
        """更新物体位姿"""
        if not (actor := self.get_object(name)):
            print(f"警告: 找不到物体 {name}")
            return

        q = euler_to_quat_sapien(rotation) if rotation else actor.pose.q
        actor.set_pose(sapien.Pose(p=position, q=q))


# ============================================================================
# 辅助函数
# ============================================================================

def _calculate_scale(obj_file: Path, auto_scale: bool, scale: float, target_size: float, name: str) -> float:
    """计算物体缩放比例"""
    if not auto_scale:
        return scale
    try:
        mesh = trimesh.load(str(obj_file), force='mesh')
        max_extent = max(mesh.extents)
        final_scale = target_size / max_extent
        print(f"  {name}: {max_extent:.3f}m → {final_scale:.3f}x → {target_size:.3f}m")
        return final_scale
    except Exception:
        print(f"  警告: 无法自动缩放 {name}，使用 scale={scale}")
        return scale


def _get_grid_position(index: int, z: float = 0.0) -> tuple:
    """获取网格排列位置"""
    x_offset = (index % 3) * 0.15 - 0.15  # 3列
    y_offset = (index // 3) * 0.15
    return (x_offset, y_offset, z)


def get_dataset_objects(
    max_objects: int = 1,
    category_filter: str = None,
    dataset_dir: str = "dataset/meshdata",
    scale: float = 1.0,
    auto_scale: bool = False,
    target_size: float = 0.1,
) -> List[ObjectConfig]:
    """
    从 dataset/meshdata 目录随机加载物体

    目录结构: dataset/meshdata/{object_id}/coacd/decomposed.obj
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"警告: dataset 目录不存在: {dataset_dir}")
        return []

    obj_files = list(dataset_path.glob("**/coacd/decomposed.obj"))
    if not obj_files:
        print(f"警告: 在 {dataset_dir} 中没有找到 decomposed.obj 文件")
        return []

    print(f"找到 {len(obj_files)} 个物体")

    if category_filter:
        obj_files = [f for f in obj_files if category_filter.lower() in f.parent.parent.name.lower()]
        print(f"过滤后剩余 {len(obj_files)} 个 '{category_filter}' 类别物体")

    if not obj_files:
        return []

    import random
    obj_files = random.sample(obj_files, min(max_objects, len(obj_files)))

    configs = []
    for i, obj_file in enumerate(obj_files):
        object_name = obj_file.parent.parent.name
        final_scale = _calculate_scale(obj_file, auto_scale, scale, target_size, object_name)

        configs.append(ObjectConfig(
            name=object_name,
            mesh_path=str(obj_file),
            scale=final_scale,
            density=1000.0,
            position=_get_grid_position(i),
            rotation=(0, 0, 0),
            use_decomposed=True,
        ))

    print(f"创建了 {len(configs)} 个物体配置")
    return configs


def list_available_categories(dataset_dir: str = "dataset/meshdata") -> dict:
    """列出 dataset 中所有可用的物体类别"""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"警告: dataset 目录不存在: {dataset_dir}")
        return {}

    categories = {}
    for obj_dir in dataset_path.iterdir():
        if obj_dir.is_dir() and '-' in obj_dir.name:
            category = obj_dir.name.split('-')[1]
            categories[category] = categories.get(category, 0) + 1

    return dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))


def load_custom_objects(
    obj_paths: List[str] | str,
    names: List[str] | str = None,
    scale: float = 1.0,
    auto_scale: bool = False,
    target_size: float = 0.1,
    density: float = 1000.0,
    positions: List[tuple] = None,
) -> List[ObjectConfig]:
    """从自定义路径加载 .obj 文件"""
    # 标准化输入
    if isinstance(obj_paths, str):
        obj_paths = [obj_paths]
    if names is None:
        names = [Path(p).stem for p in obj_paths]
    elif isinstance(names, str):
        names = [names]
    if len(names) != len(obj_paths):
        print(f"警告: 名称数量({len(names)}) 与路径数量({len(obj_paths)}) 不匹配")
        names = [Path(p).stem for p in obj_paths]
    if positions is None:
        positions = [_get_grid_position(i) for i in range(len(obj_paths))]

    # 创建配置
    configs = []
    for i, (obj_path, name) in enumerate(zip(obj_paths, names)):
        obj_file = Path(obj_path)
        if not obj_file.exists():
            print(f"警告: 文件不存在: {obj_path}")
            continue

        final_scale = _calculate_scale(obj_file, auto_scale, scale, target_size, name)

        # 检测纹理
        obj_dir = obj_file.parent
        textures = {
            'diffuse': obj_dir / "texture_diffuse.png",
            'metallic': obj_dir / "texture_metallic.png",
            'roughness': obj_dir / "texture_roughness.png",
            'normal': obj_dir / "texture_normal.png",
        }

        is_decomposed = "decomposed" in str(obj_file).lower() or "coacd" in str(obj_file).lower()

        configs.append(ObjectConfig(
            name=name,
            mesh_path=str(obj_file),
            scale=final_scale,
            density=density,
            position=positions[i] if i < len(positions) else (0, 0, 0),
            rotation=(1.5708, 0, 0),
            use_decomposed=is_decomposed,
            texture_diffuse=str(textures['diffuse']) if textures['diffuse'].exists() else None,
            texture_metallic=str(textures['metallic']) if textures['metallic'].exists() else None,
            texture_roughness=str(textures['roughness']) if textures['roughness'].exists() else None,
            texture_normal=str(textures['normal']) if textures['normal'].exists() else None,
        ))

    print(f"创建了 {len(configs)} 个自定义物体配置")
    return configs


def create_single_object_config(
    mesh_path: str,
    name: str = "custom_object",
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    scale: float = 1.0,
    density: float = 1000.0,
) -> ObjectConfig:
    """创建单个物体配置"""
    return ObjectConfig(
        name=name,
        mesh_path=mesh_path,
        scale=scale,
        density=density,
        position=(x, y, z),
        rotation=(0, 0, 0),
        use_decomposed=True,
    )
