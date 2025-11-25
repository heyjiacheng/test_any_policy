# 自定义物体加载 - 使用指南

## 📂 文件

- **`object_loader.py`** - 物体加载器（使用 trimesh）
- **`capture_custom_objects.py`** - 捕获脚本 ⭐
- **`README.md`** - 本文档

## 🚀 使用方法

### 1. 配置物体

编辑 `capture_custom_objects.py` 第 39 行：

```python
# 加载1个手机，自动缩放到10cm
return get_dataset_objects(
    max_objects=1,
    category_filter="cellphone",
    auto_scale=True,
    target_size=0.1  # 10cm
)
```

### 2. 运行

```bash
python scripts/capture/capture_custom_objects.py --max_steps 100
```

## 📝 常用配置

```python
# 手动缩放（物体太大时）
return get_dataset_objects(max_objects=1, category_filter="bottle", scale=0.1)

# 自动缩放到15cm
return get_dataset_objects(max_objects=1, category_filter="bottle", auto_scale=True, target_size=0.15)

# 加载3个杯子
return get_dataset_objects(max_objects=3, category_filter="mug", auto_scale=True)

# 加载多个不同类别
return (
    get_dataset_objects(max_objects=1, category_filter="bottle", auto_scale=True) +
    get_dataset_objects(max_objects=1, category_filter="cellphone", auto_scale=True)
)

# 手动调整位置
configs = get_dataset_objects(max_objects=2, category_filter="bottle", auto_scale=True)
configs[0].position = (0.1, 0.0, 0.05)   # 右侧
configs[1].position = (-0.1, 0.0, 0.05)  # 左侧
return configs
```

## 🎯 命令行参数

```bash
# 基础
python scripts/capture/capture_custom_objects.py --max_steps 200

# 隐藏机器人
python scripts/capture/capture_custom_objects.py --max_steps 100 --hide_robot

# 高分辨率 + 光追
python scripts/capture/capture_custom_objects.py \
    --image_width 1280 \
    --image_height 720 \
    --shader rt \
    --max_steps 100

# 固定随机种子
python scripts/capture/capture_custom_objects.py --seed 42 --max_steps 100
```

## 📍 位置坐标

```
      +y (前)
       |
       |_________ +x (右)
      /
   +z (上)

常用位置:
- 中心: (0.0, 0.0, 0.05)
- 右侧: (0.15, 0.0, 0.05)
- 左侧: (-0.15, 0.0, 0.05)
- 前方: (0.0, 0.1, 0.05)
```

## 📁 输出

```
outputs/YYYYMMDD_HHMMSS/
├── images/step_000000/*.png
├── videos/*.mp4
└── trajectory/*.json
```

## ❓ 问题

**Q: 物体太大？**
```python
# 方法1: 自动缩放（推荐）
auto_scale=True, target_size=0.1

# 方法2: 手动缩放
scale=0.1
```

**Q: 物体掉落？**
```python
position=(0.0, 0.0, 0.05)  # z >= 0.05
```

**Q: 查看类别？**
```bash
ls dataset/meshdata/ | grep bottle
```

完成！🎉
