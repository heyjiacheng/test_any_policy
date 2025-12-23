# Grasp Execution Client 使用说明

从 `grasp_client_ros_node.py` 重写的两个版本,用于将23元素的grasp pose发送给panda_moveit_allegro执行。

## 文件说明

### 1. `grasp_execution_simple.py` - 简化版本 (推荐开始使用)

**特点:**
- ✅ 不需要服务定义,只使用标准 ROS 消息
- ✅ 不需要编译 panda_moveit 包
- ✅ 可以立即运行
- ❌ 不能自动调用 execute_grasp 服务
- ❌ 需要手动控制执行流程

**工作原理:**
1. 从 Azure Kinect 采集 RGB-D 图像
2. 发送到 DexDiffuser API 生成抓取姿态
3. 发布到 ROS topics:
   - `/grasp_pose` - PoseStamped (Franka end-effector 位姿)
   - `/allegroHand_0/joint_cmd` - JointState (Allegro 手指角度)
4. 保存结果到磁盘

**使用方法:**
```bash
# 运行客户端
rosrun test_any_policy grasp_execution_simple.py --server http://100.120.117.28:8000 --objects "cup"

# 命令:
#   ENTER - 采集图像并生成抓取
#   'p'   - 重新发布最佳抓取
#   'a'   - 发布所有抓取
#   'q'   - 退出
```

**适用场景:**
- 快速测试和可视化
- 自定义执行逻辑
- 不想处理服务依赖

---

### 2. `grasp_execution_full.py` - 完整集成版本

**特点:**
- ✅ 完全集成 panda_moveit_allegro 系统
- ✅ 自动调用 execute_grasp 服务执行抓取
- ❌ 需要导入服务定义: `Empty`, `GraspArray`
- ❌ 需要先编译 panda_moveit 包

**工作原理:**
1. 从 Azure Kinect 采集 RGB-D 图像
2. 发送到 DexDiffuser API 生成抓取姿态
3. 存储抓取结果
4. 提供 `grasp_gen` ROS 服务 (GraspArray 类型)
5. 调用 `execute_grasp` 服务
6. panda_moveit_node 通过 `grasp_gen` 服务获取抓取数据
7. panda_moveit_node 自动执行完整抓取流程

**使用前准备:**
```bash
# 1. 编译 panda_moveit 包 (生成服务定义)
cd /home/rpl/dexdiff
catkin_make
source devel/setup.bash

# 2. 确认服务定义可用
python3 -c "from panda_moveit.srv import GraspArray; print('OK')"
```

**使用方法:**
```bash
# Terminal 1: 启动 panda_moveit 系统
roslaunch panda_moveit panda_moveit_allegro.launch

# Terminal 2: 运行完整客户端
cd /home/rpl/dexdiff
source devel/setup.bash
rosrun test_any_policy grasp_execution_full.py --server http://100.120.117.28:8000 --objects "cup"

# 命令:
#   ENTER - 采集图像并生成抓取
#   'e'   - 通过 panda_moveit 执行抓取
#   'q'   - 退出
```

**适用场景:**
- 完整的自动化抓取执行
- 利用 panda_moveit 的规划和碰撞检测
- 生产环境使用

---

## Grasp Pose 格式说明

API 返回的 `best_grasp` 是 **23 个元素**的数组:

```
元素索引    内容                    说明
0-3        [qw, qx, qy, qz]       四元数 (orientation)
4-6        [x, y, z]               位置 (position, 单位: 米)
7-22       [joint_0...joint_15]   Allegro 手指关节角度 (16个, 单位: 弧度)
```

**注意:** API 格式是 [qw, qx, qy, qz], 但 ROS geometry_msgs/Quaternion 是 [x, y, z, w]

---

## 参数说明

```bash
--server       # DexDiffuser API 服务器地址 (默认: http://100.120.117.28:8000)
--objects      # 目标物体名称 (默认: "cookie box")
--confidence   # 检测置信度阈值 (默认: 0.1)
--samples      # 生成抓取数量 (默认: 32)
--calibration  # 相机标定文件路径 (默认: ./calibration_results/eye_to_hand_calibration.npz)
--output       # 结果保存目录 (默认: ./grasp_results)
```

简化版额外参数:
```bash
--publish-all  # 发布所有生成的抓取 (默认: 只发布最佳抓取)
```

---

## 故障排除

### 问题 1: 导入 GraspArray 服务失败

**错误信息:**
```
ImportError: cannot import name 'GraspArray' from 'panda_moveit.srv'
```

**解决方法:**
```bash
# 编译 panda_moveit 包
cd /home/rpl/dexdiff
catkin_make
source devel/setup.bash

# 或者使用简化版本 (不需要服务定义)
rosrun test_any_policy grasp_execution_simple.py
```

### 问题 2: execute_grasp 服务不可用

**错误信息:**
```
Service 'execute_grasp' not available
```

**解决方法:**
```bash
# 确保 panda_moveit 系统正在运行
roslaunch panda_moveit panda_moveit_allegro.launch

# 检查服务是否存在
rosservice list | grep execute_grasp
```

### 问题 3: Azure Kinect 相机无法启动

**解决方法:**
```bash
# 检查相机连接
lsusb | grep Microsoft

# 检查 pykinect_azure 安装
pip list | grep pykinect

# 尝试运行原始客户端测试
rosrun test_any_policy grasp_client_ros_node.py
```

---

## 对比原始版本的改进

### 原始版本 (`grasp_client_ros_node.py`):
- 直接发布到 topics
- 可选的 Franka 控制 (使用 franky library)
- 没有与 panda_moveit 集成

### 简化版本 (`grasp_execution_simple.py`):
- 移除了 franky 依赖
- 只发布到 ROS topics
- 支持重新发布抓取
- 更简单的依赖

### 完整版本 (`grasp_execution_full.py`):
- 完全集成 panda_moveit 系统
- 使用 ROS 服务架构
- 自动化执行流程
- 利用 MoveIt 的规划能力

---

## 工作流程图

```
简化版本流程:
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Azure Kinect│────>│ DexDiffuser  │────>│ Publish Topics  │
│   Camera    │     │     API      │     │ (手动执行)      │
└─────────────┘     └──────────────┘     └─────────────────┘

完整版本流程:
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Azure Kinect│────>│ DexDiffuser  │────>│ Store Results   │
│   Camera    │     │     API      │     │ + Provide Srv   │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                    ┌──────────────────────────────┘
                    │ Call execute_grasp
                    ▼
        ┌──────────────────────┐
        │  panda_moveit_node   │
        │  1. Call grasp_gen   │◄────┐
        │  2. Plan trajectory  │     │ Provide grasps
        │  3. Execute motion   │─────┘
        └──────────────────────┘
```

---

## 推荐使用流程

1. **开始测试**: 使用 `grasp_execution_simple.py`
   - 快速验证 API 连接和抓取生成
   - 可视化抓取姿态
   - 不需要处理服务依赖

2. **集成执行**: 切换到 `grasp_execution_full.py`
   - 编译 panda_moveit 包
   - 启动完整系统
   - 自动化执行抓取

---

## 相关文件

- `grasp_client_ros_node.py` - 原始版本 (直接控制)
- `grasp_service_node.py` - API 服务端
- `README_GRASP_SERVICE.md` - API 服务说明
- `README_GRASP_ROS_NODE.md` - 原始客户端说明
