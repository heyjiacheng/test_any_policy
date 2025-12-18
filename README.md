# 🤖 test_any_policy

A robotic manipulation framework integrating vision-language-action models with simulation (ManiSkill) and real robot deployment (Franka + Azure Kinect).

## 📋 Components

### Simulation & Data Capture
**[ManiSkill Data Capture →](scripts/capture/README_MANISKILL.md)**
- Visualize policies and trajectories in ManiSkill
- Capture multi-view images/videos with custom or articulated objects
- Generate training datasets with camera parameters

### GraspVLA Policy
**[GraspVLA Integration →](scripts/graspvla/README_GRASPVLA_CLIENT.md)**
- Run vision-language-action model in ManiSkill
- Support for custom objects and YCB datasets
- Client-server architecture for policy inference

### Real Robot Control
**[Eye-to-Hand Calibration →](scripts/real_control/README_EYE2HAND_CALIBRATION.md)**
- Calibrate Azure Kinect to Franka base frame
- Interactive and automatic modes with ArUco markers

**[Grasp ROS Node →](scripts/real_control/README_GRASP_ROS_NODE.md)**
- ROS-integrated grasp execution
- RGB-D capture and DexDiffuser API integration
- Allegro Hand control via joint commands

## 🚀 Quick Start

```bash
# 1. Test in simulation
python scripts/graspvla/run_graspvla_ycb.py --instruction "pick up the mug"

# 2. Calibrate real robot
python scripts/real_control/eye_to_hand_calibration.py --mode interactive

# 3. Deploy to real robot
python scripts/real_control/grasp_client_ros_node.py --objects "cup"
```

## 📂 Structure

```
test_any_policy/
├── scripts/
│   ├── capture/          # ManiSkill data capture
│   ├── graspvla/         # GraspVLA integration
│   ├── real_control/     # Real robot deployment
│   └── contact_graspnet/ # Pose conversion utilities
├── envs/                 # Environment interfaces
├── policies/             # Policy algorithms
└── data/                 # Training data & weights
```

