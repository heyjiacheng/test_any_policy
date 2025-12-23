# Grasp Client ROS Node

This script is a ROS-integrated. It can:
1. Capture RGB-D data from Azure Kinect camera
2. Send data to DexDiffuser API server to generate grasp poses
3. **Publish the best grasp pose (last 16 joint angles) to ROS topic `/allegroHand_0/joint_cmd`**

## Features

- Added ROS Publisher to publish `sensor_msgs/JointState` messages
- Automatically extracts the last 16 dimensions from best_grasp (dexterous hand joint positions)
- Publishes to `/allegroHand_0/joint_cmd` topic to control Allegro Hand

## Dependencies

### Conda Environment Setup
```bash
conda create -n ros_franka python=3.9 -y
conda activate ros_franka
pip install opencv-python pykinect_azure requests
conda install ros-noetic-rospy ros-noetic-common-msgs
```

**trouble conda install**
```bash
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --set channel_priority strict
conda install ros-noetic-rospy ros-noetic-common-msgs
```

### install franky
due to old firm of franka arm, use:
```bash
VERSION=0-9-2
wget https://github.com/TimSchneider42/franky/releases/latest/download/libfranka_${VERSION}_wheels.zip
unzip libfranka_${VERSION}_wheels.zip
pip install numpy
pip install --no-index --find-links=./dist franky-control
```

## Usage

### 1. Launch ROS Master and Panda MoveIt Node
First, start the launch file that receives joint commands in a terminal:
```bash
cd ~/dexdiff
source devel/setup.bash
roslaunch allegro_hand_controllers allegro_hand_franka.launch
```

### 2. Start API Server
In another terminal, start the DexDiffuser API server (if not already running):
```bash
# Depending on your API server startup method
python api_server.py
```

### 3. Run Grasp Client ROS Node

#### Basic Usage
```bash
python scripts/real_control/grasp_client_ros_node.py
```
if you're on SSH
```bash
DISPLAY=:0 python scripts/real_control/grasp_client_ros_node.py
```

#### Specify Server and Object
```bash
python scripts/real_control/grasp_client_ros_node.py \
    --server http://100.120.117.28:8000 \
    --objects "cup"
```

## Output

### 1. Saved Files
- `grasp_results/point_cloud_{object}.ply`: Point cloud file
- `grasp_results/grasp_poses_{object}.npz`: Grasp pose arrays

### 2. ROS Topic
- **Topic Name**: `/allegroHand_0/joint_cmd`
- **Message Type**: `sensor_msgs/JointState`
- **Joint Names**: `['joint_0_0', 'joint_1_0', ..., 'joint_15_0']` (16 joints)
- **Joint Positions**: Last 16 dimensions extracted from best_grasp (radians)

## Important Notes

1. **Safety First**: Ensure there are no obstacles around the robot and dexterous hand before running
2. **Calibration**: Use correct eye-to-hand calibration file for accurate grasp poses
3. **Joint Limits**: API returned joint angles should be within Allegro Hand's limit range
4. **ROS Master**: Ensure ROS master and related nodes are started before running the script
