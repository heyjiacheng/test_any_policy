# Grasp Client

DexDiffuser grasp generation client with Franka + Allegro Hand control.

## Pre

launch ros node to control allegro hand

```bash
cd ~/dexdiff
source devel/setup.bash
./switch_to_pcan.sh
roslaunch allegro_hand_controllers allegro_hand_franka.launch
```

## Quick Start

```bash
conda activate ros_franka
```

```bash
# Basic usage (uses defaults)
python3 grasp_client.py

# Custom object
python3 grasp_client.py --objects "cup"
```

## Defaults

- Server: `http://100.120.117.28:8000`
- Objects: `"cookie box"`
- Calibration: `./calibration_results/eye_to_hand_calibration.npz`
- Robot IP: `172.16.1.22`

## Workflow

1. Press ENTER to capture RGB-D image
2. Server generates grasp pose
3. Choose whether to execute robot motion (y/n)
4. Robot executes: Home → Pre-grasp → Grasp → Close hand → Lift

## Requirements

- ROS node running
- Azure Kinect connected
- Franka robot + Allegro Hand
- Calibration file (optional, for eye-to-hand transform)
