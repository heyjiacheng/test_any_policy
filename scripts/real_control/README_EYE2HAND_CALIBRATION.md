# 🤖 Eye-to-Hand Calibration for Franka Robot + Azure Kinect

Calibrate the transformation between Azure Kinect camera and Franka robot base frame using ArUco markers.

---

## 📋 Prerequisites

- Ubuntu 20 or 22 (tested)
- Franka Panda robot
- Azure Kinect camera
- ArUco marker (ID: 24, size: 100mm) attached to robot end-effector

---

## 🛠️ Installation

### 1. Create Conda Environment

```bash
conda create -n real_control python=3.12 -y
conda activate real_control
```

### 2. Install Azure Kinect SDK

Follow instructions here: https://blog.csdn.net/qq_48152826/article/details/137008989

SDK packages for Ubuntu 18/20/22 available at: https://packages.microsoft.com/ubuntu/18.04/multiarch/prod/

Install additional dependency:
```bash
sudo apt install libsoundio1
```

### 3. Install pykinect-azure

Follow instructions here: https://github.com/ibaiGorordo/pyKinectAzure

```bash
pip install pykinect-azure
```

### 4. Install franky-control

Follow instructions here: https://github.com/TimSchneider42/franky

**Note:** Requires an older version of libfranka. Follow the installation guide carefully.

```bash
pip install franky-control
```

### 5. Install Additional Python Dependencies

```bash
pip install opencv-contrib-python numpy scipy
```

---

## 🚀 Quick Start

### Interactive Mode (Recommended for First-Time Setup)

Move the robot manually to different poses and capture calibration data:

```bash
python eye_to_hand_calibration.py --mode interactive
```

**Controls:**
- `c` - Capture current pose (auto-saves)
- `r` - Remove last pose
- `d` - Run calibration (when ≥10 poses captured)
- `q` - Quit

**Tips:**
- Vary robot orientation around all axes
- Keep ArUco marker visible to camera
- Capture at least 10 diverse poses
- Ensure good lighting

### Automatic Mode (Replay Saved Trajectory)

Automatically replay previously saved poses and re-calibrate:

```bash
python eye_to_hand_calibration.py --mode auto
```

**Note:** Requires pose data from previous interactive session.

---

## 📂 Output Files

Calibration results saved to `./calibration_results/`:
- `eye_to_hand_calibration_YYYYMMDD_HHMMSS.npz` - Calibration matrices (NumPy format)
- `eye_to_hand_calibration_YYYYMMDD_HHMMSS.json` - Calibration data (human-readable)
- `pose_data.npz` - Saved robot poses for auto mode replay

---

