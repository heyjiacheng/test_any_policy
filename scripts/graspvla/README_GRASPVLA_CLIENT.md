# GraspVLA + ManiSkill Integration

This directory integrates the GraspVLA Vision-Language-Action (VLA) model with the ManiSkill simulation environment, replacing the original LIBERO environment used in GraspVLA playground.

## System Architecture

```
┌──────────────────────────────────────────────────┐
│  GraspVLA Policy Server (ZMQ Port: 6666)         │
│  - Receives: images + proprioception history     │
│  - Outputs: delta actions (relative movements)   │
└────────────────┬─────────────────────────────────┘
                 │ ZMQ Communication
┌────────────────▼─────────────────────────────────┐
│  RemoteAgent (remote_agent.py)                   │
│  - Collects observations from ManiSkill          │
│  - Converts: world frame → robot base frame      │
│  - Converts: delta actions → absolute actions    │
│  - Converts: GraspVLA gripper ↔ ManiSkill        │
└────────────────┬─────────────────────────────────┘
                 │ Absolute poses
┌────────────────▼─────────────────────────────────┐
│  ManiSkill Environment                           │
│  - IK solver: pose → joint positions             │
│  - Physics simulation + rendering                │
└──────────────────────────────────────────────────┘
```

## Files

- **`remote_agent.py`**: Communicates with GraspVLA server, handles all data conversions
- **`run_graspvla.py`**: Run GraspVLA with custom objects
- **`run_graspvla_ycb.py`**: Run GraspVLA with standard PickClutterYCB-v1 environment


## Usage

### Step 1: Start the GraspVLA Policy Server

First, launch the GraspVLA policy server in a separate terminal:

```bash
cd others/GraspVLA/vla_network
python scripts/serve.py --port 6666 --path <path_to_your_model_checkpoint>
```

The server will:
- Load the VLA model
- Warm up with test samples
- Listen on port 6666 for ZMQ requests

### Step 2: Run the ManiSkill Integration

Once the server is running, launch the ManiSkill integration:

#### Basic Usage

```bash
# Run with default mug object
python scripts/graspvla/run_graspvla_ycb.py --instruction "pick up the mug"
```
