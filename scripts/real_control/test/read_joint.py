from franky import Robot, JointMotion
import numpy as np

# Joint 7 (last joint) limits in radians
JOINT_7_MIN = -0.58  # minimum angle
JOINT_7_MAX = 2.49   # maximum angle

robot = Robot("172.16.1.22")
robot.relative_dynamics_factor = 0.1


def check_joint_limits(joint_positions, joint_idx, min_val, max_val):
    """Check if a joint position is within limits."""
    return min_val <= joint_positions[joint_idx] <= max_val

# Example: Read current joint state
while True:
    joint_state = robot.current_joint_state
    print(f"Current joints: {joint_state.position}")

    # Check if joint 7 is within limits
    if not check_joint_limits(joint_state.position, 6, JOINT_7_MIN, JOINT_7_MAX):
        print(f"WARNING: Joint 7 out of bounds: {joint_state.position[6]:.3f} rad")
        break



