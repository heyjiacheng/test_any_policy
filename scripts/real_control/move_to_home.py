#!/usr/bin/env python3
"""
Move Franka Robot and Allegro Hand to Home Position
====================================================
Simple script to move both Franka robot and Allegro Hand to their home positions.
"""

import numpy as np
import time
import argparse

# ROS imports
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# Franky imports
from franky import Robot, JointMotion

# ============================================================================
# Configuration
# ============================================================================
DEFAULT_ROBOT_IP = "172.16.1.22"
DEFAULT_ROBOT_DYNAMICS_FACTOR = 0.1

# Franka home position (joint angles in radians)
FRANKA_HOME_POSITION = [0.0, -0.78, 0.0, -2.36, 0.0, 1.57, 0.78]

# Allegro Hand home position (joint angles in degrees, will be converted to radians)
ALLEGRO_HOME_POSITION_DEG = [
    0., -10., 45., 45.,   # Index finger
    0., -10., 45., 45.,   # Middle finger
    5., -5., 50., 45.,    # Ring finger
    5., 5., 5., 5.        # Thumb
]

# Allegro Hand joint names
ALLEGRO_JOINT_NAMES = [
    'joint_0_0', 'joint_1_0', 'joint_2_0', 'joint_3_0',
    'joint_4_0', 'joint_5_0', 'joint_6_0', 'joint_7_0',
    'joint_8_0', 'joint_9_0', 'joint_10_0', 'joint_11_0',
    'joint_12_0', 'joint_13_0', 'joint_14_0', 'joint_15_0'
]


class AllegroHandPublisher:
    """ROS publisher for Allegro Hand joint commands."""

    def __init__(self):
        self.joint_cmd_pub = rospy.Publisher('/allegroHand_0/joint_cmd', JointState, queue_size=10)
        # Convert home position from degrees to radians
        self.home_position = np.array([deg / 180.0 * np.pi for deg in ALLEGRO_HOME_POSITION_DEG])

    def publish_joint_command(self, joint_positions: np.ndarray):
        """Publish joint command to Allegro Hand."""
        if len(joint_positions) != 16:
            rospy.logerr(f"Expected 16 joint positions, got {len(joint_positions)}")
            return

        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = ALLEGRO_JOINT_NAMES
        joint_state.position = joint_positions.tolist()
        self.joint_cmd_pub.publish(joint_state)
        print("--> Allegro Hand command published.")

    def move_to_home(self):
        """Move Allegro Hand to home position."""
        print("Moving Allegro Hand to home position...")
        self.publish_joint_command(self.home_position)
        print("--> Allegro Hand moved to home position.")


class FrankaRobotController:
    """Franka robot controller for moving to home position."""

    def __init__(self, robot_ip: str, dynamics_factor: float = DEFAULT_ROBOT_DYNAMICS_FACTOR):
        self.robot_ip = robot_ip
        self.dynamics_factor = dynamics_factor
        self.robot = None

    def connect(self):
        """Connect to Franka robot."""
        print(f"Connecting to Franka robot at {self.robot_ip}...")
        self.robot = Robot(self.robot_ip)
        self.robot.recover_from_errors()
        self.robot.relative_dynamics_factor = self.dynamics_factor
        print("Connected to Franka robot.")

    def disconnect(self):
        """Disconnect from Franka robot."""
        self.robot = None
        print("Disconnected from Franka robot.")

    def move_to_home(self):
        """Move Franka robot to home position."""
        if self.robot is None:
            print("Error: Robot not connected")
            return False

        print("Moving Franka robot to home position...")
        try:
            joint_motion = JointMotion(FRANKA_HOME_POSITION)
            self.robot.move(joint_motion)
            print("--> Franka robot moved to home position.")
            return True
        except Exception as e:
            print(f"Error moving to home position: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Move Franka and Allegro Hand to Home Position')
    parser.add_argument('--robot-ip', type=str, default=DEFAULT_ROBOT_IP,
                        help=f'Franka robot IP address (default: {DEFAULT_ROBOT_IP})')
    parser.add_argument('--dynamics-factor', type=float, default=DEFAULT_ROBOT_DYNAMICS_FACTOR,
                        help=f'Robot dynamics factor (default: {DEFAULT_ROBOT_DYNAMICS_FACTOR})')
    args = parser.parse_args()

    # Initialize ROS node
    rospy.init_node('move_to_home_node', anonymous=True)

    # Create controllers
    allegro_publisher = AllegroHandPublisher()
    franka_controller = FrankaRobotController(args.robot_ip, args.dynamics_factor)

    try:
        print("\n" + "="*60)
        print("Moving Franka Robot and Allegro Hand to Home Position")
        print("="*60 + "\n")

        # Connect to Franka robot
        franka_controller.connect()

        # Give ROS publisher time to initialize
        time.sleep(0.5)

        # Move Allegro Hand to home position
        allegro_publisher.move_to_home()
        time.sleep(1.0)  # Wait for hand to reach home position

        # Move Franka robot to home position
        success = franka_controller.move_to_home()

        if success:
            print("\n" + "="*60)
            print("Successfully moved both to home position!")
            print("="*60 + "\n")
        else:
            print("\n" + "="*60)
            print("Failed to move Franka to home position.")
            print("="*60 + "\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
    finally:
        franka_controller.disconnect()


if __name__ == '__main__':
    main()
