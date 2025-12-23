#!/usr/bin/env python3
"""
DexDiffuser Grasp Generation Client with ROS Integration (Enhanced Sequence)
============================================================================
Modified to perform Approach -> Grasp -> Lift sequence.
"""

import cv2
import numpy as np
import pykinect_azure as pykinect
import requests
import os
import argparse
import io
import base64
import time
from typing import Optional, Tuple, Dict, Any, List

# ROS imports
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# Franky imports
from franky import Affine, Robot, CartesianMotion, JointMotion

# ============================================================================
# Default Configuration
# ============================================================================
DEFAULT_SERVER_URL = "http://100.120.117.28:8000"
DEFAULT_TARGET_OBJECTS = "cookie box"
DEFAULT_CONFIDENCE_THRESHOLD = 0.1
DEFAULT_NUM_SAMPLES = 32
CALIBRATION_FILE = "./calibration_results/eye_to_hand_calibration.npz"
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

# ============================================================================
# Math Helper Functions
# ============================================================================
def quaternion_to_matrix(q):
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2,  2*qx*qy - 2*qz*qw,      2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,      1 - 2*qx**2 - 2*qz**2,  2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,      2*qy*qz + 2*qx*qw,      1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def create_transform_matrix(pos, quat):
    """Create 4x4 transformation matrix from pos [x,y,z] and quat [qx,qy,qz,qw]."""
    T = np.eye(4)
    T[:3, :3] = quaternion_to_matrix(quat)
    T[:3, 3] = pos
    return T

def matrix_to_pos_quat(T):
    """Convert 4x4 matrix back to pos [x,y,z] and quat [qx,qy,qz,qw]."""
    pos = T[:3, 3]
    R = T[:3, :3]
    
    # Rotation matrix to quaternion conversion
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    elif (R[1,1] > R[2,2]):
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S
        
    return pos, np.array([qx, qy, qz, qw])


class AzureKinectClient:
    """Azure Kinect camera client for capturing RGB-D data."""
    # ... (Same as original code) ...
    def __init__(self, calibration_file: Optional[str] = None):
        pykinect.initialize_libraries()
        self.device_config = pykinect.default_configuration
        self.device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
        self.device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1440P
        self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
        self.device = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_extrinsics = None
        self.calibration_file = calibration_file

    def start(self):
        print("Starting Azure Kinect camera...")
        self.device = pykinect.start_device(config=self.device_config)
        self._get_intrinsics()
        if self.calibration_file and os.path.exists(self.calibration_file):
            self._load_extrinsics()
        else:
            print("No calibration file provided or file not found. Extrinsics will be None.")

    def stop(self):
        if self.device is not None:
            self.device = None
        print("Azure Kinect camera stopped.")

    def _get_intrinsics(self):
        calibration = self.device.calibration
        color_params = calibration._handle.color_camera_calibration.intrinsics.parameters.param
        self.camera_matrix = np.array([
            [color_params.fx, 0, color_params.cx],
            [0, color_params.fy, color_params.cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def _load_extrinsics(self):
        try:
            data = np.load(self.calibration_file)
            self.camera_extrinsics = data['T_base_cam']
        except Exception as e:
            print(f"Warning: Failed to load extrinsics: {e}")
            self.camera_extrinsics = None

    def capture_rgbd(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        capture = self.device.update()
        ret_color, color_image = capture.get_color_image()
        if not ret_color: return False, None, None
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
        ret_depth, depth_image = capture.get_transformed_depth_image()
        if not ret_depth: return False, None, None
        return True, color_image, depth_image

    def get_intrinsics_3x3(self) -> np.ndarray:
        return self.camera_matrix

    def get_extrinsics_4x4(self) -> Optional[np.ndarray]:
        return self.camera_extrinsics


class GraspGenerationClient:
    """Client for DexDiffuser Grasp Generation API."""
    # ... (Same as original code) ...
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')

    def process_grasp(self, rgb_image, depth_data, camera_intrinsics, target_objects, 
                      camera_extrinsics=None, confidence_threshold=0.1, num_samples=32):
        url = f"{self.server_url}/process_grasp"
        files = {}
        success, rgb_encoded = cv2.imencode('.png', rgb_image)
        files['rgb_image'] = ('rgb_image.png', rgb_encoded.tobytes(), 'image/png')
        
        depth_bytes = io.BytesIO()
        np.save(depth_bytes, depth_data)
        depth_bytes.seek(0)
        files['depth_data'] = ('depth_data.npy', depth_bytes, 'application/octet-stream')
        
        intrinsics_bytes = io.BytesIO()
        np.save(intrinsics_bytes, camera_intrinsics)
        intrinsics_bytes.seek(0)
        files['camera_intrinsics'] = ('camera_intrinsics.npy', intrinsics_bytes, 'application/octet-stream')

        if camera_extrinsics is not None:
            extrinsics_bytes = io.BytesIO()
            np.save(extrinsics_bytes, camera_extrinsics)
            extrinsics_bytes.seek(0)
            files['camera_extrinsics'] = ('camera_extrinsics.npy', extrinsics_bytes, 'application/octet-stream')

        data = {
            'target_objects': target_objects,
            'confidence_threshold': confidence_threshold,
            'num_samples': num_samples
        }
        
        print(f"Sending request to {url}...")
        response = requests.post(url, files=files, data=data, timeout=60)
        response.raise_for_status()
        return response.json()


class AllegroHandPublisher:
    """ROS publisher for Allegro Hand joint commands."""
    def __init__(self):
        self.joint_cmd_pub = rospy.Publisher('/allegroHand_0/joint_cmd', JointState, queue_size=10)
        # Convert home position from degrees to radians
        self.home_position = np.array([deg / 180.0 * np.pi for deg in ALLEGRO_HOME_POSITION_DEG])

    def publish_joint_command(self, joint_positions: np.ndarray):
        if len(joint_positions) != 16:
            rospy.logerr(f"Expected 16 joint positions, got {len(joint_positions)}")
            return
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = ALLEGRO_JOINT_NAMES
        joint_state.position = joint_positions.tolist()
        self.joint_cmd_pub.publish(joint_state)
        print(f"--> Allegro Hand command published.")

    def move_to_home(self):
        """Move Allegro Hand to home position."""
        print("Moving Allegro Hand to home position...")
        self.publish_joint_command(self.home_position)
        print("--> Allegro Hand moved to home position.")


class FrankaRobotController:
    """Franka robot controller for executing grasp poses with approach and lift."""

    def __init__(self, robot_ip: str, dynamics_factor: float = DEFAULT_ROBOT_DYNAMICS_FACTOR):
        self.robot_ip = robot_ip
        self.dynamics_factor = dynamics_factor
        self.robot = None

    def connect(self):
        print(f"Connecting to Franka robot at {self.robot_ip}...")
        self.robot = Robot(self.robot_ip)
        self.robot.recover_from_errors()
        self.robot.relative_dynamics_factor = self.dynamics_factor
        print("Connected to Franka robot.")

    def disconnect(self):
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

    def execute_grasp_sequence(self, grasp_pos, grasp_quat, hand_publisher, joint_angles):
        """
        Executes the full grasp sequence with HAND LENGTH COMPENSATION.
        """
        if self.robot is None:
            print("Error: Robot not connected")
            return

        # ==========================================
        # 1. 定义手掌长度偏移 (关键修改)
        # ==========================================
        # Allegro Hand + 法兰转接件的大致长度 (单位: 米)
        # 建议先设大一点 (如 0.20)，测试没问题后再减小到 0.15 或 0.13
        HAND_LENGTH_OFFSET = 0.18

        # 1. Calculate Grasp Matrix (原始物体坐标)
        T_object_grasp = create_transform_matrix(grasp_pos, grasp_quat)

        # 2. 计算修正后的机器人目标位姿 (Robot Flange Target)
        # 抓取坐标系的 X 轴指向物体 
        # 我们需要沿着局部 X 轴向后退 HAND_LENGTH_OFFSET
        T_back_off = np.eye(4)
        T_back_off[2, 3] = -HAND_LENGTH_OFFSET  # Local X translation
        
        # 新的“真”抓取点：手腕的位置
        T_robot_grasp = np.dot(T_object_grasp, T_back_off)
        
        # 将矩阵转回 pos, quat
        robot_grasp_pos, robot_grasp_quat = matrix_to_pos_quat(T_robot_grasp)

        # ==========================================
        # 3. 计算 Pre-Grasp (基于修正后的手腕位置继续后退)
        # ==========================================
        # 保持你原有的 Pre-grasp 逻辑，但基准点变成了 robot_grasp_pos
        # 如果你希望 Pre-grasp 离物体更远，可以保持原有逻辑
        
        T_offset_pre = np.eye(4)
        # 这里是相对于“已经补偿过手长”的位置再后退 15cm，下沉 5cm
        # 如果觉得太远，可以将 -0.15 改为 -0.05
        T_offset_pre[:3, 3] = np.array([-0.015, 0.0, -0.05]) 
        
        T_pre_grasp = np.dot(T_robot_grasp, T_offset_pre)
        pre_pos, pre_quat = matrix_to_pos_quat(T_pre_grasp)

        # 4. Calculate Lift Pose
        lift_pos = robot_grasp_pos.copy() # 使用修正后的位置抬起
        lift_pos[2] += 0.15  # Global Z lift (稍微抬高一点以防万一)
        lift_quat = robot_grasp_quat.copy()

        print("\n=== Starting Grasp Sequence (With Offset) ===")
        print(f"Original Target: {grasp_pos}")
        print(f"Compensated Target (Wrist): {robot_grasp_pos}")
        
        # --- Step A: Move to Pre-Grasp ---
        print(f"1. Moving to Pre-Grasp Pose...")
        affine_pre = Affine(pre_pos.tolist(), pre_quat.tolist())
        self.robot.move(CartesianMotion(affine_pre))
        
        # --- Step B: Move to Grasp Pose (Linear/Approach) ---
        print(f"2. Approaching Grasp Pose...")
        # 注意：这里使用的是修正后的 robot_grasp_pos
        affine_grasp = Affine(robot_grasp_pos.tolist(), robot_grasp_quat.tolist())
        self.robot.move(CartesianMotion(affine_grasp))
        
        # --- Step C: Close Hand ---
        print(f"3. Closing Hand...")
        hand_publisher.publish_joint_command(joint_angles)
        time.sleep(2.0)
        
        # --- Step D: Lift ---
        print(f"4. Lifting Object...")
        affine_lift = Affine(lift_pos.tolist(), lift_quat.tolist())
        self.robot.move(CartesianMotion(affine_lift))
        
        print("=== Sequence Complete ===\n")


def save_results(result, output_dir, target_objects):
    os.makedirs(output_dir, exist_ok=True)
    grasp_qt = np.array(result['grasp_qt'])
    np.savez(os.path.join(output_dir, f"grasp_poses_{target_objects}.npz"),
             grasp_qt=grasp_qt,
             best_grasp=result['best_grasp'])
    return target_objects


def main():
    parser = argparse.ArgumentParser(description='DexDiffuser Grasp Client with ROS Sequence')
    parser.add_argument('--server', type=str, default=DEFAULT_SERVER_URL)
    parser.add_argument('--objects', type=str, default=DEFAULT_TARGET_OBJECTS)
    parser.add_argument('--calibration', type=str, default=CALIBRATION_FILE)
    parser.add_argument('--robot-ip', type=str, default=DEFAULT_ROBOT_IP)
    args = parser.parse_args()

    rospy.init_node('grasp_client_ros_node', anonymous=True)

    camera = AzureKinectClient(calibration_file=args.calibration)
    api_client = GraspGenerationClient(server_url=args.server)
    allegro_publisher = AllegroHandPublisher()

    franka_controller = FrankaRobotController(args.robot_ip)
    franka_connected = False

    try:
        camera.start()
        print("\nPress ENTER to capture, 'q' to quit")

        while not rospy.is_shutdown():
            if input("\nCapture? (ENTER/q): ").strip().lower() == 'q': break

            success, rgb, depth = camera.capture_rgbd()
            if not success: continue

            intrinsics = camera.get_intrinsics_3x3()
            extrinsics = camera.get_extrinsics_4x4()

            result = api_client.process_grasp(
                rgb, depth, intrinsics, args.objects, camera_extrinsics=extrinsics
            )
            
            save_results(result, './grasp_results', args.objects)

            # Data Extraction
            best_grasp = np.array(result['best_grasp'])

            # API returns Quaternion as [qw, qx, qy, qz]
            qw, qx, qy, qz = best_grasp[0:4]
            # Convert to [qx, qy, qz, qw] for Franky/Scipy standard
            quat_xyzw = np.array([qx, qy, qz, qw])

            position = best_grasp[4:7]
            joint_positions = best_grasp[7:]

            # Calculate compensated target (same logic as in execute_grasp_sequence)
            HAND_LENGTH_OFFSET = 0.2
            T_object_grasp = create_transform_matrix(position, quat_xyzw)
            T_back_off = np.eye(4)
            T_back_off[0, 3] = -HAND_LENGTH_OFFSET
            T_robot_grasp = np.dot(T_object_grasp, T_back_off)
            robot_grasp_pos, robot_grasp_quat = matrix_to_pos_quat(T_robot_grasp)

            # Prompt user whether to execute Franka motion and grasp
            print("\n" + "="*60)
            print("Grasp pose received from server!")
            print(f"Original Target (Object): {position}")
            print(f"Compensated Target (Wrist): {robot_grasp_pos}")
            print(f"Orientation (quat): {quat_xyzw}")
            print("="*60)

            execute_motion = input("Execute Franka motion and grasp? (y/n): ").strip().lower()

            if execute_motion == 'y':
                # Connect to robot if not already connected
                if not franka_connected:
                    print("Connecting to Franka robot...")
                    franka_controller.connect()
                    franka_connected = True

                # Move to home position first
                print("\n" + "="*60)
                print("Moving to home position...")
                print("="*60)

                # Move Allegro hand to home position
                allegro_publisher.move_to_home()
                time.sleep(1.0)  # Wait for hand to reach home position

                # Move Franka to home position
                if not franka_controller.move_to_home():
                    print("Failed to move Franka to home position. Skipping grasp execution.")
                    continue

                time.sleep(0.5)  # Brief pause before starting grasp sequence

                # Execute the grasp sequence
                franka_controller.execute_grasp_sequence(
                    grasp_pos=position,
                    grasp_quat=quat_xyzw,
                    hand_publisher=allegro_publisher,
                    joint_angles=joint_positions
                )
            else:
                print("Skipping Franka motion. Publishing hand command only...")
                allegro_publisher.publish_joint_command(joint_positions)

    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
    finally:
        camera.stop()
        if franka_connected:
            franka_controller.disconnect()

if __name__ == '__main__':
    main()