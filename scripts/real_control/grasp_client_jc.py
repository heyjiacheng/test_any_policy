#!/usr/bin/env python3
"""
DexDiffuser Grasp Generation Client with ROS Integration (Interactive Selection)
================================================================================
Modified to work with api_service.py:
1. Upload RGB-D -> Server visualizes grasps (POST /process_grasp).
2. User inputs index -> Client retrieves specific grasp (POST /select_grasp).
3. Client aligns Quaternion (wxyz -> xyzw) and Frame (Offset).
4. Approach -> Grasp -> Lift sequence.

Updates:
- Added --eye-in-hand support.
- Calculates T_base_camera dynamically if camera is on the end-effector.
- REFACTORED: Strict calibration key handling and 6D Hand-to-Flange transform.
"""

import cv2
import numpy as np
import pykinect_azure as pykinect
import requests
import os
import argparse
import io
import time
import sys
from typing import Optional, Tuple, List

# # ROS imports
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# Franky imports
from franky import Affine, Robot, CartesianMotion, JointMotion

# ============================================================================
# Default Configuration
# ============================================================================
DEFAULT_SERVER_URL = "http://100.122.228.55:8000"
DEFAULT_TARGET_OBJECTS = "blue box"
DEFAULT_CONFIDENCE_THRESHOLD = 0.1
DEFAULT_NUM_SAMPLES = 32

# Calibration Files
CALIBRATION_FILE_EYE_TO_HAND = "./calibration_results/eye_to_hand_calibration.npz"
CALIBRATION_FILE_EYE_ON_HAND = "./calibration_results/eye_on_hand_calibration.npz"

DEFAULT_ROBOT_IP = "172.16.1.22"
DEFAULT_ROBOT_DYNAMICS_FACTOR = 0.03

# ----------------------------------------------------------------------------
# Flange to Palm Transformation Configuration
# ----------------------------------------------------------------------------
# Translation: [x, y, z] in meters
FLANGE_TO_PALM_TRANSLATION = [0.026, -0.026, 0.07]
# Rotation: [r, p, y] in degrees (Euler XYZ)
FLANGE_TO_PALM_ROTATION_DEG = [0, -90, 135] 

# Franka home position (joint angles in radians)
FRANKA_HOME_POSITION = [0.0, -0.78, 0.0, -2.36, 0.0, 1.57, 0.78]

# Allegro Hand home position (joint angles in degrees -> radians)
ALLEGRO_HOME_POSITION_DEG = [
    0., 0., 45., 45.,   # Index
    0., 0., 45., 45.,   # Middle
    5., 5., 50., 45.,    # Ring
    5., 5., 5., 5.        # Thumb
]

ALLEGRO_JOINT_NAMES = [
    'joint_0_0', 'joint_1_0', 'joint_2_0', 'joint_3_0',
    'joint_4_0', 'joint_5_0', 'joint_6_0', 'joint_7_0',
    'joint_8_0', 'joint_9_0', 'joint_10_0', 'joint_11_0',
    'joint_12_0', 'joint_13_0', 'joint_14_0', 'joint_15_0'
]

# ============================================================================
# Math Helper Functions
# ============================================================================
def euler_to_matrix(r_deg, p_deg, y_deg):
    """Convert Euler angles (degrees, XYZ sequence) to 3x3 Rotation Matrix."""
    r = np.radians(r_deg)
    p = np.radians(p_deg)
    y = np.radians(y_deg)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r), np.cos(r)]])
    
    Ry = np.array([[np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y), np.cos(y), 0],
                   [0, 0, 1]])
    
    # Sequence XYZ: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx

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

def get_flange_to_palm_transform():
    """
    Constructs the static transform T_flange_palm from configuration.
    """
    T = np.eye(4)
    # Set Rotation
    R = euler_to_matrix(*FLANGE_TO_PALM_ROTATION_DEG)
    T[:3, :3] = R
    # Set Translation
    T[:3, 3] = FLANGE_TO_PALM_TRANSLATION
    return T


class AzureKinectClient:
    """Azure Kinect camera client for capturing RGB-D data."""
    def __init__(self, calibration_file: Optional[str] = None, eye_in_hand: bool = False):
        pykinect.initialize_libraries()
        self.device_config = pykinect.default_configuration
        self.device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
        self.device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1440P
        self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
        self.device = None
        self.camera_matrix = None
        self.T_ref_cam = None  # T_base_cam (eye-to-hand) or T_flange_cam (eye-in-hand)
        self.calibration_file = calibration_file
        self.eye_in_hand = eye_in_hand

    def start(self):
        print("Starting Azure Kinect camera...")
        self.device = pykinect.start_device(config=self.device_config)
        self._get_intrinsics()
        if self.calibration_file and os.path.exists(self.calibration_file):
            self._load_calibration()
        else:
            print(f"Warning: Calibration file {self.calibration_file} not found. Extrinsics will be None.")

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

    def _load_calibration(self):
        """
        Loads calibration matrix based on strictly defined keys.
        Eye-to-Hand: 'T_base_cam' (camera to base transform, static)
        Eye-in-Hand: 'T_flange_cam' (camera to flange transform, static)
        """
        try:
            data = np.load(self.calibration_file)
            print(f"Loading calibration from: {self.calibration_file}")
            print(f"Keys found: {list(data.keys())}")

            if self.eye_in_hand:
                if 'T_flange_cam' in data:
                    self.T_ref_cam = data['T_flange_cam']
                    print("Loaded 'T_flange_cam' (Eye-in-Hand: flange to camera).")
                elif 'T_ee_cam' in data:
                    self.T_ref_cam = data['T_ee_cam']
                    print("Loaded 'T_ee_cam' (Fallback for Eye-in-Hand).")
                else:
                    print("Error: Eye-in-Hand mode selected but 'T_flange_cam' or 'T_ee_cam' key not found.")
            else:
                if 'T_base_cam' in data:
                    self.T_ref_cam = data['T_base_cam']
                    print("Loaded 'T_base_cam' (Eye-to-Hand: base to camera).")
                else:
                    print("Error: Eye-to-Hand mode selected but 'T_base_cam' key not found.")

        except Exception as e:
            print(f"Warning: Failed to load calibration: {e}")
            self.T_ref_cam = None

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

    def get_calibration_transform(self) -> Optional[np.ndarray]:
        """Returns T_base_cam (eye-to-hand) or T_flange_cam (eye-in-hand)."""
        return self.T_ref_cam


class GraspGenerationClient:
    """Client for DexDiffuser Grasp Generation API."""
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')

    def process_grasp(self, rgb_image, depth_data, camera_intrinsics, target_objects, 
                      camera_extrinsics=None, confidence_threshold=0.1, num_samples=32):
        """
        Triggers grasp generation on server.
        IMPORTANT: camera_extrinsics MUST be T_base_camera (Camera to Robot Base).
        """
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
        response = requests.post(url, files=files, data=data, timeout=120)
        response.raise_for_status()
        return response.json()

    def select_grasp(self, index: int) -> dict:
        """Retrieves the grasp pose for a specific index."""
        url = f"{self.server_url}/select_grasp"
        payload = {"grasp_index": index}
        
        print(f"Retrieving grasp pose for index {index}...")
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()


class AllegroHandPublisher:
    """ROS publisher for Allegro Hand joint commands."""
    def __init__(self):
        self.joint_cmd_pub = rospy.Publisher('/allegroHand_0/joint_cmd', JointState, queue_size=10)
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
        self.publish_joint_command(self.home_position)
        print("--> Allegro Hand moved to home position.")


class FrankaRobotController:
    """Franka robot controller for executing grasp poses."""

    def __init__(self, robot_ip: str, dynamics_factor: float = DEFAULT_ROBOT_DYNAMICS_FACTOR):
        self.robot_ip = robot_ip
        self.dynamics_factor = dynamics_factor
        self.robot = None
        # Load the constant Hand-Flange transform
        self.T_flange_palm = get_flange_to_palm_transform()
        self.T_palm_flange = np.linalg.inv(self.T_flange_palm) # Inverse for calculation

    def connect(self):
        if self.robot is not None:
            return
        print(f"Connecting to Franka robot at {self.robot_ip}...")
        self.robot = Robot(self.robot_ip)
        self.robot.recover_from_errors()
        self.robot.relative_dynamics_factor = self.dynamics_factor
        print("Connected to Franka robot.")

    def disconnect(self):
        self.robot = None
        print("Disconnected from Franka robot.")

    def get_current_pose_matrix(self) -> np.ndarray:
        """Returns the current 4x4 homogenous transformation matrix of the end-effector (T_base_flange)."""
        if self.robot is None:
            raise ConnectionError("Robot is not connected")
        try:
            current_affine = self.robot.current_cartesian_state.pose.end_effector_pose
            return np.array(current_affine.matrix).reshape(4, 4, order='F')
        except AttributeError:
            print("Warning: Could not access .current_pose")
            return np.eye(4) 

    def move_to_home(self):
        if self.robot is None: return False
        try:
            joint_motion = JointMotion(FRANKA_HOME_POSITION)
            self.robot.move(joint_motion)
            print("--> Franka robot moved to home position.")
            return True
        except Exception as e:
            print(f"Error moving to home position: {e}")
            return False

    def execute_grasp_sequence(self, grasp_pos, grasp_quat_xyzw, hand_publisher, joint_angles):
        """
        Executes the full grasp sequence using precise HAND TRANSFORMATION.
        grasp_pos, grasp_quat_xyzw: Target PALM Pose in World Frame (T_base_palm).
        """
        if self.robot is None:
            print("Error: Robot not connected")
            return

        # 1. Target Palm Pose (T_base_palm)
        T_base_palm = create_transform_matrix(grasp_pos, grasp_quat_xyzw)

        # 2. Calculate Pre-Grasp Pose in PALM FRAME
        # We want to be 15cm back from the object *along the palm's approach vector*
        # Assuming Palm X is approach.
        T_palm_pre = np.eye(4)
        T_palm_pre[0, 3] = -0.15 

        T_palm_post = np.eye(4)
        T_palm_post[0, 3] = 0.08  # 8cm forward for post-grasp lift
        
        # Calculate T_base_palm_pre (Where the palm should be for pre-grasp)
        T_base_palm_pre = np.dot(T_base_palm, T_palm_pre)

        # Calculate T_base_palm_post (Where the palm should be for post-grasp)
        T_base_palm_post = np.dot(T_base_palm, T_palm_post)

        # 3. Solve for Flange Target (T_base_flange)
        # Relation: T_base_palm = T_base_flange * T_flange_palm
        # Therefore: T_base_flange = T_base_palm * inv(T_flange_palm)
        
        # Final Grasp Flange Pose
        T_base_flange_grasp = np.dot(T_base_palm, self.T_palm_flange)
        robot_grasp_pos, robot_grasp_quat = matrix_to_pos_quat(T_base_flange_grasp)
        print(f"grasp_pose:{robot_grasp_pos}")
        print(f"rotation:{robot_grasp_quat}")

        # Pre-Grasp Flange Pose
        T_base_flange_pre = np.dot(T_base_palm_pre, self.T_palm_flange)
        pre_pos, pre_quat = matrix_to_pos_quat(T_base_flange_pre)
        
        # Post-Grasp pose
        T_base_flange_post = np.dot(T_base_palm_post, self.T_palm_flange)
        post_pos, post_quat = matrix_to_pos_quat(T_base_flange_post)

        # 4. Calculate Lift Pose (World Frame, relative to final flange pose)
        lift_pos = post_pos.copy() 
        lift_pos[2] += 0.15  # Global Z lift
        lift_quat = post_quat.copy()


        print("\n=== Starting Grasp Sequence (Precise Hand Transform) ===")
        print(f"Palm Target: {grasp_pos}")
        print(f"Flange Target: {robot_grasp_pos}")
        
        # --- Step A: Move to Pre-Grasp ---
        print(f"1. Moving to Pre-Grasp Pose...")
        affine_pre = Affine(pre_pos.tolist(), pre_quat.tolist())
        self.robot.move(CartesianMotion(affine_pre))
        
        # --- Step B: Approach ---
        print(f"2. Approaching Grasp Pose...")
        affine_grasp = Affine(post_pos.tolist(), post_quat.tolist())
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


def main():
    parser = argparse.ArgumentParser(description='DexDiffuser Grasp Client')
    parser.add_argument('--server', type=str, default=DEFAULT_SERVER_URL)
    parser.add_argument('--objects', type=str, default=DEFAULT_TARGET_OBJECTS)
    parser.add_argument('--robot-ip', type=str, default=DEFAULT_ROBOT_IP)
    parser.add_argument('--eye-in-hand', action='store_true', 
                        help='Enable eye-in-hand mode (camera on robot end-effector)')
    args = parser.parse_args()

    # Select calibration file based on mode
    calib_file = CALIBRATION_FILE_EYE_ON_HAND if args.eye_in_hand else CALIBRATION_FILE_EYE_TO_HAND
    print(f"Using calibration file: {calib_file}")

    rospy.init_node('grasp_client_ros_node', anonymous=True)

    camera = AzureKinectClient(calibration_file=calib_file, eye_in_hand=args.eye_in_hand)
    api_client = GraspGenerationClient(server_url=args.server)
    allegro_publisher = AllegroHandPublisher()
    franka_controller = FrankaRobotController(args.robot_ip)
    
    # Track connection state explicitly
    franka_connected = False

    # For Eye-in-Hand, we MUST connect to the robot immediately to get poses during capture
    if args.eye_in_hand:
        try:
            franka_controller.connect()
            franka_connected = True
        except Exception as e:
            print(f"CRITICAL: Could not connect to robot for eye-in-hand pose retrieval: {e}")
            sys.exit(1)

    try:
        camera.start()
        print("\nReady. Press ENTER to capture, 'q' to quit")

        while True:
            cmd = input("\nCapture? (ENTER/q): ").strip().lower()
            if cmd == 'q': break

            # 1. Capture RGB-D
            success, rgb, depth = camera.capture_rgbd()
            if not success: continue

            # 2. Determine Camera Extrinsics (T_base_cam)
            # This is the transformation from base frame to camera frame
            T_base_cam = None
            T_ref_cam = camera.get_calibration_transform()
            print("Calibration transform:", T_ref_cam)

            if T_ref_cam is not None:
                if args.eye_in_hand:
                    # Logic for Eye-in-Hand:
                    # T_flange_cam: Static transform from flange to camera (from calibration)
                    # T_base_flange: Dynamic transform from base to flange (current robot pose)
                    # T_base_cam = T_base_flange @ T_flange_cam
                    try:
                        T_base_flange = franka_controller.get_current_pose_matrix()
                        print("T_base_flange (current robot pose):\n", T_base_flange)

                        T_flange_cam = T_ref_cam
                        print("T_flange_cam (from calibration):\n", T_flange_cam)

                        T_base_cam = np.dot(T_base_flange, T_flange_cam)
                        print("T_base_cam (computed):\n", T_base_cam)
                    except Exception as e:
                        print(f"Error calculating eye-in-hand transform: {e}")
                        continue
                else:
                    # Logic for Eye-to-Hand:
                    # T_base_cam: Static transform from base to camera (from calibration)
                    T_base_cam = T_ref_cam
                    print("T_base_cam (static, from calibration):\n", T_base_cam)

            # 3. Process (Server Visualization)
            try:
                print("Sending data to server...")
                api_client.process_grasp(
                    rgb, depth, camera.get_intrinsics_3x3(), args.objects, 
                    camera_extrinsics=T_base_cam
                )
                print(">>> Visualization launched on server.")
            except Exception as e:
                print(f"API Error: {e}")
                continue

            # 4. Interactive Selection Loop
            while True:
                idx_str = input("\n[Interact] Enter Grasp INDEX from screen (or 's' to skip/rescan): ").strip()
                if idx_str.lower() == 's': break
                
                try:
                    idx = int(idx_str)
                    
                    # Retrieve Grasp
                    data = api_client.select_grasp(idx)
                    pose_arr = np.array(data['grasp_pose'])
                    
                    qw, qx, qy, qz = pose_arr[0:4]
                    position = pose_arr[4:7]
                    joint_positions = pose_arr[7:]
                    quat_xyzw = np.array([qx, qy, qz, qw])
                    
                    # Display target
                    print(f"Selected Index: {idx}")

                    # Execute
                    if input("Execute? (y/n): ").strip().lower() == 'y':
                        if not franka_connected:
                            franka_controller.connect()
                            franka_connected = True

                        allegro_publisher.move_to_home()
                        time.sleep(1.0)
                        
                        if franka_controller.move_to_home():
                            time.sleep(0.5)
                            # make joint_position all joint angle bigger
                            joint_positions = np.array(joint_positions) * 1.3
                            franka_controller.execute_grasp_sequence(
                                position, quat_xyzw, allegro_publisher, joint_positions
                            )
                        break # Done with this capture
                    
                except Exception as e:
                    print(f"Error: {e}")

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        camera.stop()
        if franka_connected: franka_controller.disconnect()

if __name__ == '__main__':
    main()