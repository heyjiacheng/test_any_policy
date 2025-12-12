#!/usr/bin/env python3
"""
Eye-to-Hand Calibration using Azure Kinect and Franka Robot
============================================================

This script performs Eye-to-Hand (camera fixed in workspace) calibration
using ArUco marker detection and OpenCV's calibrateHandEye function.

Requirements:
    - pykinect-azure
    - franky-control
    - opencv-contrib-python (for ArUco detection)
    - numpy
    - scipy

Usage:
    1. Mount an ArUco marker on the robot's end-effector
    2. Run this script and follow the prompts
    3. The script will save the calibration result (camera-to-base transformation)

Author: Generated for Eye-to-Hand calibration
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import pykinect_azure as pykinect
from franky import Affine, Robot, CartesianMotion, JointMotion, ReferenceType
import time
import json
import os
from typing import Tuple, List, Optional
from datetime import datetime


# ============================================================================
# ArUco Marker Configuration
# ============================================================================
ARUCO_DICT_TYPE = cv2.aruco.DICT_ARUCO_ORIGINAL  # ArUco dictionary type
ARUCO_MARKER_ID = 24  # ID of the marker to detect
ARUCO_MARKER_SIZE = 0.1  # Marker side length in meters (100mm)


# ============================================================================
# Robot Configuration
# ============================================================================
ROBOT_IP = "172.16.1.22"  # Franka robot IP address (modify as needed)
ROBOT_DYNAMICS_FACTOR = 0.1  # Robot speed factor (0.0-1.0)


# ============================================================================
# Camera Configuration
# ============================================================================
CAMERA_RESOLUTION = pykinect.K4A_COLOR_RESOLUTION_1080P


# ============================================================================
# Calibration Configuration
# ============================================================================
MIN_POSES = 10  # Minimum number of poses for calibration
CALIBRATION_SAVE_DIR = "./calibration_results"
POSE_DATA_FILE = "./calibration_results/pose_data.npz"  # Default pose data file


class ArucoDetector:
    """ArUco marker detector with camera intrinsics calibration support."""

    def __init__(self):
        """Initialize ArUco detector."""
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self.detector_params
        )

        # Camera intrinsics (will be set from Azure Kinect calibration)
        self.camera_matrix = None
        self.dist_coeffs = None

    def set_camera_intrinsics(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """Set camera intrinsic parameters."""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def detect_marker(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect ArUco marker and estimate its pose.

        Returns:
            success: Whether detection was successful
            rvec: Rotation vector (Rodrigues)
            tvec: Translation vector
        """
        if self.camera_matrix is None:
            raise ValueError("Camera intrinsics not set. Call set_camera_intrinsics first.")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect ArUco markers
        marker_corners, marker_ids, rejected = self.aruco_detector.detectMarkers(gray)

        if marker_ids is None or len(marker_ids) == 0:
            return False, None, None

        # Find the target marker ID
        target_idx = None
        for i, mid in enumerate(marker_ids):
            if mid[0] == ARUCO_MARKER_ID:
                target_idx = i
                break

        if target_idx is None:
            return False, None, None

        # Get the corners of the detected marker
        corners = marker_corners[target_idx][0]  # Shape: (4, 2)

        # Define 3D coordinates of marker corners in marker frame
        # OpenCV ArUco convention: origin at center, X-right, Y-up, Z-out (towards camera)
        # Corner order from detectMarkers: top-left, top-right, bottom-right, bottom-left
        half_size = ARUCO_MARKER_SIZE / 2.0
        obj_points = np.array([
            [-half_size,  half_size, 0],  # Top-left (in image, which is -X, +Y in marker frame)
            [ half_size,  half_size, 0],  # Top-right (+X, +Y)
            [ half_size, -half_size, 0],  # Bottom-right (+X, -Y)
            [-half_size, -half_size, 0]   # Bottom-left (-X, -Y)
        ], dtype=np.float32)

        # Estimate pose using solvePnP
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            corners,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE  # Good for planar markers
        )

        if not success:
            return False, None, None

        return True, rvec, tvec

    def draw_detection(self, image: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Draw detected marker and pose on image."""
        output = image.copy()

        # Draw coordinate axes
        cv2.drawFrameAxes(
            output, self.camera_matrix, self.dist_coeffs,
            rvec, tvec, ARUCO_MARKER_SIZE * 0.5
        )

        return output


class AzureKinectCamera:
    """Azure Kinect camera wrapper for calibration."""

    def __init__(self):
        """Initialize Azure Kinect camera."""
        pykinect.initialize_libraries()

        # Configure device
        self.device_config = pykinect.default_configuration
        self.device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        self.device_config.color_resolution = CAMERA_RESOLUTION
        self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED

        self.device = None
        self.camera_matrix = None
        self.dist_coeffs = None

    def start(self):
        """Start the camera device."""
        self.device = pykinect.start_device(config=self.device_config)
        self._get_intrinsics()
        print("Azure Kinect camera started.")

    def stop(self):
        """Stop the camera device."""
        if self.device is not None:
            self.device = None
        print("Azure Kinect camera stopped.")

    def _get_intrinsics(self):
        """Get camera intrinsic parameters from Azure Kinect calibration."""
        # Get calibration from device
        calibration = self.device.calibration

        # Get color camera intrinsics
        color_params = calibration._handle.color_camera_calibration.intrinsics.parameters.param

        # Extract intrinsic parameters
        fx = color_params.fx
        fy = color_params.fy
        cx = color_params.cx
        cy = color_params.cy

        # Build camera matrix
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # Get distortion coefficients
        # Azure Kinect uses Brown-Conrady model: k1, k2, p1, p2, k3, k4, k5, k6
        k1 = color_params.k1
        k2 = color_params.k2
        k3 = color_params.k3
        k4 = color_params.k4
        k5 = color_params.k5
        k6 = color_params.k6
        p1 = color_params.p1
        p2 = color_params.p2

        self.dist_coeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)

        print(f"Camera intrinsics loaded:")
        print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    def get_color_image(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture a color image."""
        capture = self.device.update()
        ret, color_image = capture.get_color_image()

        if ret:
            # Convert BGRA to BGR
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

        return ret, color_image


class FrankaRobotController:
    """Franka robot controller wrapper for calibration."""

    def __init__(self, robot_ip: str):
        """Initialize Franka robot controller."""
        self.robot_ip = robot_ip
        self.robot = None

    def connect(self):
        """Connect to the robot."""
        print(f"Connecting to Franka robot at {self.robot_ip}...")
        self.robot = Robot(self.robot_ip)
        self.robot.recover_from_errors()
        self.robot.relative_dynamics_factor = ROBOT_DYNAMICS_FACTOR
        print("Connected to Franka robot.")

    def disconnect(self):
        """Disconnect from the robot."""
        self.robot = None
        print("Disconnected from Franka robot.")

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current end-effector pose in robot base frame.

        Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation vector
        """
        cartesian_state = self.robot.current_cartesian_state
        ee_pose = cartesian_state.pose.end_effector_pose

        # Get 4x4 transformation matrix (matrix is a property, not a method)
        T = np.array(ee_pose.matrix).reshape(4, 4, order='F')

        R = T[:3, :3]
        t = T[:3, 3].reshape(3, 1)

        return R, t

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        joint_state = self.robot.current_joint_state
        return np.array(joint_state.position)

    def move_to_joint_position(self, joint_positions: List[float]):
        """Move robot to specified joint positions."""
        motion = JointMotion(joint_positions)
        self.robot.move(motion)

    def move_relative(self, dx: float = 0, dy: float = 0, dz: float = 0,
                     drx: float = 0, dry: float = 0, drz: float = 0):
        """Move robot end-effector relative to current pose."""
        quat = Rotation.from_euler('xyz', [drx, dry, drz]).as_quat()
        motion = CartesianMotion(Affine([dx, dy, dz], quat), ReferenceType.Relative)
        self.robot.move(motion)


class EyeToHandCalibrator:
    """Eye-to-Hand calibration manager."""

    def __init__(self):
        """Initialize calibrator."""
        self.camera = AzureKinectCamera()
        self.robot = FrankaRobotController(ROBOT_IP)
        self.detector = ArucoDetector()

        # Storage for calibration data
        self.joint_positions_list = []  # Robot joint angles (more accurate than computed poses)
        self.R_gripper2base_list = []  # Robot end-effector rotations (base frame)
        self.t_gripper2base_list = []  # Robot end-effector translations (base frame)
        self.R_target2cam_list = []    # Board-to-camera rotations
        self.t_target2cam_list = []    # Board-to-camera translations

        # Calibration result
        self.R_cam2base = None
        self.t_cam2base = None

    def setup(self):
        """Setup camera and robot."""
        self.camera.start()
        self.detector.set_camera_intrinsics(
            self.camera.camera_matrix,
            self.camera.dist_coeffs
        )
        self.robot.connect()

    def cleanup(self):
        """Cleanup resources."""
        self.camera.stop()
        self.robot.disconnect()

    def capture_calibration_pose(self) -> bool:
        """
        Capture a single calibration pose.

        Returns:
            success: Whether capture was successful
        """
        # Capture image
        ret, image = self.camera.get_color_image()
        if not ret:
            print("Failed to capture image.")
            return False

        # Detect ArUco marker
        success, rvec, tvec = self.detector.detect_marker(image)
        if not success:
            print("ArUco marker not detected in image.")
            return False

        # Get robot joint positions (more accurate than computed pose)
        joint_positions = self.robot.get_joint_positions()

        # Get robot pose (for calibration computation)
        R_ee, t_ee = self.robot.get_ee_pose()

        # Convert rotation vector to matrix
        R_board2cam, _ = cv2.Rodrigues(rvec)
        t_board2cam = tvec.reshape(3, 1)

        # Store data (including joint angles for trajectory replay)
        self.joint_positions_list.append(joint_positions)
        self.R_gripper2base_list.append(R_ee)
        self.t_gripper2base_list.append(t_ee)
        self.R_target2cam_list.append(R_board2cam)
        self.t_target2cam_list.append(t_board2cam)

        print(f"Pose {len(self.R_gripper2base_list)} captured successfully.")

        return True

    def run_interactive_calibration(self, pose_data_file: str = POSE_DATA_FILE):
        """Run interactive calibration with user guidance."""
        print("\n" + "="*60)
        print("Eye-to-Hand Calibration - Interactive Mode")
        print("="*60)
        print(f"\nMinimum poses required: {MIN_POSES}")
        print(f"Pose data will be saved to: {pose_data_file}")
        print("\nControls:")
        print("  'c' - Capture current pose (auto-saves)")
        print("  'r' - Remove last pose")
        print("  'd' - Run calibration (when enough poses)")
        print("  'q' - Quit")
        print("\nTips for good calibration:")
        print("  - Move the robot to diverse poses")
        print("  - Vary rotation around all axes")
        print("  - Keep the marker visible to the camera")
        print("  - Ensure good lighting")
        print("\n" + "="*60)

        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)

        while True:
            # Get camera image
            ret, image = self.camera.get_color_image()
            if not ret:
                continue

            # Try to detect marker
            success, rvec, tvec = self.detector.detect_marker(image)

            # Draw detection if successful
            if success:
                display_image = self.detector.draw_detection(image, rvec, tvec)
                status = f"Marker {ARUCO_MARKER_ID} DETECTED"
                color = (0, 255, 0)
            else:
                display_image = image.copy()
                status = f"Marker {ARUCO_MARKER_ID} NOT detected"
                color = (0, 0, 255)

            # Add status text
            cv2.putText(display_image, status, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(display_image, f"Poses captured: {len(self.R_gripper2base_list)}/{MIN_POSES}",
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Calibration', display_image)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                print("\nCalibration aborted.")
                break

            elif key == ord('c'):
                if success:
                    if self.capture_calibration_pose():
                        # Auto-save pose data after each capture
                        self.save_pose_data(pose_data_file)
                else:
                    print("Cannot capture - marker not detected.")

            elif key == ord('r'):
                if len(self.R_gripper2base_list) > 0:
                    self.joint_positions_list.pop()
                    self.R_gripper2base_list.pop()
                    self.t_gripper2base_list.pop()
                    self.R_target2cam_list.pop()
                    self.t_target2cam_list.pop()
                    print(f"Removed last pose. {len(self.R_gripper2base_list)} poses remaining.")
                    # Update saved pose data
                    if len(self.R_gripper2base_list) > 0:
                        self.save_pose_data(pose_data_file)
                    else:
                        # Remove pose data file if no poses left
                        if os.path.exists(pose_data_file):
                            os.remove(pose_data_file)
                            print(f"Pose data file removed (no poses left).")
                else:
                    print("No poses to remove.")

            elif key == ord('d'):
                if len(self.R_gripper2base_list) >= MIN_POSES:
                    self.compute_calibration()
                else:
                    print(f"Need at least {MIN_POSES} poses. Current: {len(self.R_gripper2base_list)}")

        cv2.destroyAllWindows()

    def run_automatic_calibration(self, pose_data_file: str = POSE_DATA_FILE):
        """
        Run automatic calibration using trajectory from interactive mode.
        Replays the joint positions saved during interactive mode and re-captures poses.

        Args:
            pose_data_file: Path to saved pose data file (required)
        """
        print("\n" + "="*60)
        print("Eye-to-Hand Calibration - Automatic Mode")
        print("="*60)

        # Check if pose data file exists
        if not os.path.exists(pose_data_file):
            print(f"\nERROR: Pose data file not found: {pose_data_file}")
            print("Auto mode requires previously saved pose data from interactive mode.")
            print("Please run interactive mode first to capture initial poses.")
            return

        # Load previously saved pose data to get joint trajectory
        try:
            data = np.load(pose_data_file)
            saved_joint_positions = list(data['joint_positions_list'])
            print(f"Loaded trajectory with {len(saved_joint_positions)} poses from {pose_data_file}")
            print(f"Will replay trajectory and re-capture poses...")
        except Exception as e:
            print(f"\nERROR: Failed to load pose data: {e}")
            return

        # Clear current calibration data (will re-capture from scratch)
        self.joint_positions_list = []
        self.R_gripper2base_list = []
        self.t_gripper2base_list = []
        self.R_target2cam_list = []
        self.t_target2cam_list = []

        # Replay trajectory and capture poses
        for i, joint_pos in enumerate(saved_joint_positions):
            print(f"\nMoving to pose {i+1}/{len(saved_joint_positions)}...")

            try:
                self.robot.move_to_joint_position(joint_pos)
                time.sleep(1.5)  # Wait for robot to settle

                # Try to capture pose
                success = self.capture_calibration_pose()
                if success:
                    # Save updated pose data after each capture
                    self.save_pose_data(pose_data_file)
                else:
                    print(f"Warning: Could not capture pose {i+1}")

            except Exception as e:
                print(f"Error at pose {i+1}: {e}")
                continue

        print(f"\nCaptured {len(self.R_gripper2base_list)}/{len(saved_joint_positions)} poses")

        if len(self.R_gripper2base_list) >= MIN_POSES:
            self.compute_calibration()
        else:
            print(f"\nCalibration failed: Only {len(self.R_gripper2base_list)} valid poses captured.")

    def compute_calibration(self):
        """Compute the eye-to-hand calibration using collected poses."""
        print("\n" + "="*60)
        print("Computing calibration...")
        print("="*60)

        # For Eye-to-Hand calibration:
        # We have: T_cam_marker (from vision) and T_base_ee (from robot)
        # We want: T_base_cam (camera pose in robot base frame)
        #
        # The relationship is: T_cam_marker = T_cam_base * T_base_ee * T_ee_marker
        # Since marker is attached to EE: T_ee_marker is constant
        # Rearranging: T_cam_base * T_base_ee = T_cam_marker * T_marker_ee
        # This is: A * X = Z * B form where X = T_cam_base^(-1) = T_base_cam

        # OpenCV's calibrateHandEye solves: A_i * X = X * B_i for Eye-in-Hand
        # For Eye-to-Hand, we need to swap inputs appropriately

        # Method: Use the relationship that for eye-to-hand:
        # T_base_cam is constant, T_ee_marker is constant
        # T_base_ee_i * T_ee_marker = T_base_cam * T_cam_marker_i
        # Rearranging: (T_base_ee_j * T_base_ee_i^-1) * T_base_cam = T_base_cam * (T_cam_marker_j * T_cam_marker_i^-1)
        # This is: A * X = X * B where X = T_base_cam

        R_gripper2base_inv_list = []
        t_gripper2base_inv_list = []

        for R, t in zip(self.R_gripper2base_list, self.t_gripper2base_list):
            R_inv = R.T
            t_inv = -R_inv @ t
            
            R_gripper2base_inv_list.append(R_inv)
            t_gripper2base_inv_list.append(t_inv)
        
        try:
            # Use cv2.calibrateHandEye with Eye-to-Hand configuration
            # For Eye-to-Hand: swap gripper and target
            R_cam2base, t_cam2base = cv2.calibrateHandEye(
                R_gripper2base_inv_list,  # R_gripper2base
                t_gripper2base_inv_list,  # t_gripper2base
                self.R_target2cam_list,    # R_target2cam
                self.t_target2cam_list,    # t_target2cam
                method=cv2.CALIB_HAND_EYE_DANIILIDIS  # Can also try PARK, HORAUD, ANDREFF, DANIILIDIS
            )

            self.R_cam2base = R_cam2base
            self.t_cam2base = t_cam2base

            print("\nCalibration successful!")
            print("\n--- Camera to Base Transformation (T_base_cam) ---")
            print(f"Rotation matrix:\n{R_cam2base}")
            print(f"\nTranslation vector (meters):\n{t_cam2base.flatten()}")

            # Convert to Euler angles for readability
            euler = Rotation.from_matrix(R_cam2base).as_euler('xyz', degrees=True)
            print(f"\nEuler angles (degrees): roll={euler[0]:.2f}, pitch={euler[1]:.2f}, yaw={euler[2]:.2f}")

            # Compute reprojection error
            self._compute_reprojection_error()

            # Save calibration
            self.save_calibration()

        except Exception as e:
            print(f"Calibration failed: {e}")

    def _compute_reprojection_error(self):
        """Compute and print reprojection error statistics."""
        if self.R_cam2base is None:
            return

        errors = []

        for i in range(len(self.R_gripper2base_list)):
            # Compute T_cam_marker from calibration
            # T_cam_marker = T_cam_base * T_base_ee * T_ee_marker
            # We can verify consistency

            R_ee2base = self.R_gripper2base_list[i]
            t_ee2base = self.t_gripper2base_list[i]
            R_marker2cam = self.R_target2cam_list[i]
            t_marker2cam = self.t_target2cam_list[i]

            # T_base_cam @ T_cam_marker should give consistent T_base_marker
            # T_base_marker = T_base_ee @ T_ee_marker

            # For now, compute angular error between relative transformations
            if i > 0:
                R_ee_rel = self.R_gripper2base_list[i] @ self.R_gripper2base_list[i-1].T
                R_marker_rel = self.R_target2cam_list[i] @ self.R_target2cam_list[i-1].T

                # Compute rotation error
                R_error = R_ee_rel @ self.R_cam2base - self.R_cam2base @ R_marker_rel
                error = np.linalg.norm(R_error, 'fro')
                errors.append(error)

        if errors:
            print(f"\nCalibration quality:")
            print(f"  Mean rotation consistency error: {np.mean(errors):.6f}")
            print(f"  Max rotation consistency error: {np.max(errors):.6f}")

    def save_pose_data(self, filepath: str = POSE_DATA_FILE):
        """Save intermediate pose data to file."""
        if len(self.R_gripper2base_list) == 0:
            print("No pose data to save.")
            return

        # Create save directory
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save pose data (joint angles are the primary trajectory data)
        np.savez(
            filepath,
            joint_positions_list=np.array(self.joint_positions_list),
            R_gripper2base_list=np.array(self.R_gripper2base_list),
            t_gripper2base_list=np.array(self.t_gripper2base_list),
            R_target2cam_list=np.array(self.R_target2cam_list),
            t_target2cam_list=np.array(self.t_target2cam_list),
            num_poses=len(self.R_gripper2base_list)
        )

        print(f"\nPose data saved: {len(self.R_gripper2base_list)} poses -> {filepath}")
        print(f"  - Joint angles trajectory saved for accurate replay")

    def load_pose_data(self, filepath: str = POSE_DATA_FILE):
        """Load previously saved pose data from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pose data file not found: {filepath}")

        data = np.load(filepath)

        # Load arrays (joint positions are the primary trajectory data)
        self.joint_positions_list = list(data['joint_positions_list'])
        self.R_gripper2base_list = list(data['R_gripper2base_list'])
        self.t_gripper2base_list = list(data['t_gripper2base_list'])
        self.R_target2cam_list = list(data['R_target2cam_list'])
        self.t_target2cam_list = list(data['t_target2cam_list'])

        print(f"\nPose data loaded: {len(self.R_gripper2base_list)} poses from {filepath}")
        print(f"  - Joint angles trajectory loaded for accurate replay")
        return len(self.R_gripper2base_list)

    def save_calibration(self, filename: str = None):
        """Save calibration result to file."""
        if self.R_cam2base is None:
            print("No calibration to save.")
            return

        # Create save directory
        os.makedirs(CALIBRATION_SAVE_DIR, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eye_to_hand_calibration_{timestamp}.npz"

        filepath = os.path.join(CALIBRATION_SAVE_DIR, filename)

        # Build 4x4 transformation matrix
        T_base_cam = np.eye(4)
        T_base_cam[:3, :3] = self.R_cam2base
        T_base_cam[:3, 3] = self.t_cam2base.flatten()

        # Save as NPZ (including joint trajectory for reproducibility)
        np.savez(
            filepath,
            R_cam2base=self.R_cam2base,
            t_cam2base=self.t_cam2base,
            T_base_cam=T_base_cam,
            camera_matrix=self.camera.camera_matrix,
            dist_coeffs=self.camera.dist_coeffs,
            joint_positions_list=np.array(self.joint_positions_list),
            num_poses=len(self.R_gripper2base_list)
        )

        print(f"\nCalibration saved to: {filepath}")
        print(f"  - Joint trajectory included for reproducibility")

        # Also save as JSON for readability
        json_filepath = filepath.replace('.npz', '.json')
        calibration_dict = {
            'R_cam2base': self.R_cam2base.tolist(),
            't_cam2base': self.t_cam2base.flatten().tolist(),
            'T_base_cam': T_base_cam.tolist(),
            'camera_matrix': self.camera.camera_matrix.tolist(),
            'dist_coeffs': self.camera.dist_coeffs.tolist(),
            'joint_positions_list': [jp.tolist() for jp in self.joint_positions_list],
            'num_poses': len(self.R_gripper2base_list),
            'aruco_config': {
                'dict_type': str(ARUCO_DICT_TYPE),
                'marker_id': ARUCO_MARKER_ID,
                'marker_size': ARUCO_MARKER_SIZE
            }
        }

        with open(json_filepath, 'w') as f:
            json.dump(calibration_dict, f, indent=2)

        print(f"Calibration (JSON) saved to: {json_filepath}")

    @staticmethod
    def load_calibration(filepath: str) -> dict:
        """Load calibration from file."""
        if filepath.endswith('.npz'):
            data = np.load(filepath)
            return {
                'R_cam2base': data['R_cam2base'],
                't_cam2base': data['t_cam2base'],
                'T_base_cam': data['T_base_cam'],
                'camera_matrix': data['camera_matrix'],
                'dist_coeffs': data['dist_coeffs']
            }
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return {
                'R_cam2base': np.array(data['R_cam2base']),
                't_cam2base': np.array(data['t_cam2base']),
                'T_base_cam': np.array(data['T_base_cam']),
                'camera_matrix': np.array(data['camera_matrix']),
                'dist_coeffs': np.array(data['dist_coeffs'])
            }
        else:
            raise ValueError("Unsupported file format. Use .npz or .json")


def generate_calibration_poses() -> List[List[float]]:
    """
    Generate a set of diverse joint poses for automatic calibration.

    Returns:
        List of joint position arrays
    """
    # Define base pose (neutral position with board facing camera)
    base_pose = [0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.0]

    poses = [base_pose]

    # Generate variations around base pose
    variations = [
        # Vary joints 1, 5, 7 for diverse orientations
        [0.3, -0.3, 0.0, -2.0, 0.0, 1.7, 0.3],
        [-0.3, -0.3, 0.0, -2.0, 0.0, 1.7, -0.3],
        [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0],
        [0.0, -0.1, 0.0, -2.2, 0.0, 1.9, 0.0],
        [0.2, -0.4, 0.1, -1.8, -0.1, 1.6, 0.2],
        [-0.2, -0.4, -0.1, -1.8, 0.1, 1.6, -0.2],
        [0.15, -0.2, 0.0, -2.1, 0.0, 1.8, 0.15],
        [-0.15, -0.2, 0.0, -2.1, 0.0, 1.8, -0.15],
        [0.0, -0.35, 0.15, -1.9, -0.15, 1.65, 0.0],
        [0.0, -0.35, -0.15, -1.9, 0.15, 1.65, 0.0],
        [0.25, -0.25, 0.0, -2.0, 0.0, 1.75, 0.25],
        [-0.25, -0.25, 0.0, -2.0, 0.0, 1.75, -0.25],
    ]

    poses.extend(variations)
    return poses


def main():
    """Main function to run eye-to-hand calibration."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Eye-to-Hand Calibration using ArUco marker',
        epilog=f'Note: Attach ArUco marker (ID: {ARUCO_MARKER_ID}, size: {ARUCO_MARKER_SIZE*1000}mm) to robot end-effector before running.'
    )
    parser.add_argument('--mode', choices=['interactive', 'auto'],
                       default='interactive',
                       help='Calibration mode (default: interactive). Auto mode requires saved pose data.')
    parser.add_argument('--robot_ip', type=str, default=ROBOT_IP,
                       help=f'Robot IP address (default: {ROBOT_IP})')
    parser.add_argument('--marker_id', type=int, default=None,
                       help=f'ArUco marker ID to detect (default: {ARUCO_MARKER_ID})')
    parser.add_argument('--marker_size', type=float, default=None,
                       help=f'ArUco marker size in meters (default: {ARUCO_MARKER_SIZE})')
    parser.add_argument('--pose_data', type=str, default=POSE_DATA_FILE,
                       help=f'Path to pose data file for saving/loading (default: {POSE_DATA_FILE})')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename for calibration result')

    args = parser.parse_args()

    # Update global config if provided
    # if args.marker_id is not None:
    #     global ARUCO_MARKER_ID
    #     ARUCO_MARKER_ID = args.marker_id

    # if args.marker_size is not None:
    #     global ARUCO_MARKER_SIZE
    #     ARUCO_MARKER_SIZE = args.marker_size

    # Create calibrator
    calibrator = EyeToHandCalibrator()

    try:
        calibrator.setup()

        if args.mode == 'interactive':
            calibrator.run_interactive_calibration(pose_data_file=args.pose_data)
        elif args.mode == 'auto':
            calibrator.run_automatic_calibration(pose_data_file=args.pose_data)

    except KeyboardInterrupt:
        print("\nCalibration interrupted by user.")
    except Exception as e:
        print(f"\nError during calibration: {e}")
        raise
    finally:
        calibrator.cleanup()


if __name__ == '__main__':
    main()
