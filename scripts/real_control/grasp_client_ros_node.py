#!/usr/bin/env python3
"""
DexDiffuser Grasp Generation Client with ROS Integration
========================================================

This client controls an Azure Kinect camera to capture RGB-D data, sends
it to the DexDiffuser Grasp Generation API server, and publishes the
generated grasp poses to the Allegro Hand via ROS topics.

Requirements:
    - pykinect-azure
    - numpy
    - opencv-python
    - requests
    - Pillow
    - rospy
    - sensor_msgs

Usage:
    rosrun test_any_policy grasp_client_ros_node.py --server http://localhost:8000 --objects "cup"

Author: Generated for DexDiffuser API client with ROS integration
"""

import cv2
import numpy as np
import pykinect_azure as pykinect
import requests
import os
import argparse
import io
import base64
from typing import Optional, Tuple, Dict, Any

# ROS imports
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# ============================================================================
# Default Configuration
# ============================================================================
DEFAULT_SERVER_URL = "http://100.120.117.28:8000"
DEFAULT_TARGET_OBJECTS = "cup"
DEFAULT_CONFIDENCE_THRESHOLD = 0.1
DEFAULT_NUM_SAMPLES = 32
CALIBRATION_FILE = "./calibration_results/eye_to_hand_calibration.npz"

# Allegro Hand joint names
ALLEGRO_JOINT_NAMES = [
    'joint_0_0', 'joint_1_0', 'joint_2_0', 'joint_3_0',
    'joint_4_0', 'joint_5_0', 'joint_6_0', 'joint_7_0',
    'joint_8_0', 'joint_9_0', 'joint_10_0', 'joint_11_0',
    'joint_12_0', 'joint_13_0', 'joint_14_0', 'joint_15_0'
]


class AzureKinectClient:
    """Azure Kinect camera client for capturing RGB-D data."""

    def __init__(self, calibration_file: Optional[str] = None):
        """
        Initialize Azure Kinect camera client.

        Args:
            calibration_file: Optional path to eye-to-hand calibration file
        """
        pykinect.initialize_libraries()

        # Configure device
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
        """Start the camera device and load calibration."""
        print("Starting Azure Kinect camera...")
        self.device = pykinect.start_device(config=self.device_config)
        self._get_intrinsics()

        # Load extrinsics if calibration file provided
        if self.calibration_file and os.path.exists(self.calibration_file):
            self._load_extrinsics()
        else:
            print("No calibration file provided or file not found. Extrinsics will be None.")

        print("Azure Kinect camera started successfully.")

    def stop(self):
        """Stop the camera device."""
        if self.device is not None:
            self.device = None
        print("Azure Kinect camera stopped.")

    def _get_intrinsics(self):
        """Get camera intrinsic parameters from Azure Kinect calibration."""
        calibration = self.device.calibration
        color_params = calibration._handle.color_camera_calibration.intrinsics.parameters.param

        # Extract intrinsic parameters
        fx = color_params.fx
        fy = color_params.fy
        cx = color_params.cx
        cy = color_params.cy

        # Build camera matrix (3x3)
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # Get distortion coefficients
        k1 = color_params.k1
        k2 = color_params.k2
        k3 = color_params.k3
        k4 = color_params.k4
        k5 = color_params.k5
        k6 = color_params.k6
        p1 = color_params.p1
        p2 = color_params.p2

        self.dist_coeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)

        print(f"Camera intrinsics loaded: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    def _load_extrinsics(self):
        """Load camera extrinsics from calibration file."""
        try:
            data = np.load(self.calibration_file)
            # T_base_cam is the 4x4 transformation matrix from camera to robot base
            self.camera_extrinsics = data['T_base_cam']
            print(f"Camera extrinsics loaded from {self.calibration_file}")
            print(f"Extrinsics matrix shape: {self.camera_extrinsics.shape}")
        except Exception as e:
            print(f"Warning: Failed to load extrinsics: {e}")
            self.camera_extrinsics = None

    def capture_rgbd(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture RGB and depth images.

        Returns:
            success: Whether capture was successful
            rgb_image: RGB image (H x W x 3) in BGR format
            depth_image: Depth image (H x W) in millimeters
        """
        capture = self.device.update()

        # Get color image
        ret_color, color_image = capture.get_color_image()
        if not ret_color:
            print("Failed to capture color image")
            return False, None, None

        # Convert BGRA to BGR
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

        # Get depth image
        ret_depth, depth_image = capture.get_transformed_depth_image()
        if not ret_depth:
            print("Failed to capture depth image")
            return False, None, None

        return True, color_image, depth_image

    def get_intrinsics_3x3(self) -> np.ndarray:
        """Get 3x3 camera intrinsic matrix."""
        return self.camera_matrix

    def get_extrinsics_4x4(self) -> Optional[np.ndarray]:
        """Get 4x4 camera extrinsic matrix (camera to base transform)."""
        return self.camera_extrinsics


class GraspGenerationClient:
    """Client for DexDiffuser Grasp Generation API."""

    def __init__(self, server_url: str):
        """
        Initialize grasp generation client.

        Args:
            server_url: Base URL of the API server (e.g., http://localhost:8000)
        """
        self.server_url = server_url.rstrip('/')

    def process_grasp(
        self,
        rgb_image: np.ndarray,
        depth_data: np.ndarray,
        camera_intrinsics: np.ndarray,
        target_objects: str,
        camera_extrinsics: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.1,
        num_samples: int = 32
    ) -> Dict[str, Any]:
        """
        Send RGB-D data to server for grasp generation.

        Args:
            rgb_image: RGB image array (H x W x 3)
            depth_data: Depth data array (H x W)
            camera_intrinsics: 3x3 camera intrinsic matrix
            target_objects: Comma-separated list of target objects
            camera_extrinsics: Optional 4x4 camera extrinsic matrix
            confidence_threshold: Detection confidence threshold
            num_samples: Number of grasp samples to generate

        Returns:
            Response dictionary containing grasp results
        """
        url = f"{self.server_url}/process_grasp"

        # Prepare files for multipart upload
        files = {}

        # 1. RGB image as PNG
        success, rgb_encoded = cv2.imencode('.png', rgb_image)
        if not success:
            raise ValueError("Failed to encode RGB image")
        files['rgb_image'] = ('rgb_image.png', rgb_encoded.tobytes(), 'image/png')

        # 2. Depth data as .npy
        depth_bytes = io.BytesIO()
        np.save(depth_bytes, depth_data)
        depth_bytes.seek(0)
        files['depth_data'] = ('depth_data.npy', depth_bytes, 'application/octet-stream')

        # 3. Camera intrinsics as .npy
        intrinsics_bytes = io.BytesIO()
        np.save(intrinsics_bytes, camera_intrinsics)
        intrinsics_bytes.seek(0)
        files['camera_intrinsics'] = ('camera_intrinsics.npy', intrinsics_bytes, 'application/octet-stream')

        # 4. Camera extrinsics as .npy (optional)
        if camera_extrinsics is not None:
            extrinsics_bytes = io.BytesIO()
            np.save(extrinsics_bytes, camera_extrinsics)
            extrinsics_bytes.seek(0)
            files['camera_extrinsics'] = ('camera_extrinsics.npy', extrinsics_bytes, 'application/octet-stream')

        # Prepare form data
        data = {
            'target_objects': target_objects,
            'confidence_threshold': confidence_threshold,
            'num_samples': num_samples
        }

        print(f"\nSending request to {url}")
        print(f"Target objects: {target_objects}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Number of samples: {num_samples}")
        print(f"RGB image shape: {rgb_image.shape}")
        print(f"Depth data shape: {depth_data.shape}")
        print(f"Camera intrinsics shape: {camera_intrinsics.shape}")
        if camera_extrinsics is not None:
            print(f"Camera extrinsics shape: {camera_extrinsics.shape}")

        # Send POST request
        try:
            response = requests.post(url, files=files, data=data, timeout=60)
            response.raise_for_status()

            result = response.json()

            print(f"\n✓ Grasp generation successful!")
            print(f"Number of grasps: {len(result['grasp_qt'])}")
            print(f"Best grasp index: {result['best_grasp_index']}")
            print(f"Best grasp score: {result['best_score']:.4f}")

            return result

        except requests.exceptions.RequestException as e:
            print(f"\n✗ Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            raise


class AllegroHandPublisher:
    """ROS publisher for Allegro Hand joint commands."""

    def __init__(self):
        """Initialize ROS publisher for Allegro Hand."""
        self.joint_cmd_pub = rospy.Publisher(
            '/allegroHand_0/joint_cmd',
            JointState,
            queue_size=10
        )
        rospy.loginfo("Allegro Hand publisher initialized")

    def publish_joint_command(self, joint_positions: np.ndarray):
        """
        Publish joint command to Allegro Hand.

        Args:
            joint_positions: Array of 16 joint positions (in radians)
        """
        if len(joint_positions) != 16:
            rospy.logerr(f"Expected 16 joint positions, got {len(joint_positions)}")
            return

        # Create JointState message
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = ALLEGRO_JOINT_NAMES
        joint_state.position = joint_positions.tolist()

        # Publish
        self.joint_cmd_pub.publish(joint_state)
        rospy.loginfo(f"Published joint command to Allegro Hand")
        rospy.loginfo(f"Joint positions: {joint_positions}")


def save_results(result: Dict[str, Any], output_dir: str, target_objects: str) -> str:
    """
    Save grasp generation results to disk.

    Args:
        result: Response from process_grasp API
        output_dir: Directory to save results
        target_objects: Target object name(s)

    Returns:
        target_objects: Target object string used for filenames
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save .ply file if present in metadata
    if 'metadata' in result and 'ply_file_base64' in result['metadata']:
        try:
            # Decode base64 .ply file
            ply_data = result['metadata']['ply_file_base64']
            ply_bytes = base64.b64decode(ply_data)
            ply_path = os.path.join(output_dir, f"point_cloud_{target_objects}.ply")
            with open(ply_path, 'wb') as f:
                f.write(ply_bytes)
            print(f"Point cloud saved to {ply_path}")
        except Exception as e:
            print(f"Warning: Failed to save .ply file: {e}")

    # Save grasp poses as numpy array
    grasp_qt = np.array(result['grasp_qt'])
    scores = np.array(result['scores'])

    npz_path = os.path.join(output_dir, f"grasp_poses_{target_objects}.npz")
    np.savez(
        npz_path,
        grasp_qt=grasp_qt,
        scores=scores,
        best_grasp_index=result['best_grasp_index'],
        best_grasp=result['best_grasp'],
        best_score=result['best_score']
    )
    print(f"Grasp poses saved to {npz_path}")

    # Print summary
    print("\n" + "="*60)
    print("GRASP GENERATION SUMMARY")
    print("="*60)
    print(f"Target object: {target_objects}")
    print(f"Total grasps generated: {len(grasp_qt)}")
    print(f"Grasp pose shape: {grasp_qt.shape}")
    print(f"Best grasp index: {result['best_grasp_index']}")
    print(f"Best grasp score: {result['best_score']:.4f}")
    print(f"\nBest grasp pose (23 dims, last 16 are joint positions):")
    best_grasp = np.array(result['best_grasp'])
    print(f"  Full: {best_grasp}")
    print(f"\nJoint positions (last 16 dims):")
    joint_positions = best_grasp[9:]  # Extract last 16 dimensions
    print(f"  {joint_positions}")
    print("="*60)

    return target_objects


def visualize_capture(rgb_image: np.ndarray, depth_image: np.ndarray, wait_time: int = 3000):
    """
    Visualize captured RGB and depth images.

    Args:
        rgb_image: RGB image array
        depth_image: Depth image array
        wait_time: Time to display in milliseconds (0 = wait for key)
    """
    # Normalize depth for visualization
    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # Create side-by-side visualization
    _, w = rgb_image.shape[:2]
    combined = np.hstack([rgb_image, depth_colored])

    # Add labels
    cv2.putText(combined, "RGB Image", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(combined, "Depth Image", (w + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    cv2.namedWindow('Captured RGB-D Data', cv2.WINDOW_NORMAL)
    cv2.imshow('Captured RGB-D Data', combined)
    cv2.waitKey(wait_time)
    cv2.destroyAllWindows()


def main():
    """Main function to run grasp generation client with ROS integration."""
    parser = argparse.ArgumentParser(
        description='DexDiffuser Grasp Generation Client with ROS Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  rosrun test_any_policy grasp_client_ros_node.py --server http://localhost:8000 --objects "cup"

  # With calibration file
  rosrun test_any_policy grasp_client_ros_node.py --server http://192.168.1.100:8000 --objects "cup" \\
      --calibration ./calibration_results/eye_to_hand_calibration.npz

  # With custom parameters
  rosrun test_any_policy grasp_client_ros_node.py --server http://localhost:8000 --objects "bottle" \\
      --confidence 0.2 --samples 64 --output ./results --no-visualize
        """
    )

    parser.add_argument('--server', type=str, default=DEFAULT_SERVER_URL,
                       help=f'API server URL (default: {DEFAULT_SERVER_URL})')
    parser.add_argument('--objects', type=str, default=DEFAULT_TARGET_OBJECTS,
                       help=f'Comma-separated list of target objects (default: {DEFAULT_TARGET_OBJECTS})')
    parser.add_argument('--confidence', type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                       help=f'Detection confidence threshold (default: {DEFAULT_CONFIDENCE_THRESHOLD})')
    parser.add_argument('--samples', type=int, default=DEFAULT_NUM_SAMPLES,
                       help=f'Number of grasp samples (default: {DEFAULT_NUM_SAMPLES})')
    parser.add_argument('--calibration', type=str, default=CALIBRATION_FILE,
                       help=f'Path to calibration file (default: {CALIBRATION_FILE})')
    parser.add_argument('--output', type=str, default='./grasp_results',
                       help='Output directory for results (default: ./grasp_results)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization of captured images')

    args = parser.parse_args()

    # Initialize ROS node
    rospy.init_node('grasp_client_ros_node', anonymous=True)
    rospy.loginfo("Starting DexDiffuser Grasp Generation Client with ROS")

    # Initialize camera, API client, and ROS publisher
    camera = AzureKinectClient(calibration_file=args.calibration)
    api_client = GraspGenerationClient(server_url=args.server)
    allegro_publisher = AllegroHandPublisher()

    try:
        # Start camera
        camera.start()

        print("\n" + "="*60)
        print("Press ENTER to capture image, or 'q' to quit")
        print("="*60)

        while not rospy.is_shutdown():
            user_input = input("\nCapture image? (ENTER to capture, 'q' to quit): ").strip().lower()

            if user_input == 'q':
                print("Exiting...")
                break

            # Capture RGB-D data
            print("\nCapturing RGB-D data...")
            success, rgb_image, depth_image = camera.capture_rgbd()

            if not success:
                print("Failed to capture images. Please try again.")
                continue

            print(f"✓ Captured RGB image: {rgb_image.shape}")
            print(f"✓ Captured depth image: {depth_image.shape}")

            # Visualize if requested
            if not args.no_visualize:
                print("Displaying captured images for 3 seconds...")
                visualize_capture(rgb_image, depth_image)

            # Get camera calibration
            intrinsics = camera.get_intrinsics_3x3()
            extrinsics = camera.get_extrinsics_4x4()

            # Send to API
            print("\nSending data to grasp generation API...")
            result = api_client.process_grasp(
                rgb_image=rgb_image,
                depth_data=depth_image,
                camera_intrinsics=intrinsics,
                target_objects=args.objects,
                camera_extrinsics=extrinsics,
                confidence_threshold=args.confidence,
                num_samples=args.samples
            )

            # Save results
            target_obj = save_results(result, args.output, args.objects)

            # Extract last 16 dimensions (joint positions) from best_grasp
            best_grasp = np.array(result['best_grasp'])
            joint_positions = best_grasp[7:]  # Last 16 dimensions

            print("\n" + "="*60)
            print("PUBLISHING TO ALLEGRO HAND")
            print("="*60)
            print(f"Extracted joint positions (16 dims):")
            print(f"  {joint_positions}")
            print("="*60)

            # Publish to Allegro Hand
            allegro_publisher.publish_joint_command(joint_positions)

            print("\n✓ Processing complete!")
            print(f"✓ Joint command published to /allegroHand_0/joint_cmd")
            print(f"\nContinue capturing? (Results saved for target object: {target_obj})")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        raise
    finally:
        # Cleanup
        camera.stop()
        print("\nGrasp generation client terminated.")


if __name__ == '__main__':
    main()
