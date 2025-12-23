#!/usr/bin/env python3
"""
DexDiffuser Grasp Execution Client - Simplified Version
========================================================

This is a simplified version that captures grasps and stores them,
allowing you to manually execute them or integrate with your own execution logic.

The 23-element grasp pose format:
- Elements 0-3: Quaternion [qw, qx, qy, qz]
- Elements 4-6: Position [x, y, z] in meters
- Elements 7-22: Allegro joint angles (16 values)

Requirements:
    - pykinect-azure
    - numpy
    - opencv-python
    - requests
    - rospy
    - geometry_msgs
    - sensor_msgs

Usage:
    rosrun test_any_policy grasp_execution_simple.py --server http://localhost:8000 --objects "cup"

Author: Generated for DexDiffuser + Panda-Allegro integration
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
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Point
from std_msgs.msg import Header

# ============================================================================
# Default Configuration
# ============================================================================
DEFAULT_SERVER_URL = "http://100.120.117.28:8000"
DEFAULT_TARGET_OBJECTS = "cookie box"
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

# Frame ID for grasp poses
GRASP_FRAME_ID = "panda_link0"


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


class GraspPublisher:
    """
    Publishes grasp poses and joint commands to ROS topics.

    This provides a way to visualize and store grasps without requiring
    service definitions. You can manually execute them or integrate with
    custom execution logic.
    """

    def __init__(self):
        """Initialize ROS publishers."""
        # Publisher for Franka end-effector grasp poses (for visualization/execution)
        self.grasp_pose_pub = rospy.Publisher(
            '/grasp_pose',
            PoseStamped,
            queue_size=10
        )

        # Publisher for Allegro Hand joint commands
        self.allegro_joint_pub = rospy.Publisher(
            '/allegroHand_0/joint_cmd',
            JointState,
            queue_size=10
        )

        rospy.loginfo("Grasp publishers initialized")
        rospy.loginfo("  - Publishing grasp poses to: /grasp_pose")
        rospy.loginfo("  - Publishing joint commands to: /allegroHand_0/joint_cmd")

    def publish_grasp(self, grasp_23: np.ndarray, grasp_index: int = 0):
        """
        Publish a single grasp pose and joint command.

        Args:
            grasp_23: 23-element grasp array [qw,qx,qy,qz,x,y,z,joints(16)]
            grasp_index: Index of this grasp (for header seq)
        """
        # Parse grasp format
        qw, qx, qy, qz = grasp_23[0:4]
        x, y, z = grasp_23[4:7]
        joint_angles = grasp_23[7:23]

        # Create and publish PoseStamped for Franka
        pose_stamped = PoseStamped()
        pose_stamped.header = Header()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = GRASP_FRAME_ID
        pose_stamped.header.seq = grasp_index

        pose_stamped.pose.position = Point(x=x, y=y, z=z)
        pose_stamped.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        self.grasp_pose_pub.publish(pose_stamped)

        # Create and publish JointState for Allegro
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.header.seq = grasp_index
        joint_state.name = ALLEGRO_JOINT_NAMES
        joint_state.position = joint_angles.tolist()

        self.allegro_joint_pub.publish(joint_state)

        rospy.loginfo(f"Published grasp {grasp_index}:")
        rospy.loginfo(f"  Position: [{x:.3f}, {y:.3f}, {z:.3f}]")
        rospy.loginfo(f"  Orientation (quat): [{qw:.3f}, {qx:.3f}, {qy:.3f}, {qz:.3f}]")
        rospy.loginfo(f"  Joint angles: {joint_angles}")


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
    print(f"\nBest grasp pose (23 dims):")
    best_grasp = np.array(result['best_grasp'])
    print(f"  Quaternion [qw,qx,qy,qz]: {best_grasp[0:4]}")
    print(f"  Position [x,y,z]: {best_grasp[4:7]}")
    print(f"  Joint angles (16): {best_grasp[7:23]}")
    print("="*60)

    return target_objects


def main():
    """Main function to run simplified grasp execution client."""
    parser = argparse.ArgumentParser(
        description='DexDiffuser Grasp Execution Client - Simplified Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  rosrun test_any_policy grasp_execution_simple.py --server http://localhost:8000 --objects "cup"

  # With custom parameters
  rosrun test_any_policy grasp_execution_simple.py --server http://192.168.1.100:8000 \\
      --objects "bottle" --confidence 0.2 --samples 64

This simplified version:
  1. Captures RGB-D images from Azure Kinect
  2. Sends to DexDiffuser API for grasp generation
  3. Publishes grasp poses to /grasp_pose topic
  4. Publishes joint commands to /allegroHand_0/joint_cmd topic
  5. Saves results to disk

You can:
  - Visualize grasps in RViz
  - Manually execute grasps using your own execution logic
  - Integrate with custom control systems
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
    parser.add_argument('--publish-all', action='store_true',
                       help='Publish all generated grasps (default: only best grasp)')

    args = parser.parse_args()

    # Initialize ROS node
    rospy.init_node('grasp_execution_simple', anonymous=True)
    rospy.loginfo("="*60)
    rospy.loginfo("DexDiffuser Grasp Execution Client - Simplified")
    rospy.loginfo("="*60)

    # Initialize components
    camera = AzureKinectClient(calibration_file=args.calibration)
    api_client = GraspGenerationClient(server_url=args.server)
    grasp_publisher = GraspPublisher()

    # Storage for latest results
    latest_results = None

    try:
        # Start camera
        camera.start()

        print("\n" + "="*60)
        print("READY TO CAPTURE AND PUBLISH GRASPS")
        print("="*60)
        print("Commands:")
        print("  ENTER - Capture image and generate grasps")
        print("  'p'   - Re-publish last best grasp")
        print("  'a'   - Publish all grasps from last generation")
        print("  'q'   - Quit")
        print("="*60)

        while not rospy.is_shutdown():
            user_input = input("\nCommand (ENTER=capture, 'p'=publish, 'a'=publish all, 'q'=quit): ").strip().lower()

            if user_input == 'q':
                print("Exiting...")
                break

            elif user_input == 'p':
                # Re-publish the best grasp
                if latest_results is None:
                    print("No grasps available! Capture and generate grasps first.")
                    continue

                best_grasp = np.array(latest_results['best_grasp'])
                best_index = latest_results['best_grasp_index']

                print(f"\nRe-publishing best grasp (index {best_index})...")
                grasp_publisher.publish_grasp(best_grasp, best_index)
                print("✓ Best grasp published to topics")
                continue

            elif user_input == 'a':
                # Publish all grasps
                if latest_results is None:
                    print("No grasps available! Capture and generate grasps first.")
                    continue

                grasp_qt_list = latest_results['grasp_qt']
                print(f"\nPublishing all {len(grasp_qt_list)} grasps...")

                for i, grasp_23 in enumerate(grasp_qt_list):
                    grasp_publisher.publish_grasp(np.array(grasp_23), i)
                    rospy.sleep(0.1)  # Small delay between publishes

                print(f"✓ Published all {len(grasp_qt_list)} grasps")
                continue

            # Capture and generate grasps (ENTER or empty input)
            print("\nCapturing RGB-D data...")
            success, rgb_image, depth_image = camera.capture_rgbd()

            if not success:
                print("Failed to capture images. Please try again.")
                continue

            print(f"✓ Captured RGB image: {rgb_image.shape}")
            print(f"✓ Captured depth image: {depth_image.shape}")

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

            # Store latest results
            latest_results = result

            # Publish grasps
            if args.publish_all:
                # Publish all grasps
                grasp_qt_list = result['grasp_qt']
                print(f"\nPublishing all {len(grasp_qt_list)} grasps...")
                for i, grasp_23 in enumerate(grasp_qt_list):
                    grasp_publisher.publish_grasp(np.array(grasp_23), i)
                    rospy.sleep(0.1)
                print(f"✓ Published all {len(grasp_qt_list)} grasps")
            else:
                # Publish only best grasp
                best_grasp = np.array(result['best_grasp'])
                best_index = result['best_grasp_index']
                print(f"\nPublishing best grasp (index {best_index})...")
                grasp_publisher.publish_grasp(best_grasp, best_index)
                print("✓ Best grasp published to topics")

            print("\n✓ Processing complete!")
            print(f"✓ Results saved to {args.output}")
            print(f"✓ Grasp poses published to /grasp_pose")
            print(f"✓ Joint commands published to /allegroHand_0/joint_cmd")
            print("\nNext steps:")
            print("  - Press 'p' to re-publish the best grasp")
            print("  - Press 'a' to publish all grasps")
            print("  - Press ENTER to capture new image")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        camera.stop()
        print("\nGrasp execution client terminated.")


if __name__ == '__main__':
    main()
