#!/usr/bin/env python3
"""
Azure Kinect RGB-D Capture and Save
====================================
Simple script to capture RGB image, depth data, and camera intrinsics
from Azure Kinect and save them to disk.

Usage:
    python capture_rgbd.py --output_dir ./captured_data
"""

import cv2
import numpy as np
import pykinect_azure as pykinect
import argparse
import os
from datetime import datetime
from typing import Tuple, Optional


class AzureKinectCapture:
    """Simple Azure Kinect camera client for capturing and saving RGB-D data."""

    def __init__(self):
        pykinect.initialize_libraries()
        self.device_config = pykinect.default_configuration
        self.device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
        self.device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
        self.device = None
        self.camera_matrix = None

    def start(self):
        """Start the camera and get intrinsics."""
        print("Starting Azure Kinect camera...")
        self.device = pykinect.start_device(config=self.device_config)
        self._get_intrinsics()
        print("Camera started successfully.")
        print(f"Camera intrinsics:\n{self.camera_matrix}")

    def stop(self):
        """Stop the camera."""
        if self.device is not None:
            self.device = None
        print("Azure Kinect camera stopped.")

    def _get_intrinsics(self):
        """Extract camera intrinsics matrix."""
        calibration = self.device.calibration
        color_params = calibration._handle.color_camera_calibration.intrinsics.parameters.param
        self.camera_matrix = np.array([
            [color_params.fx, 0, color_params.cx],
            [0, color_params.fy, color_params.cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def capture_rgbd(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture RGB and depth images.

        Returns:
            tuple: (success, rgb_image, depth_image)
                - success: bool indicating if capture was successful
                - rgb_image: numpy array with shape (H, W, 3) in BGR format
                - depth_image: numpy array with shape (H, W) in m
        """
        capture = self.device.update()

        # Get color image
        ret_color, color_image = capture.get_color_image()
        if not ret_color:
            print("Failed to capture color image")
            return False, None, None
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

        # Get depth image (aligned to color)
        ret_depth, depth_image = capture.get_transformed_depth_image()
        if not ret_depth:
            print("Failed to capture depth image")
            return False, None, None
        
        # translate from mm to m
        depth_image = depth_image.astype(np.float32) / 1000.0

        return True, color_image, depth_image

    def get_intrinsics(self) -> np.ndarray:
        """Return the 3x3 camera intrinsics matrix."""
        return self.camera_matrix


def save_capture(output_dir: str, rgb_image: np.ndarray, depth_image: np.ndarray,
                 intrinsics: np.ndarray, prefix: str = ""):
    """
    Save RGB image, depth data, and camera intrinsics to files.

    Args:
        output_dir: Directory to save files
        rgb_image: RGB image array
        depth_image: Depth image array
        intrinsics: Camera intrinsics matrix
        prefix: Optional prefix for filenames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save RGB image as PNG
    rgb_path = os.path.join(output_dir, "0_color.png")
    cv2.imwrite(rgb_path, rgb_image)
    print(f"Saved RGB image to: {rgb_path}")

    # Save depth data as numpy array
    depth_path = os.path.join(output_dir, "0_depth_aligned_rgb.npy")
    np.save(depth_path, depth_image)
    print(f"Saved depth data to: {depth_path}")

    # Save camera intrinsics
    intrinsics_path = os.path.join(output_dir, "0_camerainfo.npy")
    np.save(intrinsics_path, intrinsics)
    print(f"Saved camera intrinsics to: {intrinsics_path}")

    # Print summary
    print(f"\nCapture Summary:")
    print(f"  RGB shape: {rgb_image.shape}")
    print(f"  Depth shape: {depth_image.shape}")
    print(f"  Depth range: {depth_image.min():.1f} - {depth_image.max():.1f} m")
    print(f"  Intrinsics shape: {intrinsics.shape}")


def main():
    parser = argparse.ArgumentParser(description='Capture RGB-D images from Azure Kinect')
    parser.add_argument('--output_dir', type=str, default='./captured_data',
                        help='Directory to save captured images (default: ./captured_data)')
    parser.add_argument('--prefix', type=str, default='',
                        help='Optional prefix for saved filenames')
    parser.add_argument('--preview', action='store_true',
                        help='Show preview window before saving')
    args = parser.parse_args()

    # Initialize camera
    camera = AzureKinectCapture()

    try:
        camera.start()
        print("\nReady to capture. Press ENTER to capture, 'q' to quit")

        capture_count = 0
        while True:
            cmd = input("\nCapture? (ENTER/q): ").strip().lower()
            if cmd == 'q':
                break

            # Capture RGB-D
            success, rgb, depth = camera.capture_rgbd()
            if not success:
                print("Capture failed, please try again")
                continue

            # Optional preview
            if args.preview:
                # Show RGB
                cv2.imshow('RGB Preview', rgb)

                # Show depth (normalized for visualization)
                depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                cv2.imshow('Depth Preview', depth_colored)

                print("Preview displayed. Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # Save the capture
            capture_count += 1
            prefix = f"{args.prefix}_capture{capture_count:03d}" if args.prefix else f"capture{capture_count:03d}"
            save_capture(args.output_dir, rgb, depth, camera.get_intrinsics(), prefix=prefix)

            print(f"\nTotal captures: {capture_count}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == '__main__':
    main()
