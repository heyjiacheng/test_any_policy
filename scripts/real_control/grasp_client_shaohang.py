#!/usr/bin/env python3
"""
DexDiffuser Grasp Generation Client - Point Cloud Version
=========================================================
Sends point cloud data to grasp generation server.
"""

import numpy as np
import requests
import os
import argparse
import io
import json
from typing import Optional

# ============================================================================
# Default Configuration
# ============================================================================
DEFAULT_SERVER_URL = "http://100.120.117.28:8000"
DEFAULT_NUM_SAMPLES = 32
CALIBRATION_FILE = "./calibration_results/eye_to_hand_calibration.npz"
# DEFAULT_POINT_CLOUD_PATH = "./scripts/real_control/tmp_data/results/Orange_obj.pt"
DEFAULT_POINT_CLOUD_PATH = "./scripts/real_control/tmp_data/results/cup_obj.pt"


class GraspGenerationClient:
    """Client for DexDiffuser Grasp Generation API."""

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')

    def process_grasp(self, point_cloud_path, camera_extrinsics, num_samples=32):
        """
        Send point cloud to server for grasp generation.

        Args:
            point_cloud_path: path to .pt point cloud file
            camera_extrinsics: 4x4 transformation matrix from camera to base (required)
            num_samples: number of grasp samples to generate
        """
        url = f"{self.server_url}/process_pcd"
        files = {}

        # Send point cloud file directly
        with open(point_cloud_path, 'rb') as f:
            files['point_cloud'] = ('point_cloud.pt', f.read(), 'application/octet-stream')

        # Send camera extrinsics (required by server)
        extrinsics_bytes = io.BytesIO()
        np.save(extrinsics_bytes, camera_extrinsics)
        extrinsics_bytes.seek(0)
        files['camera_extrinsics'] = ('camera_extrinsics.npy', extrinsics_bytes.getvalue(), 'application/octet-stream')

        data = {
            'num_samples': num_samples
        }

        print(f"Sending request to {url}...")
        print(f"Point cloud file: {point_cloud_path}")
        print(f"Num samples: {num_samples}")

        response = requests.post(url, files=files, data=data, timeout=60)
        response.raise_for_status()
        return response.json()


def load_camera_extrinsics(calibration_file: str) -> np.ndarray:
    """Load camera extrinsics from calibration file."""
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")

    try:
        data = np.load(calibration_file)
        camera_extrinsics = data['T_base_cam']
        print(f"Loaded camera extrinsics from {calibration_file}")
        return camera_extrinsics
    except Exception as e:
        raise RuntimeError(f"Failed to load extrinsics: {e}")


def save_results(result, output_dir, point_cloud_filename):
    """Save grasp results to disk as JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    result_dict = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            result_dict[key] = value.tolist()
        elif isinstance(value, list):
            # Check if list contains numpy arrays
            result_dict[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
        else:
            result_dict[key] = value

    # Generate output filename based on point cloud name
    base_name = os.path.splitext(os.path.basename(point_cloud_filename))[0]
    output_file = os.path.join(output_dir, f"grasp_result_{base_name}.json")

    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Display summary
    if 'best_grasp' in result_dict:
        print(f"Best grasp: {result_dict['best_grasp']}")
    if 'grasp_qt' in result_dict:
        print(f"Total grasps: {len(result_dict['grasp_qt'])}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description='DexDiffuser Grasp Client - Point Cloud Version')
    parser.add_argument('--server', type=str, default=DEFAULT_SERVER_URL,
                        help='Server URL')
    parser.add_argument('--point-cloud', type=str, default=DEFAULT_POINT_CLOUD_PATH,
                        help='Path to point cloud .pt file')
    parser.add_argument('--calibration', type=str, default=CALIBRATION_FILE,
                        help='Path to calibration file with camera extrinsics')
    parser.add_argument('--num-samples', type=int, default=DEFAULT_NUM_SAMPLES,
                        help='Number of grasp samples to generate')
    parser.add_argument('--output-dir', type=str, default='./grasp_results',
                        help='Output directory for results')
    args = parser.parse_args()

    try:
        # Check if point cloud file exists
        if not os.path.exists(args.point_cloud):
            raise FileNotFoundError(f"Point cloud file not found: {args.point_cloud}")

        # Load camera extrinsics
        camera_extrinsics = load_camera_extrinsics(args.calibration)

        # Create API client
        api_client = GraspGenerationClient(server_url=args.server)

        # Send request to server
        print("\n" + "="*60)
        print("Sending grasp generation request...")
        print("="*60)

        result = api_client.process_grasp(
            point_cloud_path=args.point_cloud,
            camera_extrinsics=camera_extrinsics,
            num_samples=args.num_samples
        )
        # import pdb;pdb.set_trace()

        # Save results
        print("\n" + "="*60)
        print("Grasp generation complete!")
        print("="*60)

        save_results(result, args.output_dir, args.point_cloud)

        # Display best grasp
        if 'best_grasp' in result:
            best_grasp = np.array(result['best_grasp'])
            print("\nBest Grasp Details:")
            print(f"  Quaternion (qw, qx, qy, qz): {best_grasp[0:4]}")
            print(f"  Position (x, y, z): {best_grasp[4:7]}")
            print(f"  Joint angles: {best_grasp[7:]}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
