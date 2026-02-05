#!/usr/bin/env python3
"""
Dreamer Grasp Client — capture → server → execute.

1. Capture RGB + depth (aligned to color) + intrinsics from Azure Kinect.
2. POST the three files to the remote grasp server (POST /run_grasp).
   The server runs the full Allegro grasp pipeline and returns the
   grasp pose in CAMERA frame.
3. Transform the grasp from camera frame to robot base frame using
   the eye-to-hand calibration (T_base_cam).
4. Execute on the local Franka + Allegro:
   Home → Pre-Grasp → Approach → Close Hand → Lift.

Usage:
    python grasp_client_dreamer.py [--server http://host:port] [--robot-ip 172.16.1.22] [--calibration path.npz]
"""

import io
import os
import sys
import time
import argparse

import cv2
import numpy as np
import pykinect_azure as pykinect
import requests

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from franky import Affine, Robot, CartesianMotion, JointMotion

# ============================================================================
# Configuration
# ============================================================================
DEFAULT_SERVER_URL  = "http://100.122.228.55:8000"
DEFAULT_ROBOT_IP    = "172.16.1.22"
DEFAULT_ROBOT_DYNAMICS_FACTOR = 0.06

FLANGE_TO_PALM_TRANSLATION  = [0.026, -0.026, 0.07]
FLANGE_TO_PALM_ROTATION_DEG = [0, -90, 135]  # Euler XYZ degrees

FRANKA_HOME_POSITION = [0.0, -0.78, 0.0, -2.36, 0.0, 1.57, 0.78]

ALLEGRO_HOME_POSITION_DEG = [
    0., 10., 15., 15.,   # Index
    0., 10., 15., 15.,   # Middle
    5., 15., 10., 15.,   # Ring
    90., 5., 5., 5.      # Thumb
]

ALLEGRO_JOINT_NAMES = [
    'joint_0_0',  'joint_1_0',  'joint_2_0',  'joint_3_0',
    'joint_4_0',  'joint_5_0',  'joint_6_0',  'joint_7_0',
    'joint_8_0',  'joint_9_0',  'joint_10_0', 'joint_11_0',
    'joint_12_0', 'joint_13_0', 'joint_14_0', 'joint_15_0',
]

# Applied to raw joint angles from the server (same offset as dexdiffuser / execute_grasp)
JOINT_ANGLE_OFFSET = 0.3

# Calibration – pipeline output is in CAMERA frame; client must transform to base
CALIBRATION_FILE_EYE_TO_HAND = "./calibration_results/eye_to_hand_calibration.npz"

# ============================================================================
# Math helpers  (identical conventions to dexdiffuser client)
# ============================================================================
def euler_to_matrix(r_deg, p_deg, y_deg):
    r, p, y = np.radians([r_deg, p_deg, y_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def quaternion_to_matrix(q):
    """q = [qx, qy, qz, qw]"""
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,  2*qx*qy - 2*qz*qw,      2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,      1 - 2*qx**2 - 2*qz**2,  2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,      2*qy*qz + 2*qx*qw,      1 - 2*qx**2 - 2*qy**2]
    ])

def create_transform_matrix(pos, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = quaternion_to_matrix(quat_xyzw)
    T[:3, 3] = pos
    return T

def matrix_to_pos_quat(T):
    """Returns pos [x,y,z] and quat [qx,qy,qz,qw]."""
    pos = T[:3, 3]
    R   = T[:3, :3]
    tr  = np.trace(R)
    if tr > 0:
        S  = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S  = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S  = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S  = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S
    return pos, np.array([qx, qy, qz, qw])

def get_flange_to_palm_transform():
    T = np.eye(4)
    T[:3, :3] = euler_to_matrix(*FLANGE_TO_PALM_ROTATION_DEG)
    T[:3, 3]  = FLANGE_TO_PALM_TRANSLATION
    return T

def load_T_base_cam(calibration_file: str) -> np.ndarray:
    """Load T_base_cam (camera -> base) from eye-to-hand calibration npz."""
    if not os.path.exists(calibration_file):
        print(f"Error: Calibration file not found: {calibration_file}")
        sys.exit(1)
    data = np.load(calibration_file)
    if 'T_base_cam' not in data:
        print(f"Error: 'T_base_cam' key not found in {calibration_file}. Keys: {list(data.keys())}")
        sys.exit(1)
    T_base_cam = data['T_base_cam']
    print(f"Loaded T_base_cam from {calibration_file}")
    print(f"T_base_cam:\n{T_base_cam}")
    return T_base_cam

# ============================================================================
# Azure Kinect Camera
# ============================================================================
class AzureKinectClient:
    def __init__(self):
        pykinect.initialize_libraries()
        self.device_config = pykinect.default_configuration
        self.device_config.camera_fps     = pykinect.K4A_FRAMES_PER_SECOND_15
        self.device_config.color_format   = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        self.device_config.depth_mode     = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
        self.device        = None
        self.camera_matrix = None

    def start(self):
        print("Starting Azure Kinect...")
        self.device = pykinect.start_device(config=self.device_config)
        self._get_intrinsics()
        print(f"Camera started. Intrinsics:\n{self.camera_matrix}")

    def stop(self):
        if self.device is not None:
            self.device = None
        print("Azure Kinect stopped.")

    def _get_intrinsics(self):
        cal = self.device.calibration
        p   = cal._handle.color_camera_calibration.intrinsics.parameters.param
        self.camera_matrix = np.array([
            [p.fx, 0,    p.cx],
            [0,    p.fy, p.cy],
            [0,    0,    1   ]
        ], dtype=np.float64)

    def capture_rgbd(self):
        """Returns (success, rgb_bgr uint8, depth_m float32)."""
        capture = self.device.update()

        ret_color, color_image = capture.get_color_image()
        if not ret_color:
            print("Failed to capture color image")
            return False, None, None
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

        ret_depth, depth_image = capture.get_transformed_depth_image()
        if not ret_depth:
            print("Failed to capture depth image")
            return False, None, None
        depth_image = depth_image.astype(np.float32) / 1000.0  # mm → m

        return True, color_image, depth_image

    def get_intrinsics(self) -> np.ndarray:
        return self.camera_matrix

# ============================================================================
# Allegro Hand Publisher
# ============================================================================
class AllegroHandPublisher:
    def __init__(self):
        self.joint_cmd_pub = rospy.Publisher('/allegroHand_0/joint_cmd', JointState, queue_size=10)
        self.home_position = np.array([deg * np.pi / 180.0 for deg in ALLEGRO_HOME_POSITION_DEG])

    def publish_joint_command(self, joint_positions: np.ndarray):
        if len(joint_positions) != 16:
            rospy.logerr(f"Expected 16 joint positions, got {len(joint_positions)}")
            return
        msg = JointState()
        msg.header        = Header()
        msg.header.stamp  = rospy.Time.now()
        msg.name          = ALLEGRO_JOINT_NAMES
        msg.position      = joint_positions.tolist()
        self.joint_cmd_pub.publish(msg)
        print("--> Allegro hand command published.")

    def move_to_home(self):
        self.publish_joint_command(self.home_position)
        print("--> Allegro hand at home.")

# ============================================================================
# Franka Robot Controller
# ============================================================================
class FrankaRobotController:
    def __init__(self, robot_ip: str, dynamics_factor: float = DEFAULT_ROBOT_DYNAMICS_FACTOR):
        self.robot_ip       = robot_ip
        self.dynamics_factor = dynamics_factor
        self.robot          = None
        self.T_flange_palm  = get_flange_to_palm_transform()
        self.T_palm_flange  = np.linalg.inv(self.T_flange_palm)

    def connect(self):
        if self.robot is not None:
            return
        print(f"Connecting to Franka at {self.robot_ip}...")
        self.robot = Robot(self.robot_ip)
        self.robot.recover_from_errors()
        self.robot.relative_dynamics_factor = self.dynamics_factor
        print("Connected.")

    def disconnect(self):
        self.robot = None
        print("Disconnected from Franka.")

    def move_to_home(self):
        if self.robot is None:
            return False
        try:
            self.robot.move(JointMotion(FRANKA_HOME_POSITION))
            print("--> Franka at home.")
            return True
        except Exception as e:
            print(f"Error moving to home: {e}")
            return False

    def execute_grasp_sequence(self, grasp_pos, grasp_quat_xyzw, hand_pub, joint_angles):
        """
        Full grasp sequence.
        grasp_pos / grasp_quat_xyzw: palm pose in robot BASE frame.
        """
        if self.robot is None:
            print("Error: robot not connected")
            return

        T_base_palm = create_transform_matrix(grasp_pos, grasp_quat_xyzw)

        # Pre-grasp: 10 cm back along palm X
        T_palm_pre      = np.eye(4)
        T_palm_pre[0,3] = -0.1
        T_base_palm_pre = T_base_palm @ T_palm_pre

        # Flange targets  (T_base_flange = T_base_palm @ T_palm_flange)
        T_base_flange_grasp = T_base_palm     @ self.T_palm_flange
        T_base_flange_pre   = T_base_palm_pre @ self.T_palm_flange
        robot_grasp_pos, robot_grasp_quat = matrix_to_pos_quat(T_base_flange_grasp)
        pre_pos,         pre_quat         = matrix_to_pos_quat(T_base_flange_pre)

        # Lift: 15 cm up in global Z from grasp flange pose
        lift_pos  = robot_grasp_pos.copy()
        lift_pos[2] += 0.15
        lift_quat = robot_grasp_quat.copy()

        print("\n=== Grasp Sequence ===")
        print(f"  Palm target   (base): {grasp_pos}")
        print(f"  Flange target (base): {robot_grasp_pos}")

        # 1. Pre-grasp
        print("1. Pre-grasp (10 cm back)...")
        self.robot.move(CartesianMotion(Affine(pre_pos.tolist(), pre_quat.tolist())))

        # 2. Approach
        print("2. Approach...")
        self.robot.move(CartesianMotion(Affine(robot_grasp_pos.tolist(), robot_grasp_quat.tolist())))

        # 3. Close hand (10 interpolation steps)
        print("3. Closing hand...")
        home_j = hand_pub.home_position
        for i in range(1, 11):
            alpha = i / 10.0
            hand_pub.publish_joint_command(home_j + alpha * (joint_angles - home_j))
            time.sleep(0.3)
        time.sleep(1.0)

        # 4. Lift
        print("4. Lifting...")
        self.robot.move(CartesianMotion(Affine(lift_pos.tolist(), lift_quat.tolist())))

        print("=== Sequence complete ===\n")

# ============================================================================
# Server communication
# ============================================================================
def send_to_server(server_url: str, rgb_bgr: np.ndarray,
                   depth_m: np.ndarray, intrinsics: np.ndarray) -> dict:
    """POST the 3 sensor files to /run_grasp, return parsed JSON."""
    url = f"{server_url.rstrip('/')}/run_grasp"

    # RGB → PNG bytes (BGR, same encoding as cv2.imwrite used in capture_rgbd)
    _, png_buf = cv2.imencode('.png', rgb_bgr)

    # depth & intrinsics → .npy bytes
    depth_buf = io.BytesIO()
    np.save(depth_buf, depth_m)
    depth_buf.seek(0)

    intr_buf = io.BytesIO()
    np.save(intr_buf, intrinsics)
    intr_buf.seek(0)

    files = {
        "color_png":             ("0_color.png",             png_buf.tobytes(), "image/png"),
        "camerainfo_npy":        ("0_camerainfo.npy",        intr_buf,          "application/octet-stream"),
        "depth_aligned_rgb_npy": ("0_depth_aligned_rgb.npy", depth_buf,         "application/octet-stream"),
    }

    print(f"Sending to {url} ...")
    resp = requests.post(url, files=files, timeout=300)
    resp.raise_for_status()
    return resp.json()

# ============================================================================
# Main loop
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Dreamer grasp client: capture → server → execute")
    parser.add_argument("--server",      type=str, default=DEFAULT_SERVER_URL,
                        help="Grasp-server base URL")
    parser.add_argument("--robot-ip",   type=str, default=DEFAULT_ROBOT_IP,
                        help="Franka robot IP")
    parser.add_argument("--calibration", type=str, default=CALIBRATION_FILE_EYE_TO_HAND,
                        help="Eye-to-hand calibration .npz (must contain T_base_cam)")
    args = parser.parse_args()

    # Load calibration once at startup – pipeline output is in camera frame
    T_base_cam = load_T_base_cam(args.calibration)

    rospy.init_node("grasp_client_dreamer_node", anonymous=True)

    camera   = AzureKinectClient()
    allegro  = AllegroHandPublisher()
    franka   = FrankaRobotController(args.robot_ip)
    franka_connected = False

    try:
        camera.start()
        print("\nReady. Press ENTER to capture & run, 'q' to quit.\n")

        while True:
            cmd = input("Capture? (ENTER / q): ").strip().lower()
            if cmd == "q":
                break

            # --- 1. Capture ---
            ok, rgb, depth = camera.capture_rgbd()
            if not ok:
                print("Capture failed, try again.")
                continue
            print(f"Captured  rgb={rgb.shape}  depth={depth.shape} "
                  f"[{depth.min():.2f} – {depth.max():.2f} m]")

            # --- 2. Send to remote grasp server & parse response ---
            try:
                result = send_to_server(args.server, rgb, depth, camera.get_intrinsics())
                grasp_raw = np.array(result["grasp_in_base"])  # (23,) in CAMERA frame
            except Exception as e:
                print(f"Server error: {e}")
                continue

            # --- 3. Parse & transform camera frame → base frame ---
            qw, qx, qy, qz  = grasp_raw[0:4]
            position_cam     = grasp_raw[4:7]
            joint_angles     = grasp_raw[7:23] + JOINT_ANGLE_OFFSET
            quat_xyzw_cam    = np.array([qx, qy, qz, qw])

            print(f"\n[Camera frame] Position:   {position_cam}")
            print(f"[Camera frame] Quat(xyzw): {quat_xyzw_cam}")

            # T_base_palm = T_base_cam @ T_cam_palm  (same as execute_grasp.py)
            T_cam_palm           = create_transform_matrix(position_cam, quat_xyzw_cam)
            T_base_palm          = T_base_cam @ T_cam_palm
            position, quat_xyzw  = matrix_to_pos_quat(T_base_palm)

            print(f"\n{'='*50}")
            print(f"[Base frame]   Position:   {position}")
            print(f"[Base frame]   Quat(xyzw): {quat_xyzw}")
            print(f"Joint angles (+{JOINT_ANGLE_OFFSET}):  {joint_angles}")
            print(f"{'='*50}")

            # --- 4. Confirm & execute ---
            if input("Execute grasp? (y/n): ").strip().lower() != "y":
                print("Skipped.")
                continue

            if not franka_connected:
                franka.connect()
                franka_connected = True

            allegro.move_to_home()
            time.sleep(1.0)

            if not franka.move_to_home():
                print("Failed to home Franka. Aborting this attempt.")
                continue
            time.sleep(0.5)

            franka.execute_grasp_sequence(position, quat_xyzw, allegro, joint_angles)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        camera.stop()
        if franka_connected:
            franka.disconnect()


if __name__ == "__main__":
    main()
