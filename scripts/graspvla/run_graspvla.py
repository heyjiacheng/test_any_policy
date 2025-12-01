"""
GraspVLA Policy Integration with ManiSkill

This script integrates the GraspVLA robot motion policy with the ManiSkill simulation environment.
It replaces the LIBERO environment used in the original GraspVLA playground with ManiSkill.

Usage:
    # Run with default mug object
    python scripts/graspvla/run_graspvla.py --instruction "pick up the mug"

    # Run with custom object
    python scripts/graspvla/run_graspvla.py \
        --instruction "pick up the bottle" \
        --object-mesh-path dataset/customize/bottle/base.obj

    # Run with articulation object from PartNet-Mobility
    python scripts/graspvla/run_graspvla.py \
        --instruction "pick up the cabinet door" \
        --use-articulation \
        --articulation-id 12536

    # Customize positions
    python scripts/graspvla/run_graspvla.py \
        --instruction "pick up the mug" \
        --object-position -0.05 0 0.15 \
        --robot-position 0.5 0 0
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from scipy.spatial.transform import Rotation

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mani_skill.envs.sapien_env import BaseEnv
from src.object_loader import ObjectConfig, ObjectLoader, load_custom_objects
from src.camera_utils import capture_images, save_camera_params, setup_cameras
from src.camera_config import GRASPVLA_CAMERA_VIEWS
from src.trajectory_executor import initialize_ik_solver
from src.video_utils import generate_videos
from src.env_utils import (
    create_maniskill_env,
    remove_default_objects,
    hide_goal_markers,
    set_robot_base_pose
)

# Import our GraspVLA agent
from remote_agent import RemoteAgent


@dataclass
class Args:
    """Command line arguments"""
    # Environment
    env_id: str = "PickCube-v1"
    sim_backend: str = "auto"
    render_backend: str = "gpu"
    shader: str = "rt"  # Options: "rt", "rt-fast", "default"
    seed: Optional[int] = None

    # GraspVLA Policy Server
    port: int = 6666  # ZMQ server port where GraspVLA policy is running
    instruction: str = "pick up the mug"  # Task instruction

    # Object Loading (choose one mode)
    # Mode 1: Load regular mesh object
    object_mesh_path: str = "dataset/customize/mug_obj/base.obj"
    # Mode 2: Load articulation object
    use_articulation: bool = False
    articulation_id: Optional[str] = None
    articulation_dataset: str = "partnet-mobility"

    # Object pose
    object_position: tuple = (-0.1, 0.0, 0.0)  # [x, y, z] in meters
    object_rotation: tuple = (90, 0, 0)  # [rx, ry, rz] in degrees

    # Robot pose
    robot_position: Optional[tuple] = None  # [x, y, z] in meters, None = use default
    robot_rotation: tuple = (0, 0, 0)  # [rx, ry, rz] in degrees

    # Camera and rendering
    image_width: int = 256  # GraspVLA expects 256x256 images
    image_height: int = 256
    hide_goal: bool = True

    # Episode settings
    max_steps: int = 300 
    save_video: bool = True
    video_fps: int = 10

    # Output
    output_root: str = "outputs/graspvla"

    # Debug
    debug: bool = False


def load_single_object(mesh_path: str, position: tuple, rotation_deg: tuple) -> ObjectConfig:
    """Load a single regular object.

    Args:
        mesh_path: Path to the mesh file
        position: Initial position [x, y, z]
        rotation_deg: Rotation euler angles in degrees

    Returns:
        ObjectConfig: Object configuration
    """
    configs = load_custom_objects(mesh_path, auto_scale=True, target_size=0.1)
    if not configs:
        raise ValueError(f"Failed to load object: {mesh_path}")

    config = configs[0]
    config.position = position
    config.rotation = tuple(np.deg2rad(rotation_deg))
    return config


def load_articulation(env: BaseEnv, model_id: str, dataset: str,
                     position: tuple, rotation_deg: tuple):
    """Load articulation object from dataset.

    Args:
        env: ManiSkill environment
        model_id: Model ID (e.g., "12536")
        dataset: Dataset name (e.g., "partnet-mobility")
        position: Initial position [x, y, z]
        rotation_deg: Initial rotation euler angles (rx, ry, rz) in degrees

    Returns:
        Loaded articulation object
    """
    import sapien
    from mani_skill.utils import sapien_utils

    print(f"  Dataset: {dataset}")
    print(f"  Model ID: {model_id}")
    print(f"  Position: {position}")
    print(f"  Rotation (degrees): {rotation_deg}")

    # Build URDF file path
    dataset_dir = dataset.replace("-", "_")
    model_dir = Path(f"dataset/{dataset_dir}/{model_id}")

    # Find URDF file by priority
    urdf_names = ["mobility_cvx.urdf", "mobility_fixed.urdf", "mobility.urdf"]
    urdf_path = None
    for urdf_name in urdf_names:
        candidate_path = model_dir / urdf_name
        if candidate_path.exists():
            urdf_path = candidate_path
            break

    if urdf_path is None:
        raise FileNotFoundError(
            f"No URDF file found in {model_dir} ({', '.join(urdf_names)})"
        )

    print(f"  URDF file: {urdf_path}")

    # Create URDF loader
    scene = env.unwrapped.scene
    loader = scene.create_urdf_loader()
    loader.fix_root_link = False
    loader.scale = 1.0
    loader.load_multiple_collisions_from_file = True

    # Apply URDF config
    urdf_config = sapien_utils.parse_urdf_config(
        dict(material=dict(static_friction=1, dynamic_friction=1, restitution=0))
    )
    sapien_utils.apply_urdf_config(loader, urdf_config)

    # Parse URDF
    articulation_builders = loader.parse(str(urdf_path))["articulation_builders"]
    builder = articulation_builders[0]

    # Convert angles to quaternion
    rotation_rad = np.deg2rad(rotation_deg)
    quat = Rotation.from_euler('xyz', rotation_rad).as_quat()  # [x, y, z, w]
    quaternion = [quat[3], quat[0], quat[1], quat[2]]  # Convert to [w, x, y, z]

    # Set initial pose
    builder.initial_pose = sapien.Pose(p=position, q=quaternion)

    # Build articulation
    articulation = builder.build(name="articulation_object")

    print(f"✓ Loaded: {articulation.name}")
    print(f"  - Joints: {len(articulation.joints)}")
    print(f"  - Links: {len(articulation.links)}\n")

    return articulation


def get_observation(env: BaseEnv, cameras: dict, agent: RemoteAgent) -> dict:
    """Get observation from ManiSkill environment in GraspVLA format.

    Args:
        env: ManiSkill environment
        cameras: Dictionary of camera objects
        agent: RemoteAgent instance

    Returns:
        Dictionary containing:
            - 'front_view_image': np.ndarray (H, W, 3)
            - 'side_view_image': np.ndarray (H, W, 3)
            - 'tcp_pose': np.ndarray (7,) [x, y, z, qw, qx, qy, qz] in robot base frame
            - 'gripper_state': float (0-1, where 0=closed, 1=open)
    """
    # Get robot state from ManiSkill
    robot = env.unwrapped.agent

    # IMPORTANT: Get TCP pose in robot base frame (not world frame)
    # This matches LIBERO's convention where proprioception is in robot frame
    tcp_pose_world = robot.tcp.pose.raw_pose[0].cpu().numpy()  # [x, y, z, qw, qx, qy, qz]
    base_pose_world = robot.robot.pose.raw_pose[0].cpu().numpy()  # [x, y, z, qw, qx, qy, qz]

    # Transform TCP pose from world frame to robot base frame
    import sapien
    tcp_world = sapien.Pose(p=tcp_pose_world[:3], q=tcp_pose_world[3:])
    base_world = sapien.Pose(p=base_pose_world[:3], q=base_pose_world[3:])
    tcp_base = base_world.inv() * tcp_world

    # Convert back to numpy array [x, y, z, qw, qx, qy, qz]
    tcp_pose = np.concatenate([tcp_base.p, tcp_base.q])

    # Get gripper state
    # ManiSkill gripper: qpos in range [0, 0.04] typically
    # We normalize to [0, 1] where 0=closed, 1=open
    gripper_qpos = robot.robot.get_qpos()[0, -2:].cpu().numpy()  # Get finger joint positions
    gripper_state = np.mean(gripper_qpos) / 0.04  # Normalize assuming max opening is 0.04
    gripper_state = np.clip(gripper_state, 0.0, 1.0)

    # Render camera images (update only human render cameras, not sensors)
    env.unwrapped.scene.update_render(update_sensors=False, update_human_render_cameras=True)

    # Get front view image (main camera)
    front_camera = cameras.get('front', cameras.get('behind'))
    front_camera.capture()
    front_obs = front_camera.get_obs(rgb=True, depth=False, position=False, segmentation=False)
    # Convert from [0, 1] float to [0, 255] uint8
    from src.image_utils import to_numpy_uint8
    front_rgb = to_numpy_uint8(front_obs["rgb"])

    # Get side view image
    side_camera = cameras.get('right', cameras.get('left'))
    side_camera.capture()
    side_obs = side_camera.get_obs(rgb=True, depth=False, position=False, segmentation=False)
    side_rgb = to_numpy_uint8(side_obs["rgb"])

    return {
        'front_view_image': front_rgb,
        'side_view_image': side_rgb,
        'tcp_pose': tcp_pose,
        'gripper_state': gripper_state,
    }


def execute_action(env: BaseEnv, action: np.ndarray, kinematics) -> None:
    """Execute an action in ManiSkill environment using IK.

    Args:
        env: ManiSkill environment
        action: Action array (7,) [x, y, z, rx, ry, rz, gripper] in robot base frame
        kinematics: IK solver
    """
    import transforms3d as t3d
    from mani_skill.utils.structs import Pose

    # Extract target pose and gripper command
    target_pos = action[:3]
    target_euler = action[3:6]
    target_gripper = action[6]

    # Convert euler angles to quaternion
    # IMPORTANT: Use 'sxyz' convention to match GraspVLA training (same as LIBERO)
    # transforms3d uses 'sxyz' by default, so we convert euler -> matrix -> quaternion
    rot_mat = t3d.euler.euler2mat(*target_euler)  # euler (sxyz) -> rotation matrix
    quat_wxyz = t3d.quaternions.mat2quat(rot_mat)  # rotation matrix -> quaternion [w, x, y, z]

    # IMPORTANT: Action is already in robot base frame (matching LIBERO convention)
    # No need to transform from world to base frame
    # Create target pose directly from action
    device = env.unwrapped.device
    p_tensor = torch.tensor([target_pos], dtype=torch.float32, device=device)
    q_tensor = torch.tensor([quat_wxyz], dtype=torch.float32, device=device)
    target_pose = Pose.create_from_pq(p=p_tensor, q=q_tensor)

    # Solve IK
    current_qpos = env.unwrapped.agent.robot.get_qpos()
    target_qpos = kinematics.compute_ik(
        target_pose,
        q0=current_qpos,
        use_delta_ik_solver=False,
    )

    if target_qpos is None:
        print("  Warning: IK solution not found, skipping this action")
        return

    # Get joint positions (first 7 joints for arm)
    joint_pos = target_qpos[0, :7].cpu().numpy()

    # Create full action: [joint_positions (7,), gripper (1,)]
    # ManiSkill Panda gripper expects 1 value that controls both fingers
    # gripper value: 0=closed, 1=open
    full_action = np.concatenate([joint_pos, [target_gripper]])

    # Execute action
    action_tensor = torch.tensor(full_action, dtype=torch.float32).unsqueeze(0).to(device)
    env.step(action_tensor)


def run_episode(env: BaseEnv, agent: RemoteAgent, cameras: dict,
                kinematics, output_dir: Path, args: Args) -> dict:
    """Run one episode with GraspVLA policy.

    Args:
        env: ManiSkill environment
        agent: RemoteAgent instance
        cameras: Dictionary of camera objects
        kinematics: IK solver
        output_dir: Output directory for saving images
        args: Command line arguments

    Returns:
        Episode statistics dictionary
    """
    print("\n" + "=" * 60)
    print("Running episode with GraspVLA policy...")
    print(f"Task instruction: '{args.instruction}'")
    print(f"Max steps: {args.max_steps}")
    print("=" * 60 + "\n")

    step = 0
    done = False

    try:
        while step < args.max_steps and not done:
            # Get observation
            if args.debug and step == 0:
                print("Getting first observation...")

            obs = get_observation(env, cameras, agent)

            if args.debug and step == 0:
                print(f"✓ Observation received:")
                print(f"  - Front image shape: {obs['front_view_image'].shape}")
                print(f"  - Side image shape: {obs['side_view_image'].shape}")
                print(f"  - TCP pose: {obs['tcp_pose']}")
                print(f"  - Gripper state: {obs['gripper_state']}\n")

            # Get action from policy
            if args.debug and step == 0:
                print("Requesting action from GraspVLA server...")

            action, _ = agent.step(obs, debug=args.debug)

            if args.debug and step == 0:
                print(f"✓ Action received: {action}\n")

            if args.debug:
                print(f"Step {step}:")
                print(f"  TCP pos: [{obs['tcp_pose'][0]:.3f}, {obs['tcp_pose'][1]:.3f}, {obs['tcp_pose'][2]:.3f}]")
                print(f"  Action pos: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}], gripper: {action[6]:.2f}")

            # Execute action
            execute_action(env, action, kinematics)

            # Capture images
            if step % 5 == 0:  # Capture every 5 steps to save disk space
                capture_images(cameras, env.unwrapped.scene, output_dir, step // 5)
                if args.debug and step % 20 == 0:
                    print(f"  ✓ Captured images at step {step}")

            step += 1

            # Simple termination check (you can customize this)
            # For example, check if gripper is closed and lifted
            if step > 100 and obs['gripper_state'] < 0.2:
                tcp_height = obs['tcp_pose'][2]
                if tcp_height > 0.3:  # Lifted above a threshold
                    print(f"\n✓ Task completed at step {step}")
                    done = True

    except KeyboardInterrupt:
        print("\nUser interrupted")
    except Exception as e:
        print(f"\n✗ Error during episode: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nEpisode finished after {step} steps")

    return {
        'steps': step,
        'success': done,
    }


def main(args: Args):
    """Main function."""
    print("=" * 60)
    print("GraspVLA + ManiSkill Integration")
    print("=" * 60)

    # Create environment
    print("\nCreating ManiSkill environment...")
    env: BaseEnv = create_maniskill_env(
        env_id=args.env_id,
        control_mode="pd_joint_pos",
        num_envs=1,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        shader=args.shader,
        enable_shadow=False,
        seed=args.seed,
    )

    # Remove default objects
    remove_default_objects(env)

    # Set robot base pose (if specified)
    if args.robot_position is not None:
        print("\nSetting robot base pose...")
        set_robot_base_pose(env, args.robot_position, args.robot_rotation)

    # Load object
    print("\n" + "=" * 60)
    if args.use_articulation:
        print("Loading articulation object...")
        print("=" * 60)
        if not args.articulation_id:
            raise ValueError("Must specify --articulation-id when using --use-articulation")

        load_articulation(
            env=env,
            model_id=args.articulation_id,
            dataset=args.articulation_dataset,
            position=args.object_position,
            rotation_deg=args.object_rotation
        )
    else:
        print("Loading regular object...")
        print("=" * 60)
        print(f"  Mesh file: {args.object_mesh_path}")
        print(f"  Position: {args.object_position}")
        print(f"  Rotation (degrees): {args.object_rotation}")

        object_config = load_single_object(
            mesh_path=args.object_mesh_path,
            position=args.object_position,
            rotation_deg=args.object_rotation
        )

        loader = ObjectLoader(env.unwrapped.scene)
        loader.load_object(object_config)
        print(f"✓ Loaded: {object_config.name}\n")

    # Setup cameras (with robot coordinate frame)
    print("\nSetting up cameras...")
    robot = env.unwrapped.agent
    cameras = setup_cameras(
        env.unwrapped.scene,
        GRASPVLA_CAMERA_VIEWS,
        args.shader,
        args.image_width,
        args.image_height,
        robot=robot  # Pass robot for coordinate transformation
    )
    print(f"✓ Cameras ready: {list(cameras.keys())}")

    # Hide goal markers
    if args.hide_goal:
        hide_goal_markers(env)

    # Initialize IK solver
    print("\nInitializing IK solver...")
    kinematics = initialize_ik_solver(env)
    print("✓ IK solver ready")

    # Initialize GraspVLA agent
    print("\n" + "=" * 60)
    print("Connecting to GraspVLA policy server...")
    print("=" * 60)
    print(f"  Port: {args.port}")
    print(f"  Instruction: '{args.instruction}'")

    try:
        agent = RemoteAgent(
            instruction=args.instruction,
            port=args.port,
            kinematics=kinematics
        )
        print("✓ Connected to policy server")
    except Exception as e:
        print(f"\n✗ Failed to connect to policy server: {e}")
        print("\nMake sure the GraspVLA server is running:")
        print(f"  python others/GraspVLA/vla_network/scripts/serve.py \\")
        print(f"    --port {args.port} \\")
        print(f"    --path <path_to_model>")
        env.close()
        return

    # Prepare output directory
    output_dir = Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run episode
        stats = run_episode(env, agent, cameras, kinematics, output_dir, args)

        # Save results
        print("\n" + "=" * 60)
        print("Saving results...")
        print("=" * 60)
        print(f"✓ Images: {output_dir / 'images'}")

        # Save camera parameters
        camera_params = save_camera_params(cameras, output_dir)
        print(f"✓ Camera params: {output_dir / 'camera_params'}")
        print(f"  - {len(camera_params)} camera views")

        # Generate videos
        if args.save_video:
            generate_videos(output_dir, list(GRASPVLA_CAMERA_VIEWS.keys()), args.video_fps)

        print(f"\n✓ Episode statistics:")
        print(f"  - Steps: {stats['steps']}")
        print(f"  - Success: {stats['success']}")

    finally:
        agent.close()
        env.close()
        print("\n✓ Done!")

        # Force exit to avoid segfault during cleanup
        os._exit(0)


if __name__ == "__main__":
    main(tyro.cli(Args))
