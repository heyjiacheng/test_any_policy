"""
Remote Robot Control Agent for ManiSkill Environment

This module implements a remote robot control agent that communicates with the GraspVLA
policy server via ZMQ to execute robot manipulation tasks in ManiSkill simulation.

The agent processes visual observations from ManiSkill cameras, maintains proprioceptive
history, and converts model predictions into robot actions.
"""

import numpy as np
import transforms3d as t3d
import zmq
from collections import deque
from typing import Dict, Tuple, Optional, Any
import numpy.typing as npt


class RemoteAgent:
    """Agent that communicates with GraspVLA policy server for ManiSkill environment."""

    # Constants
    PROPRIO_HISTORY_SIZE = 4
    GRIPPER_OPEN = 1.0
    GRIP_TRANSITION_ACTIONS = 4

    def __init__(self, instruction: str, port: int, kinematics=None) -> None:
        """Initialize the RemoteAgent.

        Args:
            instruction: Task instruction text (e.g., "pick up the mug")
            port: ZMQ server port number
            kinematics: ManiSkill IK solver for forward kinematics
        """
        self._validate_inputs(instruction, port)
        self._setup_zmq_connection(port)
        self._initialize_state(instruction)
        self.kinematics = kinematics

    def _validate_inputs(self, instruction: str, port: int) -> None:
        """Validate initialization parameters."""
        if not instruction.strip():
            raise ValueError("Instruction cannot be empty")
        if not (1 <= port <= 65535):
            raise ValueError(f"Port must be between 1-65535, got {port}")

    def _setup_zmq_connection(self, port: int) -> None:
        """Set up ZMQ connection to model server."""
        try:
            self.zmq_context = zmq.Context()
            self.socket = self.zmq_context.socket(zmq.REQ)
            self.socket.connect(f"tcp://127.0.0.1:{port}")
            # Set socket timeout to prevent hanging
            self.socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 second timeout
        except Exception as e:
            raise ConnectionError(f"Failed to establish ZMQ connection: {e}")

    def _initialize_state(self, instruction: str) -> None:
        """Initialize agent state variables."""
        self.proprio_history = deque(maxlen=self.PROPRIO_HISTORY_SIZE)
        self.instruction = instruction
        self.pred_actions = deque()
        self.finger_state = self.GRIPPER_OPEN

    def get_current_proprio(self, tcp_pose: np.ndarray, gripper_state: float = 0.0) -> npt.NDArray[np.float64]:
        """Get the current proprioceptive state of the robot.

        Note: Following LIBERO implementation, we use the internally tracked finger_state
        rather than the observed gripper_state. This ensures consistency with the action
        commands sent to the robot.

        Args:
            tcp_pose: Tool center point pose [x, y, z, qw, qx, qy, qz]
            gripper_state: Current gripper state (0=closed, 1=open in ManiSkill convention)
                          Note: This parameter is kept for API compatibility but not used.

        Returns:
            current_proprio: Array of shape (7,) containing [x, y, z, rx, ry, rz, gripper_state].
        """
        # Extract position and quaternion
        position = tcp_pose[:3]
        quaternion = tcp_pose[3:]  # [qw, qx, qy, qz]

        # Convert quaternion to euler angles (in 'sxyz' convention)
        # Note: transforms3d expects [qw, qx, qy, qz] format
        euler = t3d.euler.quat2euler(quaternion, axes='sxyz')

        # Use internally tracked finger_state (matching LIBERO implementation)
        # finger_state is already in GraspVLA convention: -1=closed, 1=open
        current_proprio = np.concatenate([
            position,
            euler,
            np.array([self.finger_state])
        ])
        return current_proprio

    def step(self, obs: Dict[str, Any], debug: bool = False) -> Tuple[npt.NDArray[np.float64], Optional[Any]]:
        """Execute one step of the robot control loop.

        This method processes the current observation, maintains proprioceptive history,
        and either executes a cached action or requests new actions from the model server.

        Args:
            obs: Observation dictionary containing:
                - 'front_view_image': np.ndarray of shape (H, W, 3)
                - 'side_view_image': np.ndarray of shape (H, W, 3)
                - 'tcp_pose': np.ndarray of shape (7,) [x, y, z, qw, qx, qy, qz]
                - 'gripper_state': float (0-1, where 0=closed, 1=open)
            debug: Whether to print debug information

        Returns:
            Tuple containing:
                - action: Robot action array (7,) [x, y, z, rx, ry, rz, gripper]
                - bbox: Bounding box information for visualization (optional)
        """
        self._process_proprio(obs)

        # If the last action chunk is all executed, request new actions
        if len(self.pred_actions) == 0:
            self._post_and_get(obs, debug=debug)

        action, bbox = self.pred_actions.popleft()

        # Convert gripper action back to ManiSkill convention
        # GraspVLA: -1 (close), 0 (no change), 1 (open)
        # ManiSkill: 0 (close), 1 (open)
        if action[6] == -1:
            action[6] = 0.0  # Close
        elif action[6] == 1:
            action[6] = 1.0  # Open
        else:  # action[6] == 0 (no change)
            # Use internally tracked finger_state instead of observed gripper_state
            # This prevents gripper oscillation from observation noise
            # Convert finger_state from GraspVLA convention (-1/1) to ManiSkill (0/1)
            action[6] = (self.finger_state + 1) / 2

        return action, bbox

    def _process_proprio(self, obs: Dict[str, Any]) -> None:
        """Process and update the proprioceptive history buffer.

        Maintains a rolling history of the robot's proprioception. If the history is shorter
        than maxlen, it pads with copies of the most recent observation.

        Args:
            obs: Observation dictionary
        """
        current_proprio = self.get_current_proprio(obs['tcp_pose'], obs['gripper_state'])
        self.proprio_history.append(current_proprio)

        while len(self.proprio_history) < self.proprio_history.maxlen:
            self.proprio_history.append(self.proprio_history[-1])

    def _post_and_get(self, obs: Dict[str, Any], debug: bool = False) -> None:
        """Send observation data to the model server and convert delta actions to absolute actions."""
        # Prepare data in GraspVLA format
        data = {
            'front_view_image': [obs["front_view_image"]],
            'side_view_image': [obs["side_view_image"]],
            'proprio_array': [np.copy(proprio) for proprio in self.proprio_history],
            'text': self.instruction,
        }

        # Send request and receive response
        self.socket.send_pyobj(data)
        response = self.socket.recv_pyobj()

        # Extract debug info (bbox for visualization)
        bbox = response['debug'].get('bbox', None)

        # Convert delta actions to absolute actions
        last_finger_state = self.finger_state
        current_pose = self.proprio_history[-1][:6]

        for delta_action in response['result']:
            abs_action = self._delta_to_abs(delta_action, current_pose)
            current_pose = abs_action[:6]

            if abs_action[6] == 0 or abs_action[6] == last_finger_state:
                # Gripper state doesn't change
                self.pred_actions.append([abs_action, bbox])
            else:
                # Gripper state is changing - split into arm movement + grip actions
                arm_action = np.copy(abs_action)
                arm_action[6] = 0
                self.pred_actions.append([arm_action, bbox])

                # Add multiple grip transition actions for smooth gripper control
                for _ in range(self.GRIP_TRANSITION_ACTIONS):
                    self.pred_actions.append([np.copy(abs_action), bbox])

                self.finger_state = abs_action[6]

            last_finger_state = abs_action[6] if abs_action[6] != 0 else last_finger_state

        if debug:
            print('-' * 40)
            print('Proprio transition:',
                  [round(p, 4) for p in data['proprio_array'][-4][:3]],
                  [round(p, 4) for p in data['proprio_array'][-1][:3]])
            print('Proprio gripper:',
                  data['proprio_array'][-4][-1],
                  data['proprio_array'][-1][-1])

            actions = response['result'][1::2]
            z_actions = [round(action[2], 4) for action in actions]
            gripper_actions = [action[6] for action in actions]
            print('Z actions:', z_actions)
            print('Gripper actions:', gripper_actions)

    def _delta_to_abs(self, delta_action: np.ndarray,
                     current_pose: np.ndarray) -> np.ndarray:
        """Convert delta action to absolute action.

        Args:
            delta_action: Array of shape (7,) representing delta pose
                         [dx, dy, dz, drx, dry, drz, gripper_state]
            current_pose: Array of shape (6,) representing current absolute pose
                         [x, y, z, rx, ry, rz]

        Returns:
            abs_action: Array of shape (7,) representing absolute pose
                       [x, y, z, rx, ry, rz, gripper_state]
        """
        # Convert current euler angles to rotation matrix
        current_rot = t3d.euler.euler2mat(*current_pose[3:6])

        # Convert delta euler angles to rotation matrix and compose
        delta_rot = t3d.euler.euler2mat(*delta_action[3:6])
        next_rot = delta_rot @ current_rot

        # Add delta translation to current position
        current_trans = current_pose[:3]
        next_trans = current_trans + delta_action[:3]

        # Combine into absolute action
        new_action = np.concatenate([
            next_trans,
            t3d.euler.mat2euler(next_rot),
            [delta_action[6]]
        ])
        return new_action

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'zmq_context'):
            self.zmq_context.term()
