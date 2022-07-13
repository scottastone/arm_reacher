"""Mujoco environment to control a two-link planar arm with two targets."""

from typing import Tuple, Sequence, Dict, Any
import numpy as np
import cv2
from dm_control import mujoco

class Arm():
    """Environment to allow control of a two-link planar arm in mujoco.

    Adapted version of reacher in dm_control suite (https://github.com/deepmind/dm_control).
    """

    def __init__(self,
                 xml_path: str,
                 target_names: Sequence[str] = ["left_target", "right_target"],
                 max_steps_per_episode: int = 600,
                 ) -> None:
        """
        Initialize Arm class with .xml file and target names.

        Args:
            xml_path: Path to xml file with arm and targets.
            target_names: Names of each target in xml file.
            max_steps_per_episode: Max env steps before Done == True.

        Returns:
            None
        """
        self._xml_path = xml_path
        self._physics = mujoco.Physics.from_xml_path(self._xml_path)

        self._target_names = target_names
        self._target_distance = dict.fromkeys(self._target_names)
        self._target_pos = dict.fromkeys(self._target_names)
        self._target_contact = dict.fromkeys(self._target_names)

        self.steps_taken = 0
        self.max_steps_per_episode = max_steps_per_episode
        self._JOINT_ANGLE = 0.47612 # NOTE: This value makes it so we can centre the hand at the starting point. Don't know why.

        self.reset()

    def render(self,
        update_rate: float = 1/60,
        width: int = 640,
        height: int = 320,
        camera_id: str = "fixed",
        ) -> None:
        """Optionally render arm in a window (to be called on every step)."""
        pixels = self._physics.render(height=height, width=width, camera_id=camera_id)
        cv2.imshow("arm", pixels)
        cv2.moveWindow("arm", 0, 0)
        cv2.waitKey(int(update_rate * 1000))

    def _is_contacting_target(self):
        """Update self._target_contact with booleans for finger contacting each target."""
        for target_name in self._target_names:
            target_pos = self._physics.named.data.geom_xpos[target_name, :2]
            finger_pos = self._physics.named.data.geom_xpos['finger', :2]
            self._target_distance[target_name] = np.linalg.norm(target_pos - finger_pos)
            radii = self._physics.named.model.geom_size[[target_name, 'finger'], 0].sum()
            if self._target_distance[target_name] < radii:
                self._target_contact[target_name] = True
            else:
                self._target_contact[target_name] = False

    def step(
        self,
        action: np.ndarray
        ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Step the mujoco environment.

        Args:
            action: Two joint torque forces between -1 and +1.

        Returns:
            obs: Dictionary of numpy arrays for agent input.
            reward: Scalar reward value for this step.
            done: Boolean for terminal episode state.
            info: Dictionary of episode information.
        """
        self._physics.set_control(control=action)
        self._physics.step()
        self._is_contacting_target()

        self.steps_taken += 1

        obs = self.get_observation()
        if self.steps_taken >= self.max_steps_per_episode:
            done = True
        else:
            done = False
        reward = None
        info = {
            "left_target_contact": self._target_contact["left_target"],
            "right_target_contact": self._target_contact["right_target"],
            "steps_taken": self.steps_taken,
        }

        return obs, reward, done, info

    def get_observation(self) -> Dict[str, np.ndarray]:
        """Returns environment observations for agent input."""
        obs = {}
        obs["shoulder_angle_sin"] =     np.sin(self._physics.named.data.xmat['arm'][1])
        obs["shoulder_angle_cos"] =     np.cos(self._physics.named.data.xmat['arm'][1])
        obs["wrist_angle_sin"] =        np.sin(self._physics.named.data.xmat['hand'][1])
        obs["wrist_angle_cos"] =        np.cos(self._physics.named.data.xmat['hand'][1])
        obs["shoulder_velocity"] =      self._physics.named.data.qvel['shoulder'][0]
        obs["wrist_velocity"] =         self._physics.named.data.qvel['wrist'][0]
        return obs

    def get_time(self) -> float:
        """Returns physics simulation time in seconds from episode start."""
        return self._physics.time()

    def _initialize_episode(self) -> None:
        """Set joint angles at start of episode so finger is on the start position."""
        self._physics.named.data.qpos['shoulder'] = self._JOINT_ANGLE / 2
        self._physics.named.data.qpos['wrist'] = np.pi - self._JOINT_ANGLE

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and physics simulation."""
        with self._physics.reset_context():
            self._initialize_episode()
        self.steps_taken = 0

        return self.get_observation()
