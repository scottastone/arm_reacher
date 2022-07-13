from dm_control import mujoco
import numpy as np
import cv2
import dm_control

class Arm():
    def __init__(self,
                 xml_path,
                 target_names=["left_target", "right_target"],
                 max_steps_per_episode=600,):
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

    def render(self, update_rate=1/60, width=640, height=320, camera_id="fixed"):
        pixels = self._physics.render(height=height, width=width, camera_id=camera_id)
        cv2.imshow("arm", pixels)
        cv2.moveWindow("arm", 0, 0)
        cv2.waitKey(int(update_rate * 1000))

    def _is_contacting_target(self):
        for target_name in self._target_names:
            target_pos = self._physics.named.data.geom_xpos[target_name, :2]
            finger_pos = self._physics.named.data.geom_xpos['finger', :2]
            self._target_distance[target_name] = np.linalg.norm(target_pos - finger_pos)
            radii = self._physics.named.model.geom_size[[target_name, 'finger'], 0].sum()
            if self._target_distance[target_name] < radii:
                self._target_contact[target_name] = True
            else:
                self._target_contact[target_name] = False

    def get_reward(self):
        # check if we are touching the correct target
        # TODO: figure out what target we should be reaching for
        return 0
    
    def step(self, action):
        self._physics.set_control(control=action)
        self._physics.step()
        self._is_contacting_target()

        self.steps_taken += 1

        obs = self.get_observation()
        if self.steps_taken >= self.max_steps_per_episode:
            done = True
        else: 
            done = False
        reward = self.get_reward()
        info = {
            "left_target_contact": self._target_contact["left_target"],
            "right_target_contact": self._target_contact["right_target"],
        } 

        return obs, reward, done, info

    def get_observation(self):
        obs = {}
        obs["shoulder_angle_sin"] =     np.sin(self._physics.named.data.xmat['arm'][1])
        obs["shoulder_angle_cos"] =     np.cos(self._physics.named.data.xmat['arm'][1])
        obs["wrist_angle_sin"] =        np.sin(self._physics.named.data.xmat['hand'][1])
        obs["wrist_angle_cos"] =        np.cos(self._physics.named.data.xmat['hand'][1])
        obs["shoulder_velocity"] =      self._physics.named.data.qvel['shoulder'][0]
        obs["wrist_velocity"] =         self._physics.named.data.qvel['wrist'][0]
        return obs

    def _initialize_episode(self):
        self._physics.named.data.qpos['shoulder'] = self._JOINT_ANGLE / 2
        self._physics.named.data.qpos['wrist'] = np.pi - self._JOINT_ANGLE

    def reset(self):
        with self._physics.reset_context():
            self._initialize_episode()
        self.steps_taken = 0

        return self.get_observation()
