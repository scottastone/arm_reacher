from dm_control import mujoco
import matplotlib.pyplot as plt
import numpy as np

class Arm():
    def __init__(self,
                 xml_path,
                 target_names=["left_target", "right_target"],):
        self._xml_path = xml_path
        self._physics = mujoco.Physics.from_xml_path(self._xml_path)

        self._target_names = target_names
        self._target_distance = dict.fromkeys(self._target_names)
        self._target_pos = dict.fromkeys(self._target_names)
        self._target_contact = dict.fromkeys(self._target_names)

        self.reset()

    def render(self, update_rate=0.01, width=640, height=320, camera_id="fixed"):
        pixels = self._physics.render(height=height, width=width, camera_id=camera_id)
        plt.imshow(pixels)
        plt.show(block=False)
        plt.pause(update_rate)
        plt.clf()
        plt.cla()

    def _is_contacting_target(self):
        for idx, target_name in enumerate(self._target_names):
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
        return 0
    
    def step(self, action):
        self._physics.set_control(control=action)
        self._physics.step()
        self._is_contacting_target()

        obs = self.get_observation()
        reward = self.get_reward()
        done = False    # TODO
        info = {}       # TODO

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

    def reset(self):
        with self._physics.reset_context():
            self.initialize_episode()

        return self._physics.position()

    def initialize_episode(self):
        # set the target size
        self._physics.named.model.geom_size['left_target', 0] = .015
        self._physics.named.model.geom_size['right_target', 0] = .015

        # set target position
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(.05, .20)
         
        self._physics.named.model.geom_pos['left_target', 'x']  =-0.05#radius * np.sin(angle)
        self._physics.named.model.geom_pos['left_target', 'y']  = 0.15
        self._physics.named.model.geom_pos['right_target', 'x'] = 0.05#radius * np.sin(angle)
        self._physics.named.model.geom_pos['right_target', 'y'] = 0.15#radius * np.cos(angle)
