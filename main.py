from arm import Arm
import numpy as np
from tqdm import trange

N_EPISODES = 5
STEPS_PER_EPISODE = 250

arm = Arm(xml_path="assets/arm.xml")

for episode in range(N_EPISODES):
    print(f"> Episode: {episode}")

    for step in range(STEPS_PER_EPISODE):
        action = np.random.uniform(-1, 1, size=2)
        obs, reward, done, info = arm.step(action=action)
        print(f"{arm.get_time():>16.4}s: {info}")

        # arm.render()

        if done:
            arm.reset()
            break

    arm.reset()